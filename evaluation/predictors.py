from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
import pandas as pd
from .edits import extract_edits
from .schemas import PredictionRecord, PredictionRequest
from .constants import PREDICTION_REQUIRED_COLUMNS


def middle_delta(previous: str, current: str) -> tuple[int, int, int, str, str]:
    """Return the one contiguous character span changed from previous to current."""
    prefix = 0
    while prefix < min(len(previous), len(current)) and previous[prefix] == current[prefix]:
        prefix += 1
    suffix = 0
    max_suffix = min(len(previous) - prefix, len(current) - prefix)
    while suffix < max_suffix and previous[-1 - suffix] == current[-1 - suffix]:
        suffix += 1
    previous_end = len(previous) - suffix if suffix else len(previous)
    current_end = len(current) - suffix if suffix else len(current)
    old_middle = previous[prefix:previous_end]
    new_middle = current[prefix:current_end]
    return prefix, previous_end, current_end, old_middle, new_middle


def repeat_last_edit(previous: str, current: str) -> str:
    prefix, _, current_end, old_middle, new_middle = middle_delta(previous, current)
    if not old_middle and new_middle:
        return current[:current_end] + new_middle + current[current_end:]
    replacement_end = min(prefix + len(old_middle), len(current))
    return current[:prefix] + new_middle + current[replacement_end:]


def repeat_last_edit_category(previous: str, current: str) -> str:
    """Explain whether and why the repeat heuristic changes the current code."""
    if previous == current:
        return "no_prior_change"
    prediction = repeat_last_edit(previous, current)
    if extract_edits(current, prediction):
        return "proposes_line_edit"
    if prediction != current:
        return "whitespace_only_change"
    _, _, _, old_middle, new_middle = middle_delta(previous, current)
    if old_middle and new_middle and len(old_middle) == len(new_middle):
        return "idempotent_replacement"
    if old_middle and not new_middle:
        return "deletion_replay_noop"
    return "other_noop"


def predict_no_change(task: dict) -> str:
    return task["current"]


def predict_repeat_last_edit(task: dict) -> str:
    return repeat_last_edit(task["previous"], task["current"])


BASELINE_PREDICTORS = {
    "no_change": predict_no_change,
    "repeat_last_edit": predict_repeat_last_edit,
}



class Predictor(Protocol):
    """Adapters return the complete predicted code state, never a diff."""
    def predict(self, request: PredictionRequest) -> PredictionRecord: ...


@dataclass
class BaselinePredictor:
    name: str

    def __post_init__(self):
        if self.name not in BASELINE_PREDICTORS:
            raise ValueError(f'Unknown baseline: {self.name}')

    def predict(self, request: PredictionRequest) -> PredictionRecord:
        text = BASELINE_PREDICTORS[self.name](request.as_input_dict())
        return PredictionRecord(request.example_id, self.name, request.horizon, text)


class FilePredictor:
    """Reuse an existing prediction file; missing rows are marked missing."""
    def __init__(self, name: str, path: str | Path, source_model_name: str | None = None):
        self.name = name
        path = Path(path)
        if path.suffix == '.parquet':
            predictions = pd.read_parquet(path)
        elif path.suffix == '.csv':
            predictions = pd.read_csv(path, keep_default_na=False)
        elif path.suffix in {'.jsonl', '.ndjson'}:
            predictions = pd.read_json(path, lines=True)
        else:
            raise ValueError('Prediction file must be CSV, JSONL, or Parquet')
        missing = set(PREDICTION_REQUIRED_COLUMNS) - set(predictions.columns)
        if missing:
            raise ValueError(f'Missing prediction columns: {sorted(missing)}')
        if source_model_name is not None:
            predictions = predictions[predictions['model_name'] == source_model_name]
        if predictions.duplicated(['example_id', 'horizon']).any():
            raise ValueError('Duplicate example_id/horizon in selected prediction file')
        self.records = {(str(r['example_id']), int(r['horizon'])): r
                        for r in predictions.to_dict('records')}

    def predict(self, request: PredictionRequest) -> PredictionRecord:
        record = self.records.get((request.example_id, request.horizon))
        if record is None:
            return PredictionRecord(request.example_id, self.name, request.horizon,
                                    None, generation_status='missing')
        status = str(record['generation_status'])
        value = record['prediction']
        if not isinstance(value, str):
            value = None
        return PredictionRecord(request.example_id, self.name, request.horizon,
                                value, status, str(record.get('model_version', '')),
                                str(record.get('prompt_version', '')))


class HuggingFacePredictor:
    """Local causal-LM inference; optional LoRA/PEFT adapter (optional dependencies).

    Prompt/output conventions are deliberately explicit and minimal. The model
    must produce a *full next code state*, not a patch or a chat transcript.
    """
    DEFAULT_TEMPLATE = (
        'Given the previous and current versions of a code file, predict the '
        'complete next version. Output only the full code, without markdown.\n\n'
        'PREVIOUS:\n{previous}\n\nCURRENT:\n{current}\n\nNEXT:\n'
    )

    def __init__(self, name: str, model_id: str, *, adapter_path: str | None = None,
                 prompt_template: str | None = None, max_new_tokens: int = 512,
                 device_map: str | None = None, prompt_version: str = 'v1',
                 trust_remote_code: bool = False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError('HF inference requires: pip install -e ".[models]"') from error
        self.torch = torch
        self.name = name
        self.model_id = model_id
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE
        self.max_new_tokens = max_new_tokens
        self.prompt_version = prompt_version
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        kwargs = {'trust_remote_code': trust_remote_code}
        if device_map:
            kwargs['device_map'] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise ImportError('LoRA loading requires: pip install -e ".[models]"') from error
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.model_version = f'{model_id}+{adapter_path}' if adapter_path else model_id

    def predict(self, request: PredictionRequest) -> PredictionRecord:
        prompt = self.prompt_template.format(
            previous=request.previous, current=request.current,
            history='\n\n'.join(request.history), horizon=request.horizon,
        )
        tokens = self.tokenizer(prompt, return_tensors='pt')
        tokens = {key: value.to(self.model.device) for key, value in tokens.items()}
        with self.torch.inference_mode():
            out = self.model.generate(
                **tokens, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
            )
        # Decode only newly generated tokens, excluding the prompt.
        text = self.tokenizer.decode(
            out[0, tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        return PredictionRecord(request.example_id, self.name, request.horizon,
                                text, model_version=self.model_version,
                                prompt_version=self.prompt_version)


def load_predictors(configs) -> dict[str, Predictor]:
    """Create adapters once per run; models are not reloaded per example."""
    predictors = {}
    for cfg in configs:
        get = cfg.get if isinstance(cfg, dict) else lambda k, default=None: getattr(cfg, k, default)
        kind, name = get('type'), get('name')
        if name in predictors:
            raise ValueError(f'Duplicate model name: {name}')
        if kind == 'baseline':
            predictor = BaselinePredictor(name)
        elif kind == 'file':
            predictor = FilePredictor(name, get('path'), get('source_model_name'))
        elif kind == 'hf':
            predictor = HuggingFacePredictor(
                name, get('model_id'), adapter_path=get('adapter_path'),
                prompt_template=get('prompt_template'),
                max_new_tokens=get('max_new_tokens', 512),
                device_map=get('device_map'),
                prompt_version=get('prompt_version', 'v1'),
            )
        else:
            raise ValueError(f'Unknown predictor type: {kind}')
        predictors[name] = predictor
    return predictors
