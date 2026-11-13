"""Small, validated YAML/JSON experiment configuration; no framework required."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
import json

from .metrics import DEFAULT_METRICS, validate_metric_groups


def _construct(cls, data: dict):
    if not isinstance(data, dict):
        raise ValueError(f'{cls.__name__} must be an object')
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} settings: {sorted(unknown)}')
    return cls(**data)


@dataclass(frozen=True)
class DatasetConfig:
    type: str
    path: str
    cohort: str = 'strict_no_missing_states'
    id_column: str = 'session_id'
    states_column: str = 'states'
    timestamps_column: str | None = None

    def validate(self):
        if self.type not in {'decode', 'curated', 'generic'}:
            raise ValueError(f'Invalid dataset type: {self.type}')
        if not self.path:
            raise ValueError('Dataset path is required')
        if self.type == 'decode' and self.cohort not in {'strict_no_missing_states', 'drop_none_states'}:
            raise ValueError(f'Invalid DECODE cohort: {self.cohort}')


@dataclass(frozen=True)
class ModelConfig:
    type: str
    name: str
    path: str | None = None
    source_model_name: str | None = None
    model_id: str | None = None
    adapter_path: str | None = None
    prompt_template: str | None = None
    prompt_version: str = 'v1'
    max_new_tokens: int = 512
    device_map: str | None = None

    def validate(self):
        if self.type not in {'baseline', 'hf', 'file'}:
            raise ValueError(f'Unknown model type: {self.type}')
        if not self.name:
            raise ValueError('Model name is required')
        if self.type == 'hf' and not self.model_id:
            raise ValueError(f'{self.name}: hf model requires model_id')
        if self.type == 'file' and not self.path:
            raise ValueError(f'{self.name}: file predictor requires path')
        if self.max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')


@dataclass(frozen=True)
class EvalOptions:
    horizons: tuple[int, ...] = (1, 2, 5)
    metrics: tuple[str, ...] = DEFAULT_METRICS
    persistence_min_consecutive: int = 2

    def validate(self):
        if not self.horizons or not all(isinstance(i, int) and i > 0 for i in self.horizons):
            raise ValueError('horizons must be positive integers')
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError('Duplicate horizons')
        if self.persistence_min_consecutive < 1:
            raise ValueError('persistence_min_consecutive must be >= 1')
        validate_metric_groups(self.metrics)


@dataclass(frozen=True)
class OutputConfig:
    directory: str = 'results/evaluation'
    save_predictions: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    experiment: str
    dataset: DatasetConfig
    models: tuple[ModelConfig, ...]
    evaluation: EvalOptions = field(default_factory=EvalOptions)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self):
        if not self.experiment:
            raise ValueError('experiment name must be nonempty')
        self.dataset.validate()
        self.evaluation.validate()
        if not self.models:
            raise ValueError('At least one model required')
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError('Model names must be unique')
        for model in self.models:
            model.validate()


def load_config(path: str | Path) -> EvaluationConfig:
    path = Path(path)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        try:
            import yaml
        except ImportError as error:
            raise ImportError('Install PyYAML to read YAML configs') from error
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    elif path.suffix.lower() == '.json':
        raw = json.loads(path.read_text(encoding='utf-8'))
    else:
        raise ValueError('Configuration must be YAML or JSON')
    if not isinstance(raw, dict):
        raise ValueError('Experiment configuration must be a mapping')
    allowed = {'experiment', 'dataset', 'models', 'evaluation', 'output'}
    if set(raw) - allowed:
        raise ValueError(f'Unknown top-level keys: {sorted(set(raw)-allowed)}')
    if 'dataset' not in raw or 'models' not in raw:
        raise ValueError('dataset and models are required')
    opts = dict(raw.get('evaluation', {}))
    opts['horizons'] = tuple(opts.get('horizons', (1, 2, 5)))
    opts['metrics'] = tuple(opts.get('metrics', DEFAULT_METRICS))
    cfg = EvaluationConfig(
        experiment=str(raw.get('experiment', path.stem)),
        dataset=_construct(DatasetConfig, raw['dataset']),
        models=tuple(_construct(ModelConfig, row) for row in raw['models']),
        evaluation=_construct(EvalOptions, opts),
        output=_construct(OutputConfig, raw.get('output', {})),
    )
    cfg.validate()
    return cfg
