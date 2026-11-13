"""Small data contracts. DataFrames remain suitable for analysis/aggregation.

Legacy scoring functions accept dictionaries; EvalTask.to_legacy() provides that
bridge without exposing target/future information to a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Edit:
    tag: str
    start: int
    end: int
    payload: tuple[str, ...]

    @property
    def anchor(self) -> int:
        return self.start


@dataclass(frozen=True)
class Session:
    session_id: str
    states: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamps: tuple[str, ...] | None = None

    @classmethod
    def from_legacy(cls, item: Mapping[str, Any]) -> 'Session':
        if 'session_id' not in item or 'states' not in item:
            raise ValueError('Session requires session_id and states')
        meta = dict(item.get('metadata', {}))
        for key, value in item.items():
            if key not in {'session_id', 'states', 'metadata', 'timestamps'}:
                meta[key] = value
        stamps = item.get('timestamps')
        session = cls(str(item['session_id']), tuple(item['states']), meta,
                      tuple(stamps) if stamps is not None else None)
        session.validate()
        return session

    def validate(self) -> None:
        if not self.session_id:
            raise ValueError('session_id must not be empty')
        if len(self.states) < 3:
            raise ValueError(f'{self.session_id}: requires at least 3 states')
        if not all(isinstance(s, str) for s in self.states):
            raise ValueError(f'{self.session_id}: states must all be strings')
        if self.timestamps is not None and len(self.timestamps) != len(self.states):
            raise ValueError(f'{self.session_id}: timestamps must align with states')

    def to_legacy(self) -> dict:
        return {'session_id': self.session_id, 'states': list(self.states),
                **dict(self.metadata)}


@dataclass(frozen=True)
class PredictionRequest:
    example_id: str
    session_id: str
    anchor: int
    horizon: int
    previous: str
    current: str
    history: tuple[str, ...]
    # No target, future or outcome_timestamp: hard inference/reference boundary.

    def as_input_dict(self) -> dict:
        """A restricted dictionary for existing callable baseline predictors."""
        return {
            'example_id': self.example_id, 'session_id': self.session_id,
            'anchor': self.anchor, 'horizon': self.horizon,
            'previous': self.previous, 'current': self.current,
            'history': self.history,
        }


@dataclass(frozen=True)
class EvalTask:
    example_id: str
    session_id: str
    anchor: int
    horizon: int
    previous: str
    current: str
    target: str
    future: tuple[str, ...]
    history: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, task: Mapping[str, Any]) -> 'EvalTask':
        return cls(
            example_id=str(task['example_id']), session_id=str(task['session_id']),
            anchor=int(task['anchor']), horizon=int(task['horizon']),
            previous=task['previous'], current=task['current'], target=task['target'],
            future=tuple(task['future']), history=tuple(task.get('history', (task['previous'], task['current']))),
            metadata={k: task.get(k, '') for k in ('user_id', 'source_model', 'outcome_timestamp')},
        )

    def prediction_request(self) -> PredictionRequest:
        return PredictionRequest(
            self.example_id, self.session_id, self.anchor, self.horizon,
            self.previous, self.current, self.history,
        )

    def to_legacy(self) -> dict:
        return {
            'example_id': self.example_id, 'session_id': self.session_id,
            'anchor': self.anchor, 'horizon': self.horizon,
            'previous': self.previous, 'current': self.current,
            'target': self.target, 'future': list(self.future),
            'history': self.history, **dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionRecord:
    example_id: str
    model_name: str
    horizon: int
    prediction: str | None
    generation_status: str = 'ok'
    model_version: str = ''
    prompt_version: str = ''

    def as_dict(self) -> dict:
        return {
            'example_id': self.example_id, 'model_name': self.model_name,
            'horizon': self.horizon, 'prediction': self.prediction,
            'generation_status': self.generation_status,
            'model_version': self.model_version, 'prompt_version': self.prompt_version,
        }
