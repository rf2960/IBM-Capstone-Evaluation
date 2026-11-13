"""INACTIVE SCAFFOLD: prediction caching (not wired into runner).

Cache key should include dataset fingerprint, example ID, model/checkpoint
revision, prompt version, generation settings and parser version. Never use
example ID alone, and do not cache failed generations indefinitely by default.
"""
from typing import Protocol
from evaluation.schemas import PredictionRecord, PredictionRequest


class PredictionCache(Protocol):
    def get(self, key: str) -> PredictionRecord | None: ...
    def put(self, key: str, record: PredictionRecord) -> None: ...


def build_cache_key(request: PredictionRequest, model_spec: dict,
                    dataset_fingerprint: str) -> str:
    raise NotImplementedError('Scaffold only: define canonical serialization and hashing')
