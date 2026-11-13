"""INACTIVE SCAFFOLD: centralized eligibility and failure-policy validation.

Current frozen scoring semantics remain in metrics.py: invalid generations
are unscored; future states are censored according to existing rules.
Future work: explicit policy for short histories, timestamp anomalies,
unknown status strings, unsupported horizons, and missing/empty predictions.
"""
from dataclasses import dataclass
from evaluation.schemas import Session, PredictionRecord


@dataclass(frozen=True)
class ValidityIssue:
    code: str
    example_id: str
    message: str
    fatal: bool = False


def audit_inputs(sessions: list[Session], predictions: list[PredictionRecord]) -> list[ValidityIssue]:
    raise NotImplementedError('Scaffold only: do not alter frozen eligibility rules yet')
