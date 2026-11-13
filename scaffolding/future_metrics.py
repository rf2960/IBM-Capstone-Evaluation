"""INACTIVE SCAFFOLD: future@k, anticipation, edit survival, terminal similarity.

Register each implementation via evaluation.metrics.register_metric_group.
Each group must define per-task fields and its aggregation denominator / censoring.
Do not reuse frozen stable_f1 for a gap-tolerant or time-normalized variant.
"""
import pandas as pd


def score_future(shared: dict) -> dict:
    raise NotImplementedError('Define future-state matching / target choice first')


def aggregate_future(group: pd.DataFrame, scored: pd.DataFrame) -> dict:
    raise NotImplementedError('Define eligibility, censoring and pooling first')

# When definitions are frozen and tested, register explicitly:
# register_metric_group('future_v1', score=score_future, aggregate=aggregate_future)
