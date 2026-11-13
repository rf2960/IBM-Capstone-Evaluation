from __future__ import annotations

from difflib import SequenceMatcher
from statistics import mean
from typing import Callable
import numpy as np
from scipy.optimize import linear_sum_assignment
from .schemas import Edit


def clean_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def line_set(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def whole_file_line_jaccard(left: str, right: str) -> float:
    return jaccard(line_set(left), line_set(right))


def line_edit_cost(source: str, destination: str) -> int:
    """Count added plus deleted lines in a shortest line-level diff.

    A replacement counts both the removed source lines and inserted destination
    lines. This makes oversized rewrites visibly more expensive than a compact
    patch while remaining language agnostic.
    """
    source_lines = clean_lines(source)
    destination_lines = clean_lines(destination)
    matcher = SequenceMatcher(None, source_lines, destination_lines, autojunk=False)
    cost = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            cost += (i2 - i1) + (j2 - j1)
        elif tag == "delete":
            cost += i2 - i1
        elif tag == "insert":
            cost += j2 - j1
    return cost


def extract_edits(source: str, destination: str) -> list[Edit]:
    source_lines = clean_lines(source)
    destination_lines = clean_lines(destination)
    matcher = SequenceMatcher(None, source_lines, destination_lines, autojunk=False)
    edits: list[Edit] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        payload = (
            tuple(destination_lines[j1:j2])
            if tag in {"insert", "replace"}
            else tuple(source_lines[i1:i2])
        )
        edits.append(Edit(tag=tag, start=i1, end=i2, payload=payload))
    return edits


def location_similarity(left: Edit, right: Edit, tolerant: bool = False) -> float:
    distance = abs(left.anchor - right.anchor)
    if distance == 0:
        return 1.0
    if tolerant and distance == 1:
        return 0.5
    return 0.0


def payload_similarity(left: Edit, right: Edit) -> float:
    left_text = "\n".join(line.strip() for line in left.payload if line.strip())
    right_text = "\n".join(line.strip() for line in right.payload if line.strip())
    if left.tag == right.tag == "delete" and not left_text and not right_text:
        return 1.0
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()


def edit_match_score(left: Edit, right: Edit, tolerant: bool = False) -> float:
    return location_similarity(left, right, tolerant=tolerant) * payload_similarity(left, right)


def operation_similarity(left: Edit, right: Edit) -> float:
    return float(left.tag == right.tag)


def greedy_match_credit(
    predicted: list[Edit],
    observed: list[Edit],
    scorer: Callable[[Edit, Edit], float],
) -> float:
    """One-to-one maximum-first credit under a supplied edit comparison."""
    candidates = sorted(
        (
            (scorer(predicted[p], observed[o]), p, o)
            for p in range(len(predicted))
            for o in range(len(observed))
        ),
        reverse=True,
    )
    used_predicted: set[int] = set()
    used_observed: set[int] = set()
    credit = 0.0
    for score, p_index, o_index in candidates:
        if score <= 0:
            break
        if p_index in used_predicted or o_index in used_observed:
            continue
        used_predicted.add(p_index)
        used_observed.add(o_index)
        credit += score
    return credit


def optimal_match_credit(
    predicted: list[Edit],
    observed: list[Edit],
    scorer: Callable[[Edit, Edit], float],
) -> float:
    """Maximum total one-to-one edit credit via linear assignment."""
    if not predicted or not observed:
        return 0.0
    scores = np.array(
        [
            [scorer(predicted_edit, observed_edit) for observed_edit in observed]
            for predicted_edit in predicted
        ],
        dtype=float,
    )
    predicted_indices, observed_indices = linear_sum_assignment(-scores)
    return float(scores[predicted_indices, observed_indices].sum())


def prf_from_credit(
    credit: float,
    predicted_count: int,
    observed_count: int,
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 with an explicit zero-division policy."""
    precision = credit / predicted_count if predicted_count else float("nan")
    recall = credit / observed_count if observed_count else float("nan")
    denominator = predicted_count + observed_count
    f1 = 2 * credit / denominator if denominator else float("nan")
    return precision, recall, f1


def soft_edit_prf(
    predicted: list[Edit],
    observed: list[Edit],
    *,
    tolerant: bool = False,
) -> tuple[float, float, float]:
    if not predicted and not observed:
        return 1.0, 1.0, 1.0
    if not predicted or not observed:
        return 0.0, 0.0, 0.0

    credit = optimal_match_credit(
        predicted,
        observed,
        lambda left, right: edit_match_score(
            left, right, tolerant=tolerant
        ),
    )
    return prf_from_credit(credit, len(predicted), len(observed))


def edit_type(edits: list[Edit]) -> str:
    if not edits:
        return "no_edit"
    tags = {edit.tag for edit in edits}
    return next(iter(tags)) if len(tags) == 1 else "mixed"


def persistent_edits(
    edits: list[Edit],
    future_states: list[str],
    min_consecutive: int,
) -> list[Edit]:
    """Edits retained in the target and required following observed states."""
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")
    if len(future_states) < min_consecutive:
        return []
    window = future_states[:min_consecutive]
    return [
        edit
        for edit in edits
        if all(
            predicted_edit_retention([edit], state) >= 1.0 - 1e-12
            for state in window
        )
    ]


def predicted_edit_retention(predicted: list[Edit], future_state: str) -> float:
    """Check whether predicted payload survives near its original edit location.

    This deliberately does not penalize additional unrelated future edits.
    A two-line location window allows small shifts caused by nearby insertions.
    """
    if not predicted:
        return 1.0
    future_lines = clean_lines(future_state)
    edit_scores = []
    for edit in predicted:
        payload = [line.strip() for line in edit.payload if line.strip()]
        start = max(0, edit.anchor - 2)
        stop = min(len(future_lines), edit.anchor + max(1, len(payload)) + 3)
        window = [line.strip() for line in future_lines[start:stop] if line.strip()]
        if not payload:
            edit_scores.append(1.0)
            continue
        similarities = [float(line in window) for line in payload]
        payload_score = mean(similarities)
        edit_scores.append(1.0 - payload_score if edit.tag == "delete" else payload_score)
    return mean(edit_scores)


def durability_score(
    current: str,
    prediction: str,
    future_states: list[str],
    decay: float = 0.8,
) -> float:
    predicted_edits = extract_edits(current, prediction)
    if not predicted_edits:
        return float("nan")
    future_states = future_states[:5]
    weights = [decay**index for index in range(len(future_states))]
    scores = [predicted_edit_retention(predicted_edits, future) for future in future_states]
    return sum(weight * score for weight, score in zip(weights, scores)) / sum(weights)

