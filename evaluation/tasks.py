from __future__ import annotations

import pandas as pd
from .edits import extract_edits
from .predictors import repeat_last_edit_category
from .schemas import Session


def build_tasks(sessions: list[dict], horizon: int):
    for session in sessions:
        states = session.states if isinstance(session, Session) else session["states"]
        session_info = session.to_legacy() if isinstance(session, Session) else session
        for anchor in range(1, len(states) - horizon):
            example_id = f"{session_info['session_id']}::a{anchor}::h{horizon}"
            yield {
                "example_id": example_id,
                "session_id": session_info["session_id"],
                "user_id": session_info.get("user_id", ""),
                "source_model": session_info.get("source_model", ""),
                "outcome_timestamp": session_info.get("outcome_timestamp", ""),
                "anchor": anchor,
                "horizon": horizon,
                "previous": states[anchor - 1],
                "current": states[anchor],
                "target": states[anchor + horizon],
                "future": states[anchor + horizon :],
                "history": tuple(states[:anchor + 1]),
            }


def build_model_inputs(sessions: list[dict], horizons: tuple[int, ...]) -> pd.DataFrame:
    """Export only information available at prediction time."""
    rows = []
    for horizon in horizons:
        for task in build_tasks(sessions, horizon):
            rows.append(
                {
                    "example_id": task["example_id"],
                    "session_id": task["session_id"],
                    "anchor": task["anchor"],
                    "horizon": horizon,
                    "previous": task["previous"],
                    "current": task["current"],
                }
            )
    return pd.DataFrame(rows)


def repeat_activity_audit(sessions: list[dict], horizon: int = 1) -> pd.DataFrame:
    """Count why repeat-last-edit does or does not propose a new edit."""
    tasks = list(build_tasks(sessions, horizon))
    counts: dict[str, int] = {}
    for task in tasks:
        category = repeat_last_edit_category(task["previous"], task["current"])
        counts[category] = counts.get(category, 0) + 1
    order = [
        "proposes_line_edit",
        "no_prior_change",
        "idempotent_replacement",
        "deletion_replay_noop",
        "whitespace_only_change",
        "other_noop",
    ]
    return pd.DataFrame(
        [
            {
                "horizon": horizon,
                "category": category,
                "count": counts.get(category, 0),
                "rate": counts.get(category, 0) / len(tasks) if tasks else float("nan"),
            }
            for category in order
            if counts.get(category, 0)
        ]
    )


def is_exact_one_step_reversion(task: dict) -> bool:
    """True when the state after an edited h=1 target returns exactly to current."""
    return (
        task["target"] != task["current"]
        and len(task["future"]) >= 2
        and task["future"][1] == task["current"]
    )

