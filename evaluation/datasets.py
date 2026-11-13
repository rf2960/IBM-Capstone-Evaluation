from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
from .schemas import Session


def load_sessions(frame: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    strict_sessions: list[dict] = []
    drop_none_sessions: list[dict] = []
    state_type_counts: dict[str, int] = {}
    rows_with_non_string = 0
    rows_with_short_state_list = 0
    total_non_string_elements = 0
    none_positions = {"leading": 0, "interior": 0, "trailing": 0}

    for row in frame.sort_values("outcome_completion_id").itertuples(index=False):
        states = list(row.states)
        non_string_types = [type(state).__name__ for state in states if not isinstance(state, str)]
        for name in non_string_types:
            state_type_counts[name] = state_type_counts.get(name, 0) + 1
        if non_string_types:
            rows_with_non_string += 1
            total_non_string_elements += len(non_string_types)
            none_positions["leading"] += int(states[0] is None)
            none_positions["trailing"] += int(states[-1] is None)
            none_positions["interior"] += int(any(state is None for state in states[1:-1]))
        if len(states) < 3:
            rows_with_short_state_list += 1
        if len(states) >= 3 and not non_string_types:
            strict_sessions.append(
                {
                    "session_id": row.outcome_completion_id,
                    "user_id": row.user_id,
                    "source_model": row.model,
                    "outcome_timestamp": row.outcome_timestamp,
                    "states": states,
                }
            )
        string_states = [state for state in states if isinstance(state, str)]
        if len(string_states) >= 3:
            drop_none_sessions.append(
                {
                    "session_id": row.outcome_completion_id,
                    "user_id": row.user_id,
                    "source_model": row.model,
                    "outcome_timestamp": row.outcome_timestamp,
                    "states": string_states,
                }
            )

    audit = {
        "raw_trajectory_groups": int(len(frame)),
        "total_snapshots": int(frame["n_snapshots"].sum()),
        "total_states_including_s0": int((frame["n_snapshots"] + 1).sum()),
        "unique_users": int(frame["user_id"].nunique()),
        "unique_source_models": int(frame["model"].nunique()),
        "snapshot_count_mean": float(frame["n_snapshots"].mean()),
        "snapshot_count_median": float(frame["n_snapshots"].median()),
        "snapshot_count_max": int(frame["n_snapshots"].max()),
        "states_equals_s0_plus_snapshots_for_all_rows": bool(
            all(
                list(row.states) == [row.s0] + list(row.snapshots)
                for row in frame.itertuples(index=False)
            )
        ),
        "strict_eligible_sessions": len(strict_sessions),
        "eligible_after_dropping_none": len(drop_none_sessions),
        "additional_sessions_if_none_is_dropped": len(drop_none_sessions) - len(strict_sessions),
        "rows_with_fewer_than_3_states": rows_with_short_state_list,
        "rows_with_non_string_state": rows_with_non_string,
        "total_non_string_state_elements": total_non_string_elements,
        "non_string_element_types": state_type_counts,
        "rows_with_none_by_position": none_positions,
        "eligibility_rule": "at least 3 states and every state is a string",
    }
    return strict_sessions, drop_none_sessions, audit


def load_segmented_sessions(
    frame: pd.DataFrame,
    *,
    id_column: str = "segment_id",
    states_column: str = "states",
) -> list[dict]:
    """Load a future normalized/segmented table without changing metric code."""
    sessions = []
    for row in frame.to_dict("records"):
        states = list(row[states_column])
        if len(states) >= 3 and all(isinstance(state, str) for state in states):
            sessions.append({"session_id": row[id_column], "states": states})
    return sessions



def _read_table(path: str | Path) -> pd.DataFrame:
    """Read *already constructed trajectories*, never preprocess raw event data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in {'.jsonl', '.ndjson'}:
        return pd.read_json(path, lines=True)
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'sessions' in data:
            data = data['sessions']
        if not isinstance(data, list):
            raise ValueError('JSON input must contain a list of trajectories')
        return pd.DataFrame(data)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported dataset format: {path.suffix}')


def _as_sequence(value: object, label: str) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f'{label} must be a JSON array, not a plain string') from error
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{label} must be an array')
    return list(value)


def load_curated(
    path: str | Path, *, id_column: str = 'session_id',
    states_column: str = 'states', timestamps_column: str | None = None,
) -> list[Session]:
    """Load normalized JSONL/JSON/CSV/Parquet. Does not curate or segment data."""
    frame = _read_table(path)
    if not {id_column, states_column}.issubset(frame.columns):
        raise ValueError(f'Curated dataset requires {id_column!r} and {states_column!r}')
    sessions = []
    for row in frame.to_dict('records'):
        states = _as_sequence(row[states_column], 'states')
        timestamps = (
            _as_sequence(row[timestamps_column], 'timestamps')
            if timestamps_column and timestamps_column in row and row[timestamps_column] is not None
            else None
        )
        extras = {k: v for k, v in row.items()
                  if k not in {id_column, states_column, timestamps_column}}
        sessions.append(Session.from_legacy({
            'session_id': row[id_column], 'states': states,
            'timestamps': timestamps, 'metadata': extras,
        }))
    _validate_unique_ids(sessions)
    return sessions


def _validate_unique_ids(sessions: list[Session]) -> None:
    seen = set()
    for session in sessions:
        if session.session_id in seen:
            raise ValueError(f'Duplicate session_id: {session.session_id}')
        seen.add(session.session_id)


def load_dataset(config) -> list[Session]:
    """Common entry point. config may be a DatasetConfig or plain mapping.

    DECODE tables are assumed preprocessed into per-trajectory state lists.
    Evaluation cohorts are selected here, independently of later metrics.
    """
    get = config.get if isinstance(config, dict) else lambda k, default=None: getattr(config, k, default)
    kind = get('type')
    path = get('path')
    cohort = get('cohort', 'strict_no_missing_states')
    if kind == 'decode':
        if cohort not in {'strict_no_missing_states', 'drop_none_states'}:
            raise ValueError(f'Unknown DECODE cohort: {cohort}')
        frame = _read_table(path)
        strict, drop_none, _audit = load_sessions(frame)
        chosen = strict if cohort == 'strict_no_missing_states' else drop_none
        sessions = [Session.from_legacy(item) for item in chosen]
    elif kind in {'curated', 'generic'}:
        sessions = load_curated(
            path, id_column=get('id_column', 'session_id'),
            states_column=get('states_column', 'states'),
            timestamps_column=get('timestamps_column'),
        )
    else:
        raise ValueError(f'Unknown dataset type: {kind}')
    _validate_unique_ids(sessions)
    return sessions
