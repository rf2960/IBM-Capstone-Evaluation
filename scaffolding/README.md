# Inactive high-priority scaffolds

These files document **interfaces and decisions still pending**, not completed
features. They are intentionally outside the installed `evaluation` package and
are never invoked by the CLI. Some functions explicitly raise `NotImplementedError`.

- `cache.py`: cache protocol; requires key design and invalidation policy.
- `manifest.py`: verified run provenance (only config/protocol copies are saved now).
- `validity.py`: explicit audit-policy object, without changing current scoring.
- `ids.py`: versioned dataset-namespaced example IDs, after migration plan.
- `future_metrics.py`: placeholders for new research metrics; definitions TBD.
- `tests/test_golden_fixture.py`: skipped golden fixture snapshot placeholder.

The runnable smoke tests and legacy comparison tests verify the **core refactor**;
these are separate from the deferred additional assurance systems.
