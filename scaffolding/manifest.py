"""INACTIVE SCAFFOLD: full experiment provenance manifest.

The runner currently saves effective_config.json and evaluation_protocol.json.
A future verified manifest should include content fingerprints, source code
revision, environment versions, seeds, resolved model revision, prompt/parser
versions, timestamps and input/output artifact hashes.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RunManifest:
    protocol_version: str
    dataset_fingerprint: str
    model_revision: str
    config_hash: str
    code_revision: str
    seed: int | None = None


def create_verified_manifest(*args, **kwargs) -> RunManifest:
    raise NotImplementedError('Scaffold only: fingerprints and provenance not yet verified')
