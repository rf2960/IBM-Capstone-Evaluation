"""INACTIVE SCAFFOLD: dataset-namespaced stable identifiers.

Current ids remain '<session_id>::a<anchor>::h<horizon>' for compatibility.
A future scheme should namespace dataset/version and normalized trajectory
identity; migrating IDs will require a versioned join/crosswalk policy.
"""


def make_versioned_example_id(dataset_id: str, trajectory_id: str,
                              anchor: int, horizon: int) -> str:
    raise NotImplementedError('Scaffold only: preserve v1 IDs until migration policy is tested')
