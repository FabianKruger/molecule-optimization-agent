from typing import Protocol, TypedDict


class OracleResult(TypedDict, total=False, extra_items=str):
    """
    Minimal oracle output:
      - score: scalar the optimizer uses (higher/lower as you define).
      - explanation: short human-readable explanation.
    """

    score: float
    explanation: str


class Oracle(Protocol):
    def __call__(self, smiles: str) -> OracleResult: ...
