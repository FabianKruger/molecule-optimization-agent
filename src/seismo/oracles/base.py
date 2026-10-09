from typing import Protocol, TypedDict


class OracleScore(TypedDict, total=False):
    """
    Single score field of an OracleResult.
    """

    value: float
    explanation: str


# Keys should be a "what is this" description of the score.
OracleResult = dict[str, OracleScore]


class Oracle(Protocol):
    def __call__(self, smiles: str) -> OracleResult: ...
