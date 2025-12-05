from typing import Dict, Type

from .base import Oracle, OracleResult
from .opioid_ki import MolEncoderOpioidKiOracle
from .pubchem_novelty import PubChemNoveltyOracle
from .qed import ExplainableQedOracle
from .tanimoto_similarity import TanimotoSimilarityOracle
from .xgboost_sarscov2 import XGBoostMaccsSARSCoV2Oracle
from ..config import OracleConfig


ORACLE_REGISTRY: Dict[str, Type[Oracle]] = {
    "opioid_ki": MolEncoderOpioidKiOracle,
    "pubchem_novelty": PubChemNoveltyOracle,
    "qed": ExplainableQedOracle,
    "tanimoto_similarity": TanimotoSimilarityOracle,
    "xgboost_sarscov2":XGBoostMaccsSARSCoV2Oracle,
}


def build_oracle_from_config(cfg: OracleConfig) -> Oracle:
    if cfg.name not in ORACLE_REGISTRY:
        available = ", ".join(sorted(ORACLE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown oracle '{cfg.name}'. Available: {available}"
        )
    cls = ORACLE_REGISTRY[cfg.name]
    try:
        return cls(**cfg.params)
    except TypeError as e:
        raise ValueError(
            f"Failed to construct oracle '{cfg.name}' with params {cfg.params}: {e}"
        ) from e


__all__ = ["Oracle", "OracleResult", "build_oracle_from_config", "ORACLE_REGISTRY"]

