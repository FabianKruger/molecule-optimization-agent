from typing import Dict, Type

from .base import Oracle, OracleResult
from .composite import CompositeOracle
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
    "composite": CompositeOracle,
}


def build_oracle_from_config(cfg: OracleConfig) -> Oracle:
    # Special handling for composite oracle with nested sub-oracles
    if cfg.name == "composite":
        oracles_cfg = cfg.params.get("oracles", [])
        if not oracles_cfg:
            raise ValueError("CompositeOracle requires 'oracles' list in params")
        
        weights = cfg.params.get("weights")
        if not weights:
            raise ValueError("CompositeOracle requires 'weights' list in params")
        
        sub_oracles = [
            build_oracle_from_config(OracleConfig(name=sub["name"], params=sub.get("params", {})))
            for sub in oracles_cfg
        ]
        names = cfg.params.get("names")
        return CompositeOracle(oracles=sub_oracles, weights=weights, names=names)

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

