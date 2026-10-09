from typing import Dict, Type

from .base import Oracle, OracleResult
from .composite import CompositeOracle
from .qed import ExplainableQedOracle
from .similarity import SimilarityOracle

try:
    from .boltz2 import Boltz2Oracle
    _boltz2_available = True
except Exception:
    _boltz2_available = False

try:
    from .ic50mpro import IC50MproOracle
    _ic50mpro_available = True
except Exception:
    _ic50mpro_available = False

try:
    from .novel import NovelOracle
    _novel_available = True
except Exception:
    _novel_available = False

try:
    from .opioid_ki import MolEncoderOpioidKiOracle
    _opioid_ki_available = True
except Exception:
    _opioid_ki_available = False

try:
    from .tdc_tasks import TdcOracle
    _tdc_available = True
except Exception:
    _tdc_available = False

from ..config import OracleConfig


ORACLE_REGISTRY: Dict[str, Type[Oracle]] = {
    "qed": ExplainableQedOracle,
    "similarity": SimilarityOracle,
    "composite": CompositeOracle,
    **({"boltz2": Boltz2Oracle} if _boltz2_available else {}),
    **({"ic50mpro": IC50MproOracle} if _ic50mpro_available else {}),
    **({"novel": NovelOracle} if _novel_available else {}),
    **({"opioid_ki": MolEncoderOpioidKiOracle} if _opioid_ki_available else {}),
    **({"tdc_tasks": TdcOracle} if _tdc_available else {}),
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
