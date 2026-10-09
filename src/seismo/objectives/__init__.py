from typing import Dict, Type

from ..config import ObjectiveConfig
from ..oracles.base import Oracle
from .base import Objective
from .boltz2 import Boltz2Objective
from .ic50mpro import IC50MproObjective
from .ic50mpro_novel import IC50MproNovelObjective
from .ic50mpro_qed_novel import IC50MproQedNovelObjective
from .novel import NovelObjective
from .qed import QedObjective
from .similarity import SimilarityObjective
from .similarity_qed import SimilarityQedObjective

try:
    from .opioid_ki import OpioidKiObjective

    _opioid_ki_available = True
except Exception:
    _opioid_ki_available = False

try:
    from .tdc_tasks import TdcObjective

    _tdc_available = True
except Exception:
    _tdc_available = False


OBJECTIVE_REGISTRY: Dict[str, Type[Objective]] = {
    "boltz2": Boltz2Objective,
    "ic50mpro": IC50MproObjective,
    "ic50mpro_novel": IC50MproNovelObjective,
    "ic50mpro_qed_novel": IC50MproQedNovelObjective,
    "novel": NovelObjective,
    "qed": QedObjective,
    "similarity": SimilarityObjective,
    "similarity_qed": SimilarityQedObjective,
    **({"opioid_ki": OpioidKiObjective} if _opioid_ki_available else {}),
    **({"tdc_tasks": TdcObjective} if _tdc_available else {}),
}


def build_objective_from_config(cfg: ObjectiveConfig, oracle: Oracle) -> Objective:
    if cfg.name not in OBJECTIVE_REGISTRY:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown objective name '{cfg.name}'. Available objectives: {available}"
        )

    cls = OBJECTIVE_REGISTRY[cfg.name]
    try:
        return cls(oracle=oracle, **cfg.params)
    except TypeError as e:
        raise ValueError(
            f"Failed to construct objective '{cfg.name}' with params {cfg.params}: {e}"
        ) from e
