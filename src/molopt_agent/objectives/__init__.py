from typing import Dict, Type

from .base import Objective
from .ic50mpro import IC50MproObjective
from .ic50mpro_novel import IC50MproNovelObjective
from .ic50mpro_qed_novel import IC50MproQedNovelObjective
from .novel import NovelObjective
from .opioid_ki import OpioidKiObjective
from .qed import QedObjective
from .similarity import SimilarityObjective
from .similarity_qed import SimilarityQedObjective
from .tdc_tasks import TdcObjective
from ..config import ObjectiveConfig
from ..oracles.base import Oracle


OBJECTIVE_REGISTRY: Dict[str, Type[Objective]] = {
    "ic50mpro": IC50MproObjective,
    "ic50mpro_novel": IC50MproNovelObjective,
    "ic50mpro_qed_novel": IC50MproQedNovelObjective,
    "novel": NovelObjective,
    "opioid_ki": OpioidKiObjective,
    "qed": QedObjective,
    "similarity": SimilarityObjective,
    "similarity_qed": SimilarityQedObjective,
    "tdc_tasks": TdcObjective,
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
