from typing import Dict, Type

from .base import Objective
from .opioid_ki import OpioidKiObjective
from .pubchem_novelty import PubChemNoveltyObjective
from .qed import QedObjective
from .tanimoto_similarity import TanimotoSimilarityObjective
from ..config import ObjectiveConfig
from ..oracles.base import Oracle


OBJECTIVE_REGISTRY: Dict[str, Type[Objective]] = {
    "opioid_ki": OpioidKiObjective,
    "pubchem_novelty": PubChemNoveltyObjective,
    "qed": QedObjective,
    "tanimoto_similarity": TanimotoSimilarityObjective,
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
