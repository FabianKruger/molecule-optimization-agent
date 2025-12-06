from typing import Dict, Type

from .base import Objective
from .ic50_and_novel import IC50AndNovelObjective
from .ic50_qed_novel import IC50QedNovelObjective
from .opioid_ki import OpioidKiObjective
from .pubchem_novelty import PubChemNoveltyObjective
from .qed import QedObjective
from .similarity_and_qed import SimilarityAndQedObjective
from .tanimoto_similarity import TanimotoSimilarityObjective
from .xgboost_sarscov2 import XGBoostSARSCoV2Objective
from ..config import ObjectiveConfig
from ..oracles.base import Oracle


OBJECTIVE_REGISTRY: Dict[str, Type[Objective]] = {
    "ic50_and_novel": IC50AndNovelObjective,
    "ic50_qed_novel": IC50QedNovelObjective,
    "opioid_ki": OpioidKiObjective,
    "pubchem_novelty": PubChemNoveltyObjective,
    "qed": QedObjective,
    "similarity_and_qed": SimilarityAndQedObjective,
    "tanimoto_similarity": TanimotoSimilarityObjective,
    "xgboost_sarscov2": XGBoostSARSCoV2Objective,
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
