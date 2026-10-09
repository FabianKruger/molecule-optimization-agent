"""Fixed preset catalog for the web PoC job API."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from seismo.config import ExperimentConfig, LLMConfig
from seismo.objectives.base import Objective
from seismo.objectives.boltz2 import Boltz2Objective
from seismo.objectives.ic50mpro import IC50MproObjective
from seismo.objectives.ic50mpro_novel import IC50MproNovelObjective
from seismo.objectives.ic50mpro_qed_novel import IC50MproQedNovelObjective
from seismo.objectives.novel import NovelObjective
from seismo.objectives.qed import QedObjective
from seismo.objectives.similarity import SimilarityObjective
from seismo.objectives.similarity_qed import SimilarityQedObjective
from seismo.oracles.boltz2 import Boltz2Oracle
from seismo.oracles.composite import CompositeOracle
from seismo.oracles.ic50mpro import IC50MproOracle
from seismo.oracles.novel import NovelOracle
from seismo.oracles.qed import ExplainableQedOracle
from seismo.oracles.similarity import SimilarityOracle

DEFAULT_PROTEIN_SEQUENCE = "GSGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
QUERCETIN_SMILES = "O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O"
IC50_MODEL = "data/xgboost/xgb_maccs_best"

_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

FieldType = Literal["text", "textarea", "number", "select"]


@dataclass(frozen=True)
class PresetField:
    name: str
    label: str
    type: FieldType
    default: Any
    required: bool = True
    minimum: float | int | None = None
    maximum: float | int | None = None
    options: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "required": self.required,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.options is not None:
            payload["options"] = self.options
        return payload


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    description: str
    higher_is_better: bool
    score_label: str
    fields: tuple[PresetField, ...]
    builder: Callable[[dict[str, Any], Path], tuple[Objective, ExperimentConfig]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "higher_is_better": self.higher_is_better,
            "score_label": self.score_label,
            "fields": [field.to_dict() for field in self.fields],
        }


def _max_iterations_field(*, default: int, maximum: int) -> PresetField:
    return PresetField(
        name="max_iterations",
        label="Max iterations",
        type="number",
        default=default,
        required=True,
        minimum=1,
        maximum=maximum,
    )


def _llm_config() -> LLMConfig:
    import os

    return LLMConfig(
        model=os.environ.get("SEISMO_MODEL", "gpt-5.1"),
        temperature=0.1,
        reasoning_effort=None,
    )


def _experiment_config(name: str, max_iterations: int) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=name,
        llm=_llm_config(),
        recursion_limit=max(max_iterations * 8, 100),
        log_dir="data/runs",
    )


def _require_number(
    params: dict[str, Any], name: str, *, default: float | None = None
) -> float:
    if name not in params or params[name] is None or params[name] == "":
        if default is not None:
            return float(default)
        raise ValueError(f"Missing required parameter: {name}")
    try:
        return float(params[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {name} must be a number") from exc


def _require_int(
    params: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if name not in params or params[name] is None or params[name] == "":
        value = default
    else:
        try:
            value = int(params[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Parameter {name} must be an integer") from exc
        if float(params[name]) != value:
            raise ValueError(f"Parameter {name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"Parameter {name} must be between {minimum} and {maximum}")
    return value


def _require_text(
    params: dict[str, Any],
    name: str,
    *,
    default: str | None = None,
    required: bool = True,
) -> str:
    if name not in params or params[name] is None:
        if default is not None:
            return default
        if not required:
            return ""
        raise ValueError(f"Missing required parameter: {name}")
    value = str(params[name])
    if required and not value.strip():
        raise ValueError(f"Parameter {name} must not be empty")
    return value


def validate_smiles(smiles: str) -> None:
    # SimilarityOracle parses with RDKit; reuse it so this module does not
    # import rdkit.Chem (incomplete stubs confuse ty).
    try:
        SimilarityOracle(target_smiles=smiles)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def validate_protein_sequence(sequence: str) -> None:
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("Protein sequence must not be empty")
    invalid = sorted({ch for ch in cleaned if ch not in _AA_ALPHABET})
    if invalid:
        raise ValueError(
            "Protein sequence contains non-standard amino-acid characters: "
            + ", ".join(invalid)
        )


def boltz2_score_label(score_mode: str) -> str:
    if score_mode == "affinity_pred_value":
        return "Predicted affinity"
    return "Binding probability"


def _build_boltz2(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    protein = _require_text(
        params, "protein_sequence", default=DEFAULT_PROTEIN_SEQUENCE
    )
    validate_protein_sequence(protein)
    protein = "".join(protein.split()).upper()

    target = _require_number(params, "target_binding_probability", default=0.999)
    score_mode = _require_text(
        params, "score_mode", default="affinity_probability_binary"
    )
    if score_mode not in {"affinity_probability_binary", "affinity_pred_value"}:
        raise ValueError("Invalid score_mode")
    additional = _require_text(
        params, "additional_information", default="", required=False
    )
    max_iterations = _require_int(
        params, "max_iterations", default=5, minimum=1, maximum=15
    )

    oracle = Boltz2Oracle(
        protein_sequence=protein,
        output_dir_base=job_dir / "boltz",
        binding_score_name=score_mode,
        timeout=300,
        with_explanation=True,
        transform_probability="none",
    )
    objective = Boltz2Objective(
        oracle=oracle,
        target_binding_probability=target,
        max_iterations=max_iterations,
        additional_information=additional or None,
        hide_protein=False,
    )
    return objective, _experiment_config("boltz2", max_iterations)


def _build_qed(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    target = _require_number(params, "target_qed", default=0.9)
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    objective = QedObjective(
        ExplainableQedOracle(),
        target_qed=target,
        max_iterations=max_iterations,
    )
    return objective, _experiment_config("qed", max_iterations)


def _build_similarity(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    target_smiles = _require_text(params, "target_smiles", default=ASPIRIN_SMILES)
    validate_smiles(target_smiles)
    target_similarity = _require_number(params, "target_similarity", default=0.9)
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    oracle = SimilarityOracle(target_smiles=target_smiles)
    objective = SimilarityObjective(
        oracle,
        target_similarity=target_similarity,
        max_iterations=max_iterations,
    )
    return objective, _experiment_config("similarity", max_iterations)


def _build_similarity_qed(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    target_smiles = _require_text(params, "target_smiles", default=QUERCETIN_SMILES)
    validate_smiles(target_smiles)
    target_score = _require_number(params, "target_score", default=0.75)
    min_similarity = _require_number(params, "min_similarity", default=0.6)
    min_qed = _require_number(params, "min_qed", default=0.5)
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    oracle = CompositeOracle(
        oracles=[
            SimilarityOracle(target_smiles=target_smiles),
            ExplainableQedOracle(),
        ],
        components=[
            {"oracle": 0, "score": "Similarity", "weight": 0.5},
            {"oracle": 1, "score": "QED", "weight": 0.5},
        ],
    )
    objective = SimilarityQedObjective(
        oracle,
        target_score=target_score,
        min_similarity=min_similarity,
        min_qed=min_qed,
        max_iterations=max_iterations,
        xai="full",
    )
    return objective, _experiment_config("similarity_qed", max_iterations)


def _build_novel(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    objective = NovelObjective(NovelOracle(), max_iterations=max_iterations)
    return objective, _experiment_config("novel", max_iterations)


def _build_ic50mpro(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    target = _require_number(params, "target_ic50_nM", default=10)
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    oracle = IC50MproOracle(model_name=IC50_MODEL)
    objective = IC50MproObjective(
        oracle, target_ic50_nM=target, max_iterations=max_iterations
    )
    return objective, _experiment_config("ic50mpro", max_iterations)


def _build_ic50mpro_novel(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    target = _require_number(params, "target_ic50_nM", default=10)
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    oracle = CompositeOracle(
        oracles=[IC50MproOracle(model_name=IC50_MODEL), NovelOracle()],
        components=[
            {"oracle": 0, "score": "IC50", "weight": 1.0},
            {"oracle": 1, "score": "Novelty", "weight": 0.0},
        ],
    )
    objective = IC50MproNovelObjective(
        oracle,
        target_ic50_nM=target,
        max_iterations=max_iterations,
        xai="full",
    )
    return objective, _experiment_config("ic50mpro_novel", max_iterations)


def _build_ic50mpro_qed_novel(
    params: dict[str, Any], job_dir: Path
) -> tuple[Objective, ExperimentConfig]:
    del job_dir
    target = _require_number(params, "target_ic50_nM", default=10)
    min_qed = _require_number(params, "min_qed", default=0.5)
    max_iterations = _require_int(
        params, "max_iterations", default=20, minimum=1, maximum=30
    )
    oracle = CompositeOracle(
        oracles=[
            IC50MproOracle(model_name=IC50_MODEL),
            ExplainableQedOracle(),
            NovelOracle(),
        ],
        components=[
            {"oracle": 0, "score": "IC50", "weight": 1.0},
            {"oracle": 1, "score": "QED", "weight": 0.0},
            {"oracle": 2, "score": "Novelty", "weight": 0.0},
        ],
    )
    objective = IC50MproQedNovelObjective(
        oracle,
        target_ic50_nM=target,
        min_qed=min_qed,
        max_iterations=max_iterations,
        xai="full",
    )
    return objective, _experiment_config("ic50mpro_qed_novel", max_iterations)


PRESETS: tuple[Preset, ...] = (
    Preset(
        id="boltz2",
        title="Boltz-2 binding",
        description="Optimize a ligand for predicted binding to a protein sequence.",
        higher_is_better=True,
        score_label="Binding probability",
        fields=(
            PresetField(
                name="protein_sequence",
                label="Protein sequence",
                type="textarea",
                default=DEFAULT_PROTEIN_SEQUENCE,
                required=True,
            ),
            PresetField(
                name="target_binding_probability",
                label="Target binding probability",
                type="number",
                default=0.999,
                required=True,
            ),
            PresetField(
                name="score_mode",
                label="Score mode",
                type="select",
                default="affinity_probability_binary",
                required=True,
                options=[
                    {
                        "value": "affinity_probability_binary",
                        "label": "Binding probability",
                    },
                    {
                        "value": "affinity_pred_value",
                        "label": "Predicted affinity",
                    },
                ],
            ),
            PresetField(
                name="additional_information",
                label="Additional information",
                type="textarea",
                default="",
                required=False,
            ),
            _max_iterations_field(default=5, maximum=15),
        ),
        builder=_build_boltz2,
    ),
    Preset(
        id="qed",
        title="QED",
        description="Maximize the Quantitative Estimate of Drug-likeness (QED).",
        higher_is_better=True,
        score_label="QED",
        fields=(
            PresetField(
                name="target_qed",
                label="Target QED",
                type="number",
                default=0.9,
                required=True,
            ),
            _max_iterations_field(default=20, maximum=30),
        ),
        builder=_build_qed,
    ),
    Preset(
        id="similarity",
        title="Similarity",
        description="Maximize Tanimoto similarity to a target molecule.",
        higher_is_better=True,
        score_label="Tanimoto similarity",
        fields=(
            PresetField(
                name="target_smiles",
                label="Target SMILES",
                type="text",
                default=ASPIRIN_SMILES,
                required=True,
            ),
            PresetField(
                name="target_similarity",
                label="Target similarity",
                type="number",
                default=0.9,
                required=True,
            ),
            _max_iterations_field(default=20, maximum=30),
        ),
        builder=_build_similarity,
    ),
    Preset(
        id="similarity_qed",
        title="Similarity + QED",
        description="Balance structural similarity and drug-likeness.",
        higher_is_better=True,
        score_label="Combined score",
        fields=(
            PresetField(
                name="target_smiles",
                label="Target SMILES",
                type="text",
                default=QUERCETIN_SMILES,
                required=True,
            ),
            PresetField(
                name="target_score",
                label="Target combined score",
                type="number",
                default=0.75,
                required=True,
            ),
            PresetField(
                name="min_similarity",
                label="Minimum similarity",
                type="number",
                default=0.6,
                required=True,
            ),
            PresetField(
                name="min_qed",
                label="Minimum QED",
                type="number",
                default=0.5,
                required=True,
            ),
            _max_iterations_field(default=20, maximum=30),
        ),
        builder=_build_similarity_qed,
    ),
    Preset(
        id="novel",
        title="Novelty",
        description="Find molecules that are absent from PubChem.",
        higher_is_better=True,
        score_label="Novelty",
        fields=(_max_iterations_field(default=20, maximum=30),),
        builder=_build_novel,
    ),
    Preset(
        id="ic50mpro",
        title="IC50 Mpro",
        description="Minimize predicted SARS-CoV-2 Mpro IC50.",
        higher_is_better=False,
        score_label="IC50 (nM)",
        fields=(
            PresetField(
                name="target_ic50_nM",
                label="Target IC50 in nM",
                type="number",
                default=10,
                required=True,
            ),
            _max_iterations_field(default=20, maximum=30),
        ),
        builder=_build_ic50mpro,
    ),
    Preset(
        id="ic50mpro_novel",
        title="IC50 + novelty",
        description="Minimize Mpro IC50 while requiring a novel final molecule.",
        higher_is_better=False,
        score_label="IC50 (nM)",
        fields=(
            PresetField(
                name="target_ic50_nM",
                label="Target IC50 in nM",
                type="number",
                default=10,
                required=True,
            ),
            _max_iterations_field(default=20, maximum=30),
        ),
        builder=_build_ic50mpro_novel,
    ),
    Preset(
        id="ic50mpro_qed_novel",
        title="IC50 + QED + novelty",
        description="Minimize Mpro IC50 with QED and novelty constraints.",
        higher_is_better=False,
        score_label="IC50 (nM)",
        fields=(
            PresetField(
                name="target_ic50_nM",
                label="Target IC50 in nM",
                type="number",
                default=10,
                required=True,
            ),
            PresetField(
                name="min_qed",
                label="Minimum QED",
                type="number",
                default=0.5,
                required=True,
            ),
            _max_iterations_field(default=20, maximum=30),
        ),
        builder=_build_ic50mpro_qed_novel,
    ),
)

PRESET_BY_ID: dict[str, Preset] = {preset.id: preset for preset in PRESETS}


def list_presets() -> list[dict[str, Any]]:
    return [preset.to_dict() for preset in PRESETS]


def get_preset(preset_id: str) -> Preset | None:
    return PRESET_BY_ID.get(preset_id)


def build_preset_job(
    preset_id: str, params: dict[str, Any], job_dir: Path
) -> tuple[Preset, Objective, ExperimentConfig, dict[str, Any]]:
    """Validate params and construct the objective for a preset job."""
    preset = get_preset(preset_id)
    if preset is None:
        raise KeyError(preset_id)

    # Drop unknown keys early so typos surface as 400s.
    known = {field.name for field in preset.fields}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")

    normalized = dict(params)
    for field in preset.fields:
        if field.name not in normalized or normalized[field.name] is None:
            if field.required and field.default is None:
                raise ValueError(f"Missing required parameter: {field.name}")
            normalized[field.name] = field.default

    objective, cfg = preset.builder(normalized, job_dir)
    return preset, objective, cfg, normalized


def preview_first_message(preset_id: str, params: dict[str, Any]) -> str:
    """Render ``objective.first_message()`` for the given preset and params.

    Uses a temporary job directory so builders that write under ``job_dir``
    (e.g. Boltz-2) do not touch the real jobs root.
    """
    with tempfile.TemporaryDirectory(prefix="seismo-preview-") as tmp:
        _preset, objective, _cfg, _normalized = build_preset_job(
            preset_id, params, Path(tmp)
        )
        return objective.first_message()


def score_label_for_job(preset: Preset, params: dict[str, Any]) -> str:
    if preset.id == "boltz2":
        return boltz2_score_label(
            str(params.get("score_mode", "affinity_probability_binary"))
        )
    return preset.score_label
