import json
import logging
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from sklearn.metrics import pairwise_distances

from molopt_agent.oracles.base import OracleResult

from .base import OracleResult

logger = logging.getLogger(__name__)

INPUT_TEMPLATE = """version: 1
sequences:
  - protein:
      id: [A]
      sequence: {sequence}
  - ligand:
      id: [B]
      smiles: {smiles}

properties:
    - affinity:
        binder: B
"""


class Boltz2Oracle:
    """
    Oracle for predicting protein-ligand binding probability using Boltz-2.

    Returns binding affinity score, either affinity_probability_binary between 0 and 1 or affinity_pred_value
    """

    def __init__(
        self,
        protein_sequence: str,
        output_dir_base: str | Path,
        binding_score_name: str = "affinity_probability_binary",
        timeout: int = 300,
        with_explanation: bool = True,
    ):
        self.protein_sequence = protein_sequence
        self.output_dir_base = Path(output_dir_base)
        self.binding_score_name = binding_score_name
        self.timeout = timeout
        self.with_explanation = with_explanation

        mode = (
            "probability"
            if binding_score_name == "affinity_probability_binary"
            else "affinity"
        )
        logger.info(
            f"Boltz2Oracle initialized in {mode} mode (scoring: {binding_score_name})"
        )
        logger.info(f"Timeout: {timeout}s ({timeout / 60:.1f} minutes)")

    def _generate_boltz_input(self, smiles: str, path: Path) -> None:
        with open(path, "w") as f:
            _ = f.write(
                INPUT_TEMPLATE.format(sequence=self.protein_sequence, smiles=smiles)
            )

    def _get_close_residues(
        self,
        fn: Path,
        threshold_angstrom: float = 2.0,
    ) -> list[tuple[str, int]]:
        complex: struc.AtomArray = PDBFile.read(fn).get_structure(model=1)
        protein: struc.AtomArray = complex[complex.chain_id == "A"]  # pyright: ignore[reportAssignmentType]
        ligand: struc.AtomArray = complex[complex.chain_id == "B"]  # pyright: ignore[reportAssignmentType]

        distances = pairwise_distances(
            protein.coord,
            ligand.coord,
            metric="euclidean",
        )

        min_distances = distances.min(axis=1)
        mask = min_distances < threshold_angstrom

        return sorted(
            set((atom.res_name.item(), atom.res_id.item()) for atom in protein[mask]),  # pyright: ignore[reportGeneralTypeIssues]
            key=lambda x: x[1],
        )

    def _get_confidence_scores(
        self,
        path: Path,
    ) -> dict[str, float]:
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        return {
            key: data.get(key, float("nan"))
            for key in [
                "confidence_score",
                "ptm",
                "iptm",
                "ligand_iptm",
                "complex_plddt",
                "complex_iplddt",
                "complex_pde",
                "complex_ipde",
            ]
        }

    def _build_explanation(
        self,
        predict_dir: Path,
        run_name: str,
        additional_scores: dict[str, float] | None = None,
    ) -> str:
        additional_scores = additional_scores or {}
        threshold_angstrom = 5.0
        pdb_path = predict_dir / f"{run_name}_model_0.pdb"
        close_residues = self._get_close_residues(pdb_path, threshold_angstrom)
        confidence_path = predict_dir / f"confidence_{run_name}_model_0.json"
        confidence_scores = self._get_confidence_scores(confidence_path)
        confidence_scores.update(additional_scores)

        explanation_parts = [
            f"Close Residues (within {threshold_angstrom} Å): "
            + ", ".join(f"{res_name}-{res_id}" for res_name, res_id in close_residues)
        ]
        explanation_parts.append("Confidence Scores:")
        for key, value in confidence_scores.items():
            explanation_parts.append(f"  {key}: {value:.4f}")

        return "\n".join(explanation_parts)

    def __call__(self, smiles: str) -> OracleResult:
        output_dir = self.output_dir_base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = "affinity_prediction"

        input_path = output_dir / f"{run_name}.yaml"
        self._generate_boltz_input(smiles, input_path)

        cmds = [
            "pixi",
            "run",
            "-e",
            "boltz2",
            "boltz",
            "predict",
            str(input_path),
            "--out_dir",
            str(output_dir),
            "--output_format",
            "pdb",
            "--use_msa_server",
        ]

        logger.info(f"Starting Boltz-2 prediction for SMILES: {smiles}")

        try:
            _ = run(
                cmds,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            logger.info("Boltz-2 prediction completed successfully")
        except TimeoutExpired as e:
            raise ValueError(
                f"Boltz-2 subprocess timed out after {self.timeout}s ({self.timeout / 60:.1f} minutes).\n"
            )
        except CalledProcessError as e:
            raise ValueError(
                f"Boltz-2 subprocess failed with code {e.returncode}.\nstderr: {e.stderr}"
            )
        except Exception as e:
            raise ValueError(f"Boltz-2 subprocess failed: {e}")

        predict_dir = (
            output_dir / f"boltz_results_{run_name}" / "predictions" / run_name
        )
        affinity_path = predict_dir / f"affinity_{run_name}.json"

        try:
            with open(affinity_path) as f:
                affinity_data = json.load(f)
        except FileNotFoundError:
            affinity_data = {
                "affinity_pred_value": float("nan"),
                "affinity_probability_binary": float("nan"),
            }

        score: float = affinity_data[self.binding_score_name]
        # If using affinity mode, also include probability for constraint checking
        additional_scores = {}
        if self.binding_score_name == "affinity_pred_value":
            binding_prob = affinity_data["affinity_probability_binary"]
            additional_scores = {"affinity_probability_binary": binding_prob}

        explanation = (
            self._build_explanation(predict_dir, run_name, additional_scores)
            if self.with_explanation
            else ""
        )

        result: OracleResult = {
            "score": score,
            "explanation": explanation,
            "extra_scores": json.dumps(additional_scores),  # pyright: ignore[reportAssignmentType]
        }

        logger.info(f"Boltz-2 prediction result: {result}")

        return result
