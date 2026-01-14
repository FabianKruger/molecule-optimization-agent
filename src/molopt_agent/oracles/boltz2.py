import json
import logging
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run

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
            timeout: int = 300
    ):
        self.protein_sequence = protein_sequence
        self.output_dir_base = Path(output_dir_base)
        self.binding_score_name = binding_score_name
        self.timeout = timeout

        mode = "probability" if binding_score_name == "affinity_probability_binary" else "affinity"
        logger.info(f"Boltz2Oracle initialized in {mode} mode (scoring: {binding_score_name})")
        logger.info(f"Timeout: {timeout}s ({timeout/60:.1f} minutes)")

    def _generate_boltz_input(self, smiles: str, path: Path) -> None:
        with open(path, "w") as f:
            _ = f.write(
                INPUT_TEMPLATE.format(sequence=self.protein_sequence, smiles=smiles)
            )

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
            result = run(
                cmds,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            logger.info("Boltz-2 prediction completed successfully")
        except TimeoutExpired as e:
            raise ValueError(
                f"Boltz-2 subprocess timed out after {self.timeout}s ({self.timeout/60:.1f} minutes).\n"
            )
        except CalledProcessError as e:
            raise ValueError(
                f"Boltz-2 subprocess failed with code {e.returncode}.\nstderr: {e.stderr}"
            )
        except Exception as e:
            raise ValueError(f"Boltz-2 subprocess failed: {e}")

        affinity_path = (
            output_dir
            / f"boltz_results_{run_name}"
            / "predictions"
            / run_name
            / f"affinity_{run_name}.json"
        )

        with open(affinity_path) as f:
            affinity_data = json.load(f)

        score = affinity_data[self.binding_score_name]

        result = {
            "score": score,
            "explanation": "",
        }

        # If using affinity mode, also include probability for constraint checking
        if self.binding_score_name == "affinity_pred_value":
            binding_prob = affinity_data["affinity_probability_binary"]
            result["affinity_probability_binary"] = binding_prob

        logger.info(f"Boltz-2 prediction result: {result}")

        return result
