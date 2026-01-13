import json
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run

from .base import OracleResult

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

    Returns binding probability as a score between 0 and 1.
    """

    def __init__(self, protein_sequence: str, output_dir_base: str | Path):
        self.protein_sequence = protein_sequence
        self.output_dir_base = Path(output_dir_base)

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

        try:
            result = run(
                cmds,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            print(result.stdout)
            print(result.stderr)
        except (CalledProcessError, TimeoutExpired) as e:
            raise ValueError(
                f"Boltz-2 subprocess failed with code.\nstderr: {e.stderr}"
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

        score = affinity_data["affinity_probability_binary"]

        return {
            "score": score,
            "explanation": "",
        }
