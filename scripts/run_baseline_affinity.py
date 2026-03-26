"""
Run Boltz-2 binding affinity predictions for all baseline molecules against
mgyp001550541752 protein, distributed across 8 GPUs.

Usage:
    python scripts/run_baseline_affinity.py \
        --csv data/molecules/baseline_molecules.csv \
        --out data/boltz2_baseline_mgyp001550541752/results.csv \
        --gpus 8 \
        --workers-per-gpu 1
"""

import argparse
import csv
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROTEIN_SEQUENCE = (
    "ILRRYCMEEPAQALIDQLAAQLEADNWELAPLFKTLFMSEAFYSQIARTGFIKSPVEHALGFIHATGMTVQLER"
    "GNIGFGFENGLDRFFNDMDNRPTQPPVVDGWPEGTGWLSAQALVDRANMLEFITASRDFQASEGFNVASLLPA"
    "GTPTAEQVVESLALLLGITLTAPEVADLAAYLGKDNSGVADPFDPSNTAQVEERVRGLLYILGQHPQYMLR"
)

INPUT_TEMPLATE = """\
version: 1
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

TIMEOUT = 600  # seconds per ligand


def predict_one(task: dict) -> dict:
    """Run a single Boltz-2 prediction. Called in a worker process."""
    smiles: str = task["smiles"]
    gpu_id: int = task["gpu_id"]
    out_base: Path = Path(task["out_base"])
    row_id: str = task["row_id"]

    work_dir = out_base / row_id
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = work_dir / "affinity_prediction.yaml"
    input_path.write_text(
        INPUT_TEMPLATE.format(sequence=PROTEIN_SEQUENCE, smiles=smiles)
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmds = [
        "pixi", "run", "-e", "boltz2",
        "boltz", "predict", str(input_path),
        "--out_dir", str(work_dir),
        "--output_format", "pdb",
        "--use_msa_server",
        "--accelerator", "gpu",
        "--devices", "1",
    ]

    result = {
        "row_id": row_id,
        "smiles": smiles,
        "gpu_id": gpu_id,
        "affinity_probability_binary": float("nan"),
        "affinity_pred_value": float("nan"),
        "error": "",
    }

    try:
        run(cmds, check=True, capture_output=True, text=True, timeout=TIMEOUT, env=env)
    except TimeoutExpired:
        result["error"] = f"timeout after {TIMEOUT}s"
        logger.warning("TIMEOUT row_id=%s smiles=%s", row_id, smiles[:40])
        return result
    except CalledProcessError as e:
        result["error"] = f"returncode={e.returncode}: {e.stderr[:200]}"
        logger.warning("FAILED row_id=%s: %s", row_id, result["error"])
        return result
    except Exception as e:
        result["error"] = str(e)
        logger.warning("ERROR row_id=%s: %s", row_id, e)
        return result

    predict_dir = (
        work_dir
        / "boltz_results_affinity_prediction"
        / "predictions"
        / "affinity_prediction"
    )
    affinity_path = predict_dir / "affinity_affinity_prediction.json"

    try:
        with open(affinity_path) as f:
            affinity_data = json.load(f)
        result["affinity_probability_binary"] = affinity_data.get(
            "affinity_probability_binary", float("nan")
        )
        result["affinity_pred_value"] = affinity_data.get(
            "affinity_pred_value", float("nan")
        )
        logger.info(
            "OK row_id=%s prob=%.4f pred=%.4f",
            row_id,
            result["affinity_probability_binary"],
            result["affinity_pred_value"],
        )
    except FileNotFoundError:
        result["error"] = f"affinity JSON not found: {affinity_path}"
        logger.warning("NO JSON row_id=%s", row_id)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/molecules/baseline_molecules.csv",
        help="Input CSV with a 'smiles' column",
    )
    parser.add_argument(
        "--out",
        default="data/boltz2_baseline_mgyp001550541752/results.csv",
        help="Output CSV path",
    )
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Parallel workers per GPU (keep at 1 unless GPU has headroom)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_base = out_path.parent / "runs"

    # Read tasks
    tasks = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            smiles = row["smiles"].strip()
            if not smiles:
                continue
            # Build a stable row_id from the row fields
            model = row.get("model", "")
            replicate = row.get("replicate", "")
            iteration = row.get("iteration", "")
            row_id = f"{model}_r{replicate}_i{iteration}_{i:04d}"
            tasks.append(
                {
                    "smiles": smiles,
                    "gpu_id": i % args.gpus,
                    "out_base": str(out_base),
                    "row_id": row_id,
                    # carry original CSV fields for output
                    "model": model,
                    "replicate": replicate,
                    "iteration": iteration,
                }
            )

    total = len(tasks)
    max_workers = args.gpus * args.workers_per_gpu
    logger.info("Running %d ligands with %d workers (%d GPUs)", total, max_workers, args.gpus)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(predict_one, t): t for t in tasks}
        for done_idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
            except Exception as e:
                t = futures[future]
                res = {
                    "row_id": t["row_id"],
                    "smiles": t["smiles"],
                    "gpu_id": t["gpu_id"],
                    "affinity_probability_binary": float("nan"),
                    "affinity_pred_value": float("nan"),
                    "error": str(e),
                }
            # Merge original CSV fields back
            t = futures[future]
            res["model"] = t["model"]
            res["replicate"] = t["replicate"]
            res["iteration"] = t["iteration"]
            results.append(res)
            logger.info("Progress: %d/%d", done_idx, total)

    # Write output CSV
    fieldnames = [
        "row_id", "model", "replicate", "iteration", "smiles",
        "gpu_id", "affinity_probability_binary", "affinity_pred_value", "error",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by original order (row_id encodes index)
        results.sort(key=lambda r: r["row_id"])
        writer.writerows(results)

    n_ok = sum(1 for r in results if not r["error"])
    logger.info("Done. %d/%d succeeded. Results saved to %s", n_ok, total, out_path)


if __name__ == "__main__":
    main()
