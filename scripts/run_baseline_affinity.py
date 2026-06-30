"""
Run Boltz-2 binding affinity predictions for all baseline molecules, distributed across GPUs.
The target protein is read from the CSV 'target' column (supported: MGYP, TRIB2).

Usage:
    python scripts/run_baseline_affinity.py \
        --csv data/molecules/baseline_molecules.csv \
        --out data/boltz2_baseline/results.csv \
        --gpus 8 \
        --workers-per-gpu 1
"""

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_MSA = {
    "MGYP": "data/msa/mgyp001550541752_warmup/boltz_results_mgyp001550541752_msa_input/msa/mgyp001550541752_msa_input_0.csv",
    "TRIB2": "data/msa/trib2_warmup/boltz_results_trib2_msa_input/msa/trib2_msa_input_0.csv",
}

TARGET_SEQUENCES = {
    "MGYP": (
        "ILRRYCMEEPAQALIDQLAAQLEADNWELAPLFKTLFMSEAFYSQIARTGFIKSPVEHALGFIHATGMTVQLER"
        "GNIGFGFENGLDRFFNDMDNRPTQPPVVDGWPEGTGWLSAQALVDRANMLEFITASRDFQASEGFNVASLLPA"
        "GTPTAEQVVESLALLLGITLTAPEVADLAAYLGKDNSGVADPFDPSNTAQVEERVRGLLYILGQHPQYMLR"
    ),
    "TRIB2": (
        "MNIHRSTPITIARYGRSRNKTQDFEELSSIRSAEPSQSFSPNLGSPSPPETPNLSHCVSCIGKYLLLEPLEGDH"
        "VFRAVHLHSGEELVCKVFDISCYQESLAPCFCLSAHSNINQITEIILGETKAYVFFERSYGDMHSFVRTCKKL"
        "REEEAARLFYQIASAVAHCHDGGLVLRDLKLRKFIFKDEERTRVKLESLEDAYILRGDDDSLSDKHGCPAYVS"
        "PEILNTSGSYSGKAADVWSLGVMLYTMLVGRYPFHDIEPSSLFSKIRRGQFNIPETLSPKAKCLIRSILRREP"
        "SERLTSQEILDHPWFSTDFSVSNSAYGAKEVSDQLVPDVNMEENLDPFFN"
    ),
}

INPUT_TEMPLATE = """\
version: 1
sequences:
  - protein:
      id: [A]
      sequence: {sequence}
      msa: {msa_path}
  - ligand:
      id: [B]
      smiles: {smiles}

properties:
    - affinity:
        binder: B
"""

TIMEOUT = 600  # seconds per ligand


def _affinity_path(work_dir: Path) -> Path:
    return (
        work_dir
        / "boltz_results_affinity_prediction"
        / "predictions"
        / "affinity_prediction"
        / "affinity_affinity_prediction.json"
    )


def predict_one(task: dict) -> dict:
    """Run a single Boltz-2 prediction. Called in a worker process.
    Raises on any failure so the main process can abort early."""
    smiles: str = task["smiles"]
    gpu_id: int = task["gpu_id"]
    out_base: Path = Path(task["out_base"])
    row_id: str = task["row_id"]

    work_dir = out_base / row_id
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = work_dir / "affinity_prediction.yaml"
    input_path.write_text(
        INPUT_TEMPLATE.format(
            sequence=task["sequence"],
            smiles=smiles,
            msa_path=task["msa_path"],
        )
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmds = [
        "pixi", "run", "-e", "boltz2",
        "boltz", "predict", str(input_path),
        "--out_dir", str(work_dir),
        "--output_format", "pdb",
        "--accelerator", "gpu",
        "--devices", "1",
    ]

    logger.info("START row_id=%s gpu=%d smiles=%s", row_id, gpu_id, smiles[:60])
    run(cmds, check=True, timeout=TIMEOUT, env=env)
    logger.info("BOLTZ DONE row_id=%s", row_id)

    with open(_affinity_path(work_dir)) as f:
        affinity_data = json.load(f)

    result = {
        "row_id": row_id,
        "smiles": smiles,
        "gpu_id": gpu_id,
        "affinity_probability_binary": affinity_data["affinity_probability_binary"],
        "affinity_pred_value": affinity_data["affinity_pred_value"],
    }
    logger.info(
        "OK row_id=%s prob=%.4f pred=%.4f",
        row_id,
        result["affinity_probability_binary"],
        result["affinity_pred_value"],
    )
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

    # Read tasks, skipping already-completed ones
    all_tasks = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            smiles = row["smiles"].strip()
            if not smiles:
                continue
            model = row.get("model", "")
            replicate = row.get("replicate", "")
            iteration = row.get("iteration", "")
            target = row["target"].strip().upper()
            if target not in TARGET_SEQUENCES:
                raise ValueError(f"Unknown target '{target}' in row {i}. Known targets: {list(TARGET_SEQUENCES)}")
            msa_path = Path(TARGET_MSA[target]).resolve()
            if not msa_path.exists():
                raise FileNotFoundError(f"MSA file not found for target '{target}': {msa_path}")
            row_id = f"{model}_r{replicate}_i{iteration}_{i:04d}"
            all_tasks.append(
                {
                    "smiles": smiles,
                    "gpu_id": i % args.gpus,
                    "out_base": str(out_base),
                    "row_id": row_id,
                    "msa_path": str(msa_path),
                    "model": model,
                    "replicate": replicate,
                    "iteration": iteration,
                    "target": target,
                    "sequence": TARGET_SEQUENCES[target],
                }
            )

    def is_done(row_id: str) -> bool:
        return _affinity_path(out_base / row_id).exists()

    tasks = [t for t in all_tasks if not is_done(t["row_id"])]
    logger.info(
        "%d ligands total, %d already done, %d to run",
        len(all_tasks), len(all_tasks) - len(tasks), len(tasks),
    )

    max_workers = args.gpus * args.workers_per_gpu
    logger.info("Submitting tasks to %d workers", max_workers)
    new_results: list[dict] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(predict_one, t): t for t in tasks}
        for done_idx, future in enumerate(as_completed(futures), 1):
            t = futures[future]
            res = future.result()  # raises immediately on any error
            res["model"] = t["model"]
            res["replicate"] = t["replicate"]
            res["iteration"] = t["iteration"]
            new_results.append(res)
            logger.info("Progress: %d/%d", done_idx, len(tasks))

    all_results = new_results
    fieldnames = [
        "row_id", "model", "replicate", "iteration", "smiles",
        "gpu_id", "affinity_probability_binary", "affinity_pred_value",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        all_results.sort(key=lambda r: r["row_id"])
        writer.writerows(all_results)

    logger.info("Done. %d results saved to %s", len(all_results), out_path)


if __name__ == "__main__":
    main()
