"""
Collect Boltz-2 affinity results from completed runs and write results.csv.
Also reports how many ligands from the baseline CSV are missing results.

Usage:
    python scripts/collect_baseline_affinity.py \
        --runs data/boltz2_baseline_mgyp001550541752/runs \
        --csv data/molecules/baseline_molecules.csv \
        --out data/boltz2_baseline_mgyp001550541752/results.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        default="data/boltz2_baseline_mgyp001550541752/runs",
        help="Directory containing per-ligand run subdirectories",
    )
    parser.add_argument(
        "--csv",
        default="data/molecules/baseline_molecules.csv",
        help="Original baseline CSV (to check for missing results)",
    )
    parser.add_argument(
        "--out",
        default="data/boltz2_baseline_mgyp001550541752/results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all completed results
    results = []
    affinity_json_paths = sorted(
        runs_dir.glob("*/boltz_results_affinity_prediction/predictions/affinity_prediction/affinity_affinity_prediction.json")
    )
    logger.info("Found %d completed result files", len(affinity_json_paths))

    for json_path in affinity_json_paths:
        row_id = json_path.parts[len(runs_dir.parts)]  # first subdir = row_id
        try:
            with open(json_path) as f:
                data = json.load(f)
            results.append({
                "row_id": row_id,
                "affinity_probability_binary": data.get("affinity_probability_binary", ""),
                "affinity_pred_value": data.get("affinity_pred_value", ""),
            })
        except Exception as e:
            logger.warning("Failed to read %s: %s", json_path, e)

    # Build row_id -> result lookup
    result_by_row_id = {r["row_id"]: r for r in results}

    # Read baseline CSV to reconstruct row_ids and check coverage
    baseline_rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        meta_fields = [f for f in fieldnames if f != "smiles"]
        for i, row in enumerate(reader):
            smiles = row["smiles"].strip()
            if not smiles:
                continue
            meta = {f: row[f] for f in meta_fields}
            meta_str = "_".join(str(meta[f]) for f in meta_fields)
            row_id = f"{meta_str}_{i:04d}"
            baseline_rows.append({"row_id": row_id, "smiles": smiles, **meta})

    missing = [r for r in baseline_rows if r["row_id"] not in result_by_row_id]
    logger.info(
        "%d / %d ligands have results, %d missing",
        len(results), len(baseline_rows), len(missing),
    )
    if missing:
        logger.info("Missing row_ids (first 10): %s", [r["row_id"] for r in missing[:10]])

    # Write output CSV merging baseline metadata with affinity scores
    out_fieldnames = ["row_id", *meta_fields, "smiles", "affinity_probability_binary", "affinity_pred_value"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in baseline_rows:
            res = result_by_row_id.get(row["row_id"], {})
            writer.writerow({
                **row,
                "affinity_probability_binary": res.get("affinity_probability_binary", ""),
                "affinity_pred_value": res.get("affinity_pred_value", ""),
            })

    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
