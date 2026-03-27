import csv
from pathlib import Path

baseline = list(csv.DictReader(open("data/molecules/baseline_molecules.csv")))
results = {r["row_id"]: r for r in csv.DictReader(open("data/boltz2_baseline_mgyp001550541752/results.csv"))}

for i, row in enumerate(baseline):
    model, replicate, iteration = row["model"], row["replicate"], row["iteration"]
    row_id = f"{model}_r{replicate}_i{iteration}_{i:04d}"
    res = results.get(row_id, {})
    row["affinity_probability_binary"] = res.get("affinity_probability_binary", "")
    row["affinity_pred_value"] = res.get("affinity_pred_value", "")

out = Path("data/molecules/baseline_molecules_boltz2_pred.csv")
fieldnames = list(baseline[0].keys())
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(baseline)

print(f"Written {len(baseline)} rows to {out}")
