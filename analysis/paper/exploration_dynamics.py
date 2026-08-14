#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import warnings
from glob import glob
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── paths & constants ──────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent          # analysis/paper/
ROOT = HERE.parents[1]                           # repo root

# Focus on the full-XAI variant only (3 replicates) to keep the figures readable.
FULL_XAI_DIR = ROOT / "data/results/ic50mpro_qed/claude-opus-4.5_full_xai"

OUT_CSV       = HERE / "exploration_dynamics.csv"
OUT_RECORDS   = HERE / "figures/dynamics_records_by_basin.png"

# Hard-constraint gating — same constraints as analysis/paper/xai_vs_no_xai_ic50mpro_extended.ipynb:
# a molecule counts only if it is drug-like (QED >= 0.6) AND novel (Novelty is binary,
# 1.0 = novel / 0.0 = not). A molecule failing either gate is disqualified and assigned the
# penalty IC50 (1e9 nM → pIC50 0). The quality axis (records / best-so-far / basin quality)
# therefore reflects the best *valid* IC50, never a QED/novelty-failing one.
QED_THRESHOLD        = 0.6
INVALID_PENALTY_IC50 = 1_000_000_000.0

ECFP_RADIUS   = 2
ECFP_NBITS    = 2048
SIM_PRIMARY   = 0.65
RANDOM_STATE  = 42

# Basins are clustered *per run*, so basin ids are not comparable across panels.
# Give each replicate its own colour family to discourage cross-panel colour
# matching. Each has 20 entries (enough for the observed basin counts).
PANEL_CMAPS = ["tab20", "tab20b", "tab20c"]

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_NBITS)


# ── run loading ──────────────────────────────────────────────────────────────────

def load_run(path: Path) -> dict:
    """Iteration-ordered valid molecules with fingerprints and pIC50 scores."""
    with open(path) as f:
        data = json.load(f)
    trace = sorted(data["trace"], key=lambda e: e["iteration"])
    fps, pic50, pic50_raw, valid_mask, smiles = [], [], [], [], []
    for e in trace:
        mol = Chem.MolFromSmiles(e["smiles"])
        if mol is None:
            continue
        scores = e["scores"]
        qed = float(scores.get("QED", 0.0))
        novelty = float(scores.get("Novelty", 1.0))  # binary: 1.0 novel / 0.0 not
        valid = (qed >= QED_THRESHOLD) and (novelty > 0)
        raw = 9.0 - np.log10(max(float(scores["IC50"]), 1e-3))  # actual predicted pIC50
        fps.append(_morgan.GetFingerprint(mol))   # walk is structural: keep all parseable mols
        pic50.append(raw if valid else 0.0)       # gated quality: invalid → pIC50 0
        pic50_raw.append(raw)                     # ungated, for display only
        valid_mask.append(valid)
        smiles.append(e["smiles"])
    return {
        "fps": fps,
        "pic50": np.asarray(pic50),
        "pic50_raw": np.asarray(pic50_raw),
        "valid": np.asarray(valid_mask),
        "smiles": smiles,
        "n": len(fps),
    }


def collect_runs() -> list[dict]:
    return [
        {"variant": FULL_XAI_DIR.name, "path": Path(jp)}
        for jp in sorted(glob(str(FULL_XAI_DIR / "*.json")))
    ]


def rep_name(filename: str) -> str:
    m = re.search(r"rep\d+", filename)
    return m.group(0) if m else filename


# ── clustering (basins) ─────────────────────────────────────────────────────────

def butina_labels(fps: list, sim_thresh: float) -> np.ndarray:
    """Basin id per molecule via Butina clustering at the given Tanimoto cut-off."""
    n = len(fps)
    if n == 0:
        return np.array([], dtype=int)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, 1.0 - sim_thresh, isDistData=True)
    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


# ── analysis 1: basin segmentation ───────────────────────────────────────────────

def rle(labels: np.ndarray) -> list[tuple[int, int]]:
    segs: list[list[int]] = []
    for l in labels:
        if segs and segs[-1][0] == l:
            segs[-1][1] += 1
        else:
            segs.append([int(l), 1])
    return [(l, c) for l, c in segs]


def segmentation_metrics(labels: np.ndarray) -> dict:
    if len(labels) == 0:
        return {}
    segs = rle(labels)
    dwells = np.array([c for _, c in segs])
    n_basins = len(set(labels.tolist()))
    n_segments = len(segs)
    return {
        "n_basins": n_basins,
        "n_switches": n_segments - 1,
        "mean_dwell": float(dwells.mean()),
        "max_dwell": int(dwells.max()),
        "n_revisits": n_segments - n_basins,  # extra segments beyond one-per-basin
    }


# ── analysis 2: effort vs quality ────────────────────────────────────────────────

def effort_quality(labels: np.ndarray, pic50: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        rows.append({"basin": cid, "effort": int(mask.sum()),
                     "best_pic50": float(pic50[mask].max())})
    df = pd.DataFrame(rows)
    if df["effort"].nunique() < 2 or len(df) < 3:
        return np.nan, df
    rho, _ = spearmanr(df["effort"], df["best_pic50"])
    return float(rho), df


# ── analysis 3: records from multiple basins ─────────────────────────────────────

def record_basins(labels: np.ndarray, pic50: np.ndarray) -> dict:
    """Best-so-far record events and which basins produced them.

    Starts at the penalty floor (pIC50 0), so an invalid molecule (gated to 0)
    never counts as a record — only molecules that beat the best *valid* score do.
    """
    running = 0.0
    rec_iter, rec_val, rec_basin = [], [], []
    for i, v in enumerate(pic50):
        if v > running:
            running = v
            rec_iter.append(i)
            rec_val.append(float(v))
            rec_basin.append(int(labels[i]))
    return {
        "n_records": len(rec_iter),
        "n_record_basins": len(set(rec_basin)),
        "rec_iter": rec_iter, "rec_val": rec_val, "rec_basin": rec_basin,
    }


# ── per-run driver ──────────────────────────────────────────────────────────────

def analyse_run(run: dict) -> dict:
    data = load_run(run["path"])
    fps, pic50 = data["fps"], data["pic50"]
    labels = butina_labels(fps, SIM_PRIMARY)
    seg = segmentation_metrics(labels)
    rho, _ = effort_quality(labels, pic50)
    rec = record_basins(labels, pic50)
    return {
        "variant": run["variant"], "rep": rep_name(run["path"].name),
        "file": run["path"].name, "n_valid": data["n"],
        "n_basins": seg.get("n_basins"),
        "mean_dwell": seg.get("mean_dwell"),
        "n_switches": seg.get("n_switches"),
        "n_revisits": seg.get("n_revisits"),
        "spearman_effort_quality": rho,
        "n_record_basins": rec["n_record_basins"],
        "n_records": rec["n_records"],
        # detail for plotting (underscore keys are dropped before the CSV is written)
        "_labels": labels,
        "_pic50": pic50,               # gated (best-so-far / records)
        "_pic50_raw": data["pic50_raw"],  # ungated (display)
        "_valid": data["valid"],
    }


# ── figures (full_xai, one panel per replicate) ──────────────────────────────────

def _panels(n: int, w: float, h: float):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    return fig, (list(axes) if n > 1 else [axes])


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel basins 0..k-1 by order of first appearance.

    Cluster ids from Butina are arbitrary; remapping to appearance order makes the
    colour assignment stable and meaningful (first basin visited → first colour)
    instead of keyed to an opaque cluster index.
    """
    order: dict[int, int] = {}
    out = np.empty(len(labels), dtype=int)
    for i, l in enumerate(labels.tolist()):
        if l not in order:
            order[l] = len(order)
        out[i] = order[l]
    return out


def plot_records(results: list[dict], out: Path) -> None:
    fig, axes = _panels(len(results), 5.0, 4.2)
    for panel_i, (ax, r) in enumerate(zip(axes, results)):
        pic50 = r["_pic50"]
        pic50_raw, valid = r["_pic50_raw"], r["_valid"]
        x = np.arange(1, len(pic50) + 1)

        # basin ids are per-run → each panel gets its own colour family so the
        # eye does not match colours across panels.
        labels = _compact_labels(r["_labels"])
        cmap = PANEL_CMAPS[panel_i % len(PANEL_CMAPS)]
        norm = mcolors.Normalize(vmin=0, vmax=max(labels.max(), 1))

        # best-so-far / records use the GATED score (best valid IC50)
        best_so_far = np.maximum.accumulate(pic50)
        ax.plot(x, best_so_far, color="black", lw=1.2, zorder=1, label="best-so-far")

        # molecules plotted at their RAW pIC50; colour = basin, shape = validity
        ax.scatter(x[valid], pic50_raw[valid], c=labels[valid], cmap=cmap, norm=norm,
                   s=22, alpha=0.85, zorder=2)
        if (~valid).any():
            ax.scatter(x[~valid], pic50_raw[~valid], c=labels[~valid], cmap=cmap, norm=norm,
                       marker="x", s=40, linewidths=1.5, zorder=2,
                       label="invalid (QED<0.6 / non-novel)")

        rec = record_basins(labels, pic50)
        ax.scatter([i + 1 for i in rec["rec_iter"]], rec["rec_val"],
                   c=rec["rec_basin"], cmap=cmap, norm=norm,
                   marker="*", s=150, edgecolors="red", linewidths=0.6,
                   zorder=3, label="high-score")
        ax.set_xlabel("iteration")
        ax.set_title(r["rep"].replace("rep", "repetition "), fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("pIC50")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] records-by-basin → {out}")


def main() -> None:
    runs = collect_runs()
    print(f"[runs] analysing {len(runs)} runs\n")

    results = [analyse_run(r) for r in runs]

    # CSV (drop the heavy per-run arrays)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[csv] per-run dynamics → {OUT_CSV}\n")

    # console summary
    print(f"=== full_xai per-replicate summary (basins @{SIM_PRIMARY}) ===\n")
    for _, row in df.iterrows():
        print(f"{row['rep']}")
        print(f"   # basins           {row['n_basins']:.0f}")
        print(f"   mean dwell         {row['mean_dwell']:.2f} iterations")
        print(f"   effort∝quality ρ   {row['spearman_effort_quality']:.2f}")
        print(f"   # record basins    {row['n_record_basins']:.0f}")
        print()

    plot_records(results, OUT_RECORDS)


if __name__ == "__main__":
    main()
