#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from glob import glob
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.cm import ScalarMappable
from rdkit import Chem
from rdkit.Chem import MACCSkeys, QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
import umap as umap_module

warnings.filterwarnings("ignore")

# ── paths & constants ──────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent          # analysis/paper/
ROOT = HERE.parents[1]                           # repo root

CHEMBL_DB = Path(
    "..."
)
XGB_MODEL = ROOT / "data/xgboost/xgb_maccs_best.json"

# Same 3 runs as exploration_dynamics.py: the full-XAI variant's replicates.
FULL_XAI_DIR = ROOT / "data/results/ic50mpro_qed/claude-opus-4.5_full_xai"
FIG_DIR = HERE / "figures"

N_BACKGROUND     = 50_000
ECFP_RADIUS      = 2
ECFP_NBITS       = 2048
UMAP_METRIC      = "jaccard"   # Jaccard distance on binary ECFP4 == Tanimoto distance
UMAP_N_NEIGHBORS = 15          # smaller → more local detail / finer cluster structure
UMAP_MIN_DIST    = 0.1         # smaller → tighter packing, clearer cluster separation
RANDOM_STATE     = 42


# ── 1. Parse trajectory ────────────────────────────────────────────────────────

def parse_trajectory(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    rows = [
        {
            "iteration": e["iteration"],
            "smiles":    e["smiles"],
            "ic50_nM":   e["scores"]["IC50"],
            "qed":       e["scores"]["QED"],
        }
        for e in data["trace"]
    ]
    df = pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)
    df["pic50"] = 9.0 - np.log10(df["ic50_nM"].clip(lower=1e-3))
    print(
        f"[trajectory] {len(df)} molecules, "
        f"iterations {df.iteration.min()}–{df.iteration.max()}, "
        f"IC50 {df.ic50_nM.min():.1f}–{df.ic50_nM.max():.0f} nM"
    )
    return df


def rep_name(filename: str) -> str:
    m = re.search(r"rep\d+", filename)
    return m.group(0) if m else filename


# ── 2. Predict IC50 from SMILES (MACCS + XGBoost) ────────────────────────────

def load_model(path: Path) -> xgb.XGBRegressor:
    # Mirror IC50MproOracle._load_xgb_model: set _estimator_type before loading
    model = xgb.XGBRegressor()
    model._estimator_type = "regressor"
    model.load_model(str(path))
    return model


def predict_pic50_batch(smiles_list: list[str], model: xgb.XGBRegressor) -> np.ndarray:
    """Returns pIC50 array (NaN for invalid SMILES)."""
    fps = []
    idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.array(
            list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=np.float32
        )
        fps.append(fp)
        idx.append(i)
    out = np.full(len(smiles_list), np.nan, dtype=np.float32)
    if fps:
        preds = model.predict(np.vstack(fps))
        for k, i in enumerate(idx):
            out[i] = preds[k]
    return out  # pIC50; higher = more potent


# ── 3. Load ChEMBL background ──────────────────────────────────────────────────

def load_chembl(db_path: Path, model: xgb.XGBRegressor, n: int = N_BACKGROUND) -> pd.DataFrame:
    # Reproducible random sample: fetch eligible molregnos (ints only – fast),
    # numpy-sample with fixed seed, then retrieve SMILES via a temp-table JOIN
    # instead of a massive IN (...) clause (which is the slow part).
    # Oversample 2× to absorb invalid SMILES / failed IC50 predictions.
    conn = sqlite3.connect(db_path)
    print("[chembl] fetching eligible molregnos…")
    all_ids = pd.read_sql(
        "SELECT molregno FROM compound_structures WHERE canonical_smiles IS NOT NULL",
        conn,
    )["molregno"].values
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(all_ids, size=min(n * 2, len(all_ids)), replace=False)
    print(f"[chembl] sampled {len(chosen):,} candidates from {len(all_ids):,} eligible")

    conn.execute("CREATE TEMP TABLE _sample (molregno INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO _sample VALUES (?)", ((int(i),) for i in chosen))
    raw = pd.read_sql(
        """
        SELECT cs.canonical_smiles
        FROM   compound_structures cs
        JOIN   _sample s ON cs.molregno = s.molregno
        """,
        conn,
    )
    conn.close()
    print(f"[chembl] {len(raw)} candidates – computing QED + predicting IC50…")

    smiles, qed_vals = [], []
    for smi in raw["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smiles.append(smi)
            qed_vals.append(QED.qed(mol))
        if len(smiles) >= n:
            break

    pic50_vals = predict_pic50_batch(smiles, model)

    df = pd.DataFrame({"smiles": smiles, "qed": qed_vals, "pic50": pic50_vals})
    df = df.dropna(subset=["pic50"]).reset_index(drop=True)
    print(f"[chembl] {len(df)} valid background molecules")
    return df


# ── 4. ECFP4 fingerprints ──────────────────────────────────────────────────────

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_NBITS)


def to_fps(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    fps, valid = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        arr = np.zeros(ECFP_NBITS, dtype=np.uint8)
        ConvertToNumpyArray(_morgan.GetFingerprint(mol), arr)
        fps.append(arr)
        valid.append(i)
    return np.vstack(fps), valid  # binary bits (uint8) for Jaccard/Tanimoto


# ── 5. UMAP on binary fingerprints (co-embed bg + trajectories) ───────────────

def embed(
    bg_fps: np.ndarray, *traj_fps_list: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    
    all_fps = np.vstack([bg_fps, *traj_fps_list])
    offsets = np.cumsum([len(bg_fps), *(len(f) for f in traj_fps_list)])
    print(f"[UMAP] co-embedding {len(all_fps)} molecules "
          f"({len(bg_fps)} background + {len(all_fps) - len(bg_fps)} trajectory, "
          f"{all_fps.shape[1]} bits, metric={UMAP_METRIC})…")
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    emb = reducer.fit_transform(all_fps)
    print("[UMAP] done.")

    bg_2d = emb[: offsets[0]]
    trajs_2d = [emb[offsets[i]: offsets[i + 1]] for i in range(len(traj_fps_list))]
    return bg_2d, trajs_2d


# ── 6. Draw one panel ──────────────────────────────────────────────────────────

def draw_panel(
    ax:          plt.Axes,
    bg_2d:       np.ndarray,
    bg_values:   np.ndarray,
    traj_values: np.ndarray,   # same score as bg, for coloring trajectory dots
    cmap:        str,
    traj_2d:     np.ndarray,
    traj_df:     pd.DataFrame,
    vmin:        float | None = None,
    vmax:        float | None = None,
) -> ScalarMappable:
    ax.set_facecolor("#0d0d1a")

    # ── background scatter colored by score ───────────────────────────────────
    ax.scatter(
        bg_2d[:, 0], bg_2d[:, 1],
        c=bg_values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=3, alpha=0.6, linewidths=0,
        zorder=1, rasterized=True,
    )

    # ── trajectory arrows colored by iteration ────────────────────────────────
    iters = traj_df["iteration"].values
    iter_norm = mcolors.Normalize(vmin=iters.min(), vmax=iters.max())

    for i in range(len(traj_2d) - 1):
        x0, y0 = traj_2d[i]
        x1, y1 = traj_2d[i + 1]
        col = plt.cm.cool(iter_norm((iters[i] + iters[i + 1]) / 2))
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=col, lw=1.5, mutation_scale=10
            ),
            zorder=3,
        )

    # ── trajectory dots colored by score (same scale as background) ───────────
    ax.scatter(
        traj_2d[:, 0], traj_2d[:, 1],
        c=traj_values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=35, edgecolors="white", linewidths=0.5, zorder=4,
    )

    # ── start / end markers ─────────────────────────────────────────────────────
    ax.scatter(*traj_2d[0],  s=60, marker="*", color="lime",   ec="white", lw=0.8,
               zorder=5, label=f"Start (iter {iters[0]})")
    ax.scatter(*traj_2d[-1], s=45, marker="D", color="tomato", ec="white", lw=0.8,
               zorder=5, label=f"End (iter {iters[-1]})")


    bx0, bx1 = np.nanpercentile(bg_2d[:, 0], [1, 99])
    by0, by1 = np.nanpercentile(bg_2d[:, 1], [1, 99])
    xmin = min(bx0, np.nanmin(traj_2d[:, 0]))
    xmax = max(bx1, np.nanmax(traj_2d[:, 0]))
    ymin = min(by0, np.nanmin(traj_2d[:, 1]))
    ymax = max(by1, np.nanmax(traj_2d[:, 1]))
    mx, my = 0.04 * (xmax - xmin), 0.04 * (ymax - ymin)
    ax.set_xlim(xmin - mx, xmax + mx)
    ax.set_ylim(ymin - my, ymax + my)
    ax.set_xlabel("UMAP 1", fontsize=11, color="white")
    ax.set_ylabel("UMAP 2", fontsize=11, color="white")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.45,
                    facecolor="#111122", edgecolor="#444466")
    for t in leg.get_texts():
        t.set_color("white")

    return ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)


# ── 7. Full figure ─────────────────────────────────────────────────────────────

def plot(
    bg_2d:   np.ndarray,
    bg_df:   pd.DataFrame,
    traj_2d: np.ndarray,
    traj_df: pd.DataFrame,
    out:     Path,
    title:   str | None = None,
) -> None:
    iters = traj_df["iteration"].values
    traj_norm = mcolors.Normalize(vmin=iters.min(), vmax=iters.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.patch.set_facecolor("#0d0d1a")

    # ── left: pIC50 ───────────────────────────────────────────────────────────
    p5, p95 = np.nanpercentile(bg_df["pic50"].values, [5, 95])
    sm_ic50 = draw_panel(
        axes[0], bg_2d, bg_df["pic50"].values, traj_df["pic50"].values,
        cmap="Reds_r", traj_2d=traj_2d, traj_df=traj_df,
        vmin=p5, vmax=p95,
    )
    axes[0].set_title(
        "IC50 landscape  (pIC50 = −log₁₀[IC50 / 1 M])\nhigher = more potent",
        fontsize=11, color="white", pad=8,
    )
    cb_ic50 = fig.colorbar(sm_ic50, ax=axes[0], shrink=0.75, pad=0.02)
    cb_ic50.set_label("pIC50  (background)", fontsize=10, color="white")
    cb_ic50.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cb_ic50.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    # ── right: QED ───────────────────────────────────────────────────────────
    sm_qed = draw_panel(
        axes[1], bg_2d, bg_df["qed"].values, traj_df["qed"].values,
        cmap="YlGn_r", traj_2d=traj_2d, traj_df=traj_df,
        vmin=0.0, vmax=1.0,
    )
    axes[1].set_title(
        "QED landscape  (drug-likeness)\nhigher = more drug-like",
        fontsize=11, color="white", pad=8,
    )
    cb_qed = fig.colorbar(sm_qed, ax=axes[1], shrink=0.75, pad=0.02)
    cb_qed.set_label("QED  (background)", fontsize=10, color="white")
    cb_qed.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cb_qed.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    # ── shared trajectory colorbar ────────────────────────────────────────────
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.55])
    cb_traj = fig.colorbar(
        ScalarMappable(norm=traj_norm, cmap="cool"), cax=cax
    )
    cb_traj.set_label("Optimization iteration", fontsize=10, color="white", labelpad=10)
    cb_traj.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cb_traj.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    default_title = (
        "Optimization trajectory in chemical space  ·  IC50 Mpro + QED  ·  claude-opus-4.5\n"
        f"Background: {len(bg_2d):,} ChEMBL molecules  ·  "
        f"Trajectory: {len(traj_2d)} molecules  ·  ECFP4 → UMAP (Jaccard / Tanimoto)"
    )
    fig.suptitle(title or default_title, fontsize=11, color="white", y=1.01)

    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[plot] saved → {out}")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    run_paths = [Path(p) for p in sorted(glob(str(FULL_XAI_DIR / "*.json")))]
    reps = [rep_name(p.name) for p in run_paths]
    print(f"[runs] {len(run_paths)} full_xai runs: {', '.join(reps)}\n")
    traj_dfs = [parse_trajectory(p) for p in run_paths]

    model = load_model(XGB_MODEL)
    bg_df = load_chembl(CHEMBL_DB, model, n=N_BACKGROUND)

    print("[fps] background…")
    bg_fps, bg_ok = to_fps(bg_df["smiles"].tolist())
    bg_df = bg_df.iloc[bg_ok].reset_index(drop=True)

    traj_fps_list, clean_dfs = [], []
    for rep, df in zip(reps, traj_dfs):
        print(f"[fps] trajectory {rep}…")
        fps, ok = to_fps(df["smiles"].tolist())
        traj_fps_list.append(fps)
        clean_dfs.append(df.iloc[ok].reset_index(drop=True))

    # All trajectories share one UMAP background embedding.
    bg_2d, trajs_2d = embed(bg_fps, *traj_fps_list)

    n_bg = len(bg_2d)
    base = f"Background: {n_bg:,} ChEMBL molecules  ·  ECFP4 → UMAP (Jaccard / Tanimoto)"
    for rep, df, traj_2d in zip(reps, clean_dfs, trajs_2d):
        out = FIG_DIR / f"molecule_trajectory_{rep}.png"
        plot(
            bg_2d, bg_df, traj_2d, df, out,
            title=(
                f"Optimization trajectory in chemical space  ·  IC50 Mpro + QED  ·  full_xai {rep}\n"
                f"{base}"
            ),
        )


if __name__ == "__main__":
    main()
