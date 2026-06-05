#!/usr/bin/env python3
"""
Molecular optimization trajectory visualization in chemical space.

Two side-by-side plots sharing the same UMAP embedding:
  Left  – IC50 scatter  (ChEMBL points colored by predicted pIC50)
  Right – QED scatter   (ChEMBL points colored by QED)

Both overlaid with the optimization trajectory (arrows + points colored by
iteration).

Pipeline:
  ChEMBL SQLite  →  random background sample
                 →  predict IC50 (MACCS + XGBoost) + compute QED
  Trace JSON     →  trajectory molecules
  ECFP4          →  PCA (95 % var, fit on bg)  →  UMAP 2D (fit on bg)
  Background     →  scatter colored by score
  Trajectory     →  arrows + scatter colored by iteration

Usage:
    pixi run python scripts/visualize_trajectory.py
"""

from __future__ import annotations

import json
import sqlite3
import warnings
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
from sklearn.decomposition import PCA
import umap as umap_module

warnings.filterwarnings("ignore")

# ── paths & constants ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

CHEMBL_DB = Path(
    "/Users/fabian/Code/Helmholtz/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
)
XGB_MODEL = ROOT / "data/xgboost/xgb_maccs_best.json"
TRAJECTORY_JSON = ROOT / (
    "data/results/ic50mpro_qed/claude-opus-4.5_full_xai/"
    "ic50mpro_qed_novel_full_rep1_conversation_20260107_152318.json"
)
TRAJECTORY_JSON_2 = ROOT / (
    "data/results/ic50mpro_qed_novel_full_rep1_conversation_20260327_184935.json"
)
OUTPUT_PNG = ROOT / "molecule_trajectory.png"
OUTPUT_PNG_2 = ROOT / "molecule_trajectory_long.png"

N_BACKGROUND     = 50_000
ECFP_RADIUS      = 2
ECFP_NBITS       = 2048
PCA_VARIANCE     = 0.95
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
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
        arr = np.zeros(ECFP_NBITS, dtype=np.float32)
        ConvertToNumpyArray(_morgan.GetFingerprint(mol), arr)
        fps.append(arr)
        valid.append(i)
    return np.vstack(fps), valid


# ── 5. PCA → UMAP (fit on background only) ────────────────────────────────────

def embed(
    bg_fps: np.ndarray, *traj_fps_list: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit PCA + UMAP on background; transform each trajectory in traj_fps_list.

    Returns (bg_2d, [traj_2d_1, traj_2d_2, ...]) so all trajectories share
    the same chemical-space embedding.
    """
    print(f"[PCA]  fitting on {len(bg_fps)} background molecules…")
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_STATE)
    bg_pca = pca.fit_transform(bg_fps)
    print(f"[PCA]  {pca.n_components_} components → {pca.explained_variance_ratio_.sum()*100:.1f} % variance")

    print(f"[UMAP] fitting on PCA-reduced background ({bg_pca.shape[1]} dims)…")
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    bg_2d = reducer.fit_transform(bg_pca)
    print("[UMAP] done.")

    trajs_2d = [reducer.transform(pca.transform(fps)) for fps in traj_fps_list]
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

    # ── start / end / best markers ────────────────────────────────────────────
    ax.scatter(*traj_2d[0],  s=60, marker="*", color="lime",   ec="white", lw=0.8,
               zorder=5, label=f"Start (iter {iters[0]})")
    ax.scatter(*traj_2d[-1], s=45, marker="D", color="tomato", ec="white", lw=0.8,
               zorder=5, label=f"End (iter {iters[-1]})")

    best_idx = traj_df["pic50"].idxmax()
    bx, by   = traj_2d[best_idx]
    ax.scatter(bx, by, s=60, marker="^", color="gold", ec="white", lw=0.8,
               zorder=5, label=f"Best IC50 = {traj_df.loc[best_idx, 'ic50_nM']:.1f} nM")
    ax.annotate(
        f"{traj_df.loc[best_idx, 'ic50_nM']:.1f} nM\niter {iters[best_idx]}",
        xy=(bx, by), xytext=(bx + 1.0, by + 1.0),
        fontsize=7.5, color="white",
        arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="#111133", ec="#7777aa", alpha=0.85),
        zorder=6,
    )

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-12.5, 2.5)
    ax.set_ylim(-10, 5)
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
        cmap="magma", traj_2d=traj_2d, traj_df=traj_df,
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
        f"Trajectory: {len(traj_2d)} molecules  ·  ECFP4 → PCA ({PCA_VARIANCE*100:.0f} %) → UMAP"
    )
    fig.suptitle(title or default_title, fontsize=11, color="white", y=1.01)

    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[plot] saved → {out}")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    traj_df  = parse_trajectory(TRAJECTORY_JSON)
    traj_df2 = parse_trajectory(TRAJECTORY_JSON_2)

    model = load_model(XGB_MODEL)
    bg_df = load_chembl(CHEMBL_DB, model, n=N_BACKGROUND)

    print("[fps] background…")
    bg_fps, bg_ok = to_fps(bg_df["smiles"].tolist())
    bg_df = bg_df.iloc[bg_ok].reset_index(drop=True)

    print("[fps] trajectory 1…")
    traj_fps, traj_ok = to_fps(traj_df["smiles"].tolist())
    traj_df = traj_df.iloc[traj_ok].reset_index(drop=True)

    print("[fps] trajectory 2…")
    traj_fps2, traj_ok2 = to_fps(traj_df2["smiles"].tolist())
    traj_df2 = traj_df2.iloc[traj_ok2].reset_index(drop=True)

    bg_2d, (traj_2d, traj_2d2) = embed(bg_fps, traj_fps, traj_fps2)

    n_bg = len(bg_2d)
    base = f"Background: {n_bg:,} ChEMBL molecules  ·  ECFP4 → PCA ({PCA_VARIANCE*100:.0f} %) → UMAP"
    plot(
        bg_2d, bg_df, traj_2d, traj_df, OUTPUT_PNG,
        title=(
            "Optimization trajectory in chemical space  ·  IC50 Mpro + QED\n"
            f"{base}"
        ),
    )
    plot(
        bg_2d, bg_df, traj_2d2, traj_df2, OUTPUT_PNG_2,
        title=(
            "Optimization trajectory in chemical space  ·  IC50 Mpro + QED  ·  long run (133 iterations)\n"
            f"{base}"
        ),
    )


if __name__ == "__main__":
    main()
