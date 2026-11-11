"""Train the pIC50 (SARS-CoV-2 Mpro) XGBoost surrogate used by the IC50 oracle.

Hyperparameters are selected by k-fold cross-validation on the Polaris train
split. The Polaris test split is read exactly once, after the configuration is
fixed, to report held-out performance.

Molecules are featurised as the full 167-bit MACCS vector returned by RDKit, in
which bit k is MACCS key k and bit 0 is unused. Keeping all 167 bits means
feature index k is MACCS key k, which is the indexing the IC50 oracle relies on
when it reports SHAP attributions.

The dataset itself comes from the Polaris hub, which cannot be queried from
this environment (polaris-lib needs an httpx stack that clashes with the httpx2
required by anthropic and openai). Export it once in a throwaway environment
and pass the CSV in, or rely on the cached split CSVs next to this model:

    pixi run python scripts/xgboost_training.py --raw-csv path/to/asap_potency_raw.csv

Once the split CSVs exist, plain `pixi run python scripts/xgboost_training.py`
reproduces the model without any hub access.
"""

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    # Data
    "dataset": "asap-discovery/antiviral-potency-2025-unblinded",
    "target_column": "pIC50 (SARS-CoV-2 Mpro)",
    "smiles_column": "CXSMILES",
    # Cached copies of the two splits, so training does not need Polaris access
    # and so the train split is available as a reference set for applicability
    # domain analyses. Delete them to force a fresh download.
    "train_csv": ROOT / "data/xgboost/asap_potency_train.csv",
    "test_csv": ROOT / "data/xgboost/asap_potency_test.csv",
    # Model selection
    "n_trials": 40,
    "n_folds": 5,
    "seed": 42,
    # Output
    "model_prefix": ROOT / "data/xgboost/xgb_maccs_best",
}

# Fixed model settings, unchanged from the originally published model. Only the
# four searched hyperparameters below vary between trials.
FIXED_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2,
    "grow_policy": "lossguide",
    "max_leaves": 128,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
}

SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.5, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
}


# -------------------------
#  Data loading
# -------------------------
def polaris_to_dataframe(ds):
    root = ds.zarr_root  # opens the zarr hierarchy
    data = {}

    for col in ds.columns:
        arr = root[col][:]
        # Flatten scalar-like arrays (e.g. shape (n, 1))
        arr = np.squeeze(arr)
        data[col] = arr

    return pd.DataFrame(data)


def preprocess_for_xgb(df, column_name_target_values, column_name_smiles):
    df = df.rename(
        columns={column_name_target_values: "pIC50", column_name_smiles: "SMILES"}
    )

    df = df.dropna(subset=["pIC50"]).reset_index(drop=True)
    df["pIC50"] = df["pIC50"].astype("float64")

    # CXSMILES → SMILES (safe)
    def cx_to_smiles(s):
        if isinstance(s, str):
            base = s.split("|")[0]  # takes SMILES part
            return base.strip()  # remove left-over spaces
        return s

    df["SMILES"] = df["SMILES"].apply(cx_to_smiles)

    if "Set" not in df.columns:
        raise ValueError('Dataframe must contain a "Set" column with values "train" or "test".')

    df_train = df[df["Set"].str.lower() == "train"].reset_index(drop=True)
    df_test = df[df["Set"].str.lower() == "test"].reset_index(drop=True)

    total = len(df)
    print(f"Total usable rows: {total}")
    print(f"Train: {len(df_train)} ({len(df_train) / total * 100:.2f}%)")
    print(f"Test:  {len(df_test)} ({len(df_test) / total * 100:.2f}%)")

    return df_train, df_test


def load_splits(cfg, raw_csv=None):
    """Return the train and test splits, deriving them once if needed."""
    train_csv, test_csv = Path(cfg["train_csv"]), Path(cfg["test_csv"])

    if train_csv.exists() and test_csv.exists():
        print(f"Using cached splits: {train_csv.name}, {test_csv.name}")
        return pd.read_csv(train_csv), pd.read_csv(test_csv)

    if raw_csv is not None:
        print(f"Reading the raw dataset export from {raw_csv}")
        df = pd.read_csv(raw_csv)
    else:
        import polaris as po  # only needed when no local copy exists

        print(f"Downloading {cfg['dataset']} from Polaris")
        df = polaris_to_dataframe(po.load_dataset(cfg["dataset"]))

    df_train, df_test = preprocess_for_xgb(
        df,
        column_name_target_values=cfg["target_column"],
        column_name_smiles=cfg["smiles_column"],
    )

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    df_train[["SMILES", "pIC50"]].to_csv(train_csv, index=False)
    df_test[["SMILES", "pIC50"]].to_csv(test_csv, index=False)
    print(f"Cached splits to {train_csv.name} and {test_csv.name}")

    return df_train, df_test


# -------------------------
#  Convert SMILES → MACCS keys
# -------------------------
def smiles_to_maccs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.array(list(fp.ToBitString()), dtype=np.float32)
    return arr  # shape = (167,)


def df_to_maccs(df, smiles_column="SMILES", target_column="pIC50"):
    fingerprints = []
    valid_idx = []

    for i, smi in enumerate(df[smiles_column]):
        fp = smiles_to_maccs(smi)
        if fp is not None:
            fingerprints.append(fp)
            valid_idx.append(i)

    X = np.vstack(fingerprints)
    y = df.loc[valid_idx, target_column].values.astype(float)
    n_invalid = len(df) - len(valid_idx)
    if n_invalid:
        print(f"  dropped {n_invalid} molecules with unparseable SMILES")
    return X, y


# -------------------------
#  Metrics
# -------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE and R2. Note that sklearn's mean_squared_error returns MSE."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


# -------------------------
#  Model selection
# -------------------------
def make_model(params: dict) -> XGBRegressor:
    return XGBRegressor(**FIXED_PARAMS, **params)


def sample_configs(n_trials: int, seed: int) -> list[dict]:
    """Draw distinct hyperparameter configurations reproducibly."""
    rng = random.Random(seed)
    grid_size = int(np.prod([len(v) for v in SEARCH_SPACE.values()]))
    configs: list[dict] = []
    seen = set()
    while len(configs) < min(n_trials, grid_size):
        params = {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(params)
    return configs


def cross_validate(X, y, params: dict, n_folds: int, seed: int) -> dict:
    """Mean out-of-fold RMSE for one configuration."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_rmse = []
    for train_idx, val_idx in kfold.split(X):
        model = make_model(params)
        model.fit(X[train_idx], y[train_idx])
        fold_rmse.append(regression_metrics(y[val_idx], model.predict(X[val_idx]))["rmse"])
    return {
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse)),
        "cv_rmse_folds": [float(v) for v in fold_rmse],
    }


def search_xgb_models(X_train, y_train, cfg) -> tuple[dict, dict, list[dict]]:
    """Select hyperparameters on cross-validated train performance only."""
    configs = sample_configs(cfg["n_trials"], cfg["seed"])
    print(f"\nCross-validating {len(configs)} configurations ({cfg['n_folds']} folds)")

    trials = []
    best_params, best_cv = None, None
    for i, params in enumerate(configs, start=1):
        cv = cross_validate(X_train, y_train, params, cfg["n_folds"], cfg["seed"])
        trials.append({"params": params, **cv})
        marker = ""
        if best_cv is None or cv["cv_rmse_mean"] < best_cv["cv_rmse_mean"]:
            best_params, best_cv = deepcopy(params), cv
            marker = "  <- new best"
        print(
            f"  trial {i:>2}/{len(configs)}: CV RMSE = {cv['cv_rmse_mean']:.4f} "
            f"+/- {cv['cv_rmse_std']:.4f}  {params}{marker}"
        )

    return best_params, best_cv, trials


# -------------------------
#  Persistence
# -------------------------
def save_xgb_model(model, metadata: dict, path_prefix):
    """Save XGBoost model + metadata to disk."""
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(f"{path_prefix}.json")

    metadata = metadata.copy()
    metadata["saved_at"] = datetime.now().isoformat()
    metadata["model_format"] = "xgboost-json"

    with open(f"{path_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {path_prefix}.json")
    print(f"Metadata saved to {path_prefix}_metadata.json")


def main(cfg=CONFIG, raw_csv=None):
    df_train, df_test = load_splits(cfg, raw_csv=raw_csv)

    print("\nFeaturising train split")
    X_train, y_train = df_to_maccs(df_train)
    print("Featurising test split")
    X_test, y_test = df_to_maccs(df_test)

    best_params, best_cv = search_xgb_models(X_train, y_train, cfg)[:2]

    # Refit the selected configuration on the full train split, then read the
    # test split for the first and only time.
    model = make_model(best_params)
    model.fit(X_train, y_train)
    test_metrics = regression_metrics(y_test, model.predict(X_test))

    print("\n==============================")
    print("Selected hyperparameters:", best_params)
    print(f"CV RMSE (train split):  {best_cv['cv_rmse_mean']:.3f} +/- {best_cv['cv_rmse_std']:.3f}")
    print(f"Test RMSE:              {test_metrics['rmse']:.3f}")
    print(f"Test MAE:               {test_metrics['mae']:.3f}")
    print(f"Test R2:                {test_metrics['r2']:.3f}")
    print(f"Test molecules:         {test_metrics['n_samples']}")
    print("==============================")

    save_xgb_model(
        model,
        {
            "dataset": cfg["dataset"],
            "target": cfg["target_column"],
            "featurisation": "167-bit MACCS keys (RDKit), feature index k = MACCS key k",
            "selection": (
                f"{cfg['n_folds']}-fold cross-validation on the train split over "
                f"{cfg['n_trials']} random hyperparameter draws, seed {cfg['seed']}"
            ),
            "hyperparameters": {**FIXED_PARAMS, **best_params},
            "searched_hyperparameters": best_params,
            "cv_metrics": best_cv,
            "test_metrics": test_metrics,
            "n_train_molecules": int(len(y_train)),
            "versions": {
                "xgboost": xgboost.__version__,
                "scikit-learn": sklearn.__version__,
                "rdkit": rdBase.rdkitVersion,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        path_prefix=cfg["model_prefix"],
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="raw dataset export to derive the splits from, used only when the "
        "cached split CSVs are absent",
    )
    main(raw_csv=parser.parse_args().raw_csv)
