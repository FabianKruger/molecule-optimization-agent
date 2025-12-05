import polaris as po
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from xgboost import XGBRegressor
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import random
import itertools
from copy import deepcopy


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
    # ----------------------------------------
    # 1. Rename columns
    # ----------------------------------------
    df = df.rename(columns={
        column_name_target_values: "pIC50",
        column_name_smiles: "SMILES"
    })

   # 2. Drop NaNs in pIC50
    df = df.dropna(subset=["pIC50"]).reset_index(drop=True)
    df["pIC50"] = df["pIC50"].astype("float64")

    # 3. CXSMILES → SMILES (safe)
    def cx_to_smiles(s):
        if isinstance(s, str):
            base = s.split("|")[0]   # takes SMILES part
            return base.strip()      # remove left-over spaces
        return s

    df["SMILES"] = df["SMILES"].apply(cx_to_smiles)

    # 4. Check Set column
    if "Set" not in df.columns:
        raise ValueError('Dataframe must contain a "Set" column with values "train" or "test".')

    # 5. Train/test split
    df_train = df[df["Set"].str.lower() == "train"].reset_index(drop=True)
    df_test  = df[df["Set"].str.lower() == "test"].reset_index(drop=True)

    # 6. Print split percentages
    total = len(df)
    print(f"Total usable rows: {total}")
    print(f"Train: {len(df_train)} ({len(df_train)/total*100:.2f}%)")
    print(f"Test:  {len(df_test)} ({len(df_test)/total*100:.2f}%)")

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

def df_to_maccs(df, smiles_column):
    fingerprints = []
    valid_idx = []

    for i, smi in enumerate(df[smiles_column]):
        fp = smiles_to_maccs(smi)
        if fp is not None:
            fingerprints.append(fp)
            valid_idx.append(i)
    
    X = np.vstack(fingerprints)
    y = df.loc[valid_idx, "pIC50"].values
    return X, y, valid_idx


# -------------------------
#  Train XGBoost on MACCS features
# -------------------------
def train_xgb_maccs(
        df,
        max_depth=6,
        min_child_weight=5,
        gamma=0.1,
        colsample_bytree=0.5,
        smiles_col="SMILES",
        target_col="pIC50"
    ):
    X, y, valid_idx = df_to_maccs(df, smiles_col)

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=2,
        grow_policy="lossguide",
        max_leaves=128,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )

    model.fit(X, y)
    return model, X, y, valid_idx


def save_xgb_model(model, metadata: dict, path_prefix: str = "xgb_maccs"):
    """Save XGBoost model + metadata to disk."""
    
    # Save model
    model.save_model(f"{path_prefix}.json")
    
    # Add timestamp + versioning info
    metadata = metadata.copy()
    metadata["saved_at"] = datetime.now().isoformat()
    metadata["model_format"] = "xgboost-json"
    
    # Save metadata
    with open(f"{path_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {path_prefix}.json")
    print(f"Metadata saved to {path_prefix}_metadata.json")


def evaluate_on_test(model, df_test):
    X_test = []
    y_test = []
    valid_idx = []

    for i, row in df_test.iterrows():
        fp = smiles_to_maccs(row["SMILES"])
        if fp is None:
            continue
        X_test.append(fp)
        y_test.append(row["pIC50"])
        valid_idx.append(i)

    if len(X_test) == 0:
        raise ValueError("No valid SMILES in test set.")

    X_test = np.vstack(X_test)
    y_test = np.array(y_test, dtype=float)

    preds = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_test_samples": len(y_test),
        "n_invalid_smiles": len(df_test) - len(y_test)
    }

    return preds, metrics


def search_xgb_models(
        df_train,
        df_test,
        n_trials=40,
        save_prefix="xgb_maccs_best"
    ):
    # Hyperparameter search space
    space = {
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.5, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0]
    }

    best_rmse = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    for i in range(n_trials):
        # Sample a random combination
        params = {
            "max_depth": random.choice(space["max_depth"]),
            "min_child_weight": random.choice(space["min_child_weight"]),
            "gamma": random.choice(space["gamma"]),
            "colsample_bytree": random.choice(space["colsample_bytree"])
        }

        print(f"\n=== Trial {i+1}/{n_trials} ===")
        print("Params:", params)

        # Train model
        model, X_train, y_train, valid_idx = train_xgb_maccs(
            df_train,
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            colsample_bytree=params["colsample_bytree"],
        )

        # Evaluate on test
        preds, metrics = evaluate_on_test(model, df_test)
        rmse = metrics["rmse"]

        print(f"RMSE = {rmse:.4f}")

        # Check if best model so far
        if rmse < best_rmse:
            print("→ New best model found! Saving…")

            best_rmse = rmse
            best_model = deepcopy(model)
            best_params = params
            best_metrics = metrics

            # Save using your existing function
            metadata = {
                "hyperparameters": params,
                "metrics": metrics,
            }
            save_xgb_model(best_model, metadata, path_prefix=save_prefix)

    print("\n\n==============================")
    print("Best model found:")
    print("Hyperparameters:", best_params)
    print("Metrics:", best_metrics)
    print("==============================\n")



# Load the dataset from the Hub
dataset = po.load_dataset("asap-discovery/antiviral-potency-2025-unblinded")

# Convert to dataframe
df = polaris_to_dataframe(dataset)

# Preprocess data
df_train, df_test = preprocess_for_xgb(df, column_name_target_values="pIC50 (SARS-CoV-2 Mpro)", column_name_smiles="CXSMILES")

# Random 4x4x4x4 hyperparameter search space and save best (lowest rmse) model
search_xgb_models(df_train, df_test, save_prefix="xgb_maccs_best")