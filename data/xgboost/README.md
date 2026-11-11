# pIC50 surrogate for SARS-CoV-2 M<sup>pro</sup>

Files in this directory:

- `xgb_maccs_best.json` — the XGBoost regressor used by `IC50MproOracle`
- `xgb_maccs_best_metadata.json` — selected hyperparameters, cross-validation and
  held-out metrics, package versions
- `asap_potency_train.csv`, `asap_potency_test.csv` — the training and held-out
  splits the model is built from

## Data provenance

The data come from the ASAP Discovery antiviral potency challenge, dataset
`asap-discovery/antiviral-potency-2025-unblinded`, obtained through the Polaris
hub. The raw challenge data are deposited at Zenodo as "ASAP-Polaris-OpenADMET
blind challenge data", DOI [10.5281/zenodo.15582067](https://doi.org/10.5281/zenodo.15582067).

The Polaris dataset page states a licence of CC0 1.0 and the Zenodo deposit
states CC BY 4.0. The copies here are redistributed under the stricter of the
two, with credit to the ASAP Discovery, Polaris and OpenADMET consortia and
the contributors listed on the Zenodo record.

## How the splits were derived

Produced by `scripts/xgboost_training.py`, which keeps the dataset's own `Set`
column as the train and test assignment and applies the preprocessing in
`preprocess_for_xgb`:

- rows without a SARS-CoV-2 M<sup>pro</sup> pIC50 value are dropped, which
  reduces 1,328 raw rows to 1,105
- `CXSMILES` is reduced to the SMILES part preceding the `|` delimiter
- only the SMILES and the SARS-CoV-2 M<sup>pro</sup> pIC50 are retained, so the
  MERS-CoV values and molecule names are not redistributed here

This leaves 842 training and 263 held-out molecules.

Molecules are featurised as the full 167-bit MACCS vector returned by RDKit, in
which bit k is MACCS key k and bit 0 is unused. Keeping all 167 bits means
feature index k is MACCS key k, which is the indexing `IC50MproOracle` relies on
when it reports SHAP attributions.

## Reproducing the model

With the split CSVs present, no Polaris access is needed:

    pixi run python scripts/xgboost_training.py

Hyperparameters are selected by 5-fold cross-validation on the training split
over 40 seeded random draws. The held-out split is read once, after the
configuration is fixed.
