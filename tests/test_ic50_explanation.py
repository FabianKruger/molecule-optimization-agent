"""Regression tests for the MACCS key mapping in the IC50 SHAP explanation.

Run with `pixi run pytest tests/` or directly with
`pixi run python tests/test_ic50_explanation.py`.

The invariant under test: RDKit's GenMACCSKeys returns 167 bits in which bit k
is MACCS key k (keys 1 to 166) and bit 0 is unused. The XGBoost model is
trained on that full bit string, so SHAP attribution index k belongs to key k.
An off-by-one here silently mislabels the chemistry reported to the agent.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.MACCSkeys import smartsPatts

from molopt_agent.oracles.ic50mpro import IC50MproOracle

# MACCS key 154 is [#6]=[#8], a carbonyl.
CARBONYL_KEY = 154
# Acetaminophen, whose amide carbonyl sets that key.
CARBONYL_SMILES = "CC(=O)Nc1ccc(O)cc1"


class _StubExplainer:
    """Stands in for the SHAP explainer so no model file is needed."""

    expected_value = 0.0


def _oracle_without_model() -> IC50MproOracle:
    oracle = IC50MproOracle.__new__(IC50MproOracle)
    oracle.explainer = _StubExplainer()
    return oracle


def _fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    return np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=np.float32)


def _parse_rows(explanation: str) -> list[dict]:
    """Pull the table rows out of the rendered explanation."""
    rows = []
    for line in explanation.splitlines():
        if not line.startswith("|"):
            continue
        fields = [f.strip() for f in line.strip("|").split("|")]
        if len(fields) != 4:
            continue
        try:
            attribution = float(fields[0])
        except ValueError:
            continue  # header or separator row
        rows.append(
            {
                "attribution": attribution,
                "value": int(fields[1]),
                "key": int(fields[2]),
                "smarts": fields[3],
            }
        )
    return rows


def test_attribution_index_maps_to_the_same_maccs_key():
    fp = _fingerprint(CARBONYL_SMILES)
    attribution = np.zeros(len(fp), dtype=np.float64)
    attribution[CARBONYL_KEY] = 1.0  # the single largest attribution

    rows = _parse_rows(_oracle_without_model()._build_explanation(attribution, 6.0, fp))
    top = rows[0]

    assert top["key"] == CARBONYL_KEY, (
        f"attribution at index {CARBONYL_KEY} was reported as key {top['key']}"
    )
    assert smartsPatts[CARBONYL_KEY][0] in top["smarts"]
    # The molecule contains a carbonyl, so the key must be reported as present.
    assert top["value"] == 1


def test_reported_fingerprint_value_matches_the_reported_key():
    fp = _fingerprint(CARBONYL_SMILES)
    attribution = np.random.default_rng(0).normal(size=len(fp))

    rows = _parse_rows(_oracle_without_model()._build_explanation(attribution, 6.0, fp))

    assert len(rows) == 20  # top 10 and bottom 10
    for row in rows:
        assert 1 <= row["key"] <= 166
        assert row["value"] == int(fp[row["key"]]), (
            f"key {row['key']} reported as {row['value']} but fingerprint bit is "
            f"{int(fp[row['key']])}"
        )
        assert smartsPatts[row["key"]][0] in row["smarts"] or "rdkit:" in row["smarts"]


def test_attribution_length_is_validated():
    fp = _fingerprint(CARBONYL_SMILES)
    try:
        _oracle_without_model()._build_explanation(np.zeros(len(fp) - 1), 6.0, fp)
    except ValueError:
        return
    raise AssertionError("a mismatched attribution length should raise")


if __name__ == "__main__":
    test_attribution_index_maps_to_the_same_maccs_key()
    test_reported_fingerprint_value_matches_the_reported_key()
    test_attribution_length_is_validated()
    print("ok")
