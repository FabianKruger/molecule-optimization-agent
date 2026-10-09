import math

from rdkit import Chem, rdBase
from rdkit.Chem.QED import (
    WEIGHT_MEAN,
    StructuralAlerts,
    StructuralAlertSmarts,
    ads,
    adsParameters,
    properties,
)

from .base import OracleResult

rdBase.DisableLog("rdApp.error")


class ExplainableQedOracle:
    """
    Explainable QED (Quantitative Estimate of Drug-likeness) Oracle.
    
    QED = exp(Σ(wᵢ × log(ADSᵢ)) / Σwᵢ)
    Score range: 0-1, higher is better.
    """

    PROPERTY_NAMES = ["MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS"]
    
    # Optimal ranges where ADS ≥ 0.95
    OPTIMAL_RANGES = {
        "MW": "281-332 Da",
        "ALOGP": "1.95-3.57",
        "HBA": "2-3",
        "HBD": "1",
        "PSA": "40-69 Ų",
        "ROTB": "2-4",
        "AROM": "1-2",
        "ALERTS": "0",
    }

    def __init__(self) -> None:
        pass

    def __call__(self, smiles: str) -> OracleResult:
        mol = Chem.MolFromSmiles(smiles)
        qed_props = properties(mol)
        weights = WEIGHT_MEAN

        # Calculate weighted log contributions
        contributions = {}
        for name, prop_value, weight in zip(self.PROPERTY_NAMES, qed_props, weights):
            ads_value = ads(prop_value, adsParameters[name])
            log_contrib = math.log(ads_value) if ads_value > 0 else -float("inf")
            contributions[name] = weight * log_contrib

        qed_score = math.exp(sum(contributions.values()) / sum(weights))

        # Find triggered structural alerts
        alerts = [
            StructuralAlertSmarts[i]
            for i, pattern in enumerate(StructuralAlerts)
            if mol.HasSubstructMatch(pattern)
        ]

        # Build explanation
        explanation = self._format_explanation(qed_props, contributions, alerts)

        return {"score": qed_score, "explanation": explanation}

    def _format_explanation(self, props, contributions: dict, alerts: list) -> str:
        lines = [
            "Weighted log contributions (closer to 0 = better, negative values hurt score):",
            "",
        ]

        prop_values = dict(props._asdict())
        for name in self.PROPERTY_NAMES:
            contrib = contributions[name]
            value = prop_values[name]
            optimal = self.OPTIMAL_RANGES[name]
            
            if name in ["HBA", "HBD", "ROTB", "AROM", "ALERTS"]:
                val_str = str(int(value))
            else:
                val_str = f"{value:.2f}"
            
            lines.append(f"  {name}: {contrib:+.3f} (value={val_str}, optimal={optimal})")

        if alerts:
            lines.append("")
            lines.append(f"{len(alerts)} structural alert(s) triggered. SMARTS patterns of alerts:")
            for smarts in alerts:
                lines.append(f"  {smarts}")

        return "\n".join(lines)
