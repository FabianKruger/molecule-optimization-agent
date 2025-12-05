from xgboost import XGBRegressor
import json
import shap
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from .base import OracleResult
from molopt_agent.oracles.utils import build_maccs_key_definitions

class XGBoostMaccsSARSCoV2Oracle:
    """
    Oracle using XGBoost model trained using training script on SARS-CoV2 pIC50 data from https://polarishub.io/competitions/asap-discovery/antiviral-potency-2025 

    primary_score = ic50_nM 
    metrics = {"ic50_nM": IC50}
    explanation = markdown table with attribution values of most important MACCS keys
    """

    def _load_xgb_model(self, model_name):
        """Load XGBoost model"""
        model = XGBRegressor()
        model.load_model(f"{model_name}.json")
        return model

    def __init__(self, model_name):
        self.model = self._load_xgb_model(model_name=model_name)
        self.explainer = shap.Explainer(self.model)

    def _prediction(self, smiles: str) -> float:
        # smiles to maccs
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp = np.array(list(fp.ToBitString()), dtype=np.float32)
        if fp is None:
            raise ValueError("Invalid SMILES.")
        # predict pic50 value
        pic50 = self.model.predict(fp.reshape(1, -1))[0]
        # get attribution values
        attr = self.explainer.shap_values(fp.reshape(1, -1))
        return pic50, attr, fp
    
    def _build_explanation(
        self, 
        attribution: set[float], 
        pic50: float,
        fp: set[int]
    ) -> str:
        MACCS_KEY_DEFINITIONS = build_maccs_key_definitions()

        indexed = [(attr, idx + 1) for idx, attr in enumerate(attribution)]
        indexed_sorted = sorted(indexed, key=lambda x: x[0], reverse=True)
        top_values = indexed_sorted[:10]
        low_values = indexed_sorted[-10:]

        lines = [
            "The following attribution values are calculated using SHAP. These are dependent on the individual input:",
            "",
        ]
        lines.append("error of SHAP = prediction - (expected value + total SHAP attribution)")
        lines.append(f"prediction (pIC50) = {pic50:.6f}")
        lines.append(f"expected value = {self.explainer.expected_value:.6f}")
        lines.append(f"total SHAP attribution = {sum(attribution):.6f}")
        lines.append(f"error = {(pic50 - (self.explainer.expected_value + sum(attribution)))}")
        lines.append("")

        lines.append("Top 10: highest attribution values (pushing towards higher pIC50 values therefore lower IC50 values):")
        lines.append("")
        lines.append("| Attribution | MACCS fingerprint value | MACCS key | SMARTS substructure |")
        lines.append("|-------------|-------------------------|-----------|---------------------|")

        for attr, key_id in top_values:
            desc = MACCS_KEY_DEFINITIONS.get(key_id, f"MACCS_KEY_{key_id}")
            lines.append(
                        f"| {attr:.6f} | {int(fp[key_id-1])} | {key_id} | {desc} |"
                    )
        lines.append("")

        lines.append("Bottom 10: lowest attribution values (pushing towards lower pIC50 values therefore higher IC50 values):")
        lines.append("")
        lines.append("| Attribution | MACCS fingerprint value | MACCS key | SMARTS substructure |")
        lines.append("|-------------|-------------------------|-----------|---------------------|")

        for attr, key_id in low_values:
            desc = MACCS_KEY_DEFINITIONS.get(key_id, f"MACCS_KEY_{key_id}")
            lines.append(
                        f"| {attr:.6f} | {int(fp[key_id-1])} | {key_id} | {desc} |"
                    )
                
        return "\n".join(lines)

    def __call__(self, smiles: str) -> OracleResult:
        pic50, attr, fp = self._prediction(smiles)

        ic50_nM = 10 ** (9 - pic50)
        ic50_nM = ic50_nM.item()

        explanation = self._build_explanation(attr[0], pic50, fp)

        return {
            "score": ic50_nM, 
            "explanation": explanation,
        }
    
