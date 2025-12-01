import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import OracleResult


class MolEncoderOpioidKiOracle:
    """
    Oracle using fabikru/molencoder-D3R-simple.

    primary_score = Ki_nM 
    metrics = {"ki_nM": Ki}
    explanation = short text
    """

    def __init__(self, model_name: str = "fabikru/molencoder-D3R-simple"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def _predict_ki_nM(self, smiles: str) -> float:
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=502,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze().item()
        return 10 ** (-logits)

    def __call__(self, smiles: str) -> OracleResult:
        ki_nM = self._predict_ki_nM(smiles)
        return {
            "score": ki_nM, 
            "explanation": f"Predicted Ki ≈ {ki_nM:.2f} nM (lower is better).",
        }
