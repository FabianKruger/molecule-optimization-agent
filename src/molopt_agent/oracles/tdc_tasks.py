import tdc

from .base import OracleResult


class TdcOracle:
    """
    Generic wrapper for TDC oracles.
    
    This oracle wraps any TDC oracle (e.g., DRD2, GSK3, JNK3, etc.)
    and provides a consistent interface compatible with the Oracle protocol.
    """

    def __init__(self, oracle_name: str, target_smiles: str | None = None, **kwargs):
        """
        Initialize TDC oracle.
        
        Args:
            oracle_name: Name of the TDC oracle (e.g., 'drd2', 'gsk3b', 'jnk3')
            target_smiles: Optional target SMILES for meta-oracles
            **kwargs: Additional parameters passed to tdc.Oracle
        """
        self.oracle_name = oracle_name
        self.target_smiles = target_smiles
        
        # Initialize TDC oracle
        self.tdc_oracle = tdc.Oracle(
            name=oracle_name,
            target_smiles=target_smiles,
            **kwargs
        )

    def __call__(self, smiles: str) -> OracleResult:
        """
        Evaluate a molecule using the TDC oracle.
        
        Args:
            smiles: SMILES string of the molecule to evaluate
            
        Returns:
            OracleResult with score and empty explanation (no XAI support)
        """
        try:
            score = self.tdc_oracle(smiles)
            # Ensure score is a float (TDC may return numpy types)
            score = float(score)
        except Exception as e:
            raise ValueError(f"Failed to evaluate molecule {smiles} with TDC oracle {self.oracle_name}: {e}")
        
        return {
            "score": score,
            "explanation": "",  # No explainability for now
        }

    def get_params(self) -> dict:
        """Return oracle configuration for sharing with objectives."""
        return {
            "oracle_name": self.oracle_name,
            "target_smiles": self.target_smiles,
        }
