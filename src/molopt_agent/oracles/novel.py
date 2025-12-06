import requests

from .base import OracleResult


class NovelOracle:
    """
    Oracle that checks if a molecule is novel (not present in PubChem).
    
    Returns score=1.0 if the molecule is novel (not found in PubChem),
    and score=0.0 if it already exists in the database.
    """

    PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"

    def __init__(self) -> None:
        pass

    def __call__(self, smiles: str) -> OracleResult:
        exists = self._pubchem_exists(smiles)
        
        if exists:
            return {
                "score": 0.0,
                "explanation": (
                    "This molecule already exists in PubChem. "
                    "Please propose a structurally different molecule that is not in the database."
                ),
            }
        else:
            return {
                "score": 1.0,
                "explanation": (
                    "This molecule is novel - it does not exist in PubChem."
                ),
            }

    def _pubchem_exists(self, smiles: str) -> bool:
        """
        Return True if PubChem has at least one real CID (> 0) for the given SMILES.
        
        Raises requests.HTTPError on non-2xx responses.
        """
        r = requests.get(self.PUBCHEM_URL, params={"smiles": smiles}, timeout=10)
        r.raise_for_status()

        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])

        # PubChem CIDs are positive integers; anything <= 0 is not a real compound ID
        return any(isinstance(cid, int) and cid > 0 for cid in cids)
