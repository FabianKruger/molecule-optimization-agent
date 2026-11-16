import time

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

    MAX_ATTEMPTS = 3
    RETRY_BACKOFF_SECONDS = 1.0

    def _pubchem_exists(self, smiles: str) -> bool:
        """
        Return True if PubChem has at least one real CID (> 0) for the given SMILES.

        PubChem answers 200 with CID 0 for a SMILES it can parse but does not
        hold, so a molecule being absent is not an error. A 400
        PUGREST.BadRequest means PubChem could not parse the SMILES at all, which
        cannot correspond to a stored compound, so it is reported as absent. This
        happens for RDKit-valid but chemically unusual proposals, for example an
        [SH] in a ring, and must not abort an optimisation run. Transient network
        and server errors are retried before giving up.
        """
        last_error: Exception | None = None

        for attempt in range(self.MAX_ATTEMPTS):
            try:
                r = requests.get(self.PUBCHEM_URL, params={"smiles": smiles}, timeout=10)

                # PubChem cannot parse it, or holds no record under that query
                if r.status_code in (400, 404):
                    return False

                r.raise_for_status()
                data = r.json()
                cids = data.get("IdentifierList", {}).get("CID", [])

                # PubChem CIDs are positive integers; anything <= 0 is not a real compound ID
                return any(isinstance(cid, int) and cid > 0 for cid in cids)

            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status is not None and 400 <= status < 500:
                    raise  # a client error we did not anticipate, do not mask it
                last_error = error
            except (requests.ConnectionError, requests.Timeout, ValueError) as error:
                last_error = error

            if attempt < self.MAX_ATTEMPTS - 1:
                time.sleep(self.RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise last_error  # type: ignore[misc]
