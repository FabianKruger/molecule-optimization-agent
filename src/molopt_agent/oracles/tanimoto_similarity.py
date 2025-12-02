from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.MACCSkeys import smartsPatts
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.DataStructs import TanimotoSimilarity

from .base import OracleResult


def _build_maccs_key_definitions() -> dict[int, str]:
    """
    Build MACCS key definitions from RDKit's smartsPatts.
    
    For most keys, uses the SMARTS pattern directly from RDKit.
    For special keys (1, 125, 166) that use '?' placeholder, 
    describes the RDKit function used instead.
    """
    definitions = {}
    
    for key, (smarts, count) in smartsPatts.items():
        if smarts == '?':
            # Special cases handled by RDKit functions, not SMARTS
            if key == 1:
                definitions[key] = "ISOTOPE (rdkit: mol.GetAtoms() isotope check)"
            elif key == 125:
                definitions[key] = "Aromatic Ring > 1 (rdkit: GetRingInfo().BondRings() aromaticity check)"
            elif key == 166:
                definitions[key] = "Fragments > 1 (rdkit: Chem.GetMolFrags(mol))"
            else:
                raise ValueError(f"Unexpected '?' placeholder for MACCS key {key}")
        else:
            # Use SMARTS pattern directly, with count threshold if applicable
            if count > 0:
                definitions[key] = f"{smarts} (count > {count})"
            else:
                definitions[key] = smarts
    
    return definitions


MACCS_KEY_DEFINITIONS = _build_maccs_key_definitions()


class TanimotoSimilarityOracle:
    """
    Oracle computing Tanimoto similarity using MACCS keys against a target molecule.
    
    Returns similarity score (0-1, higher = more similar) and explanation of 
    which structural features differ between the query and target molecules.
    """

    def __init__(self, target_smiles: str):
        self.target_smiles = target_smiles
        self.target_mol = Chem.MolFromSmiles(target_smiles)
        if self.target_mol is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")
        self.target_fp = MACCSkeys.GenMACCSKeys(self.target_mol)
        self.target_on_bits = set(self.target_fp.GetOnBits())
        self.target_inchi_key = MolToInchiKey(self.target_mol)

    def __call__(self, smiles: str) -> OracleResult:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Check for identical molecule using InChIKey
        query_inchi_key = MolToInchiKey(mol)
        if query_inchi_key == self.target_inchi_key:
            return {
                "score": 0.0,
                "explanation": "IDENTICAL MOLECULE - This is the target molecule itself. "
                               "Please propose a structurally different molecule.",
            }

        fp = MACCSkeys.GenMACCSKeys(mol)
        similarity = TanimotoSimilarity(self.target_fp, fp)

        query_on_bits = set(fp.GetOnBits())

        # Bits only in target (features missing from query)
        missing_from_query = self.target_on_bits - query_on_bits
        # Bits only in query (extra features not in target)
        extra_in_query = query_on_bits - self.target_on_bits

        explanation = self._build_explanation(similarity, missing_from_query, extra_in_query)

        return {"score": similarity, "explanation": explanation}

    def _build_explanation(
        self, 
        similarity: float, 
        missing_from_query: set[int], 
        extra_in_query: set[int]
    ) -> str:
        lines = [
            f"Tanimoto Similarity (MACCS keys): {similarity:.4f}",
            "",
        ]

        if not missing_from_query and not extra_in_query:
            lines.append("All MACCS key bits match between query and target.")
            return "\n".join(lines)

        # Report features that reduce similarity
        if missing_from_query:
            lines.append(
                f"The following {len(missing_from_query)} features are present in TARGET but not in QUERY (reducing similarity):"
            )
            for bit in sorted(missing_from_query):
                desc = MACCS_KEY_DEFINITIONS.get(bit, f"MACCS_KEY_{bit}")
                lines.append(f"  Key {bit}: {desc}")
            lines.append("")

        if extra_in_query:
            lines.append(
                f"The following {len(extra_in_query)} features are present in QUERY but not in TARGET (reducing similarity):"
            )
            for bit in sorted(extra_in_query):
                desc = MACCS_KEY_DEFINITIONS.get(bit, f"MACCS_KEY_{bit}")
                lines.append(f"  Key {bit}: {desc}")

        return "\n".join(lines)

