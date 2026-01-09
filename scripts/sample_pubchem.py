import json
import random
from typing import List, Optional, Tuple
from urllib.request import urlopen

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


MAX_CID = 500_000_000 # overestimate to have unbiased sampling (just retries if not found)


def cid_to_mol(cid: int) -> Optional[Chem.Mol]:
    """
    Convert a PubChem CID to an RDKit Mol using PUG REST and canonical SMILES.
    Returns None if anything fails.
    """
    # Request multiple SMILES types as fallback
    url = f"{PUG_REST_BASE}/compound/cid/{cid}/property/CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES/JSON"
    try:
        with urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        props = data["PropertyTable"]["Properties"]
        if not props:
            return None
        
        prop_dict = props[0]
        # Try different SMILES types in order of preference
        smiles = None
        for smiles_type in ["CanonicalSMILES", "IsomericSMILES", "SMILES", "ConnectivitySMILES"]:
            if smiles_type in prop_dict and prop_dict[smiles_type]:
                smiles = prop_dict[smiles_type]
                break
        
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def sample_random_molecule_from_pubchem(
    max_tries: int = 1000,
) -> Optional[Chem.Mol]:
    """
    Sample a random RDKit Mol from PubChem by random CID in [1, MAX_CID].
    Returns None if no valid molecule is found after max_tries attempts.
    """
    for _ in range(max_tries):
        cid = random.randint(1, MAX_CID)
        mol = cid_to_mol(cid)
        if mol is not None:
            return mol
    return None


def qed_score(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))


def molecular_weight(mol: Chem.Mol) -> float:
    return float(Descriptors.MolWt(mol))


def fingerprint(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
    """
    Morgan fingerprint (ECFP-like) used for similarity.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return mfpgen.GetFingerprint(mol)


def similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    fp1 = fingerprint(mol1)
    fp2 = fingerprint(mol2)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def is_single_molecule(mol: Chem.Mol) -> bool:
    """
    Check if the molecule is a single connected component (not multiple disconnected molecules).
    """
    if mol is None:
        return False
    frags = Chem.GetMolFrags(mol, asMols=True)
    return len(frags) == 1


def sample_molecules(
    n: int,
    max_outer_tries: int = 100_000,
    qed_threshold: float = 0.5,
    mw_threshold: float = 500.0,
    similarity_threshold: float = 0.2,
) -> List[str]:
    """
    Sample `n` molecules from PubChem and return a list of canonical SMILES.

    Constraints:
      - Single connected molecule (not multiple disconnected parts)
      - QED(mol) < qed_threshold
      - MW(mol)  < mw_threshold
      - pairwise Tanimoto similarity < similarity_threshold
    """
    # Keep both Mol and SMILES internally
    samples: List[Tuple[Chem.Mol, str]] = []
    tries = 0

    while len(samples) < n and tries < max_outer_tries:
        tries += 1

        mol = sample_random_molecule_from_pubchem()
        if mol is None:
            continue

        # Check that it's a single connected molecule (not multiple disconnected parts)
        if not is_single_molecule(mol):
            continue

        if qed_score(mol) >= qed_threshold:
            continue

        if molecular_weight(mol) >= mw_threshold:
            continue

        # Similarity filter
        too_similar = False
        for existing_mol, _ in samples:
            if similarity(mol, existing_mol) > similarity_threshold:
                too_similar = True
                break

        smi = Chem.MolToSmiles(mol, canonical=True)

        if not too_similar:
            samples.append((mol, smi))
            print(f"Added molecule {len(samples)}: {smi}")

    # Return only SMILES
    return [smi for _, smi in samples]


def save_smiles_txt(smiles: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for smi in smiles:
            f.write(smi + "\n")


if __name__ == "__main__":    
    random.seed(42)
    smiles_list = sample_molecules(n=20)

    for i, smi in enumerate(smiles_list, start=1):
        print(i, smi)

    save_smiles_txt(smiles_list, "data/molecules/sampled_molecules.txt")
