from rdkit.Chem.MACCSkeys import smartsPatts

def build_maccs_key_definitions() -> dict[int, str]:
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