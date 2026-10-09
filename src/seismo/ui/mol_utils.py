"""Molecule visualization utilities."""

from io import BytesIO
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw


def smiles_to_image(smiles: str, size: tuple[int, int] = (400, 400)) -> Image.Image | None:
    """
    Convert SMILES string to 2D structure image.
    
    Args:
        smiles: SMILES string of the molecule
        size: Image size as (width, height)
    
    Returns:
        PIL Image of the 2D structure, or None if SMILES is invalid
    """
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return Draw.MolToImage(mol, size=size)


def smiles_to_image_bytes(smiles: str, size: tuple[int, int] = (400, 400)) -> bytes | None:
    """
    Convert SMILES string to PNG image bytes.
    
    Args:
        smiles: SMILES string of the molecule
        size: Image size as (width, height)
    
    Returns:
        PNG image as bytes, or None if SMILES is invalid
    """
    img = smiles_to_image(smiles, size)
    if img is None:
        return None
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def compare_molecules_image(
    smiles1: str, 
    smiles2: str, 
    labels: tuple[str, str] = ("Before", "After"),
    size: tuple[int, int] = (300, 300)
) -> Image.Image | None:
    """
    Create side-by-side comparison image of two molecules.
    
    Args:
        smiles1: SMILES string of first molecule
        smiles2: SMILES string of second molecule
        labels: Labels for the molecules
        size: Size of each individual molecule image
    
    Returns:
        PIL Image with side-by-side comparison, or None if either SMILES is invalid
    """
    mol1 = Chem.MolFromSmiles(smiles1) if smiles1 else None
    mol2 = Chem.MolFromSmiles(smiles2) if smiles2 else None
    
    mols = []
    legends = []
    
    if mol1 is not None:
        mols.append(mol1)
        legends.append(labels[0])
    if mol2 is not None:
        mols.append(mol2)
        legends.append(labels[1])
    
    if not mols:
        return None
    
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=size,
        legends=legends
    )
