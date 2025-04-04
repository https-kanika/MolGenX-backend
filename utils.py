from rdkit import Chem
import numpy as np
import pandas as pd
from optimize import DrugOptimizer

def return_vocabulary(csv="cleaned_smiles.csv"):
    try:
        clean_smiles = pd.read_csv(csv)["smiles"].tolist() # Specify the column name
    except KeyError:
        clean_smiles = pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv').iloc[:, 0].values  # Use first column
    char_to_idx, idx_to_char = create_vocabulary(clean_smiles)

def create_vocabulary(smiles_data):
    """Create character vocabulary from SMILES strings"""
    # Include all possible SMILES characters and special tokens
    chars = set("".join(smiles_data))
    char_to_idx = {char: idx +1 for idx,char in enumerate(sorted(chars))}
    char_to_idx["<PAD>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

def validate_molecule(smiles: str) -> bool:
    """
    Validate if the generated SMILES represents a valid molecule
    
    Args:
        smiles (str): SMILES string to validate
    
    Returns:
        bool: True if molecule is valid, False otherwise
    """
    # Remove the <PAD> token before validation
    smiles = smiles.replace("<PAD>", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and Chem.MolToSmiles(mol) == smiles  #this is a canonicalization check
    except:
        return False
    

def get_optimized_variants(protien_sequence,optimized_compounds,optimizer,optimization_params):
    top_compound = optimized_compounds[0]['smiles']
        
    variants = optimizer.generate_molecular_modifications(top_compound, 50)
    print(f"Generated {len(variants)} variants of top compound")

    variant_optimizer = DrugOptimizer(variants, protien_sequence)  
    optimized_variants = variant_optimizer.optimize(optimization_params)
    sorted_variants = sorted(optimized_variants, key=lambda x: x['score'], reverse=True)
    variant_optimizer.export_results(sorted_variants, "optimized_variants.csv")
    explanation = variant_optimizer.explain_results_with_gemini(sorted_variants)
    
    return sorted_variants, explanation