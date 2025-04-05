from rdkit import Chem
import numpy as np
import pandas as pd

def return_vocabulary(csv="cleaned_smiles.csv"):
    try:
        clean_smiles = pd.read_csv(csv)["smiles"].tolist() # Specify the column name
    except KeyError:
        clean_smiles = pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv').iloc[:, 0].values  # Use first column
    char_to_idx, idx_to_char = create_vocabulary(clean_smiles)
    return char_to_idx, idx_to_char

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
    
def get_pdb_id_from_sequence(sequence):
    """
    Get a PDB ID for a protein sequence by searching the PDB database
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        str: PDB ID if found, None otherwise
    """
    try:
        import requests
        
        # Using NCBI BLAST REST API to search against PDB
        url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
        
        # First submit the search
        params = {
            'PROGRAM': 'blastp',
            'DATABASE': 'pdb',
            'QUERY': sequence,
            'CMD': 'Put',
            'FORMAT_TYPE': 'JSON2'
        }
        
        print("Submitting sequence to BLAST...")
        response = requests.post(url, data=params)
        
        # Extract RID (Request ID) from the response
        import re
        rid_match = re.search(r'RID = (.*)', response.text)
        if not rid_match:
            print("Could not get BLAST request ID")
            return None
            
        rid = rid_match.group(1).strip()
        print(f"BLAST search submitted with ID: {rid}")
        
        # Wait for results (polling)
        import time
        params = {
            'CMD': 'Get',
            'RID': rid,
            'FORMAT_TYPE': 'JSON2'
        }
        
        max_tries = 30
        for i in range(max_tries):
            time.sleep(10)  # Wait 10 seconds between checks
            print(f"Checking BLAST results (attempt {i+1}/{max_tries})...")
            response = requests.get(url, params=params)
            
            if "Status=READY" in response.text:
                print("BLAST search completed")
                break
                
        # Parse the results
        if "Status=READY" in response.text:
            results_match = re.search(r'<Iteration_hits>(.*?)</Iteration_hits>', response.text, re.DOTALL)
            if results_match:
                hits_text = results_match.group(1)
                
                # Extract PDB IDs from hits
                pdb_id_matches = re.findall(r'<Hit_id>(pdb\|([a-zA-Z0-9]+))</Hit_id>', hits_text)
                
                if pdb_id_matches:
                    # Return the first (best) match
                    pdb_id = pdb_id_matches[0][1].lower()
                    print(f"Found matching PDB ID: {pdb_id}")
                    return pdb_id
        
        print("No matching PDB structures found")
        return None
        
    except Exception as e:
        print(f"Error searching for PDB ID: {str(e)}")
        return None