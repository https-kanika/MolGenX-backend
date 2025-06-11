from rdkit import Chem
import numpy as np
import pandas as pd
import os

def return_vocabulary(csv="cleaned_smiles.csv"):
    """
    Reads a CSV file containing SMILES strings and generates character-to-index and index-to-character vocabularies.

    Parameters:
        csv (str): Path to the CSV file containing a 'smiles' column. Defaults to 'cleaned_smiles.csv'.

    Returns:
        tuple: A tuple containing two dictionaries:
            - char_to_idx (dict): Mapping from characters to their corresponding indices.
            - idx_to_char (dict): Mapping from indices to their corresponding characters.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        Exception: For other exceptions raised during file reading or vocabulary creation.

    Notes:
        If the specified CSV file does not contain a 'smiles' column, the function attempts to read from
        '250k_rndm_zinc_drugs_clean_3.csv' and uses the first column as the source of SMILES strings.
    """
    try:
        clean_smiles = pd.read_csv(csv)["smiles"].tolist() # Specify the column name
    except KeyError:
        clean_smiles = pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv').iloc[:, 0].values  # Use first column
    char_to_idx, idx_to_char = create_vocabulary(clean_smiles)
    return char_to_idx, idx_to_char

def create_vocabulary(smiles_data):
    """
    Creates a character-level vocabulary from a list of SMILES strings.
    This function generates mappings from characters to indices and vice versa,
    including all unique characters present in the provided SMILES data. It also
    adds a special "<PAD>" token mapped to index 0 for padding purposes.
    Args:
        smiles_data (list of str): List of SMILES strings to extract characters from.
    Returns:
        tuple:
            - char_to_idx (dict): Mapping from character to unique index (int), with "<PAD>" as 0.
            - idx_to_char (dict): Mapping from index (int) to character.
    """
    # Include all possible SMILES characters and special tokens
    chars = set("".join(smiles_data))
    char_to_idx = {char: idx +1 for idx,char in enumerate(sorted(chars))}
    char_to_idx["<PAD>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

def validate_molecule(smiles: str) -> bool:
    """ Validates whether a given SMILES string represents a valid and canonical molecule.
    This function removes any "<PAD>" tokens from the input SMILES string, attempts to parse it into a molecule object,
    and checks if the molecule can be successfully canonicalized back to the original SMILES string.
        smiles (str): The SMILES string to validate.
        bool: True if the SMILES string is valid and canonical, False otherwise.
    
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
    
def get_compound_files(output_dir="compound_visualizations"):
    """
    Helper function to read visualization files and encode image/PDB data
    
    Args:
        output_dir: Directory containing visualization files
        
    Returns:
        Dictionary with file contents organized by compound
    """
    import base64
    import os
    
    compound_data = {
        "compounds": []
    }
    
    try:
        # Check if directory exists
        if not os.path.exists(output_dir):
            return {"error": f"Visualization directory {output_dir} not found"}
            
        # Get all files in the directory
        files = os.listdir(output_dir)
        
        # Find the maximum compound number
        compound_nums = []
        for file in files:
            if file.startswith("compound_") and "_" in file:
                try:
                    num = int(file.split("_")[1])
                    compound_nums.append(num)
                except:
                    pass
        
        max_compounds = max(compound_nums) if compound_nums else 0
        
        # Process each compound
        for i in range(1, max_compounds + 1):
            compound_info = {
                "id": i,
                "images": {},
                "models": {}
            }
            
            # Get 2D image
            image_path = os.path.join(output_dir, f"compound_{i}_2D.png")
            if os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    compound_info["images"]["2d"] = f"data:image/png;base64,{encoded_image}"
            
            # Get PDB file
            pdb_path = os.path.join(output_dir, f"compound_{i}_3D.pdb")
            if os.path.exists(pdb_path):
                with open(pdb_path, "r") as pdb_file:
                    compound_info["models"]["pdb"] = pdb_file.read()
            
            # Get SDF file
            sdf_path = os.path.join(output_dir, f"compound_{i}_3D.sdf")
            if os.path.exists(sdf_path):
                with open(sdf_path, "r") as sdf_file:
                    compound_info["models"]["sdf"] = sdf_file.read()
            
            # Add viewer path
            viewer_path = f"compound_{i}_3D_viewer.html"
            if os.path.exists(os.path.join(output_dir, viewer_path)):
                compound_info["models"]["viewer"] = viewer_path
            
            # Only add compounds that have at least an image or a model
            if compound_info["images"] or compound_info["models"]:
                compound_data["compounds"].append(compound_info)
        
        # Add grid image if available
        grid_path = os.path.join(output_dir, "all_compounds_grid.png")
        if os.path.exists(grid_path):
            with open(grid_path, "rb") as grid_file:
                encoded_grid = base64.b64encode(grid_file.read()).decode('utf-8')
                compound_data["grid"] = f"data:image/png;base64,{encoded_grid}"
        
    except Exception as e:
        return {"error": f"Error processing visualization files: {str(e)}"}
    
    return compound_data