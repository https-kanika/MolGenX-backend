Module MolGenX-backend.utils
============================

Functions
---------

`create_vocabulary(smiles_data)`
:   Create character vocabulary from SMILES strings

`get_pdb_id_from_sequence(sequence)`
:   Get a PDB ID for a protein sequence by searching the PDB database
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        str: PDB ID if found, None otherwise

`return_vocabulary(csv='cleaned_smiles.csv')`
:   

`validate_molecule(smiles: str) ‑> bool`
:   Validate if the generated SMILES represents a valid molecule
    
    Args:
        smiles (str): SMILES string to validate
    
    Returns:
        bool: True if molecule is valid, False otherwise