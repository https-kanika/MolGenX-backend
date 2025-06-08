import os
import shutil
from gradio_client import Client

def call_diffdock_api(input_structure, input_pdb, smiles_string, input_ligand):
    """
    Calls the DiffDock API with given inputs and retrieves binding affinity results.
    """
    client = Client("https://reginabarzilaygroup-diffdock-web.hf.space/--replicas/ya4cb/")
    
    # Check file paths exist
    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"PDB file not found: {input_pdb}")
    if not os.path.exists(input_ligand):
        raise FileNotFoundError(f"Ligand file not found: {input_ligand}")
    
    # The correct parameter structure based on API description
    result = client.predict(
        input_structure,  # PDB ID (str)
        input_pdb,        # Input PDB File (file path)
        smiles_string,    # SMILES string (str)
        input_ligand,     # Input Ligand (file path)
        None,             # Configuration YML file (optional) - pass None for default
        10,               # Samples Per Complex (int)
        fn_index=1
    )
    
    # Process result
    if isinstance(result, tuple) and len(result) >= 2:
        output_file_path = result[1]  # Second item is the file path
        
        # Move the file to the same directory as diffdock.py if it exists
        if output_file_path and os.path.exists(output_file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            new_output_path = os.path.join(script_dir, os.path.basename(output_file_path))
            shutil.move(output_file_path, new_output_path)
            print(f"Output saved to: {new_output_path}")
            
            # Return binding affinity (typically in result[0])
            return result[0]
        else:
            print(f"Warning: Output file not found at {output_file_path}")
            return result[0] if result[0] else None
    else:
        print(f"Unexpected result format: {result}")
        return None

if __name__ == "__main__":
    # Use raw strings or forward slashes to avoid escape sequence issues
    pdb_file = r"compound_visualizations\target_protein.pdb"
    ligand_file = r"compound_visualizations\compound_1_3D.sdf"
    smiles = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen
    
    print(f"Testing DiffDock API with:")
    print(f"- PDB ID: 6w70")
    print(f"- PDB file: {pdb_file}")
    print(f"- SMILES: {smiles}")
    print(f"- Ligand file: {ligand_file}")
    
    result = call_diffdock_api("6w70", pdb_file, smiles, ligand_file)
    print("DiffDock Binding Affinity Results:", result)