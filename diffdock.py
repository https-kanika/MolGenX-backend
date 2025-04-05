import os
import shutil
from gradio_client import Client

def call_diffdock_api(input_structure, input_pdb, smiles_string, input_ligand, 
                      num_inference_steps=10, num_samples=10, actual_inference_steps=10, no_final_step_noise=True):
    """
    Calls the DiffDock API with given inputs and retrieves binding affinity results.
    """
    client = Client("https://alphunt-diffdock-alphunt-demo.hf.space/--replicas/bqy60/")

    result = client.predict(
        input_structure,  # str in 'Input structure' Textbox component
        input_pdb,  # Local file path
        smiles_string,  # str in 'SMILES string' Textbox component
        input_ligand,  # Local file path
        num_inference_steps,  # int | float (10-40) in 'Number of inference steps' Slider component
        num_samples,  # int | float (10-40) in 'Number of samples' Slider component
        actual_inference_steps,  # int | float (10-40) in 'Number of actual inference steps' Slider component
        no_final_step_noise,  # bool in 'No final step noise' Checkbox component
        fn_index=1
    )

    output_file_path = result[2]  # This is the path where the API saves the zip file

    # Move the file to the same directory as diffdock.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_output_path = os.path.join(script_dir, os.path.basename(output_file_path))

    if os.path.exists(output_file_path):
        shutil.move(output_file_path, new_output_path)
        print(f"Output file moved to: {new_output_path}")
    else:
        print(f"Warning: Output file not found at {output_file_path}")

    return {
        "binding_affinity": result[0],
        "ranked_samples": result[1],
        "output_file_path": new_output_path
    }

if __name__ == "__main__":
    # Provide local file paths
    pdb_file = "6w70.pdb"  # Update with your actual local file path
    ligand_file = "6w70_ligand.sdf"  # Update with your actual local file path
    
    # Example SMILES
    smiles = "CC(=O)Nc1ccc(O)cc1"  # Example SMILES 
    
    # Call API with local file paths
    result = call_diffdock_api("6w70", pdb_file, smiles, ligand_file)
    print("DiffDock Binding Affinity Results:", result)
