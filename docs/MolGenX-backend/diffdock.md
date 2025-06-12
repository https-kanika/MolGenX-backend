Module MolGenX-backend.diffdock
===============================

# diffdock.py — Function Documentation

## `call_diffdock_api(input_structure, input_pdb, smiles_string, input_ligand)`

Calls the DiffDock API to predict binding affinity and generate docking results for a given protein-ligand pair.

**Parameters:**
- `input_structure` (`str`): The PDB ID of the target protein structure.
- `input_pdb` (`str`): File path to the input PDB file for the protein.
- `smiles_string` (`str`): SMILES string representing the ligand molecule.
- `input_ligand` (`str`): File path to the input ligand file (e.g., SDF format).

**Returns:**
- The binding affinity result (typically a float or string) as returned by the API, or `None` if the call fails or the result format is unexpected.

**Behavior:**
- Checks that the provided PDB and ligand files exist.
- Calls the DiffDock API using the Gradio client with the provided inputs.
- If the API returns a result tuple with a file path, moves the output file to the script directory and prints its new location.
- Returns the binding affinity result from the API response.
- Prints warnings if files are missing or the result format is unexpected.

---

## `if __name__ == "__main__":` (Test Block)

A test block that demonstrates how to use the `call_diffdock_api` function.

**Behavior:**
- Sets example file paths and a SMILES string for testing.
- Prints the test parameters.
- Calls `call_diffdock_api` with the example inputs.
- Prints the binding affinity results returned by the API.

---