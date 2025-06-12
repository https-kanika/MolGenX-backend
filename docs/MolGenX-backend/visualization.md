# visualization.py — Function Documentation

## `visualize_simple(compounds, show_protein=True, pdb_id=None)`

Generates 2D and 3D visualizations for a list of chemical compounds and optionally visualizes a target protein structure.

**Parameters:**
- `compounds` (`list`): List of compound dictionaries, each containing:
    - `molecule`: RDKit molecule object
    - `smiles`: SMILES string representation
    - `score`: Score for the compound
    - `rank` (optional): Compound ranking
    - `type` (optional): Compound type/category
- `show_protein` (`bool`, optional): Whether to visualize the target protein. Default is `True`.
- `pdb_id` (`str`, optional): PDB ID for the target protein structure. Required if `show_protein=True`.

**Returns:**
- `None`: All visualizations are saved to the `compound_visualizations` directory.

**Behavior:**
- Creates and manages the output directory `compound_visualizations`.
- For each compound:
    - Generates a 2D PNG image.
    - Generates 3D structure files in PDB and SDF formats.
    - Creates an interactive HTML viewer for the 3D structure using 3Dmol.js.
- Generates a grid image of all compounds.
- If `show_protein` is `True` and a `pdb_id` is provided:
    - Downloads the protein structure from the RCSB PDB.
    - Saves the protein structure as a PDB file.
    - Generates an interactive HTML viewer for the protein structure.
    - Optionally displays the protein structure using py3Dmol in a Jupyter environment.

**Notes:**
- Handles errors in visualization and file operations gracefully.
- Prints the status and file paths of all generated visualizations.