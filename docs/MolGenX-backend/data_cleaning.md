# RNNModel/data_cleaning.py — Function Documentation

## `preprocess_smiles(smiles_list, canonical=True, show_progress=True)`

Processes a list of SMILES strings, filters out invalid ones, and optionally canonicalizes them.

**Parameters:**
- `smiles_list` (`list`): List of SMILES strings to process.
- `canonical` (`bool`, default `True`): If `True`, converts valid SMILES to their canonical form.
- `show_progress` (`bool`, default `True`): If `True`, prints progress and statistics during processing.

**Returns:**
- `list`: List of valid (optionally canonicalized) SMILES strings.

**Behavior:**
- Iterates through the input SMILES list in batches.
- Uses RDKit to validate and optionally canonicalize each SMILES string.
- Tracks and reports the number of valid and invalid SMILES, as well as processing progress and estimated time remaining.

---

## `process_smi_file(file_path, output_path=None)`

Reads a `.smi` file, processes the SMILES strings to filter out invalid entries, and saves the valid SMILES to a new file.

**Parameters:**
- `file_path` (`str`): Path to the input `.smi` file containing SMILES strings.
- `output_path` (`str`, optional): Path to save the output file with valid SMILES. If not provided, appends `_valid.smi` to the input file name.

**Returns:**
- `list`: List of valid SMILES strings processed from the file.

**Behavior:**
- Reads SMILES strings from the input file, ignoring empty lines and comment lines starting with `//`.
- Calls `preprocess_smiles` to validate and canonicalize the SMILES.
- Writes the valid SMILES to the specified output file.
- Prints summary statistics about the number of valid and invalid SMILES processed.

---