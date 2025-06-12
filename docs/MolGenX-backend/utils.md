# RNNModel/utils.py — Functions Documentation

## `return_vocabulary(csv="cleaned_smiles.csv")`
Reads a CSV file containing SMILES strings and generates character-to-index and index-to-character vocabularies.

**Parameters:**
- `csv` (`str`): Path to the CSV file containing a 'smiles' column. Defaults to `'cleaned_smiles.csv'`.

**Returns:**
- `tuple`: (`char_to_idx`, `idx_to_char`)
  - `char_to_idx` (`dict`): Mapping from characters to their corresponding indices.
  - `idx_to_char` (`dict`): Mapping from indices to their corresponding characters.

**Raises:**
- `FileNotFoundError`: If the specified CSV file does not exist.
- `Exception`: For other exceptions during file reading or vocabulary creation.

**Notes:**
- If the specified CSV file does not contain a 'smiles' column, the function attempts to read from `'250k_rndm_zinc_drugs_clean_3.csv'` and uses the first column as the source of SMILES strings.

---

## `create_vocabulary(smiles_data)`
Creates a character-level vocabulary from a list of SMILES strings.

**Parameters:**
- `smiles_data` (`list of str`): List of SMILES strings to extract characters from.

**Returns:**
- `tuple`:
  - `char_to_idx` (`dict`): Mapping from character to unique index (with `"<PAD>"` as 0).
  - `idx_to_char` (`dict`): Mapping from index to character.

**Notes:**
- Includes all unique characters present in the provided SMILES data.
- Adds a special `"<PAD>"` token mapped to index 0 for padding.

---

## `validate_molecule(smiles: str) -> bool`
Validates whether a given SMILES string represents a valid and canonical molecule.

**Parameters:**
- `smiles` (`str`): The SMILES string to validate.

**Returns:**
- `bool`: `True` if the SMILES string is valid and canonical, `False` otherwise.

**Behavior:**
- Removes any `"<PAD>"` tokens from the input SMILES string.
- Attempts to parse it into a molecule object using RDKit.
- Checks if the molecule can be successfully canonicalized back to the original SMILES string.

---

## `get_pdb_id_from_sequence(sequence)`
Searches the PDB database for a matching structure for a given protein sequence and returns a PDB ID if found.

**Parameters:**
- `sequence` (`str`): Amino acid sequence.

**Returns:**
- `str`: PDB ID if found, `None` otherwise.

**Behavior:**
- Submits the sequence to the NCBI BLAST REST API against the PDB database.
- Polls for results and parses the response to extract the best matching PDB ID.

---

## `get_compound_files(output_dir="compound_visualizations")`
Reads visualization files for compounds and encodes image/PDB/SDF data for use in web applications.

**Parameters:**
- `output_dir` (`str`): Directory containing visualization files. Defaults to `'compound_visualizations'`.

**Returns:**
- `dict`: Dictionary with file contents organized by compound, including images, models, and optional grid images.

**Behavior:**
- Scans the output directory for compound images, PDB, SDF, and viewer files.
- Encodes images as base64 strings for web display.
- Returns a structured dictionary containing all available visualization data for each compound.

---