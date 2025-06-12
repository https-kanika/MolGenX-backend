Module MolGenX-backend.main
===========================

# main.py — Function Documentation

## `get_protein_visualization(pdb_id)`

**Endpoint:** `/api/protein/<pdb_id>`  
**Methods:** `GET`

Retrieves protein structure and visualization data for a given PDB ID.

**Parameters:**
- `pdb_id` (`str`): The 4-character PDB ID for the protein.

**Returns:**
- `JSON`: Contains protein metadata (title, description, experimental method, resolution), PDB content, HTML visualization, and relevant URLs.
- Returns error JSON if the PDB ID is invalid or visualization generation fails.

---

## `find_optimized_candidates()`

**Endpoint:** `/api/optimize`  
**Methods:** `POST`

Finds and optimizes drug candidates for a given protein and user-specified parameters.

**Accepts (JSON body):**
- `pdb_id` (`str`): PDB ID of the target protein.
- `weights` (`dict`): Weights for optimization properties (druglikeness, synthetic_accessibility, lipinski_violations, toxicity, binding_affinity, solubility).
- `num_compounds` (`int`, optional): Number of compounds to return (default: 20, max: 50).
- `binding_affinity` (`float`, optional): Target binding affinity (default: 0.7, range: 0.1–1.0).
- `generate_visualizations` (`bool`, optional): Whether to generate compound visualizations (default: False).

**Returns:**
- `JSON`: Contains optimized compounds (as JSON), overall and per-compound explanations, requested parameters, and (optionally) visualization data.
- Returns error JSON if required data is missing or processing fails.

**Behavior:**
- Fetches protein sequence from RCSB using the provided PDB ID.
- Generates molecules using a Conditional RNN (or fallback RNN).
- Optimizes molecules using multi-objective criteria.
- Generates molecular variants and explanations.
- Optionally generates and returns visualization data.

---

## `index()`

**Endpoint:** `/`  
**Methods:** `GET`

Returns a simple HTML welcome page.

**Returns:**
- Rendered `index.html` template.

---

## `handle_500_error(error)`

Handles HTTP 500 (Internal Server Error) responses.

**Parameters:**
- `error`: The error object.

**Returns:**
- `JSON`: Error message with status code 500.

---

## `handle_404_error(error)`

Handles HTTP 404 (Resource Not Found) responses.

**Parameters:**
- `error`: The error object.

**Returns:**
- `JSON`: Error message with status code 404.

---