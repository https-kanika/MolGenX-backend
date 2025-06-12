# Molecule Generation Model Loader and Evaluator for our custom conditional RNN

This module provides utilities to load trained conditional RNN models for protein-targeted molecule generation, generate molecules, and evaluate model performance and drug-likeness.

---

## Contents

- [Functions](#functions)
  - [load_model](#load_model)
  - [generate_for_target](#generate_for_target)
  - [evaluate_generation_quality](#evaluate_generation_quality)
  - [evaluate_druglikeness](#evaluate_druglikeness)
- [Command Line Usage](#command-line-usage)

---

## Functions

### `load_model(model_path, device) -> (model, protein_encoder, vocab_data)`

Loads a trained model checkpoint from a directory or `.pt` file.

**Args:**  
- `model_path`: Path to `.pt` file or directory containing `best_model.pt`  
- `device`: PyTorch device (`cpu` or `cuda`)  

**Returns:**  
- `model`: Loaded `ConditionalRNNGenerator`  
- `protein_encoder`: Loaded `ProteinEncoder`  
- `vocab_data`: SMILES and protein vocab dictionaries  

---

### `generate_for_target(model_path, target_sequence_or_file, affinity=0.7, n_molecules=10, output_folder="generated") -> list`

Generates drug-like molecules for a target protein sequence.

**Args:**  
- `model_path`: Path to trained model  
- `target_sequence_or_file`: Raw sequence or file containing sequence  
- `affinity`: Target binding affinity (0 to 1)  
- `n_molecules`: Number of molecules to generate  
- `output_folder`: Directory to save results  

**Returns:**  
- List of SMILES strings of generated molecules  

---

### `evaluate_generation_quality(model, protein_encoder, vocab_data, test_proteins=None, n_molecules=100, device='cuda') -> dict`

Evaluates quality of molecules generated for several protein sequences.

**Metrics include:**  
- Validity and uniqueness rates  
- Average molecular weight and LogP  
- Per-protein generation statistics  

**Returns:**  
- Dictionary of aggregated and per-protein metrics  

---

### `evaluate_druglikeness(molecules) -> dict`

Evaluates the drug-likeness of SMILES molecules using:

- QED (Quantitative Estimate of Drug-likeness)  
- Lipinski's Rule of Five  
- MW and LogP distributions  

**Returns:**  
- Dictionary of scores and distribution summaries  

---

## Command Line Usage

```bash
# Generate molecules for a protein
python model_loading.py --target "PROTEIN_SEQUENCE" --model_path ./models --n_molecules 10

# Evaluate model performance
python model_loading.py --target dummy --model_path ./models --evaluate
```

### Available Arguments

| Argument           | Description                                           |
|--------------------|-------------------------------------------------------|
| `--model_path`     | Path to saved checkpoint (or folder)                  |
| `--target`         | Protein sequence or path to text file                 |
| `--affinity`       | Target binding affinity (default: 0.8)                |
| `--n_molecules`    | Number of molecules to generate (default: 10)         |
| `--output_folder`  | Folder to save generated files (default: "generated") |
| `--evaluate`       | Flag to run evaluation instead of generation          |

---

## Notes

- QED > 0.3 filtering ensures generated molecules have reasonable drug-likeness.  
- Lipinski’s Rule violations >1 are considered non-drug-like.  
- Models are trained using SMILES + protein conditioning and can generate sequence-specific ligands.