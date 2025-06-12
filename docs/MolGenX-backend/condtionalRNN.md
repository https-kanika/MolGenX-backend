
# conditionalRNN.py — Function Documentation# Conditional RNN for Molecule Generation - Documentation

This module implements a protein-conditioned recurrent neural network for generating valid drug-like molecules in SMILES format. It includes data preprocessing, dataset classes, model definitions, training loop, and generation utilities.

---

## Contents

- [Functions](#functions)
- [Classes](#classes)
  - [BindingDBDataset](#bindingdbdataset)
  - [ProteinEncoder](#proteinencoder)
  - [ConditionalRNNGenerator](#conditionalrnngenerator)

---

## Functions

### `validate_molecule(smiles: str) -> bool`

Validates if a SMILES string represents a valid molecule using RDKit.

---

### `preprocess_bindingdb(filepath, max_smiles_length=100, max_protein_length=1000) -> tuple`

Loads and preprocesses the BindingDB dataset: filtering, normalizing, tokenizing, and splitting into train/val/test sets. Returns dataframes and vocabulary.

---

### `create_dataloaders(...) -> tuple`

Creates PyTorch DataLoaders for training, validation, and testing from the processed datasets.

---

### `train_model(...) -> tuple`

Trains the Conditional RNN Generator and Protein Encoder. Supports mixed precision, early stopping, gradient accumulation, and model checkpointing.

---

### `generate_molecules(...) -> list`

Generates valid SMILES strings conditioned on a given protein sequence using the trained model.

---

### `main(args)`

Main entry point to load data, train models, and optionally generate sample molecules.

---

## Classes

### `BindingDBDataset`

> `torch.utils.data.Dataset`

Custom Dataset class for handling protein-SMILES pairs from BindingDB.

**Constructor Arguments:**

- `df`: Input dataframe  
- `smiles_char_to_idx`: SMILES vocabulary mapping  
- `protein_char_to_idx`: Protein vocabulary mapping  
- `max_smiles_len`: Padding length for SMILES  
- `max_protein_len`: Padding length for proteins  
- `include_affinity`: Whether to include binding affinity  

**__getitem__ Output:**

- Input tensor (SMILES)  
- Target tensor (SMILES shifted)  
- Protein tensor  
- Optional: Affinity tensor  

---

### `ProteinEncoder`

> `torch.nn.Module`

Encodes protein sequences into fixed-length feature vectors using:

- Embedding layer  
- BiLSTM (bidirectional LSTM)  
- Attention mechanism  
- Linear projection  

**Constructor Arguments:**

- `vocab_size`: Size of protein vocabulary  
- `embed_dim`: Embedding size (default: 128)  
- `hidden_dim`: LSTM hidden units (default: 256)  
- `output_dim`: Output feature size (default: 256)  
- `num_layers`: LSTM layers (default: 3)  

**Forward Input:**

- `x`: Tensor of protein token indices  

**Forward Output:**

- `Tensor`: Encoded protein vector  

---

### `ConditionalRNNGenerator`

> `torch.nn.Module`

SMILES generator RNN conditioned on protein encodings and optionally affinity.

**Constructor Arguments:**

- `vocab_size`: Size of SMILES vocabulary  
- `embed_dim`: Character embedding size  
- `hidden_dim`: Hidden state size  
- `target_encoding_dim`: Dim. of protein encoder output  
- `use_affinity`: Whether to use affinity conditioning  

**Architecture:**

- SMILES Embedding  
- Target Encoder (MLP)  
- LSTM (with target conditioning)  
- Output Network (project to vocab)  

**Forward Input:**

- `x`: SMILES input sequence  
- `target_features`: Protein features  
- `affinity`: Optional affinity value  

**Forward Output:**

- `Tensor`: Logits for next SMILES character  

---

## Entry Point

### `if __name__ == "__main__":`

Handles command-line arguments via `argparse`, then calls `main()` to:

- Load dataset  
- Preprocess and tokenize  
- Initialize models  
- Train  
- Optionally generate molecules  

---

## Argument Summary

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to BindingDB CSV |
| `--max_smiles_len` | Max SMILES sequence length |
| `--max_protein_len` | Max protein sequence length |
| `--embed_dim` | Embedding dimension |
| `--hidden_dim` | LSTM hidden units |
| `--output_dim` | Output size from ProteinEncoder |
| `--num_layers` | LSTM layers in ProteinEncoder |
| `--batch_size` | Batch size |
| `--epochs` | Training epochs |
| `--learning_rate` | Learning rate |
| `--gradient_accumulation` | Steps to accumulate gradient |
| `--num_workers` | Dataloader workers |
| `--use_amp` | Use mixed precision training |
| `--save_dir` | Model save path |
| `--generate_examples` | Generate example molecules |

---
