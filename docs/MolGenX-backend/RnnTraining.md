# # RNNModel/RnnTraining.py — File Documentation

This script preprocesses a dataset of SMILES strings, encodes them for RNN training, and trains an RNN-based generative model for molecular SMILES sequences.

---

## Workflow Overview

1. **Imports**  
   Loads required libraries for deep learning (PyTorch), data handling (pandas, numpy), and utility modules for SMILES processing and model definition.

2. **Data Loading and Preprocessing**
   - Reads a CSV file (`250k_rndm_zinc_drugs_clean_3.csv`) containing SMILES strings.
   - Drops unnecessary columns (`logP`, `qed`, `SAS`).
   - Extracts the SMILES column as a list.
   - Cleans and canonicalizes SMILES using `preprocess_smiles`.
   - Saves the cleaned SMILES to `cleaned_smiles.csv`.

3. **Vocabulary Construction**
   - Builds character-to-index and index-to-character mappings using `return_vocabulary`.
   - Determines the maximum SMILES length in the cleaned dataset.

4. **SMILES Encoding**
   - Defines `smiles_to_sequence(smiles)`, which converts a SMILES string to a list of integer indices, padded to `max_length`.
   - Encodes all cleaned SMILES into a numpy array of sequences.

5. **Model Setup**
   - Initializes an `RNNGenerator` model with the vocabulary size, embedding dimension, and hidden dimension.
   - Converts the encoded sequences to a PyTorch tensor for training.

6. **Training Preparation**
   - Sets up the loss function (`CrossEntropyLoss`) and optimizer (`Adam`).
   - Moves the model to the appropriate device (CPU or CUDA).

7. **Training Loop**
   - Trains the model for a specified number of epochs (`num_epochs`).
   - Uses mini-batches (`batch_size`) and teacher forcing (input/target shifting).
   - Computes loss, performs backpropagation, and updates model weights.
   - Prints average loss per epoch.

8. **Model Saving**
   - Saves the trained model weights to `rnn_model.pth`.

---

## Defined Functions

### `smiles_to_sequence(smiles)`
Converts a SMILES string into a sequence of integer indices, padding the result to a fixed maximum length.

**Parameters:**
- `smiles` (`str`): The SMILES string to be converted.

**Returns:**
- `list[int]`: List of integer indices representing the SMILES string, padded with zeros to match `max_length`.

**Raises:**
- `KeyError`: If a character in the SMILES string is not found in `char_to_idx`.

---

## Notes

- All other logic in this file is procedural and not encapsulated in functions or classes.
- The script expects the input CSV file to be present in the working directory.
- The model and cleaned SMILES are saved to disk for later use.