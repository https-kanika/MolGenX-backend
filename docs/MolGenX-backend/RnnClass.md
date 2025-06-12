# RNNModel/RnnClass.py — Function Documentation

## `class RNNGenerator(nn.Module)`

A PyTorch neural network module for generating molecular SMILES sequences using an RNN (LSTM) architecture.

### `__init__(self, vocab_size, embed_dim, hidden_dim)`
Initializes the RNNGenerator model.

**Parameters:**
- `vocab_size` (`int`): Number of unique tokens in the vocabulary.
- `embed_dim` (`int`): Dimension of the embedding vectors.
- `hidden_dim` (`int`): Dimension of the LSTM hidden state.

---

### `forward(self, x)`
Defines the forward pass of the RNNGenerator.

**Parameters:**
- `x` (`torch.Tensor`): Input tensor of token indices.

**Returns:**
- `torch.Tensor`: Output logits for each token in the vocabulary.

---

## `generate_diverse_molecules(model, char_to_idx, idx_to_char, device, start_token="C", num_molecules=10, max_length=100, max_attempts=100)`

Generates multiple unique and valid molecular SMILES strings using a trained RNN model.

**Parameters:**
- `model` (`nn.Module`): Trained RNN model for sequence generation.
- `char_to_idx` (`dict`): Mapping from characters to indices.
- `idx_to_char` (`dict`): Mapping from indices to characters.
- `device` (`torch.device`): Device to run the model on (CPU or CUDA).
- `start_token` (`str`, default `"C"`): Starting token for molecule generation.
- `num_molecules` (`int`, default `10`): Number of unique molecules to generate.
- `max_length` (`int`, default `100`): Maximum length of each generated SMILES string.
- `max_attempts` (`int`, default `100`): Maximum number of generation attempts.

**Returns:**
- `List[str]`: List of unique, valid SMILES strings.

**Behavior:**
- Uses temperature-based sampling for diversity.
- Stops generation on `<PAD>` token or reaching `max_length`.
- Validates generated SMILES and ensures uniqueness.
- Returns up to `num_molecules` valid, unique SMILES strings.

---