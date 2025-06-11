# Model Architecture Technical Documentation

<i>Note: 
Folder models_550k has the weights for our final model trained on 550k data points obtained from ChEMBL.
Folder models has the SMILES and target vocabulary and weights of a smaller conditional RNN model trained on fewer data points.
See model_loading.py to learn how to load this model.
</i>

## 1. Model Components

### Protein Encoder
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Input**: Protein amino acid sequences (tokenized)
- **Output**: Fixed-dimension protein encoding vector (256 dimensions)
- **Layer Configuration**:
  - Embedding Layer: `vocab_size × 64` embedding dimensions
  - Bidirectional LSTM: 2 layers, 256 hidden units (512 total with bidirectionality)
  - Attention Mechanism: 
    - Linear projection: 512 → 256 → 1
    - Softmax normalization across sequence dimension
  - Final FC Layer: 512 → 256 dimensions
  - Dropout: 0.3 rate applied after embedding layer

### Conditional RNN Generator
- **Architecture**: Conditional character-level LSTM for sequential generation
- **Input**:
  - Protein encoding vector (256 dimensions)
  - Optional binding affinity value (scalar)
  - Previously generated SMILES characters
- **Output**: Next character probability distribution
- **Layer Configuration**:
  - SMILES Embedding Layer: `vocab_size × 64` dimensions
  - Target Processing Network:
    - Linear: 257 → 512 (including affinity dimension)
    - LayerNorm → ReLU → Dropout(0.2)
    - Linear: 512 → 512
    - LayerNorm
  - Conditional LSTM:
    - Input size: 576 (64 + 512) - concatenated embeddings
    - Hidden size: 512
    - Layers: 3 with 0.2 dropout between layers
    - Batch-first: True
  - Output Network:
    - Linear: 512 → 512
    - ReLU → Dropout(0.1)
    - Linear: 512 → `vocab_size`

## 2. Key Design Choices

- **Bidirectional Protein Encoding**: Captures information from both directions in the protein sequence, critical for capturing distant residue dependencies
- **Attention Mechanism**: Dynamically weights protein sequence tokens based on importance, achieving 15-20% improvement in molecule validity
- **Character-level Generation**: Generates SMILES strings one character at a time, maintaining chemical syntax
- **Conditioning Strategy**:
  - Protein features expanded to match sequence length
  - Concatenated with SMILES embeddings along feature dimension
  - Conditioning at every generation step for consistent target influence
- **Temperature Sampling**: Controls randomness during generation (T=0.7 default, increased by 0.2 each attempt)
- **Token Repetition Prevention**: Reduces the probability of tokens appearing >3 times by dividing their logits by 2.0

## 3. Training Strategy

- **Dataset**: 550,000 protein-molecule pairs from ChEMBL
- **Batch Size**: 32 (effective 256 with gradient accumulation)
- **Optimizer**: AdamW with weight decay 1e-6
- **Learning Rate**: 3e-4 with ReduceLROnPlateau scheduling
- **LR Scheduler Configuration**:
  - Factor: 0.5
  - Patience: 4 epochs
  - Mode: min (tracking validation loss)
  - Minimum LR: 1e-6
- **Gradient Accumulation**: 8 steps for effective batch size scaling
- **Mixed Precision Training**: FP16 computation with AMP
- **Gradient Clipping**: Maximum norm 1.0
- **Early Stopping**: 7 epochs patience, monitoring validation loss
- **Total Parameters**: ~12.5M (4.2M in protein encoder, 8.3M in generator)

## 4. Generation Process

The molecule generation follows these steps:
1. **Encode the target protein sequence**:
   - Convert amino acids to indices
   - Pad to max length (1000)
   - Process through protein encoder to get 256-dimensional representation
2. **Initialize generation with start token**:
   - Batch generation for efficiency (up to 64 molecules in parallel)
3. **Autoregressively generate each character**:
   - Combine protein features with current sequence
   - Predict next character probabilities using full network
   - Apply temperature to logits (starts at 0.7, increases with attempts)
   - Sample next character using multinomial sampling
   - Prevent repetitive patterns with frequency-based logit adjustment
4. **Handle sequence termination**:
   - Stop when all sequences in batch reach padding token
   - Force termination for sequences exceeding 80 characters
5. **Post-processing**:
   - Filter for valid molecules using RDKit
   - Remove special tokens
   - Ensure uniqueness in the result set

## 5. Performance Metrics

- **Validity Rate**:
- **Uniqueness**: 
- **Training Time**: ~6 hours on NVIDIA L4
- **Average Generation Time**: 
- **Memory Usage**:

This architecture effectively creates a targeted molecule generation system that can produce chemical structures specifically designed for given protein targets, while incorporating binding affinity information.