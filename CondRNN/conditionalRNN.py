import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem
import torch.nn.functional as F
from contextlib import nullcontext
import os
# Fix memory fragmentation issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# conditional_rnn.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem
from tqdm import tqdm
import time
import argparse
from contextlib import nullcontext
import torch.cuda.amp as amp

def validate_molecule(smiles: str) -> bool:
    """
    Validate if the generated SMILES represents a valid molecule
    """
    # Remove the <PAD> and <START> tokens before validation
    smiles = smiles.replace("<PAD>", "").replace("<START>", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        # Just check if the molecule is valid, don't require canonical matching
        return mol is not None
        # The below is too strict for generation:
        # return mol is not None and Chem.MolToSmiles(mol) == smiles
    except:
        return False

def preprocess_bindingdb(filepath, max_smiles_length=100, max_protein_length=1000):
    """
    Preprocess BindingDB dataset for conditional RNN training
    """
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df=df[:550000]
    
    # Filter out invalid SMILES and sequences
    print("Initial data size:", len(df))
    
    # Validate SMILES
    valid_mols = []
    for i, smiles in enumerate(df['smiles']):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Get canonical SMILES
                canonical_smiles = Chem.MolToSmiles(mol)
                valid_mols.append(i)
                # Update with canonical version
                df.loc[i, 'smiles'] = canonical_smiles
        except:
            pass
    
    df = df.iloc[valid_mols].reset_index(drop=True)
    print(f"After SMILES validation: {len(df)} entries")
    
    # Filter by length
    df = df[df['smiles'].str.len() <= max_smiles_length]
    df = df[df['target_seq'].str.len() <= max_protein_length]
    print(f"After length filtering: {len(df)} entries")
    
    # Filter by affinity value (remove potential errors/outliers)
    if 'affinity' in df.columns:
        # Log-transform and normalize affinity values
        df = df[df['affinity'] > 0]  # Filter out zero/negative values
        df['affinity'] = np.log10(df['affinity'])
        
        # Min-max normalize log affinities to [0,1] range
        min_aff = df['affinity'].min()
        max_aff = df['affinity'].max()
        df['affinity_normalized'] = (df['affinity'] - min_aff) / (max_aff - min_aff)
        
        # Store normalization parameters for later use
        norm_params = {'min_aff': min_aff, 'max_aff': max_aff}
    
    # Build vocabularies
    smiles_chars = set()
    for smiles in df['smiles']:
        smiles_chars.update(set(smiles))
    
    protein_chars = set()
    for seq in df['target_seq']:
        protein_chars.update(set(seq))
    
    # Create character mappings
    smiles_vocab = ['<PAD>', '<START>'] + sorted(list(smiles_chars))
    smiles_char_to_idx = {char: idx for idx, char in enumerate(smiles_vocab)}
    smiles_idx_to_char = {idx: char for char, idx in smiles_char_to_idx.items()}
    
    protein_vocab = ['<PAD>'] + sorted(list(protein_chars))
    protein_char_to_idx = {char: idx for idx, char in enumerate(protein_vocab)}
    protein_idx_to_char = {idx: char for char, idx in protein_char_to_idx.items()}
    
    print(f"SMILES vocabulary size: {len(smiles_vocab)}")
    print(f"Protein vocabulary size: {len(protein_vocab)}")
    
    # Prepare train/val/test split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Vocabularies and splits
    vocab_data = {
        'smiles_char_to_idx': smiles_char_to_idx,
        'smiles_idx_to_char': smiles_idx_to_char,
        'protein_char_to_idx': protein_char_to_idx,
        'protein_idx_to_char': protein_idx_to_char,
    }
    
    if 'affinity' in df.columns:
        vocab_data['norm_params'] = norm_params
    
    return train_df, val_df, test_df, vocab_data

class BindingDBDataset(Dataset):
    def __init__(self, df, smiles_char_to_idx, protein_char_to_idx, 
                 max_smiles_len=100, max_protein_len=1000,
                 include_affinity=True):
        self.df = df
        self.smiles_char_to_idx = smiles_char_to_idx
        self.protein_char_to_idx = protein_char_to_idx
        self.max_smiles_len = max_smiles_len
        self.max_protein_len = max_protein_len
        self.include_affinity = include_affinity
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Process SMILES (add start token and convert to indices)
        smiles = row['smiles']
        smiles_indices = [self.smiles_char_to_idx['<START>']] + [self.smiles_char_to_idx[c] for c in smiles]
        
        # Pad if necessary
        if len(smiles_indices) < self.max_smiles_len:
            smiles_indices += [self.smiles_char_to_idx['<PAD>']] * (self.max_smiles_len - len(smiles_indices))
        else:
            smiles_indices = smiles_indices[:self.max_smiles_len]
        
        # Process protein sequence
        protein = row['target_seq']
        protein_indices = [self.protein_char_to_idx[c] if c in self.protein_char_to_idx else 
                          self.protein_char_to_idx['<PAD>'] for c in protein]
        
        # Pad protein sequence
        if len(protein_indices) < self.max_protein_len:
            protein_indices += [self.protein_char_to_idx['<PAD>']] * (self.max_protein_len - len(protein_indices))
        else:
            protein_indices = protein_indices[:self.max_protein_len]
        
        # Convert to tensors
        smiles_tensor = torch.tensor(smiles_indices, dtype=torch.long)
        protein_tensor = torch.tensor(protein_indices, dtype=torch.long)
        
        # For training, we need inputs and targets (shifted by 1)
        # Input: [<START>, C, C, O, ...] -> Target: [C, C, O, ...]
        input_tensor = smiles_tensor[:-1]  # Everything except the last token
        target_tensor = smiles_tensor[1:]   # Everything except the first token
        
        if self.include_affinity and 'affinity_normalized' in self.df.columns:
            affinity = torch.tensor([row['affinity_normalized']], dtype=torch.float)
            return input_tensor, target_tensor, protein_tensor, affinity
        else:
            return input_tensor, target_tensor, protein_tensor

def create_dataloaders(train_df, val_df, test_df, vocab_data, 
                       batch_size=64, max_smiles_len=100, max_protein_len=1000,
                       include_affinity=True, num_workers=4):
    """Create PyTorch DataLoaders for training, validation and testing"""
    
    train_dataset = BindingDBDataset(
        train_df, 
        vocab_data['smiles_char_to_idx'],
        vocab_data['protein_char_to_idx'],
        max_smiles_len,
        max_protein_len,
        include_affinity
    )
    
    val_dataset = BindingDBDataset(
        val_df, 
        vocab_data['smiles_char_to_idx'],
        vocab_data['protein_char_to_idx'],
        max_smiles_len,
        max_protein_len,
        include_affinity
    )
    
    test_dataset = BindingDBDataset(
        test_df, 
        vocab_data['smiles_char_to_idx'],
        vocab_data['protein_char_to_idx'],
        max_smiles_len,
        max_protein_len,
        include_affinity
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

import torch.nn as nn
class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=256, num_layers=3):
        super(ProteinEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Increase complexity with more layers and attention
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Get LSTM output
        output, (hidden, _) = self.lstm(embedded)
        
        # Apply attention
        attention_weights = self.attention(output)
        context_vector = torch.sum(attention_weights * output, dim=1)
        
        # Use attention-weighted representations instead of just final state
        return self.fc(context_vector)  # [batch_size, output_dim]

class ConditionalRNNGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, target_encoding_dim, use_affinity=True):
        super(ConditionalRNNGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.use_affinity = use_affinity
        
        # Target protein encoder input dimension
        target_input_dim = target_encoding_dim
        if use_affinity:
            target_input_dim += 1  # Add dimension for affinity value
        
        # More robust target encoder with normalization
        self.target_encoder = nn.Sequential(
            nn.Linear(target_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalization helps training
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # More powerful LSTM configuration
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim, 
            hidden_dim, 
            batch_first=True,
            num_layers=3,  # More layers
            dropout=0.2
        )
        
        # More robust output layer with a projection network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, target_features, affinity=None):
        batch_size, seq_len = x.size()
        
        # Process molecule embedding
        x_embed = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_dim]
        
        # Process target features (combine with affinity if available)
        if self.use_affinity and affinity is not None:
            # Combine protein features with affinity
            combined_target = torch.cat([target_features, affinity], dim=1)
        else:
            combined_target = target_features
            
        target_encoded = self.target_encoder(combined_target)  # [batch_size, hidden_dim]
        
        # Expand target features to match sequence length
        target_expanded = target_encoded.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Concatenate molecule embedding with target features
        combined_input = torch.cat([x_embed, target_expanded], dim=2)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(combined_input)
        
        # Project to vocabulary using more robust output network
        output = self.output_network(lstm_output)
        return output

    # Keep the generate method as it is in your existing code

import torch.cuda.amp as amp
from tqdm import tqdm
import time
import os

def train_model(
    model, 
    protein_encoder,
    train_loader, 
    val_loader,
    vocab_data,
    device,
    epochs=50,  # More epochs
    lr=1e-4,    # Slightly lower learning rate
    save_dir='./models',
    include_affinity=True,
    use_amp=True,
    gradient_accumulation_steps=8,
    patience=7   # Early stopping patience
):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Move models to device
    model = model.to(device)
    protein_encoder = protein_encoder.to(device)
    
    # Optimization with weight decay
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(protein_encoder.parameters()),
        lr=lr,
        weight_decay=1e-6,  # Slight L2 regularization
        eps=1e-8           # For numerical stability
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=4,
        min_lr=1e-6
    )
    old_lr = lr  # Track old learning rate for logging
    
    
    # Early stopping parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Loss function - ignore padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_data['smiles_char_to_idx']['<PAD>'])
    
    # For mixed precision training
    scaler = amp.GradScaler() if use_amp else None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        protein_encoder.train()
        train_loss = 0
        
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)  # Memory-efficient gradient reset 
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            if include_affinity:
                inputs, targets, protein_sequences, affinities = [b.to(device) for b in batch]
            else:
                inputs, targets, protein_sequences = [b.to(device) for b in batch]
                affinities = None
            
            # Process with mixed precision
            with amp.autocast() if use_amp else nullcontext():
                # Encode protein sequences
                protein_features = protein_encoder(protein_sequences)
                
                # Forward pass
                outputs = model(inputs, protein_features, affinities)
                
                # Calculate loss and normalize by accumulation steps
                loss = criterion(
                    outputs.contiguous().view(-1, len(vocab_data['smiles_char_to_idx'])),
                    targets.contiguous().view(-1)
                ) / gradient_accumulation_steps
            
            # Backward pass with scaling for mixed precision
            if use_amp:
                scaler.scale(loss).backward()
                
                # Only update weights after accumulating gradients
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # Unscales gradients before optimizer step
                    scaler.unscale_(optimizer)
                    
                    # Optional: clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(protein_encoder.parameters(), max_norm=1.0)
                    
                    # Update weights and zero gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zeros
            else:
                loss.backward()
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # Optional: clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(protein_encoder.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            # Track unscaled loss (multiply to get actual loss value)
            train_loss += loss.item() * gradient_accumulation_steps
        
        # Calculate average loss
        train_loss /= len(train_loader)
        train_time = time.time() - start_time
        
        # Free GPU memory before validation
        torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        protein_encoder.eval()
        val_loss = 0
        
        # Process validation data in smaller batches if needed
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if include_affinity:
                    inputs, targets, protein_sequences, affinities = [b.to(device) for b in batch]
                else:
                    inputs, targets, protein_sequences = [b.to(device) for b in batch]
                    affinities = None
                
                # Forward pass
                protein_features = protein_encoder(protein_sequences)
                outputs = model(inputs, protein_features, affinities)
                
                # Calculate loss
                loss = criterion(
                    outputs.contiguous().view(-1, len(vocab_data['smiles_char_to_idx'])),
                    targets.contiguous().view(-1)
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check if learning rate changed
        new_lr = [group['lr'] for group in optimizer.param_groups][0]
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            old_lr = new_lr
        
        # Report training metrics
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {train_time:.2f}s")
        
        # Monitor GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'protein_encoder_state_dict': protein_encoder.state_dict(),
                'vocab_data': vocab_data,
                'epoch': epoch,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Free GPU memory after each epoch
        torch.cuda.empty_cache()
    
    return model, protein_encoder

def generate_molecules(
    model,
    protein_encoder,
    target_sequence,
    vocab_data,
    affinity_value=0.7,
    num_molecules=10,
    temperature=0.8,
    device='cuda',
    max_attempts=3  # New parameter
):
    """Generate molecules for a specified target with optional affinity"""
    model.eval()
    protein_encoder.eval()
    
    # Convert target sequence to tensor
    protein_indices = [vocab_data['protein_char_to_idx'][c] if c in vocab_data['protein_char_to_idx'] else 
                     vocab_data['protein_char_to_idx']['<PAD>'] for c in target_sequence]
    
    # Pad protein sequence
    max_protein_len = 1000  # Set to the same value used in training
    if len(protein_indices) < max_protein_len:
        protein_indices += [vocab_data['protein_char_to_idx']['<PAD>']] * (max_protein_len - len(protein_indices))
    else:
        protein_indices = protein_indices[:max_protein_len]
    
    # Convert to tensor and repeat for batch
    protein_tensor = torch.tensor([protein_indices], dtype=torch.long).to(device)
    protein_tensor = protein_tensor.repeat(num_molecules, 1)
    
    # Encode protein
    with torch.no_grad():
        protein_features = protein_encoder(protein_tensor)
        
    # Prepare affinity tensor if model uses it
    if model.use_affinity:
        affinity_tensor = torch.tensor([[affinity_value]] * num_molecules, dtype=torch.float).to(device)
    else:
        affinity_tensor = None
    
    # Generate molecules
    valid_molecules = []
    attempt = 0
    
    while len(valid_molecules) < num_molecules and attempt < max_attempts:
        # Adjust temperature based on attempt number
        current_temp = temperature * (1.0 + 0.2 * attempt)
        
        start_token_idx = vocab_data['smiles_char_to_idx']['<START>']
        pad_token_idx = vocab_data['smiles_char_to_idx']['<PAD>']
        
        with torch.no_grad():
            # Generate initial batch
            batch_size = min(num_molecules * 2, 64)  # Generate more per attempt, but cap batch size
            current_seqs = torch.tensor([[start_token_idx]] * batch_size, device=device)
            
            # Repeat protein features and affinity for the batch
            batch_protein_features = protein_features[:1].repeat(batch_size, 1)
            batch_affinity = affinity_tensor[:1].repeat(batch_size, 1) if affinity_tensor is not None else None
            
            # Track finished sequences
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for step in range(100):  # Maximum SMILES length
                # Only process unfinished sequences
                if finished.all():
                    break
                    
                # Forward pass
                outputs = model(current_seqs, batch_protein_features, batch_affinity)
                
                # Apply temperature to logits
                next_token_logits = outputs[:, -1, :] / current_temp
                
                # Block repeated generation of the same token (helps avoid loops)
                if step > 0 and step % 10 == 0:
                    for b in range(batch_size):
                        if not finished[b]:
                            # Get tokens already generated
                            prev_tokens = current_seqs[b]
                            # Get the most frequently generated token
                            token_counts = {}
                            for t in prev_tokens:
                                if t.item() in token_counts:
                                    token_counts[t.item()] += 1
                                else:
                                    token_counts[t.item()] = 1
                                    
                            for token, count in token_counts.items():
                                if count > 3:  # If a token appears more than 3 times
                                    next_token_logits[b, token] /= 2.0  # Reduce its probability
                
                # Apply softmax to convert to probabilities
                probs = F.softmax(next_token_logits, dim=1)
                
                # Sample next token for each sequence in the batch
                next_tokens = torch.multinomial(probs, 1)
                
                # Append to sequences
                current_seqs = torch.cat([current_seqs, next_tokens], dim=1)
                
                # Mark sequences that output PAD token as finished
                finished = finished | (next_tokens.squeeze(-1) == pad_token_idx)
                
                # Early stopping: if generated sequence is too long without finishing
                if step > 80:
                    # Force unfinished sequences to terminate
                    for b in range(batch_size):
                        if not finished[b]:
                            current_seqs[b, -1] = pad_token_idx
                            finished[b] = True
        
        # Convert sequences to SMILES strings
        generated_smiles = []
        for seq in current_seqs:
            smiles = ''.join([vocab_data['smiles_idx_to_char'][idx.item()] 
                           for idx in seq if idx.item() not in [pad_token_idx, start_token_idx]])
            generated_smiles.append(smiles)
        
        # Validate molecules
        for smiles in generated_smiles:
            if validate_molecule(smiles) and smiles not in valid_molecules:
                valid_molecules.append(smiles)
                if len(valid_molecules) >= num_molecules:
                    break
        
        attempt += 1
    
    # If we still don't have enough, return whatever we got
    return list(set(valid_molecules))[:num_molecules]





# Fix memory fragmentation issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Optimize CUDA operations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load and preprocess data
    print(f"Loading data from {args.data_path}")
    train_df, val_df, test_df, vocab_data = preprocess_bindingdb(
        args.data_path, 
        max_smiles_length=args.max_smiles_len,
        max_protein_length=args.max_protein_len
    )
    
    # Create dataloaders with optimized settings
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, vocab_data, 
        batch_size=args.batch_size,
        max_smiles_len=args.max_smiles_len, 
        max_protein_len=args.max_protein_len,
        include_affinity=True, 
        num_workers=args.num_workers
    )
    
    # Initialize models
    protein_vocab_size = len(vocab_data['protein_char_to_idx'])
    smiles_vocab_size = len(vocab_data['smiles_char_to_idx'])
    
    # Create protein encoder with optimized size
    protein_encoder = ProteinEncoder(
        vocab_size=protein_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )
    
    # Create conditional generator
    model = ConditionalRNNGenerator(
        vocab_size=smiles_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim*2,  # Typically RNN generator needs more capacity
        target_encoding_dim=args.output_dim,
        use_affinity=True
    )
    
    # Train model with gradient accumulation
    model, protein_encoder = train_model(
        model,
        protein_encoder,
        train_loader,
        val_loader,
        vocab_data,
        device,
        epochs=args.epochs,
        lr=args.learning_rate,
        save_dir=args.save_dir,
        include_affinity=True,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )
    
    # Generate example molecules
    if args.generate_examples:
        example_protein = test_df.iloc[0]['target_seq']
        generated_molecules = generate_molecules(
            model,
            protein_encoder,
            example_protein,
            vocab_data,
            affinity_value=0.7,
            num_molecules=10,
            device=device
        )
        
        print("Generated molecules:")
        for mol in generated_molecules:
            print(mol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train conditional RNN for molecule generation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='bindingDB.csv', 
                        help='Path to the BindingDB data')
    parser.add_argument('--max_smiles_len', type=int, default=100, 
                        help='Maximum SMILES string length')
    parser.add_argument('--max_protein_len', type=int, default=500, 
                        help='Maximum protein sequence length')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=64,  # From 32
                   help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,  # From 128
                       help='Hidden dimension') 
    parser.add_argument('--output_dim', type=int, default=256,  # From 128
                       help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=2,  # From 1
                       help='Number of LSTM layers for protein encoder')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, 
                        help='Learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=8, 
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=2, 
                        help='Number of data loader workers')
    parser.add_argument('--use_amp', action='store_true', 
                        help='Use automatic mixed precision')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./models', 
                        help='Directory to save models')
    parser.add_argument('--generate_examples', action='store_true', 
                        help='Generate example molecules after training')
    
    args = parser.parse_args()
    
    main(args)