import pandas as pd
import numpy as np
from rdkit import Chem
from contextlib import nullcontext
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def validate_molecule(smiles: str) -> bool:
    """
    Validate if a SMILES string represents a valid chemical molecule.
    
    This function removes special tokens and uses RDKit to check if the
    SMILES string can be converted to a valid molecular structure.
    
    Args:
        smiles (str): The SMILES string to validate.
        
    Returns:
        bool: True if the molecule is valid, False otherwise.
    """
    # Remove the <PAD> and <START> tokens before validation
    smiles = smiles.replace("<PAD>", "").replace("<START>", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def preprocess_bindingdb(filepath, max_smiles_length=100, max_protein_length=500, use_gpu=False, skip_validation=False):
    """
    Preprocess BindingDB dataset with visual feedback for each processing stage.
    """
    print(f"\n{'='*50}")
    print(f"🔄 PREPROCESSING PIPELINE STARTED")
    print(f"{'='*50}")
    
    print(f"📂 Loading data from {filepath}")
    start_time = time.time()
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded in {time.time() - start_time:.2f}s - {len(df)} entries found")
    
    # Skip validation if requested
    if skip_validation:
        print(f"\n{'-'*50}")
        print(f"⏩ Skipping SMILES validation as requested")
        # Assume all SMILES are valid
        valid_mols = list(range(len(df)))
    else:

    # Create a visual separator for the validation stage
        print(f"\n{'-'*50}")
        print(f"🧪 Validating SMILES strings")
        
    # Use parallel processing for SMILES validation when dataset is large
        if len(df) > 10000:
            from concurrent.futures import ThreadPoolExecutor
            from functools import partial
            
            def process_smiles(idx, smiles_series):
                smiles = smiles_series.iloc[idx]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        return idx, canonical_smiles, True
                    return idx, smiles, False
                except:
                    return idx, smiles, False
            
            valid_mols = []
            updated_smiles = {}
            
            # Progress tracking
            total_mols = len(df)
            processed = 0
            valid_count = 0
            
            # Use thread pool for RDKit operations
            print(f"🧵 Using parallel processing with {os.cpu_count()} threads")
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                process_func = partial(process_smiles, smiles_series=df['smiles'])
                futures = [executor.submit(process_func, i) for i in range(len(df))]
                
                # Show progress bar for validation
                for future in tqdm(as_completed(futures), 
                                total=len(futures), 
                                desc="Validating SMILES"):
                    idx, canonical_smiles, is_valid = future.result()
                    processed += 1
                    if is_valid:
                        valid_count += 1
                        valid_mols.append(idx)
                        updated_smiles[idx] = canonical_smiles
                        
                    # Print progress update every 10% increment
                    if processed % (total_mols // 10) == 0:
                        print(f"  ↳ Progress: {processed}/{total_mols} molecules processed, {valid_count} valid ({valid_count/processed*100:.1f}%)")
        else:
            # Original sequential processing for smaller datasets
            valid_mols = []
            print(f"🔍 Processing {len(df)} SMILES strings sequentially")
            for i, smiles in tqdm(enumerate(df['smiles']), total=len(df), desc="Validating SMILES"):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        valid_mols.append(i)
                        df.loc[i, 'smiles'] = canonical_smiles
                except:
                    pass
        
        # Update with canonical versions in larger datasets
        if len(df) > 10000:
            print("📝 Updating with canonical SMILES representations...")
            for idx, smiles in tqdm(updated_smiles.items(), desc="Updating SMILES"):
                df.loc[idx, 'smiles'] = smiles
        
        # Filter to valid molecules
        df = df.iloc[valid_mols].reset_index(drop=True)
        print(f"✅ SMILES validation complete: {len(df)} valid entries ({len(df)/len(valid_mols)*100:.1f}% of input)")
        
    # Visual separator for length filtering
    print(f"\n{'-'*50}")
    print(f"📏 Filtering by sequence length")
    print(f"  ↳ Max SMILES length: {max_smiles_length}")
    print(f"  ↳ Max protein length: {max_protein_length}")
    
    norm_params = None
    
    # Use GPU-accelerated operations when available and requested
    if use_gpu:
        try:
            print("🧪 Attempting GPU acceleration for dataframe operations")
            import cudf
            import cupy as cp
            
            print("✅ RAPIDS libraries found! Using GPU acceleration")
            # Move dataframe to GPU
            start_time = time.time()
            gpu_df = cudf.DataFrame.from_pandas(df)
            print(f"  ↳ DataFrame moved to GPU in {time.time() - start_time:.2f}s")
            
            # Filter by length using GPU operations
            print("📊 Filtering sequences by length...")
            smiles_lengths = gpu_df['smiles'].str.len()
            protein_lengths = gpu_df['target_seq'].str.len()
            mask = (smiles_lengths <= max_smiles_length) & (protein_lengths <= max_protein_length)
            gpu_df = gpu_df[mask]
            
            # Process affinity values if present
            if 'affinity' in gpu_df.columns:
                print("🧮 Processing affinity values...")
                gpu_df = gpu_df[gpu_df['affinity'] > 0]
                gpu_df['affinity'] = cp.log10(gpu_df['affinity'].values)
                
                # Min-max normalize
                min_aff = float(gpu_df['affinity'].min())
                max_aff = float(gpu_df['affinity'].max())
                gpu_df['affinity_normalized'] = (gpu_df['affinity'] - min_aff) / (max_aff - min_aff)
                norm_params = {'min_aff': min_aff, 'max_aff': max_aff}
                print(f"  ↳ Affinity range: {min_aff:.2f} to {max_aff:.2f} (log10)")
            
            # Convert back to pandas for compatibility with rest of pipeline
            start_time = time.time()
            df = gpu_df.to_pandas()
            print(f"  ↳ DataFrame moved back to CPU in {time.time() - start_time:.2f}s")
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"❌ RAPIDS libraries not found: {e}")
            print("⚠️ Falling back to CPU processing for dataframes")
            
            # Fallback to CPU processing with visual feedback
            print("📊 Filtering by length on CPU...")
            before_len = len(df)
            df = df[df['smiles'].str.len() <= max_smiles_length]
            after_smiles = len(df)
            print(f"  ↳ SMILES length filter: {before_len - after_smiles} entries removed")
            
            df = df[df['target_seq'].str.len() <= max_protein_length]
            after_protein = len(df)
            print(f"  ↳ Protein length filter: {after_smiles - after_protein} entries removed")
            
            # Process affinity values if present
            if 'affinity' in df.columns:
                print("🧮 Processing affinity values...")
                df = df[df['affinity'] > 0]
                df['affinity'] = np.log10(df['affinity'])
                
                # Min-max normalize
                min_aff = df['affinity'].min()
                max_aff = df['affinity'].max()
                df['affinity_normalized'] = (df['affinity'] - min_aff) / (max_aff - min_aff)
                norm_params = {'min_aff': min_aff, 'max_aff': max_aff}
                print(f"  ↳ Affinity range: {min_aff:.2f} to {max_aff:.2f} (log10)")
    else:
        # Standard CPU processing with visual feedback
        print("🖥️ Using CPU for dataframe operations (GPU acceleration disabled)")
        
        print("📊 Filtering by length...")
        before_len = len(df)
        df = df[df['smiles'].str.len() <= max_smiles_length]
        after_smiles = len(df)
        print(f"  ↳ SMILES length filter: {before_len - after_smiles} entries removed")
        
        df = df[df['target_seq'].str.len() <= max_protein_length]
        after_protein = len(df)
        print(f"  ↳ Protein length filter: {after_smiles - after_protein} entries removed")
        
        # Process affinity values if present
        if 'affinity' in df.columns:
            print("🧮 Processing affinity values...")
            before_aff = len(df)
            df = df[df['affinity'] > 0]
            print(f"  ↳ Removed {before_aff - len(df)} entries with non-positive affinity")
            
            df['affinity'] = np.log10(df['affinity'])
            
            # Min-max normalize
            min_aff = df['affinity'].min()
            max_aff = df['affinity'].max()
            df['affinity_normalized'] = (df['affinity'] - min_aff) / (max_aff - min_aff)
            norm_params = {'min_aff': min_aff, 'max_aff': max_aff}
            print(f"  ↳ Affinity range: {min_aff:.2f} to {max_aff:.2f} (log10)")
    
    print(f"✅ Length filtering complete: {len(df)} entries remain")
    
    # Visual separator for vocabulary building
    print(f"\n{'-'*50}")
    print("📚 Building vocabulary")
    
    # Build vocabularies
    start_time = time.time()
    smiles_chars = set()
    protein_chars = set()
    
    # Process in chunks for better memory efficiency
    chunk_size = 10000
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i in tqdm(range(0, len(df), chunk_size), total=num_chunks, desc="Building vocabulary"):
        chunk = df.iloc[i:i+chunk_size]
        # Update char sets from this chunk
        for smiles in chunk['smiles']:
            smiles_chars.update(set(smiles))
        for seq in chunk['target_seq']:
            protein_chars.update(set(seq))
    
    smiles_vocab = ['<PAD>', '<START>'] + sorted(list(smiles_chars))
    smiles_char_to_idx = {char: idx for idx, char in enumerate(smiles_vocab)}
    smiles_idx_to_char = {idx: char for char, idx in smiles_char_to_idx.items()}
    
    protein_vocab = ['<PAD>'] + sorted(list(protein_chars))
    protein_char_to_idx = {char: idx for idx, char in enumerate(protein_vocab)}
    protein_idx_to_char = {idx: char for char, idx in protein_char_to_idx.items()}
    
    print(f"⌛ Vocabulary built in {time.time() - start_time:.2f}s")
    print(f"  ↳ SMILES vocabulary size: {len(smiles_vocab)}")
    print(f"  ↳ Protein vocabulary size: {len(protein_vocab)}")
    
    # Visual separator for dataset splitting
    print(f"\n{'-'*50}")
    print("✂️ Splitting dataset into train/validation/test")
    
    # Use stratified split if affinity is available
    if 'affinity_normalized' in df.columns:
        print("📊 Using stratified splitting based on affinity values")
        try:
            # Try to create bins for stratification with duplicate handling
            df['affinity_bin'] = pd.qcut(
                df['affinity_normalized'], 
                10, 
                labels=False, 
                duplicates='drop'  # Handle duplicate bin edges
            )
            
            # In case we have NaN values due to very few unique values
            if df['affinity_bin'].isna().any():
                print("⚠️ Duplicate bin edges detected, reducing to 3 bins")
                # Simple fallback: just use 3 bins instead of 10
                df['affinity_bin'] = pd.qcut(
                    df['affinity_normalized'], 
                    3,  # Fewer bins
                    labels=False, 
                    duplicates='drop'
                )
                
            # Handle the rare case where we still have NaN values
            if df['affinity_bin'].isna().any():
                print("⚠️ Still have NaN bins, using rank-based binning")
                # Just assign arbitrary bin values to maintain equal length
                df['affinity_bin'] = df['affinity_normalized'].rank(method='first') % 3
                
            train_df, test_df = train_test_split(
                df, test_size=0.1, random_state=42, stratify=df['affinity_bin']
            )
            train_df, val_df = train_test_split(
                train_df, test_size=0.1, random_state=42, stratify=train_df['affinity_bin']
            )
            print("✅ Stratified split successful")
        except ValueError as e:
            # If binning still fails, fallback to regular split
            print(f"❌ Stratified split failed: {e}")
            print("⚠️ Falling back to regular random split")
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
            
        # Remove temporary column
        train_df = train_df.drop('affinity_bin', axis=1)
        val_df = val_df.drop('affinity_bin', axis=1)
        test_df = test_df.drop('affinity_bin', axis=1)
    else:
        print("📊 Using random split (no affinity values available)")
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    print(f"✅ Dataset split complete:")
    print(f"  ↳ Training set:   {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  ↳ Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  ↳ Test set:       {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create the vocabulary data structure
    vocab_data = {
        'smiles_char_to_idx': smiles_char_to_idx,
        'smiles_idx_to_char': smiles_idx_to_char,
        'protein_char_to_idx': protein_char_to_idx,
        'protein_idx_to_char': protein_idx_to_char,
    }
    
    # Only add norm_params if we have affinity data and it was normalized
    if 'affinity' in df.columns and norm_params is not None:
        vocab_data['norm_params'] = norm_params
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"✨ PREPROCESSING COMPLETE")
    print(f"{'='*50}")
    
    return train_df, val_df, test_df, vocab_data

class BindingDBDataset(Dataset):
    def __init__(self, df, smiles_char_to_idx, protein_char_to_idx, 
                 max_smiles_len=100, max_protein_len=1000,
                 include_affinity=True, device=None, name="dataset"):
        self.df = df
        self.smiles_char_to_idx = smiles_char_to_idx
        self.protein_char_to_idx = protein_char_to_idx
        self.max_smiles_len = max_smiles_len
        self.max_protein_len = max_protein_len
        self.include_affinity = include_affinity
        self.name = name  # Add name for logging
        
        # Pre-process all data at initialization (can use GPU)
        self.processed_data = []
        
        # Get device for preprocessing
        self.device = device or torch.device("cpu")
        print(f"🔄 Creating {self.name} dataset with {len(df)} samples on {self.device.type}")
        
        # Process entire dataset at once
        start_time = time.time()
        with torch.no_grad():
            self._preprocess_all_data()
        
        # Print memory stats
        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(f"  ↳ GPU memory allocated: {gpu_mem:.2f} GB")
        
        print(f"✅ {self.name} dataset created in {time.time() - start_time:.2f}s - {len(self.processed_data)} samples")
    
    def _preprocess_all_data(self):
        """Process all data in parallel using GPU if available"""
        
        # Process in batches to avoid OOM
        batch_size = 1000
        num_batches = (len(self.df) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Preprocessing {self.name} data"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.df))
            
            batch_data = []
            for idx in range(start_idx, end_idx):
                row = self.df.iloc[idx]
                
                # Process SMILES
                smiles = row['smiles']
                smiles_indices = [self.smiles_char_to_idx['<START>']] + [self.smiles_char_to_idx[c] for c in smiles]
                
                if len(smiles_indices) < self.max_smiles_len:
                    smiles_indices += [self.smiles_char_to_idx['<PAD>']] * (self.max_smiles_len - len(smiles_indices))
                else:
                    smiles_indices = smiles_indices[:self.max_smiles_len]
                
                # Process protein
                protein = row['target_seq']
                protein_indices = [self.protein_char_to_idx[c] if c in self.protein_char_to_idx else 
                                  self.protein_char_to_idx['<PAD>'] for c in protein]
                
                if len(protein_indices) < self.max_protein_len:
                    protein_indices += [self.protein_char_to_idx['<PAD>']] * (self.max_protein_len - len(protein_indices))
                else:
                    protein_indices = protein_indices[:self.max_protein_len]
                
                # Create tensors
                smiles_tensor = torch.tensor(smiles_indices, dtype=torch.long)
                input_tensor = smiles_tensor[:-1]  
                target_tensor = smiles_tensor[1:]
                protein_tensor = torch.tensor(protein_indices, dtype=torch.long)
                
                # Add affinity if needed
                if self.include_affinity and 'affinity_normalized' in self.df.columns:
                    affinity = torch.tensor([row['affinity_normalized']], dtype=torch.float)
                    batch_data.append((input_tensor, target_tensor, protein_tensor, affinity))
                else:
                    batch_data.append((input_tensor, target_tensor, protein_tensor))
            
            self.processed_data.extend(batch_data)
            
            # Provide update on tensor placement
            if batch_idx == 0 and self.processed_data:
                item = self.processed_data[0]
                tensors_location = "GPU" if item[0].device.type == 'cuda' else "CPU"
                print(f"  ↳ Tensors stored on: {tensors_location}")
        smiles_tensor = torch.tensor(smiles_indices, dtype=torch.long, device=self.device)
        input_tensor = smiles_tensor[:-1]  
        target_tensor = smiles_tensor[1:]
        protein_tensor = torch.tensor(protein_indices, dtype=torch.long, device=self.device)
        
        if self.include_affinity and 'affinity_normalized' in self.df.columns:
            affinity = torch.tensor([row['affinity_normalized']], dtype=torch.float, device=self.device)
            batch_data.append((input_tensor, target_tensor, protein_tensor, affinity))
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def create_dataloaders(train_df, val_df, test_df, vocab_data, 
                       batch_size=64, max_smiles_len=100, max_protein_len=1000,
                       include_affinity=True, num_workers=4, device=None):
    """Create data loaders with detailed visual feedback"""
    
    print(f"\n{'-'*50}")
    print("📊 Creating datasets and dataloaders")
    
    # Use GPU for preprocessing if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
    
    # Create each dataset with name label for visual tracking
    train_dataset = BindingDBDataset(
        train_df, 
        vocab_data['smiles_char_to_idx'],
        vocab_data['protein_char_to_idx'],
        max_smiles_len,
        max_protein_len,
        include_affinity,
        device=device,
        name="Training"
    )
    
    val_dataset = BindingDBDataset(
        val_df, 
        vocab_data['smiles_char_to_idx'],
        vocab_data['protein_char_to_idx'],
        max_smiles_len,
        max_protein_len,
        include_affinity,
        device=device,
        name="Validation"
    )
    
    test_dataset = BindingDBDataset(
        test_df, 
        vocab_data['smiles_char_to_idx'],
        vocab_data['protein_char_to_idx'],
        max_smiles_len,
        max_protein_len,
        include_affinity,
        device=device,
        name="Test"
    )
    
    # Use fewer workers when preprocessing on GPU since data is already batched
    pin_memory = device.type == 'cpu'  # Only pin memory if using CPU for preprocessing
    
    print(f"\n🔄 Creating DataLoaders with batch_size={batch_size}")
    if device.type == 'cuda':
        print(f"  ↳ Using GPU for preprocessing - no additional workers needed")
    else:
        print(f"  ↳ Using {num_workers} workers for CPU preprocessing")
    
    # Create DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if device.type == 'cuda' else num_workers,  
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type == 'cuda' else num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type == 'cuda' else num_workers,
        pin_memory=pin_memory
    )
    
    # Summary
    total_time = time.time() - start_time
    print(f"✅ All datasets and dataloaders created in {total_time:.2f}s")
    print(f"  ↳ Training:   {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  ↳ Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  ↳ Test:       {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"{'-'*50}")
    
    return train_loader, val_loader, test_loader

class ProteinEncoder(nn.Module):
    """
    Neural network module for encoding protein sequences.
    
    This module processes protein sequences through an embedding layer,
    followed by a bidirectional LSTM with attention mechanism to capture
    important sequence features.
    
    Architecture:
        1. Embedding layer to convert amino acid indices to vectors
        2. Bidirectional LSTM to capture sequence context
        3. Attention mechanism to focus on important regions
        4. Fully connected output layer to produce final protein encoding
        
    Args:
        vocab_size (int): Size of the protein vocabulary.
        embed_dim (int, optional): Dimension of embedding vectors. Default: 128.
        hidden_dim (int, optional): Hidden dimension of LSTM. Default: 256.
        output_dim (int, optional): Dimension of output encoding. Default: 256.
        num_layers (int, optional): Number of LSTM layers. Default: 3.
        """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=256, num_layers=3):
        super(ProteinEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass for the protein encoder.
        
        Args:
            x (torch.Tensor): Input protein sequences as token indices.
                             Shape: [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Encoded protein representation.
                         Shape: [batch_size, output_dim]
        """
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        output, (hidden, _) = self.lstm(embedded)
        
        attention_weights = self.attention(output)
        context_vector = torch.sum(attention_weights * output, dim=1)

        return self.fc(context_vector)  # [batch_size, output_dim]

class ConditionalRNNGenerator(nn.Module):
    """
    Conditional RNN for generating molecules based on protein target features.
    
    This module generates SMILES strings character-by-character, conditioned on:
    1. A protein encoding vector (from the ProteinEncoder)
    2. An optional binding affinity value
    3. The previously generated characters
    
    Architecture:
        1. SMILES character embedding layer
        2. Target feature processing network with normalization
        3. Conditional LSTM that combines protein features and SMILES embeddings
        4. Output network that predicts the next character
    
    Args:
        vocab_size (int): Size of the SMILES vocabulary.
        embed_dim (int): Dimension of SMILES character embeddings.
        hidden_dim (int): Hidden dimension of LSTM.
        target_encoding_dim (int): Dimension of the protein encoding vector.
        use_affinity (bool, optional): Whether to use binding affinity values. Default: True.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, target_encoding_dim, use_affinity=True, latent_dim=64):
        super(ConditionalRNNGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.use_affinity = use_affinity
        
        target_input_dim = target_encoding_dim
        if use_affinity:
            target_input_dim += 1  # Add dimension for affinity value
        
        # Add latent dimension
        self.use_latent = latent_dim > 0
        if self.use_latent:
            target_input_dim += latent_dim
        
        # Now create the target encoder with correct input dimension
        self.target_encoder = nn.Sequential(
            nn.Linear(target_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim, 
            hidden_dim, 
            batch_first=True,
            num_layers=3,  
            dropout=0.2
        )
        
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, target_features, affinity=None, latent_z=None):
        batch_size, seq_len = x.size()
        
        x_embed = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_dim]

        # Process target features
        if self.use_affinity and affinity is not None:
            combined_target = torch.cat([target_features, affinity], dim=1)
        else:
            combined_target = target_features
        
        # Add latent vector if provided
        if self.use_latent and latent_z is not None:
            # Ensure latent_z has the correct batch size
            if latent_z.size(0) != combined_target.size(0):
                latent_z = latent_z[0:1].expand(combined_target.size(0), -1)
            
            combined_target = torch.cat([combined_target, latent_z], dim=1)
        
        # Process through target encoder
        target_encoded = self.target_encoder(combined_target)  # [batch_size, hidden_dim]
        
        # Expand target features to match sequence length
        target_expanded = target_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        #print(f"x_embed shape: {x_embed.shape}")
        #print(f"target_expanded shape: {target_expanded.shape}")
        
        # Check if batch sizes match, fix if they don't
        if x_embed.size(0) != target_expanded.size(0):
            # Resize target_expanded to match x_embed's batch size
            target_expanded = target_expanded[:x_embed.size(0)]
            print(f"Adjusted target_expanded shape: {target_expanded.shape}")
        
        # Fix dimension mismatch by resizing feature dimensions if needed
        expected_hidden_dim = self.lstm.input_size - x_embed.size(2)
        if target_expanded.size(2) != expected_hidden_dim:
            # Use adaptive pooling to resize the feature dimension
            target_expanded = F.adaptive_avg_pool1d(
                target_expanded.transpose(1, 2), expected_hidden_dim
            ).transpose(1, 2)
            print(f"Resized target_expanded shape: {target_expanded.shape}")
        
        # Concatenate molecule embedding with target features
        combined_input = torch.cat([x_embed, target_expanded], dim=2)
        
        lstm_output, _ = self.lstm(combined_input)
        output = self.output_network(lstm_output)
        return output

def train_model(
    model, 
    protein_encoder,
    train_loader, 
    val_loader,
    vocab_data,
    device,
    epochs=50,  
    lr=1e-4,    
    save_dir='./models',
    include_affinity=True,
    use_amp=True,
    gradient_accumulation_steps=8,
    patience=7,   # Early stopping patience
    save_all_epochs=True  # Add this new parameter to control saving all epochs
):
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    protein_encoder = protein_encoder.to(device)
    vae_encoder = ProteinVAEEncoder(
        vocab_size=protein_encoder.embedding.num_embeddings,
        embed_dim=protein_encoder.embedding.embedding_dim,
        hidden_dim=protein_encoder.lstm.hidden_size,
        latent_dim=64,
        num_layers=protein_encoder.lstm.num_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(protein_encoder.parameters()) + list(vae_encoder.parameters()),
        lr=lr,
        weight_decay=1e-6,  
        eps=1e-8          
    )
    
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
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_data['smiles_char_to_idx']['<PAD>'])
    
    # For mixed precision training
    scaler = amp.GradScaler() if use_amp else None
    
    for epoch in range(epochs):
        model.train()
        protein_encoder.train()
        train_loss = 0
        
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)  
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            if include_affinity:
                inputs, targets, protein_sequences, affinities = [b.to(device) for b in batch]
            else:
                inputs, targets, protein_sequences = [b.to(device) for b in batch]
                affinities = None
            
            # Process with mixed precision
            with amp.autocast() if use_amp else nullcontext():
                z, mu, logvar = vae_encoder(protein_sequences)
                protein_features = protein_encoder(protein_sequences)

                outputs = model(inputs, protein_features, affinities, latent_z=z)

                ce_loss = criterion(
                    outputs.contiguous().view(-1, len(vocab_data['smiles_char_to_idx'])),
                    targets.contiguous().view(-1)
                ) / gradient_accumulation_steps
                
                kl_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/inputs.size(0)
                beta = 0.1  # KL divergence weight, can be adjusted 
                loss = ce_loss + beta * kl_loss
            
            if use_amp:
                scaler.scale(loss).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # Unscales gradients before optimizer step
                    scaler.unscale_(optimizer)
                    
                    # Optional: clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(protein_encoder.parameters(), max_norm=1.0)
                    
                    # Update weights and zero gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  
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
        
        train_loss /= len(train_loader)
        train_time = time.time() - start_time
        
        torch.cuda.empty_cache()
        
        model.eval()
        protein_encoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if include_affinity:
                    inputs, targets, protein_sequences, affinities = [b.to(device) for b in batch]
                else:
                    inputs, targets, protein_sequences = [b.to(device) for b in batch]
                    affinities = None
                
                # Generate latent vector for validation
                z, mu, logvar = vae_encoder(protein_sequences)
                
                protein_features = protein_encoder(protein_sequences)
                outputs = model(inputs, protein_features, affinities, latent_z=z)
                
                loss = criterion(
                    outputs.contiguous().view(-1, len(vocab_data['smiles_char_to_idx'])),
                    targets.contiguous().view(-1)
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        new_lr = [group['lr'] for group in optimizer.param_groups][0]
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            old_lr = new_lr
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {train_time:.2f}s")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        
        # Always save each epoch if save_all_epochs is True
        if save_all_epochs:
            epoch_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'protein_encoder_state_dict': protein_encoder.state_dict(),
                'vae_encoder_state_dict': vae_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_data': vocab_data,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, epoch_save_path)
            print(f"Saved model checkpoint for epoch {epoch+1} at {epoch_save_path}")
        
        # Always save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'protein_encoder_state_dict': protein_encoder.state_dict(),
                'vae_encoder_state_dict': vae_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_data': vocab_data,
                'epoch': epoch,
                'val_loss': val_loss,
            }, best_model_path)
            
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        torch.cuda.empty_cache()
    
    return model, protein_encoder

def generate_molecules(
    model,
    protein_encoder,
    vae_encoder,
    target_sequence,
    vocab_data,
    affinity_value=0.7,
    num_molecules=10,
    temperature=1.5,  # Increased temperature for more diversity
    device='cuda',
    max_attempts=30,  # More attempts to generate molecules
    latent_noise=1.2  # Higher noise in latent space
):
    """
    Generate molecules for a specified protein target sequence with enhanced diversity.
    
    Args:
        model (ConditionalRNNGenerator): The trained molecule generator model.
        protein_encoder (ProteinEncoder): The trained protein encoder model.
        vae_encoder (ProteinVAEEncoder): The trained VAE encoder model.
        target_sequence (str): The amino acid sequence of the target protein.
        vocab_data (dict): Dictionary containing vocabulary mappings.
        affinity_value (float, optional): Target binding affinity (0-1). Default: 0.7.
        num_molecules (int, optional): Number of molecules to generate. Default: 10.
        temperature (float, optional): Sampling temperature. Higher = more diverse. Default: 1.5.
        device (str, optional): Device to run generation on ('cuda' or 'cpu'). Default: 'cuda'.
        max_attempts (int, optional): Maximum number of generation attempts. Default: 30.
        latent_noise (float, optional): Amount of noise to add to learned latent vector. Default: 1.2.
    Returns:
        list: List of valid SMILES strings for generated molecules.
    """
    print("\n" + "="*50)
    print("🧪 STARTING MOLECULE GENERATION")
    print("="*50)
    
    model.eval()
    protein_encoder.eval()
    vae_encoder.eval()
    
    # Pad protein sequence
    max_protein_len = 1000  
    print(f"📊 Processing target protein sequence ({len(target_sequence)} amino acids)")
    protein_indices = [vocab_data['protein_char_to_idx'][c] if c in vocab_data['protein_char_to_idx'] else 
                     vocab_data['protein_char_to_idx']['<PAD>'] for c in target_sequence]
    
    if len(protein_indices) < max_protein_len:
        protein_indices += [vocab_data['protein_char_to_idx']['<PAD>']] * (max_protein_len - len(protein_indices))
    else:
        protein_indices = protein_indices[:max_protein_len]
    
    protein_tensor = torch.tensor([protein_indices], dtype=torch.long).to(device)
    
    print("🔍 Encoding protein and generating latent representation")
    with torch.no_grad():
        # Get protein encoding and latent representation for a single protein
        protein_features_single = protein_encoder(protein_tensor)
        z_mean_single, mu, logvar = vae_encoder(protein_tensor)
        
        print(f"  ↳ Protein encoding dimension: {protein_features_single.shape}")
        print(f"  ↳ Latent vector dimension: {z_mean_single.shape}")
    
    if model.use_affinity:
        print(f"🔗 Using binding affinity: {affinity_value}")
    else:
        print("🔗 Not using binding affinity")
    
    valid_molecules = []
    all_valid_mols = set()  # Keep track of all valid molecules across attempts
    attempt = 0
    
    start_token_idx = vocab_data['smiles_char_to_idx']['<START>']
    pad_token_idx = vocab_data['smiles_char_to_idx']['<PAD>']
    
    while len(valid_molecules) < num_molecules and attempt < max_attempts:
        # Gradually increase diversity parameters with failed attempts
        current_temp = temperature * (1.0 + 0.05 * (attempt // 3))  # Gentler temperature increase
        current_noise = min(0.8, latent_noise * (1.0 + 0.1 * (attempt // 5)))  # Capped noise
        
        print(f"\n🔄 Attempt {attempt+1}/{max_attempts}")
        print(f"  ↳ Temperature: {current_temp:.2f}")
        print(f"  ↳ Latent noise: {current_noise:.2f}")
        
        with torch.no_grad():
            # Larger batch size to increase chance of finding valid molecules
            batch_size = min(num_molecules * 4, 128)
            current_seqs = torch.tensor([[start_token_idx]] * batch_size, device=device)
            batch_protein_features = protein_features_single.repeat(batch_size, 1)
            
            # Prepare affinity values if needed
            if model.use_affinity:
                batch_affinity = torch.tensor([[affinity_value]] * batch_size, 
                                             dtype=torch.float).to(device)
            else:
                batch_affinity = None
            
            # Use different latent vector approaches depending on attempt number
            if attempt % 5 == 0 and attempt > 0:
                # Every 5 attempts, generate completely fresh latent vectors
                print("  ↳ Using fresh random latent vectors")
                batch_latent_z = torch.randn(batch_size, z_mean_single.size(1), device=device) * 2.0
            else:
                # Normal approach: use learned latent vector + noise
                base_latent = z_mean_single
                noise = torch.randn(batch_size, base_latent.size(1), device=device) * current_noise
                batch_latent_z = base_latent.repeat(batch_size, 1) + noise
            
            # Track finished sequences
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for step in range(125):  # Maximum SMILES length (increased from 100)
                # Only process unfinished sequences
                if finished.all():
                    break
                    
                # Forward pass with latent vector from VAE
                outputs = model(current_seqs, batch_protein_features, batch_affinity, latent_z=batch_latent_z)
                
                # Apply temperature to logits
                next_token_logits = outputs[:, -1, :] / current_temp
                
                # Only apply anti-repeat penalties after a certain sequence length
                if step > 10:
                    for b in range(batch_size):
                        if not finished[b]:
                            # Get tokens already generated
                            prev_tokens = current_seqs[b, -10:] # Check last 10 tokens only
                            
                            # Penalize repeated tokens
                            token_counts = {}
                            for t in prev_tokens:
                                token_id = t.item()
                                if token_id in token_counts:
                                    token_counts[token_id] += 1
                                else:
                                    token_counts[token_id] = 1
                                    
                            for token_id, count in token_counts.items():
                                if count > 2:  # If token appears more than twice in last 10 positions
                                    penalty = 1.0 + 0.5 * count  # Increasing penalty for more repetitions
                                    next_token_logits[b, token_id] /= penalty
                
                # Apply sampling with temperature
                probs = F.softmax(next_token_logits, dim=1)
                next_tokens = torch.multinomial(probs, 1)
                
                # Update sequences and finished status
                current_seqs = torch.cat([current_seqs, next_tokens], dim=1)
                finished = finished | (next_tokens.squeeze(-1) == pad_token_idx)
                
                # Early stopping: if generated sequence is too long
                # More generous length limit (100 instead of 80)
                if step > 100:
                    for b in range(batch_size):
                        if not finished[b]:
                            current_seqs[b, -1] = pad_token_idx
                            finished[b] = True
        
        # Process generated sequences
        generated_smiles = []
        for seq in current_seqs:
            smiles = ''.join([vocab_data['smiles_idx_to_char'][idx.item()] 
                           for idx in seq if idx.item() not in [pad_token_idx, start_token_idx]])
            generated_smiles.append(smiles)
        
        # More lenient validation - don't use QED filter initially
        new_valid_count = 0
        for smiles in generated_smiles:
            # Skip empty strings and already found molecules
            if not smiles or smiles in all_valid_mols:
                continue
                
            # Try to validate the molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                all_valid_mols.add(smiles)
                valid_molecules.append(smiles)
                new_valid_count += 1
                print(f"✅ Valid molecule found: {smiles}")
        
        print(f"📊 Attempt results: {new_valid_count} new valid molecules")
        
        # If no molecules were found in this attempt, show examples
        if new_valid_count == 0 and attempt % 2 == 0:
            print("❌ No valid molecules found in this attempt")
            print("Examples of generated strings:")
            for i, smiles in enumerate(generated_smiles[:3]):
                print(f"  {i+1}. {smiles}")
        
        attempt += 1
    
    # Final filtering and selection
    final_molecules = []
    if len(valid_molecules) > 0:
        print(f"\n✨ Generated {len(valid_molecules)} valid molecules")
        
        # Sort by drug-likeness if we have enough molecules
        if len(valid_molecules) > num_molecules:
            from rdkit.Chem import QED
            scored_mols = []
            for smiles in valid_molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    qed = QED.qed(mol)
                    scored_mols.append((smiles, qed))
            
            # Sort by QED score (higher is better)
            scored_mols.sort(key=lambda x: x[1], reverse=True)
            print("🔍 Molecules ranked by drug-likeness (QED):")
            for i, (smiles, qed) in enumerate(scored_mols[:num_molecules]):
                print(f"  {i+1}. {smiles} (QED: {qed:.3f})")
            
            final_molecules = [m[0] for m in scored_mols[:num_molecules]]
        else:
            final_molecules = valid_molecules
    else:
        print("❌ Failed to generate any valid molecules")
    
    print("="*50)
    return final_molecules

class ProteinVAEEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        
        # Add attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, _) = self.lstm(embedded)
        
        # Add attention mechanism
        attention_weights = self.attention(output)
        context = torch.sum(attention_weights * output, dim=1)
        
        # Use attention-weighted context instead of just last hidden state
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        
        # Use the reparameterization trick with temperature annealing
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        
        return z, mu, logvar

# Add new argument in the parser section
def main(args):
    """
    Main function to run the conditional RNN training pipeline.
    
    Orchestrates the complete workflow from data loading to model evaluation:
    1. Set up computation device and optimize CUDA settings
    2. Load and preprocess the dataset
    3. Create data loaders
    4. Initialize and train models
    5. Optionally generate example molecules
    
    Args:
        args: Command-line arguments with data paths, model parameters,
              training parameters, and output options
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Optimize CUDA operations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"Loading data from {args.data_path}")
    train_df, val_df, test_df, vocab_data = preprocess_bindingdb(
        args.data_path, 
        max_smiles_length=args.max_smiles_len,
        max_protein_length=args.max_protein_len,
        use_gpu=args.use_gpu_preprocessing,
        skip_validation=args.skip_validation  
    )
    use_gpu_preprocessing = args.use_gpu_preprocessing and device.type == 'cuda'
    preproc_device = device if use_gpu_preprocessing else torch.device("cpu")
    
    print(f"Preprocessing data on: {preproc_device.type}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, vocab_data, 
        batch_size=args.batch_size,
        max_smiles_len=args.max_smiles_len, 
        max_protein_len=args.max_protein_len,
        include_affinity=args.use_affinity,
        num_workers=args.num_workers,
        device=preproc_device if use_gpu_preprocessing else None
    )
    
    protein_vocab_size = len(vocab_data['protein_char_to_idx'])
    smiles_vocab_size = len(vocab_data['smiles_char_to_idx'])
    
    protein_encoder = ProteinEncoder(
        vocab_size=protein_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )
    
    model = ConditionalRNNGenerator(
        vocab_size=smiles_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim*2,  
        target_encoding_dim=args.output_dim,
        use_affinity=args.use_affinity,
        latent_dim=64 
    )
    
    
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
        include_affinity=args.use_affinity, 
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_all_epochs=args.save_all_epochs  # Pass the new argument
    )

    # Load the best model, which includes the VAE
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), map_location=device, weights_only=False)
    vae_encoder = ProteinVAEEncoder(
        vocab_size=protein_encoder.embedding.num_embeddings,
        embed_dim=protein_encoder.embedding.embedding_dim,
        hidden_dim=protein_encoder.lstm.hidden_size,
        latent_dim=64,
        num_layers=protein_encoder.lstm.num_layers
    ).to(device)
    
    if 'vae_encoder_state_dict' in checkpoint:
        vae_encoder.load_state_dict(checkpoint['vae_encoder_state_dict'])
        print("Loaded VAE encoder for generation")
    
    if args.generate_examples:
        example_protein = test_df.iloc[0]['target_seq']
        generated_molecules = generate_molecules(
            model,
            protein_encoder,
            vae_encoder,  # Pass the VAE encoder
            example_protein,
            vocab_data,
            affinity_value=0.7 if args.use_affinity else None,
            num_molecules=10,
            device=device,
            latent_noise=0.3  # Control diversity
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
    parser.add_argument('--embed_dim', type=int, default=64,  
                   help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,  
                       help='Hidden dimension') 
    parser.add_argument('--output_dim', type=int, default=256,  
                       help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=2,  
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
    
    # NEW: Add argument for optional binding affinity
    parser.add_argument('--use_affinity', action='store_true', 
                        help='Use binding affinity values during training and generation')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./models', 
                        help='Directory to save models')
    parser.add_argument('--generate_examples', action='store_true', 
                        help='Generate example molecules after training')
    parser.add_argument('--save_all_epochs', action='store_true',
                        help='Save model checkpoint after each epoch, not just the best one')
    
    # Add GPU preprocessing option
    parser.add_argument('--use_gpu_preprocessing', action='store_true', 
                        help='Use GPU for dataset preprocessing (faster for large datasets)')
    parser.add_argument('--skip_validation', action='store_true',
                    help='Skip SMILES validation step (use for pre-validated datasets)')

    args = parser.parse_args()
    
    main(args)