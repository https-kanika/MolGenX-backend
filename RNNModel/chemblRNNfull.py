import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from RnnClass import RNNGenerator
import os
import numpy as np
import time
from rdkit import Chem
import json
import argparse
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RNN on full ChEMBL dataset')
    parser.add_argument('--initial_model', type=str, default='rnn_model_chembl.pth',
                      help='Path to initial model weights')
    parser.add_argument('--data_path', type=str, default='./data/chembl_full_cleaned.smi',
                      help='Path to full ChEMBL SMILES file')
    parser.add_argument('--vocab_file', type=str, default='chembl_vocabulary.json',
                      help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                      help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='full_chembl_model',
                      help='Directory to save models')
    return parser.parse_args()

def validate_smiles(smiles):
    """Check if a SMILES string is valid using RDKit"""
    if not smiles or len(smiles) < 3:
        return False
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return True
    except:
        return False

def load_vocabulary(vocab_file):
    """Load vocabulary from JSON file"""
    if not os.path.exists(vocab_file):
        print(f"Vocabulary file {vocab_file} not found!")
        return None, None
        
    with open(vocab_file, 'r') as f:
        vocab_data = json.load(f)
        
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    
    return char_to_idx, idx_to_char

def generate_molecule(model, char_to_idx, idx_to_char, device, temperature=0.7, max_len=100):
    """Generate a molecule using the trained model"""
    model.eval()
    with torch.no_grad():
        # Start with C (carbon) if available, else use first non-padding token
        start_char = 'C' if 'C' in char_to_idx else list(char_to_idx.keys())[1]  # Skip padding token
        current_seq = torch.tensor([[char_to_idx[start_char]]], device=device)
        
        for _ in range(max_len):
            output = model(current_seq)
            next_logits = output[0, -1, :] / temperature
            next_probs = torch.softmax(next_logits, dim=0)
            next_char_idx = torch.multinomial(next_probs, 1)
            
            current_seq = torch.cat([current_seq, next_char_idx.unsqueeze(0)], dim=1)
            
            if next_char_idx.item() == 0:  # PAD token
                break
        
        # Convert to SMILES
        smiles = ''.join([idx_to_char[idx.item()] for idx in current_seq[0] 
                         if idx.item() > 0 and idx.item() in idx_to_char])
        return smiles

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")
    
    # Set up device and mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True if torch.cuda.is_available() else False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        return
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_file}")
    char_to_idx, idx_to_char = load_vocabulary(args.vocab_file)
    if char_to_idx is None:
        print("Failed to load vocabulary. Exiting.")
        return
    
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model and load pretrained weights
    model = RNNGenerator(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)
    
    if os.path.exists(args.initial_model):
        try:
            model.load_state_dict(torch.load(args.initial_model, map_location=device))
            print(f"Loaded initial weights from {args.initial_model}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting from scratch instead.")
    else:
        print(f"Initial model not found: {args.initial_model}")
    
    model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.7)
    
    # Count total lines in file
    print("Counting total lines in file...")
    with open(args.data_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines in file: {total_lines}")
    
    # Process data in chunks
    chunk_size = 100000  # Adjust based on memory
    max_length = 100  # Maximum SMILES length
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        chunk_num = 0
        start_time = time.time()
        
        # Open the file for reading
        with open(args.data_path, 'r') as f:
            while True:
                # Read a chunk of SMILES strings
                smiles_chunk = []
                for _ in range(chunk_size):
                    line = f.readline()
                    if not line:  # End of file
                        break
                    smiles_chunk.append(line.strip())
                
                if not smiles_chunk:  # No more data
                    break
                
                chunk_num += 1
                chunk_start = time.time()
                
                # Process this chunk
                print(f"Epoch {epoch+1}/{args.epochs}, Processing chunk {chunk_num} with {len(smiles_chunk)} SMILES...")
                
                # Convert SMILES to sequences
                sequences = []
                for smi in smiles_chunk:
                    try:
                        # Create sequence from SMILES
                        seq = []
                        for char in smi:
                            if char in char_to_idx:
                                seq.append(char_to_idx[char])
                        
                        # Skip if too short
                        if len(seq) < 3:
                            continue
                            
                        # Truncate or pad
                        if len(seq) > max_length:
                            seq = seq[:max_length]
                        else:
                            seq += [0] * (max_length - len(seq))
                            
                        sequences.append(seq)
                    except:
                        continue
                
                if not sequences:  # Skip empty chunks
                    continue
                    
                # Convert to tensor and train
                train_data = torch.tensor(sequences, dtype=torch.long)
                total_samples += len(train_data)
                
                # Shuffle data
                indices = torch.randperm(len(train_data))
                shuffled_data = train_data[indices]
                
                # Process in batches
                for i in range(0, len(shuffled_data), args.batch_size):
                    batch = shuffled_data[i:i + args.batch_size].to(device)
                    
                    # Skip small batches
                    if len(batch) < 4:
                        continue
                    
                    inputs, targets = batch[:, :-1], batch[:, 1:]
                    
                    # Training step with mixed precision
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item() * len(batch)
                
                # Progress update
                progress = chunk_num * chunk_size / total_lines * 100
                chunk_time = time.time() - chunk_start
                print(f"  Progress: {progress:.1f}%, Chunk time: {chunk_time:.2f}s")
        
        # Epoch summary
        avg_loss = total_loss / max(1, total_samples)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, Samples: {total_samples}, Time: {epoch_time:.2f}s")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Generate sample molecules after each epoch
        print("Generating sample molecules...")
        samples_path = os.path.join(args.save_dir, f"samples_epoch_{epoch+1}.txt")
        
        with open(samples_path, 'w') as f:
            valid_count = 0
            for i in range(10):
                molecule = generate_molecule(model, char_to_idx, idx_to_char, device)
                is_valid = validate_smiles(molecule)
                status = "VALID" if is_valid else "INVALID"
                
                if is_valid:
                    valid_count += 1
                
                print(f"Sample {i+1}: {molecule} - {status}")
                f.write(f"{molecule} - {status}\n")
            
            # Write summary
            f.write(f"\nValidity: {valid_count}/10 ({valid_count*10}%)\n")
            print(f"Samples saved to {samples_path}")
    
    # Save final model
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # Generate a larger batch of molecules with the final model
    print("\nGenerating molecules with final model...")
    final_samples_path = os.path.join(args.save_dir, "final_samples.txt")
    
    with open(final_samples_path, 'w') as f:
        valid_count = 0
        valid_mols = []
        
        for i in range(100):
            molecule = generate_molecule(model, char_to_idx, idx_to_char, device)
            is_valid = validate_smiles(molecule)
            
            if is_valid:
                valid_count += 1
                valid_mols.append(molecule)
            
            status = "VALID" if is_valid else "INVALID"
            f.write(f"{molecule} - {status}\n")
            
            if i % 10 == 0:
                print(f"Generated {i}/100 molecules...")
        
        # Write summary
        f.write(f"\nFinal validity: {valid_count}/100 ({valid_count}%)\n")
        
        # Write valid molecules separately
        f.write("\n=== VALID MOLECULES ===\n")
        for mol in valid_mols:
            f.write(f"{mol}\n")
            
    print(f"Final samples saved to {final_samples_path}")
    print(f"Validity rate: {valid_count}%")
    print("Training complete!")

if __name__ == "__main__":
    main()