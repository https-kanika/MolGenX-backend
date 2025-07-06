import torch
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
import os
from protienEncodedVAE import ProteinEncoder, ConditionalRNNGenerator, ProteinVAEEncoder, generate_molecules

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def load_model(model_path, device, use_affinity=True, embed_dim=64, hidden_dim=256, 
               output_dim=256, num_layers=2, latent_dim=64):
    """
    Load a trained conditional RNN molecule generation model with VAE from a checkpoint file.
    
    This function loads both the protein encoder and molecule generator components
    of the conditional RNN model, reconstructing them with the same architecture
    parameters used during training.
    
    Args:
        model_path (str): Path to the model checkpoint file or directory containing 
                         'best_model.pt'.
        device (torch.device): Device to load the model onto (CPU or CUDA).
        use_affinity (bool, optional): Whether the model was trained with affinity values.
                                     Default: True.
        embed_dim (int, optional): Embedding dimension. Default: 64.
        hidden_dim (int, optional): Hidden dimension. Default: 256.
        output_dim (int, optional): Output dimension. Default: 256.
        num_layers (int, optional): Number of layers. Default: 2.
        latent_dim (int, optional): Latent dimension. Default: 64.
    
    Returns:
        tuple: A tuple containing:
            - model (ConditionalRNNGenerator): The loaded molecule generator model
            - protein_encoder (ProteinEncoder): The loaded protein encoder model
            - vae_encoder (ProteinVAEEncoder): The loaded VAE encoder model
            - vocab_data (dict): Dictionary containing vocabulary mappings for 
                                SMILES and protein tokens

    IMPORTANT: Match these parameters with your training parameters
    """
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "best_model.pt")
        print(f"Model path is a directory, looking for model at: {model_path}")
    
    # Check if file exists and get file size
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_size = os.path.getsize(model_path)
    print(f"Loading model from {model_path} (size: {file_size} bytes)")
    
    # Check if file is too small (likely corrupted)
    if file_size < 1024:  # Less than 1KB is suspicious
        raise ValueError(f"Model file appears to be corrupted (size: {file_size} bytes)")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        print("This could be due to:")
        print("1. Corrupted model file")
        print("2. Model saved with different PyTorch version")
        print("3. Incomplete file download/transfer")
        print("4. File not being a valid PyTorch checkpoint")
        raise
    
    # Validate checkpoint structure
    required_keys = ['vocab_data', 'protein_encoder_state_dict', 'model_state_dict']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
    
    vocab_data = checkpoint['vocab_data']
    protein_vocab_size = len(vocab_data['protein_char_to_idx'])
    smiles_vocab_size = len(vocab_data['smiles_char_to_idx'])
    
    print(f"Using model parameters: embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
          f"output_dim={output_dim}, num_layers={num_layers}, latent_dim={latent_dim}")
    
    protein_encoder = ProteinEncoder(
        vocab_size=protein_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers
    )
    
    # Create VAE encoder
    vae_encoder = ProteinVAEEncoder(
        vocab_size=protein_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers
    )
    
    # Create model with the appropriate affinity and latent settings
    model = ConditionalRNNGenerator(
        vocab_size=smiles_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim*2, 
        target_encoding_dim=output_dim,
        use_affinity=use_affinity,
        latent_dim=latent_dim  # Add latent dimension parameter
    )
    
    try:
        protein_encoder.load_state_dict(checkpoint['protein_encoder_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model state dictionaries: {e}")
        print("This could be due to mismatched model architecture parameters.")
        print("Please verify that the architecture parameters match those used during training.")
        raise
    
    # Load VAE encoder state if available
    if 'vae_encoder_state_dict' in checkpoint:
        try:
            vae_encoder.load_state_dict(checkpoint['vae_encoder_state_dict'])
            print("VAE encoder loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading VAE encoder: {e}")
            print("Using initialized VAE encoder weights instead")
    else:
        print("WARNING: No VAE encoder state found in checkpoint, using initialized weights")
    
    protein_encoder.to(device)
    vae_encoder.to(device)
    model.to(device)
    
    protein_encoder.eval()
    vae_encoder.eval()
    model.eval()
    
    print(f"Model loaded successfully (from epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"Model {'uses' if use_affinity else 'does not use'} affinity values")
    
    return model, protein_encoder, vae_encoder, vocab_data


import random

def generate_for_target(model_path, target_sequence_or_file, affinity=0.7, 
                        n_molecules=10, output_folder="generated", use_affinity=True,
                        embed_dim=64, hidden_dim=256, output_dim=256, num_layers=2, latent_dim=64,
                        base_temperature=0.7, temp_variation=0.1, base_latent_noise=0.2, noise_variation=0.1,
                        max_generation_attempts=5):
    """
    Generate molecules for a specific target protein sequence.
    
    Args:
        # ...existing parameters...
        base_temperature (float): Base temperature for generation
        temp_variation (float): How much temperature can vary between attempts
        base_latent_noise (float): Base latent noise value
        noise_variation (float): How much latent noise can vary between attempts
        max_generation_attempts (int): Maximum number of overall generation attempts
    
    Returns:
        list: A list of SMILES strings representing the generated molecules.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, protein_encoder, vae_encoder, vocab_data = load_model(
        model_path, device, use_affinity, embed_dim, hidden_dim, 
        output_dim, num_layers, latent_dim
    )

    if os.path.isfile(target_sequence_or_file):
        with open(target_sequence_or_file, 'r') as f:
            target_sequence = f.read().strip()
    else:
        target_sequence = target_sequence_or_file
    
    print(f"Target sequence length: {len(target_sequence)}")
    if use_affinity:
        print(f"Generating {n_molecules} molecules with affinity {affinity}...")
    else:
        print(f"Generating {n_molecules} molecules (model does not use affinity values)...")
    
    valid_molecules = []
    attempt = 0
    
    while len(valid_molecules) < n_molecules and attempt < max_generation_attempts:
        # Vary temperature and noise for each attempt
        temperature = base_temperature + random.uniform(-temp_variation, temp_variation)
        latent_noise = base_latent_noise + random.uniform(-noise_variation, noise_variation)
        
        # Ensure values stay in reasonable ranges
        temperature = max(0.1, min(1.5, temperature))
        latent_noise = max(0.0, min(0.5, latent_noise))
        
        print(f"Attempt {attempt+1}/{max_generation_attempts}: Temperature={temperature:.2f}, Noise={latent_noise:.2f}")
        
        # Generate molecules with these parameters (max_attempts=1 for each call)
        molecules = generate_molecules(
            model,
            protein_encoder,
            vae_encoder,
            target_sequence,
            vocab_data,
            affinity_value=affinity if use_affinity else None,
            num_molecules=n_molecules - len(valid_molecules),  # Only generate what's still needed
            device=device,
            temperature=temperature,
            max_attempts=1,  # Only one attempt per call with these parameters
            latent_noise=latent_noise
        )
        
        # Add valid molecules to our list
        for smi in molecules:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol and smi not in valid_molecules:
                    qed = QED.qed(mol)
                    print(f"  Valid: {smi[:40]}... (QED: {qed:.2f})")
                    valid_molecules.append(smi)
                    if len(valid_molecules) >= n_molecules:
                        break
            except Exception as e:
                print(f"  Invalid molecule: {str(e)}")
                continue
        
        attempt += 1
    
    # Calculate average QED for the generated molecules
    qed_values = []
    for smi in valid_molecules:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                qed = QED.qed(mol)
                qed_values.append(qed)
        except Exception:
            continue
    
    avg_qed = sum(qed_values) / len(qed_values) if qed_values else 0
    print(f"Average QED of generated molecules: {avg_qed:.4f}")
    print(f"Generated {len(valid_molecules)} valid molecules")

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "generated_molecules.csv")
    pd.DataFrame({"SMILES": valid_molecules}).to_csv(output_file, index=False)
    print(f"Generated molecules saved to {output_file}")

    return valid_molecules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate molecules using a trained conditional RNN with VAE')
    
    parser.add_argument('--model_path', type=str, default='./models/best_model.pt',
                      help='Path to saved model checkpoint')
    parser.add_argument('--target', type=str, required=True,
                      help='Target protein sequence or path to text file with sequence')
    parser.add_argument('--affinity', type=float, default=0.8,
                      help='Target affinity (0-1 scale)')
    parser.add_argument('--n_molecules', type=int, default=10,
                      help='Number of molecules to generate')
    parser.add_argument('--output_folder', type=str, default='generated',
                      help='Output folder for generated molecules')
    
    # Add new argument for models without affinity
    parser.add_argument('--no_affinity', action='store_true',
                      help='Load model that was trained without affinity values')
    
    # Add model architecture parameters
    parser.add_argument('--embed_dim', type=int, default=64,
                      help='Embedding dimension (default: 64)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension (default: 256)')
    parser.add_argument('--output_dim', type=int, default=256,
                      help='Output dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of layers (default: 2)')
    parser.add_argument('--latent_dim', type=int, default=64,
                      help='Latent dimension (default: 64)')
    parser.add_argument('--check_model', action='store_true',
                      help='Only check if model file is valid, do not generate molecules')
                      
    # Add temperature and noise variation parameters - MOVED BEFORE parse_args()
    parser.add_argument('--base_temp', type=float, default=0.7,
                      help='Base temperature value (default: 0.7)')
    parser.add_argument('--temp_var', type=float, default=0.1,
                      help='Temperature variation (default: 0.1)')
    parser.add_argument('--base_noise', type=float, default=0.2,
                      help='Base latent noise value (default: 0.2)')
    parser.add_argument('--noise_var', type=float, default=0.1,
                      help='Latent noise variation (default: 0.1)')
    parser.add_argument('--max_attempts', type=int, default=5,
                      help='Maximum generation attempts (default: 5)')
                      
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use the no_affinity flag to determine if the model uses affinity
    use_affinity = not args.no_affinity

    if args.check_model:
        print("Checking if model can be loaded...")
        _ = load_model(args.model_path, device, use_affinity, args.embed_dim, 
                       args.hidden_dim, args.output_dim, args.num_layers, args.latent_dim)
        print("Model loaded successfully!")
    else:
        # Call generate_for_target with the correct parameter names
        generate_for_target(
            args.model_path,
            args.target,
            args.affinity,
            args.n_molecules,
            args.output_folder,
            use_affinity,
            args.embed_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_layers,
            args.latent_dim,
            args.base_temp,
            args.temp_var,
            args.base_noise,
            args.noise_var,
            args.max_attempts
        )
        