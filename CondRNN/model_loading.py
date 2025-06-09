import torch
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# Import your model definitions
from conditionalRNN import ProteinEncoder, ConditionalRNNGenerator, generate_molecules, validate_molecule


def load_model(model_path, device):
    """Load a trained conditional RNN model"""
    
    print(f"Loading model from {model_path}")
    # Load the saved dictionary
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    
    # Extract vocab data
    vocab_data = checkpoint['vocab_data']
    
    # Create models with the right dimensions
    protein_vocab_size = len(vocab_data['protein_char_to_idx'])
    smiles_vocab_size = len(vocab_data['smiles_char_to_idx'])
    
    # These should match your training parameters
    embed_dim = 32
    hidden_dim = 128
    output_dim = 128
    num_layers = 1
    
    # Initialize models
    protein_encoder = ProteinEncoder(
        vocab_size=protein_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers
    )
    
    model = ConditionalRNNGenerator(
        vocab_size=smiles_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim*2,
        target_encoding_dim=output_dim,
        use_affinity=True
    )
    
    # Load state dictionaries
    protein_encoder.load_state_dict(checkpoint['protein_encoder_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to evaluation mode
    protein_encoder.to(device)
    model.to(device)
    protein_encoder.eval()
    model.eval()
    
    print(f"Model loaded successfully (from epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model, protein_encoder, vocab_data


def visualize_molecules(smiles_list, output_file=None):
    """Generate a visual grid of molecules from SMILES strings"""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if validate_molecule(s)]
    mols = [m for m in mols if m is not None]
    
    if not mols:
        print("No valid molecules to visualize")
        return None
    
    # Calculate grid dimensions
    n_mols = len(mols)
    cols = min(5, n_mols)
    rows = (n_mols + cols - 1) // cols
    
    # Create grid image
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=cols, 
        subImgSize=(300, 200), 
        legends=[f"Mol_{i+1}" for i in range(len(mols))],
        useSVG=False
    )
    
    if output_file:
        img.save(output_file)
        print(f"Molecule visualization saved to {output_file}")
    
    # For notebooks, you can display the image directly
    # from IPython.display import display
    # display(img)
    
    return img


def generate_for_target(model_path, target_sequence_or_file, affinity=0.7, n_molecules=10, output_folder="generated"):
    """Generate molecules for a specific target protein sequence"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, protein_encoder, vocab_data = load_model(model_path, device)
    
    # Process target sequence
    if os.path.isfile(target_sequence_or_file):
        # Read sequence from file
        with open(target_sequence_or_file, 'r') as f:
            target_sequence = f.read().strip()
    else:
        # Use input directly as sequence
        target_sequence = target_sequence_or_file
    
    print(f"Target sequence length: {len(target_sequence)}")
    
    # Generate molecules
    print(f"Generating {n_molecules} molecules with affinity {affinity}...")
    molecules = generate_molecules(
        model,
        protein_encoder,
        target_sequence,
        vocab_data,
        affinity_value=affinity,
        num_molecules=n_molecules,
        device=device,
        temperature=0.7,
        max_attempts=5
    )
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save generated molecules
    output_file = os.path.join(output_folder, "generated_molecules.csv")
    pd.DataFrame({"SMILES": molecules}).to_csv(output_file, index=False)
    print(f"Generated {len(molecules)} molecules, saved to {output_file}")
    
    # Visualize molecules
    #if molecules:
        #img_file = os.path.join(output_folder, "molecule_visualization.png")
        #visualize_molecules(molecules, img_file)
    
    return molecules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate molecules using a trained conditional RNN')
    
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
    
    args = parser.parse_args()
    
    generate_for_target(
        args.model_path,
        args.target,
        args.affinity,
        args.n_molecules,
        args.output_folder
    )
