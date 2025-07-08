import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
import argparse
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your VAE model components
from cycle_loading import load_model
from cycle import generate_molecules

def calculate_molecular_properties(smiles_list):
    """
    Calculate comprehensive molecular properties for a list of SMILES strings.
    
    Args:
        smiles_list (list): List of SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame containing molecular properties
    """
    properties = []
    
    print(f"Calculating properties for {len(smiles_list)} molecules...")
    
    for smiles in tqdm(smiles_list, desc="Computing molecular properties"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Calculate various molecular properties
            mol_props = {
                'SMILES': smiles,
                'QED': QED.qed(mol),
                'LogP': Crippen.MolLogP(mol),
                'Molecular_Weight': Descriptors.MolWt(mol),
                'TPSA': CalcTPSA(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'Rotatable_Bonds': CalcNumRotatableBonds(mol),
                'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
                'Heavy_Atoms': mol.GetNumHeavyAtoms(),
                'Rings': Descriptors.RingCount(mol),
                'Formal_Charge': Chem.rdmolops.GetFormalCharge(mol),
                
                # Lipinski's Rule of Five
                'Lipinski_Violations': sum([
                    Descriptors.MolWt(mol) > 500,
                    Crippen.MolLogP(mol) > 5,
                    Lipinski.NumHDonors(mol) > 5,
                    Lipinski.NumHAcceptors(mol) > 10
                ]),
            }
            
            properties.append(mol_props)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
            continue
    
    df = pd.DataFrame(properties)
    print(f"Successfully processed {len(df)} molecules")
    return df

def plot_property_distributions(df, output_dir="evaluation_results"):
    """
    Create the 4 main property distribution plots.
    
    Args:
        df (pd.DataFrame): DataFrame containing molecular properties
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a 2x2 subplot for the main 4 properties
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drug-like Property Distributions of Generated Molecules', fontsize=16, fontweight='bold')
    
    # 1. QED Distribution
    axes[0, 0].hist(df['QED'], bins=30, alpha=0.7, color='#d1bce3', edgecolor='black')
    axes[0, 0].axvline(df['QED'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["QED"].mean():.3f}')
    axes[0, 0].axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Drug-like threshold (0.5)')
    axes[0, 0].set_xlabel('QED (Quantitative Estimate of Drug-likeness)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('QED Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. LogP Distribution
    axes[0, 1].hist(df['LogP'], bins=30, alpha=0.7, color='#84cae7', edgecolor='black')
    axes[0, 1].axvline(df['LogP'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["LogP"].mean():.2f}')
    axes[0, 1].axvline(-0.4, color='orange', linestyle='--', linewidth=2, label='Lower limit (-0.4)')
    axes[0, 1].axvline(5.6, color='orange', linestyle='--', linewidth=2, label='Upper limit (5.6)')
    axes[0, 1].set_xlabel('LogP (Lipophilicity)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('LogP Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Molecular Weight Distribution
    axes[1, 0].hist(df['Molecular_Weight'], bins=30, alpha=0.7, color='#62d1c7', edgecolor='black')
    axes[1, 0].axvline(df['Molecular_Weight'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Molecular_Weight"].mean():.1f}')
    axes[1, 0].axvline(500, color='orange', linestyle='--', linewidth=2, label='Lipinski limit (500 Da)')
    axes[1, 0].set_xlabel('Molecular Weight (Da)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Molecular Weight Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. TPSA Distribution
    axes[1, 1].hist(df['TPSA'], bins=30, alpha=0.7, color='#fbf8ea', edgecolor='black')
    axes[1, 1].axvline(df['TPSA'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["TPSA"].mean():.1f}')
    axes[1, 1].axvline(140, color='orange', linestyle='--', linewidth=2, label='Drug-like limit (140 A^2)')
    axes[1, 1].set_xlabel('TPSA (Topological Polar Surface Area, A^2)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('TPSA Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_property_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(df, output_dir):
    """
    Generate and save comprehensive summary statistics.
    """
    print("\n" + "="*60)
    print("MOLECULAR PROPERTY SUMMARY STATISTICS")
    print("="*60)
    
    # Basic statistics
    stats_summary = df.describe()
    print("\nBasic Statistics:")
    print(stats_summary.round(3))
    
    # Drug-likeness analysis
    drug_like_count = len(df[df['QED'] >= 0.5])
    lipinski_compliant = len(df[df['Lipinski_Violations'] == 0])
    
    print(f"\nDrug-likeness Analysis:")
    print(f"Total molecules analyzed: {len(df)}")
    print(f"Drug-like molecules (QED >= 0.5): {drug_like_count} ({drug_like_count/len(df)*100:.1f}%)")
    print(f"Lipinski-compliant molecules: {lipinski_compliant} ({lipinski_compliant/len(df)*100:.1f}%)")
    print(f"Mean QED score: {df['QED'].mean():.3f} +/- {df['QED'].std():.3f}")
    print(f"Mean LogP: {df['LogP'].mean():.2f} +/- {df['LogP'].std():.2f}")
    print(f"Mean Molecular Weight: {df['Molecular_Weight'].mean():.1f} +/- {df['Molecular_Weight'].std():.1f} Da")
    print(f"Mean TPSA: {df['TPSA'].mean():.1f} +/- {df['TPSA'].std():.1f} A^2")
    
    # Property ranges
    print(f"\nProperty Ranges:")
    print(f"QED range: {df['QED'].min():.3f} - {df['QED'].max():.3f}")
    print(f"LogP range: {df['LogP'].min():.2f} - {df['LogP'].max():.2f}")
    print(f"MW range: {df['Molecular_Weight'].min():.1f} - {df['Molecular_Weight'].max():.1f} Da")
    print(f"TPSA range: {df['TPSA'].min():.1f} - {df['TPSA'].max():.1f} A^2")
    
    # Save detailed statistics to file with UTF-8 encoding
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("MOLECULAR PROPERTY SUMMARY STATISTICS\n")
        f.write("="*60 + "\n\n")
        f.write("Basic Statistics:\n")
        f.write(stats_summary.round(3).to_string())
        f.write(f"\n\nDrug-likeness Analysis:\n")
        f.write(f"Total molecules analyzed: {len(df)}\n")
        f.write(f"Drug-like molecules (QED >= 0.5): {drug_like_count} ({drug_like_count/len(df)*100:.1f}%)\n")
        f.write(f"Lipinski-compliant molecules: {lipinski_compliant} ({lipinski_compliant/len(df)*100:.1f}%)\n")
        f.write(f"Mean QED score: {df['QED'].mean():.3f} +/- {df['QED'].std():.3f}\n")
        f.write(f"Mean LogP: {df['LogP'].mean():.2f} +/- {df['LogP'].std():.2f}\n")
        f.write(f"Mean Molecular Weight: {df['Molecular_Weight'].mean():.1f} +/- {df['Molecular_Weight'].std():.1f} Da\n")
        f.write(f"Mean TPSA: {df['TPSA'].mean():.1f} +/- {df['TPSA'].std():.1f} A^2\n")
        
        f.write(f"\nProperty Ranges:\n")
        f.write(f"QED range: {df['QED'].min():.3f} - {df['QED'].max():.3f}\n")
        f.write(f"LogP range: {df['LogP'].min():.2f} - {df['LogP'].max():.2f}\n")
        f.write(f"MW range: {df['Molecular_Weight'].min():.1f} - {df['Molecular_Weight'].max():.1f} Da\n")
        f.write(f"TPSA range: {df['TPSA'].min():.1f} - {df['TPSA'].max():.1f} A^2\n")
    
    # Save the DataFrame
    df.to_csv(os.path.join(output_dir, 'molecular_properties.csv'), index=False)
    print(f"\nResults saved to {output_dir}/")

def process_csv_file(csv_file_path, model_path, num_molecules=100, affinity_value=0.7, 
                    output_dir="evaluation_results", embed_dim=192, hidden_dim=384, 
                    output_dim=384, num_layers=4, latent_dim=64):
    """
    Process a CSV file containing target sequences and generate molecules for all.
    
    Expected CSV format:
    - Must have column: 'target_seq'
    - Total molecules generated = num_rows * num_molecules
    """
    
    print("="*60)
    print("PROCESSING CSV FILE - GENERATING MOLECULES FOR ALL TARGETS")
    print("="*60)
    
    # Read the CSV file
    try:
        targets_df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV with {len(targets_df)} target sequences")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None
    
    # Validate required columns
    if 'target_seq' not in targets_df.columns:
        print(f"Error: CSV must have 'target_seq' column")
        print(f"Available columns: {list(targets_df.columns)}")
        return None
    
    # Load the model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model, protein_encoder, vae_encoder, vocab_data = load_model(
            model_path, device, use_affinity=True, 
            embed_dim=embed_dim, hidden_dim=hidden_dim, 
            output_dim=output_dim, num_layers=num_layers, latent_dim=latent_dim
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Generate molecules for all target sequences
    all_molecules = []
    
    print(f"\nGenerating {num_molecules} molecules for each of {len(targets_df)} targets...")
    print(f"Total molecules to generate: {len(targets_df) * num_molecules}")
    
    for idx, row in targets_df.iterrows():
        sequence = row['target_seq']
        
        print(f"\nProcessing target {idx+1}/{len(targets_df)} (sequence length: {len(sequence)})")
        
        try:
            # Generate molecules for this target
            generated_molecules = generate_molecules(
                model, protein_encoder, vae_encoder, sequence, vocab_data,
                affinity_value=affinity_value, num_molecules=num_molecules,
                device=device, temperature=0.8, max_attempts=5, latent_noise=0.3
            )
            
            if len(generated_molecules) == 0:
                print(f"No molecules generated for target {idx+1}")
                continue
            
            print(f"Generated {len(generated_molecules)} molecules for target {idx+1}")
            all_molecules.extend(generated_molecules)
            
        except Exception as e:
            print(f"Error processing target {idx+1}: {str(e)}")
            continue
    
    if not all_molecules:
        print("No molecules were generated from any target!")
        return None
    
    print(f"\nTotal molecules generated: {len(all_molecules)}")
    
    # Calculate properties for all molecules
    df = calculate_molecular_properties(all_molecules)
    
    if len(df) == 0:
        print("No valid molecules for property calculation!")
        return None
    
    # Create the 4 main plots
    plot_property_distributions(df, output_dir)
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    
    return df

def evaluate_single_target(model_path, target_sequence, num_molecules=100, 
                          affinity_value=0.7, output_dir="evaluation_results",
                          embed_dim=192, hidden_dim=384, output_dim=384, 
                          num_layers=4, latent_dim=64):
    """
    Evaluate a single target sequence (original functionality).
    """
    print("="*60)
    print("VAE MOLECULE GENERATION AND EVALUATION - SINGLE TARGET")
    print("="*60)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, protein_encoder, vae_encoder, vocab_data = load_model(
        model_path, device, use_affinity=True, 
        embed_dim=embed_dim, hidden_dim=hidden_dim, 
        output_dim=output_dim, num_layers=num_layers, latent_dim=latent_dim
    )
    
    # Generate molecules
    print(f"\nGenerating {num_molecules} molecules for target protein...")
    generated_molecules = generate_molecules(
        model, protein_encoder, vae_encoder, target_sequence, vocab_data,
        affinity_value=affinity_value, num_molecules=num_molecules,
        device=device, temperature=0.8, max_attempts=5, latent_noise=0.3
    )
    
    if len(generated_molecules) == 0:
        print("No molecules were generated! Check your model and parameters.")
        return None
    
    print(f"Successfully generated {len(generated_molecules)} molecules")
    
    # Calculate molecular properties
    df = calculate_molecular_properties(generated_molecules)
    
    if len(df) == 0:
        print("No valid molecules for property calculation!")
        return None
    
    # Create visualizations
    plot_property_distributions(df, output_dir)
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate molecules generated by VAE model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained VAE model checkpoint')
    
    # Make these mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv_file', type=str,
                           help='Path to CSV file containing target sequences (column: target_seq)')
    input_group.add_argument('--target_sequence', type=str,
                           help='Single target protein sequence (or path to file containing sequence)')
    
    parser.add_argument('--num_molecules', type=int, default=100,
                        help='Number of molecules to generate per target')
    parser.add_argument('--affinity', type=float, default=0.7,
                        help='Target binding affinity (0-1)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    
    # Model architecture parameters
    parser.add_argument('--embed_dim', type=int, default=192,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=384,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=384,
                        help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.csv_file:
        # Process CSV file with multiple targets
        print(f"Processing CSV file: {args.csv_file}")
        df = process_csv_file(
            csv_file_path=args.csv_file,
            model_path=args.model_path,
            num_molecules=args.num_molecules,
            affinity_value=args.affinity,
            output_dir=args.output_dir,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            latent_dim=args.latent_dim
        )
        
    else:
        # Process single target (original functionality)
        # Check if target_sequence is a file path
        if os.path.isfile(args.target_sequence):
            with open(args.target_sequence, 'r') as f:
                target_sequence = f.read().strip()
        else:
            target_sequence = args.target_sequence
        
        df = evaluate_single_target(
            model_path=args.model_path,
            target_sequence=target_sequence,
            num_molecules=args.num_molecules,
            affinity_value=args.affinity,
            output_dir=args.output_dir,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            latent_dim=args.latent_dim
        )
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    if df is not None:
        print(f"Successfully evaluated {len(df)} molecules")
        print(f"Average QED score: {df['QED'].mean():.3f}")
        print(f"Drug-like molecules (QED >= 0.5): {len(df[df['QED'] >= 0.5])}/{len(df)} ({len(df[df['QED'] >= 0.5])/len(df)*100:.1f}%)")
        print(f"Lipinski-compliant molecules: {len(df[df['Lipinski_Violations'] == 0])}/{len(df)} ({len(df[df['Lipinski_Violations'] == 0])/len(df)*100:.1f}%)")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Evaluation failed")