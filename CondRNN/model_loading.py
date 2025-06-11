"""
Molecule Generation Model Loader and Evaluator

This module provides utilities for loading trained conditional RNN models
for protein-targeted molecule generation, as well as functions for generating
molecules and evaluating model performance.

The module includes:
1. Functions to load pre-trained models from checkpoints
2. Molecule generation capabilities for specific protein targets
3. Comprehensive evaluation metrics for assessing model quality
4. Drug-likeness assessment for generated molecules

Example usage:
    # Generate molecules for a protein target
    python model_loading.py --target "PROTEIN_SEQUENCE" --model_path ./models_550k --n_molecules 10
    
    # Evaluate model performance
    python model_loading.py --target dummy --evaluate
"""

import torch
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from .conditionalRNN import ProteinEncoder, ConditionalRNNGenerator, generate_molecules

def load_model(model_path, device):
    """
    Load a trained conditional RNN molecule generation model from a checkpoint file.
    
    This function loads both the protein encoder and molecule generator components
    of the conditional RNN model, reconstructing them with the same architecture
    parameters used during training.
    
    Args:
        model_path (str): Path to the model checkpoint file or directory containing 
                         'best_model.pt'.
        device (torch.device): Device to load the model onto (CPU or CUDA).
    
    Returns:
        tuple: A tuple containing:
            - model (ConditionalRNNGenerator): The loaded molecule generator model
            - protein_encoder (ProteinEncoder): The loaded protein encoder model
            - vocab_data (dict): Dictionary containing vocabulary mappings for 
                                SMILES and protein tokens

    IMPORTANT: Match these parameters with your training parameters
        embed_dim, hidden_dim, output_dim, num_layers 
    
    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If there's an issue loading the model state dictionaries.
    """
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "best_model.pt")
        print(f"Model path is a directory, looking for model at: {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab_data = checkpoint['vocab_data']
    protein_vocab_size = len(vocab_data['protein_char_to_idx'])
    smiles_vocab_size = len(vocab_data['smiles_char_to_idx'])
    
    # IMPORTANT: Match these parameters with your training parameters
    embed_dim = 64     
    hidden_dim = 256   
    output_dim = 256   
    num_layers = 2     
    
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
    
    protein_encoder.load_state_dict(checkpoint['protein_encoder_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    protein_encoder.to(device)
    model.to(device)
    protein_encoder.eval()
    model.eval()
    
    print(f"Model loaded successfully (from epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model, protein_encoder, vocab_data


def generate_for_target(model_path, target_sequence_or_file, affinity=0.7, n_molecules=10, output_folder="generated"):
    """
    Generate molecules for a specific target protein sequence.
    
    This function loads the model and generates molecules optimized for binding to
    the specified protein target with the given affinity. Generated molecules are
    saved to a CSV file.
    
    Args:
        model_path (str): Path to the model checkpoint file.
        target_sequence_or_file (str): Protein sequence as a string or path to a file
                                      containing the sequence.
        affinity (float, optional): Target binding affinity on a scale of 0-1. 
                                  Default: 0.7.
        n_molecules (int, optional): Number of molecules to generate. Default: 10.
        output_folder (str, optional): Directory to save generated molecules.
                                      Default: "generated".
    
    Returns:
        list: A list of SMILES strings representing the generated molecules.
    
    Notes:
        - Higher affinity values typically produce molecules with stronger predicted binding.
        - The function handles both direct sequence input and sequence files.
        - Only valid molecules (according to RDKit) are included in the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, protein_encoder, vocab_data = load_model(model_path, device)

    if os.path.isfile(target_sequence_or_file):
        with open(target_sequence_or_file, 'r') as f:
            target_sequence = f.read().strip()
    else:
        target_sequence = target_sequence_or_file
    
    print(f"Target sequence length: {len(target_sequence)}")
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
    
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, "generated_molecules.csv")
    pd.DataFrame({"SMILES": molecules}).to_csv(output_file, index=False)
    print(f"Generated {len(molecules)} molecules, saved to {output_file}")
    
    return molecules


def evaluate_generation_quality(model, protein_encoder, vocab_data, test_proteins=None, n_molecules=100, device='cuda'):
    """
    Evaluate model performance metrics on molecule generation.
    
    This function assesses the quality of molecules generated by the model for a set
    of test proteins. It evaluates validity rate, uniqueness, molecular properties,
    and other quality metrics.
    
    Args:
        model (ConditionalRNNGenerator): The molecule generator model.
        protein_encoder (ProteinEncoder): The protein encoder model.
        vocab_data (dict): Dictionary containing vocabulary mappings for tokens.
        test_proteins (list, optional): List of protein sequences to test. If None,
                                      a default set of test proteins is used.
        n_molecules (int, optional): Total number of molecules to generate per protein.
                                    Default: 100.
        device (str, optional): Device to run evaluation on ('cuda' or 'cpu').
                              Default: 'cuda'.
    
    Returns:
        dict: A dictionary containing detailed evaluation metrics:
            - total_generated: Total number of generated molecules
            - valid_molecules: Number of valid molecules
            - unique_molecules: Number of unique molecules
            - validity_rate: Proportion of valid molecules
            - uniqueness_rate: Proportion of unique molecules
            - avg_mol_weight: Average molecular weight
            - avg_logp: Average LogP value
            - failed_proteins: Count of proteins that failed to generate molecules
            - per_protein_results: List of dictionaries with per-protein metrics
    
    Notes:
        - Multiple affinity values (0.3, 0.5, 0.7, 0.9) are used for each protein
        - Metrics include validity rate, uniqueness, and basic molecular properties
    """
    if test_proteins is None:
        test_proteins = [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",  # Hemoglobin
            "MDKNELVQKAKLAEQAERYDDMAACMKSVTEQGAELSNEERNLLSVAYKNVVGARRSSWRVVSSIEQKTEGAEKKQQMAREYREKIETELRDICNDVLSLLEKFLIPNASQAESKVFYLKMKGDYYRYLAEVAAGDDKKGIVDQSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEILNSPEKACSLAKTAFDEAIAELDTLSEESYKDSTLIMQLLRDNLTLWTSDTQGDEAEAGEGGEN",  # 14-3-3 protein
            "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # GB1 domain
        ]
    
    results = {
        "total_generated": 0,
        "valid_molecules": 0,
        "unique_molecules": 0,
        "validity_rate": 0,
        "uniqueness_rate": 0,
        "avg_mol_weight": 0,
        "avg_logp": 0,
        "failed_proteins": 0,
        "per_protein_results": []
    }
    
    all_molecules = []
    
    for i, protein in enumerate(test_proteins):
        print(f"Evaluating protein {i+1}/{len(test_proteins)}: length {len(protein)}")
        
        # Try with different affinities
        affinities = [0.3, 0.5, 0.7, 0.9]
        protein_molecules = []
        
        for affinity in affinities:
            molecules = generate_molecules(
                model,
                protein_encoder,
                protein,
                vocab_data,
                affinity_value=affinity,
                num_molecules=n_molecules // len(affinities),  
                device=device,
                temperature=0.7,
                max_attempts=5
            )
            protein_molecules.extend(molecules)
        
        if len(protein_molecules) == 0:
            results["failed_proteins"] += 1
            continue
            
        valid_mols = [Chem.MolFromSmiles(s) for s in protein_molecules]
        valid_mols = [m for m in valid_mols if m is not None]
        
        mol_weights = [Descriptors.MolWt(m) for m in valid_mols if m is not None]
        logp_values = [Descriptors.MolLogP(m) for m in valid_mols if m is not None]
        
        protein_results = {
            "protein_length": len(protein),
            "generated": len(protein_molecules),
            "valid": len(valid_mols),
            "unique": len(set(protein_molecules)),
            "validity_rate": len(valid_mols) / max(1, len(protein_molecules)),
            "uniqueness_rate": len(set(protein_molecules)) / max(1, len(protein_molecules)),
            "avg_mol_weight": sum(mol_weights) / max(1, len(mol_weights)) if mol_weights else 0,
            "avg_logp": sum(logp_values) / max(1, len(logp_values)) if logp_values else 0
        }
        
        results["per_protein_results"].append(protein_results)
        all_molecules.extend(protein_molecules)
    
    results["total_generated"] = len(all_molecules)
    
    if results["total_generated"] > 0:
        valid_mols = [Chem.MolFromSmiles(s) for s in all_molecules]
        valid_mols = [m for m in valid_mols if m is not None]
        
        results["valid_molecules"] = len(valid_mols)
        results["unique_molecules"] = len(set(all_molecules))
        results["validity_rate"] = results["valid_molecules"] / results["total_generated"]
        results["uniqueness_rate"] = results["unique_molecules"] / results["total_generated"]
        
        mol_weights = [Chem.Descriptors.MolWt(m) for m in valid_mols if m is not None]
        logp_values = [Chem.Descriptors.MolLogP(m) for m in valid_mols if m is not None]
        
        if mol_weights:
            results["avg_mol_weight"] = sum(mol_weights) / len(mol_weights)
        if logp_values:
            results["avg_logp"] = sum(logp_values) / len(logp_values)
    
    return results

def evaluate_druglikeness(molecules):
    """
    Evaluate the drug-like properties of generated molecules.
    
    This function analyzes a set of molecules and calculates various medicinal
    chemistry metrics to assess their potential as drug candidates, including
    Lipinski's Rule of Five compliance, QED scores, and property distributions.
    
    Args:
        molecules (list): List of SMILES strings representing molecules to evaluate.
    
    Returns:
        dict: A dictionary containing drug-likeness metrics:
            - qed_scores: List of QED scores for each molecule
            - lipinski_pass: Number of molecules passing Lipinski's Rule of Five
            - avg_qed: Average QED score across all molecules
            - lipinski_pass_rate: Proportion of molecules passing Lipinski's rules
            - sa_scores: List of synthetic accessibility scores (if available)
            - avg_sa: Average synthetic accessibility score
            - mw_distribution: Distribution of molecular weights in percentage
            - logp_distribution: Distribution of LogP values in percentage
    
    Notes:
        - Lipinski's Rule of Five criteria: MW<500, LogP<5, H-donors<5, H-acceptors<10
        - QED (Quantitative Estimate of Drug-likeness) ranges from 0-1, with higher
          values indicating more drug-like molecules
        - One violation of Lipinski's rules is still considered acceptable
    """
    from rdkit.Chem import Descriptors, Lipinski, QED
    
    results = {
        "qed_scores": [],
        "lipinski_pass": 0,
        "avg_qed": 0,
        "lipinski_pass_rate": 0,
        "sa_scores": [],
        "avg_sa": 0,
        "mw_distribution": {
            "<250": 0,
            "250-350": 0,
            "350-500": 0,
            ">500": 0
        },
        "logp_distribution": {
            "<1": 0,
            "1-3": 0,
            "3-5": 0,
            ">5": 0
        }
    }
    
    try:
        # Import SA score calculation (uncomment if available)
        # from rdkit.Contrib.SA_Score import sascorer
        sa_available = False
    except ImportError:
        sa_available = False
    
    valid_count = 0
    
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        valid_count += 1
        
        # Calculate QED (quantitative estimate of drug-likeness)
        try:
            qed = QED.qed(mol)
            results["qed_scores"].append(qed)
        except:
            pass
        
        # Check Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if h_donors > 5: violations += 1
        if h_acceptors > 10: violations += 1
        
        if violations <= 1:  
            results["lipinski_pass"] += 1
        
        # Calculate SA Score if available
        if sa_available:
            try:
                sa = sascorer.calculateScore(mol)
                results["sa_scores"].append(sa)
            except:
                pass
        
        if mw < 250:
            results["mw_distribution"]["<250"] += 1
        elif mw < 350:
            results["mw_distribution"]["250-350"] += 1
        elif mw < 500:
            results["mw_distribution"]["350-500"] += 1
        else:
            results["mw_distribution"][">500"] += 1
            
        if logp < 1:
            results["logp_distribution"]["<1"] += 1
        elif logp < 3:
            results["logp_distribution"]["1-3"] += 1
        elif logp < 5:
            results["logp_distribution"]["3-5"] += 1
        else:
            results["logp_distribution"][">5"] += 1
    
    if results["qed_scores"]:
        results["avg_qed"] = sum(results["qed_scores"]) / len(results["qed_scores"])
    
    if valid_count > 0:
        results["lipinski_pass_rate"] = results["lipinski_pass"] / valid_count
    
    if results["sa_scores"]:
        results["avg_sa"] = sum(results["sa_scores"]) / len(results["sa_scores"])
    
    for key in results["mw_distribution"]:
        if valid_count > 0:
            results["mw_distribution"][key] = results["mw_distribution"][key] * 100 / valid_count
            
    for key in results["logp_distribution"]:
        if valid_count > 0:
            results["logp_distribution"][key] = results["logp_distribution"][key] * 100 / valid_count
    
    return results


if __name__ == "__main__":
    """
        Main entry point for the model loading and evaluation script.
        
        This script provides command-line functionality for:
        1. Loading trained conditional RNN models
        2. Generating molecules for specific protein targets
        3. Evaluating model performance on test proteins
        4. Analyzing drug-likeness of generated molecules
        
        Command line arguments:
            --model_path: Path to the trained model checkpoint
            --target: Target protein sequence or path to file with sequence
            --affinity: Target binding affinity value (0-1 scale)
            --n_molecules: Number of molecules to generate
            --output_folder: Folder to save results
            --evaluate: Flag to run model evaluation instead of generation
        """

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
    
    parser.add_argument('--evaluate', action='store_true',
                      help='Run evaluation instead of generation')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, protein_encoder, vocab_data = load_model(args.model_path, device)
    
    if args.evaluate:
        print("Evaluating model performance...")
        results = evaluate_generation_quality(
            model, 
            protein_encoder, 
            vocab_data, 
            n_molecules=100,
            device=device
        )
        
        print("\n=== Generation Quality Results ===")
        print(f"Validity rate: {results['validity_rate']:.2%}")
        print(f"Uniqueness rate: {results['uniqueness_rate']:.2%}")
        print(f"Average molecular weight: {results['avg_mol_weight']:.2f}")
        print(f"Average LogP: {results['avg_logp']:.2f}")
        print(f"Failed proteins: {results['failed_proteins']}")
        
        print("\nGenerating molecules for drug-likeness evaluation...")
        all_mols = []
        for protein_idx in range(min(3, len(results['per_protein_results']))):
            protein = results["per_protein_results"][protein_idx]
            print(f"  Protein {protein_idx+1}: Generated {protein['generated']} molecules, "
                  f"validity rate: {protein['validity_rate']:.2%}")
            
            if protein_idx == 0 and len(all_mols) < 100:
                test_protein = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
                additional_mols = generate_molecules(
                    model,
                    protein_encoder,
                    test_protein,
                    vocab_data,
                    affinity_value=0.7,
                    num_molecules=100,
                    device=device,
                    temperature=0.7,
                    max_attempts=5
                )
                all_mols.extend(additional_mols)
        
        if all_mols:
            drug_results = evaluate_druglikeness(all_mols)
            
            print("\n=== Drug-likeness Evaluation ===")
            print(f"Average QED score: {drug_results['avg_qed']:.3f}")
            print(f"Lipinski Rule of Five pass rate: {drug_results['lipinski_pass_rate']:.2%}")
            
            print("\nMolecular Weight Distribution:")
            for key, value in drug_results['mw_distribution'].items():
                print(f"  {key}: {value:.1f}%")
                
            print("\nLogP Distribution:")
            for key, value in drug_results['logp_distribution'].items():
                print(f"  {key}: {value:.1f}%")
        
        os.makedirs(args.output_folder, exist_ok=True)
        output_file = os.path.join(args.output_folder, "evaluation_results.json")
        with open(output_file, 'w') as f:
            import json
            combined_results = {
                "generation_quality": results,
                "drug_likeness": drug_results if all_mols else {}
            }
            json.dump(combined_results, f, indent=2)
        
        print(f"\nEvaluation results saved to {output_file}")
        
    elif args.target:
        generate_for_target(
            args.model_path,
            args.target,
            args.affinity,
            args.n_molecules,
            args.output_folder
        )
    else:
        print("Error: You must either specify --evaluate or provide a --target sequence")
