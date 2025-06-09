from rdkit import Chem
import numpy as np
import pandas as pd
import os
import time

def preprocess_smiles(smiles_list, canonical=True, show_progress=True):
    """
    Process a list of SMILES strings and return valid ones
    
    Args:
        smiles_list: List of SMILES strings to process
        canonical: Whether to convert to canonical SMILES
        show_progress: Whether to show progress updates
    
    Returns:
        List of valid SMILES strings
    """
    clean_smiles = []
    invalid_count = 0
    total = len(smiles_list)
    
    start_time = time.time()
    
    if show_progress:
        print(f"Processing {total} SMILES strings...")
    
    # Process in batches of 5000 for progress reporting
    batch_size = 5000
    for i in range(0, total, batch_size):
        batch = smiles_list[i:i+batch_size]
        
        for smi in batch:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                if canonical:
                    clean_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                else:
                    clean_smiles.append(smi)
            else:
                invalid_count += 1
        
        if show_progress and i > 0:
            progress = min(i + batch_size, total) / total * 100
            elapsed = time.time() - start_time
            est_total = elapsed / progress * 100
            remaining = est_total - elapsed
            
            print(f"Progress: {progress:.1f}% ({i+batch_size}/{total}) | "
                  f"Valid: {len(clean_smiles)} | Invalid: {invalid_count} | "
                  f"Time remaining: {remaining:.1f}s")
    
    # Final stats
    if show_progress:
        print(f"Completed in {time.time() - start_time:.1f}s")
        print(f"Total SMILES: {total}")
        print(f"Valid SMILES: {len(clean_smiles)} ({len(clean_smiles)/total*100:.1f}%)")
        print(f"Invalid SMILES: {invalid_count} ({invalid_count/total*100:.1f}%)")
    
    return clean_smiles

def process_smi_file(file_path, output_path=None):
    """Process a .smi file and save valid SMILES"""
    # Read the file
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip() and not line.startswith('//')]
    
    print(f"Loaded {len(smiles_list)} SMILES from {file_path}")
    
    # Process SMILES
    clean_smiles = preprocess_smiles(smiles_list)
    
    # Save results
    if output_path is None:
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}_valid.smi"
    
    with open(output_path, 'w') as f:
        for smi in clean_smiles:
            f.write(f"{smi}\n")
    
    print(f"Saved {len(clean_smiles)} valid SMILES to {output_path}")
    return clean_smiles

process_smi_file('./data/chembl.mini.smi')