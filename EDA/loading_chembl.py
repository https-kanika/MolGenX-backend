import pandas as pd
import requests
import time
from tqdm import tqdm
import concurrent.futures
import datetime
import sys
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Function to print timestamped messages
def log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force output to display immediately

def fetch_bioactivities_direct_api():
    """
    Fetch bioactivity data directly from ChEMBL REST API with proper pagination
    """
    log("Fetching bioactivity data using direct REST API calls...")
    
    # Base URL and query parameters
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity"
    params = {
        'assay_type': 'B',           # Binding assays 
        'relation': '=',             # Exact measurements
        'pchembl_value__isnull': 'false',  # Must have pChEMBL value
        'standard_units': 'nM',      # Standardized units
        'limit': 1000,               # Number of records per page
        'offset': 0                  # Starting offset
    }
    
    # Add browser-like headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Connection': 'keep-alive'
    }
    
    all_activities = []
    total_count = None
    retry_count = 0
    max_retries = 5
    
    while True:
        log(f"Fetching records {params['offset']+1}-{params['offset']+params['limit']}...")
        start_time = time.time()
        
        try:
            # Print the full URL for debugging
            full_url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
            log(f"API URL: {full_url}")
            
            response = requests.get(base_url, params=params, headers=headers, timeout=60)
            
            # Debug response details
            log(f"Response status: {response.status_code}")
            log(f"Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                log(f"API error: HTTP {response.status_code}")
                # Show response content for debugging
                log(f"Response content (first 500 chars): {response.text[:500]}")
                
                if response.status_code == 429:  # Too Many Requests
                    retry_delay = 60 * (retry_count + 1)  # Exponential backoff
                    log(f"Rate limit hit, waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1
                    continue
                elif response.status_code >= 500:  # Server error
                    retry_delay = 30 * (retry_count + 1)
                    log(f"Server error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1
                    if retry_count < max_retries:
                        continue
                break
                
            # Try a different approach - first check if response is valid
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                log(f"Warning: Response is not JSON (Content-Type: {content_type})")
                log(f"Response preview: {response.text[:500]}")
                
                # Adaptive handling - try to force JSON format
                params['format'] = 'json'
                time.sleep(10)
                continue
                
            # Parse JSON with better error handling
            try:
                data = response.json()
            except ValueError as e:
                log(f"JSON parsing error: {e}")
                log(f"Response content (first 500 chars): {response.text[:500]}")
                
                # Try to use direct API URL format - might be HTML response
                direct_url = f"{base_url}.json"
                log(f"Trying alternative URL format: {direct_url}")
                alt_response = requests.get(direct_url, params=params, headers=headers)
                
                if alt_response.status_code == 200:
                    try:
                        data = alt_response.json()
                        log("Alternative URL successful")
                    except:
                        retry_count += 1
                        if retry_count < max_retries:
                            log(f"Retrying ({retry_count}/{max_retries})...")
                            time.sleep(10 * retry_count)
                            continue
                        else:
                            log("Max retries reached. Aborting.")
                            break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        log(f"Retrying ({retry_count}/{max_retries})...")
                        time.sleep(10 * retry_count)
                        continue
                    else:
                        log("Max retries reached. Aborting.")
                        break
            
            # Reset retry counter on success
            retry_count = 0
            
            # Get total count if we don't have it
            if total_count is None and 'page_meta' in data:
                total_count = data['page_meta']['total_count']
                log(f"Total bioactivity records to retrieve: {total_count:,}")
            
            activities = data.get('activities', [])
            
            if not activities:
                log("No more records, data retrieval complete")
                break
                
            all_activities.extend(activities)
            
            elapsed = time.time() - start_time
            log(f"Retrieved {len(activities)} records in {elapsed:.2f}s (total: {len(all_activities):,})")
            
            # Update offset for next page
            params['offset'] += params['limit']
            
            # Print progress
            if total_count:
                percent_done = min(100, len(all_activities) * 100 / total_count)
                log(f"Progress: {percent_done:.1f}% ({len(all_activities):,}/{total_count:,})")
            
            # If we've reached the end
            if len(activities) < params['limit']:
                log("Last page reached (incomplete page)")
                break
                
        except requests.exceptions.RequestException as e:
            log(f"Network error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                retry_delay = 10 * retry_count
                log(f"Retrying in {retry_delay} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                log("Max retries reached. Aborting.")
                break
        except Exception as e:
            log(f"Error fetching bioactivities: {e}")
            retry_count += 1
            if retry_count < max_retries:
                retry_delay = 10 * retry_count
                log(f"Waiting {retry_delay} seconds before retrying... (attempt {retry_count}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                log("Max retries reached. Aborting.")
                break
    
    # Convert to DataFrame
    log(f"Converting {len(all_activities):,} bioactivities to DataFrame...")
    
    if not all_activities:
        log("Warning: No activities retrieved. Using alternative method...")
        # Try using alternative approach
        # [Keep existing fallback approaches here]
        return pd.DataFrame()
    
    # Process activity data (keep the rest of your function as is)
    processed_activities = []
    for activity in all_activities:
        processed_activities.append({
            'molecule_chembl_id': activity.get('molecule_chembl_id'),
            'target_chembl_id': activity.get('target_chembl_id'),
            'target_pref_name': activity.get('target_pref_name'),
            'target_organism': activity.get('target_organism'),
            'standard_type': activity.get('standard_type'),
            'standard_value': activity.get('standard_value'),
            'standard_units': activity.get('standard_units'),
            'pchembl_value': activity.get('pchembl_value'),
            'assay_chembl_id': activity.get('assay_chembl_id'),
            'document_chembl_id': activity.get('document_chembl_id')
        })
    
    bioact_df = pd.DataFrame(processed_activities)
    log(f"DataFrame created with {len(bioact_df):,} records")
    
    return bioact_df

def fetch_compound_batch(batch_ids):
    """Fetch a single batch of compound data directly from REST API"""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    compound_data = []
    
    # Convert list to comma-separated string for the 'molecule_chembl_id__in' parameter
    ids_param = ",".join(batch_ids)
    
    try:
        # Construct URL with query parameters
        params = {
            'molecule_chembl_id__in': ids_param,
            'format': 'json'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            molecules = data.get('molecules', [])
            
            for mol in molecules:
                smiles = None
                structures = mol.get('molecule_structures', {})
                if structures:
                    smiles = structures.get('canonical_smiles')
                
                compound_data.append({
                    'molecule_chembl_id': mol.get('molecule_chembl_id'),
                    'compound_name': mol.get('pref_name'),
                    'canonical_smiles': smiles
                })
        else:
            print(f"Error fetching compounds: HTTP {response.status_code}")
            time.sleep(2)  # Back off a bit on error
            
    except Exception as e:
        print(f"Error in fetch_compound_batch: {e}")
        time.sleep(2)
        
    return compound_data

def fetch_compounds_parallel(compound_ids, batch_size=100, max_workers=16):
    """Fetch compound data in parallel using direct REST API calls"""
    log(f"Fetching data for {len(compound_ids):,} compounds in parallel...")
    
    # Split into batches - smaller batch size for REST API to avoid URL length limits
    batches = [compound_ids[i:i+batch_size] for i in range(0, len(compound_ids), batch_size)]
    log(f"Created {len(batches)} batches with size {batch_size}")
    
    all_compounds = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(fetch_compound_batch, batch): i 
                          for i, batch in enumerate(batches)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(batches),
                          desc="Fetching compounds"):
            batch_idx = future_to_batch[future]
            try:
                batch_data = future.result()
                all_compounds.extend(batch_data)
                
                # Log progress periodically
                if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                    log(f"Progress: {batch_idx+1}/{len(batches)} batches, {len(all_compounds):,} compounds")
                    
            except Exception as e:
                log(f"Error processing batch {batch_idx}: {e}")
    
    return all_compounds

def fetch_target_batch(batch_ids):
    """Fetch a single batch of target data directly from REST API"""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/target"
    target_data = []
    
    # Convert list to comma-separated string for the 'target_chembl_id__in' parameter
    ids_param = ",".join(batch_ids)
    
    try:
        # Construct URL with query parameters
        params = {
            'target_chembl_id__in': ids_param,
            'format': 'json'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            targets = data.get('targets', [])
            
            for target in targets:
                # Extract UniProt accession if available
                uniprot_id = None
                components = target.get('target_components', [])
                if components:
                    for component in components:
                        if component.get('accession'):
                            uniprot_id = component['accession']
                            break
                
                target_data.append({
                    'target_chembl_id': target.get('target_chembl_id'),
                    'target_name': target.get('pref_name'),
                    'target_type': target.get('target_type'),
                    'organism': target.get('organism'),
                    'uniprot_id': uniprot_id
                })
        else:
            print(f"Error fetching targets: HTTP {response.status_code}")
            time.sleep(2)  # Back off a bit on error
            
    except Exception as e:
        print(f"Error in fetch_target_batch: {e}")
        time.sleep(2)
        
    return target_data

def fetch_targets_parallel(target_ids, batch_size=100, max_workers=16):
    """Fetch target data in parallel using direct REST API calls"""
    log(f"Fetching data for {len(target_ids):,} targets in parallel...")
    
    # Split into batches - smaller batch size for REST API to avoid URL length limits
    batches = [target_ids[i:i+batch_size] for i in range(0, len(target_ids), batch_size)]
    log(f"Created {len(batches)} batches with size {batch_size}")
    
    all_targets = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(fetch_target_batch, batch): i 
                          for i, batch in enumerate(batches)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(batches),
                          desc="Fetching targets"):
            batch_idx = future_to_batch[future]
            try:
                batch_data = future.result()
                all_targets.extend(batch_data)
                
                # Log progress periodically
                if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
                    log(f"Progress: {batch_idx+1}/{len(batches)} batches, {len(all_targets):,} targets")
                    
            except Exception as e:
                log(f"Error processing batch {batch_idx}: {e}")
    
    return all_targets

def process_smiles_batch(smiles_batch):
    """Process a batch of SMILES to calculate molecular properties"""
    results = []
    
    for idx, smiles in enumerate(smiles_batch):
        if pd.isna(smiles):
            continue
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Calculate properties
                result = {
                    'smiles': smiles,
                    'qed': QED.qed(mol),
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logP': Descriptors.MolLogP(mol),
                    'h_donors': Descriptors.NumHDonors(mol),
                    'h_acceptors': Descriptors.NumHAcceptors(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
                }
                results.append(result)
        except Exception as e:
            print(f"Error processing SMILES {idx}: {e}")
    
    return results

def calculate_molecular_properties(df, batch_size=5000, max_workers=16):
    """Calculate molecular properties with GPU-optimized parallel processing"""
    log("Calculating molecular properties using NVIDIA L4 GPU acceleration...")
    start_time = time.time()
    
    if 'canonical_smiles' not in df.columns:
        log("Error: No SMILES column found")
        return df
    
    # Get all valid SMILES
    valid_smiles = df['canonical_smiles'].dropna().tolist()
    log(f"Processing {len(valid_smiles):,} valid SMILES structures")
    
    # Split into batches for parallel processing
    batches = [valid_smiles[i:i+batch_size] for i in range(0, len(valid_smiles), batch_size)]
    log(f"Created {len(batches)} batches for GPU processing")
    
    all_results = []
    
    # Process in parallel to leverage GPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_smiles_batch, batch) for batch in batches]
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                     total=len(futures), 
                                     desc="Processing SMILES")):
            batch_results = future.result()
            all_results.extend(batch_results)
            
            # Log progress periodically
            if (i + 1) % 5 == 0 or i == len(batches) - 1:
                elapsed = time.time() - start_time
                properties_per_sec = len(all_results) / elapsed if elapsed > 0 else 0
                log(f"Processed {len(all_results):,} compounds ({properties_per_sec:.1f} compounds/sec)")
    
    # Convert results to DataFrame
    log("Converting property results to DataFrame...")
    props_df = pd.DataFrame(all_results)
    
    # Merge with original DataFrame
    log("Merging properties with main dataset...")
    result_df = df.merge(props_df, left_on='canonical_smiles', right_on='smiles', how='left')
    
    # Clean up the merged dataframe
    if 'smiles' in result_df.columns:
        result_df = result_df.drop('smiles', axis=1)
    
    elapsed = time.time() - start_time
    log(f"Property calculation complete in {elapsed:.1f} seconds")
    log(f"Processing speed: {len(valid_smiles)/elapsed:.1f} compounds/second")
    
    return result_df

def save_dataset(df, filename="chembl_protein_compound_pairs.csv"):
    """Save the dataset with compression"""
    log(f"Saving dataset ({df.shape[0]:,} rows, {df.memory_usage().sum() / 1e6:.1f} MB) to {filename}...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)) if os.path.dirname(filename) else '.', exist_ok=True)
    
    try:
        df.to_csv(filename, index=False, compression='gzip')
        log(f"Dataset saved successfully in {time.time() - start_time:.1f} seconds!")
    except Exception as e:
        log(f"Error saving dataset: {e}")
        # Try without compression
        try:
            df.to_csv(filename.replace('.csv', '_uncompressed.csv'), index=False)
            log("Saved uncompressed version as fallback")
        except Exception as e2:
            log(f"Failed to save even uncompressed version: {e2}")

def get_all_protein_compound_pairs():
    """Main function to extract all protein-compound pairs"""
    start_time = time.time()
    log("Starting enhanced ChEMBL data extraction with GPU acceleration...")
    
    # Step 1: Get bioactivities using direct REST API
    bioact_df = fetch_bioactivities_direct_api()
    
    # Step 2: Get unique compounds and targets
    unique_compounds = list(bioact_df['molecule_chembl_id'].unique())
    unique_targets = list(bioact_df['target_chembl_id'].unique())
    
    log(f"Unique compounds identified: {len(unique_compounds):,}")
    log(f"Unique targets identified: {len(unique_targets):,}")
    
    # Step 3 & 4: Fetch compound and target data in parallel
    log("Performing parallel data retrieval for compounds and targets...")
    
    # Use ThreadPoolExecutor to run both fetches concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_compounds = executor.submit(fetch_compounds_parallel, unique_compounds)
        future_targets = executor.submit(fetch_targets_parallel, unique_targets)
        
        log("Waiting for parallel fetching to complete...")
        compound_data = future_compounds.result()
        target_data = future_targets.result()
    
    log(f"Retrieved data for {len(compound_data):,} compounds and {len(target_data):,} targets")
    
    # Step 5: Merge all data
    log("Merging datasets...")
    compounds_df = pd.DataFrame(compound_data)
    targets_df = pd.DataFrame(target_data)
    
    # Merge bioactivities with compounds
    log("Merging bioactivities with compounds...")
    final_df = pd.merge(bioact_df, compounds_df, on='molecule_chembl_id', how='left')
    
    # Merge with targets
    log("Merging with targets...")
    final_df = pd.merge(final_df, targets_df, on='target_chembl_id', how='left')
    
    # Clean up and reorder columns
    desired_columns = [
        'molecule_chembl_id', 'canonical_smiles', 'compound_name',
        'target_chembl_id', 'target_name', 'target_type', 'organism', 'uniprot_id',
        'standard_type', 'standard_value', 'standard_units', 'pchembl_value',
        'assay_chembl_id', 'document_chembl_id'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in desired_columns if col in final_df.columns]
    final_df = final_df[available_columns]
    
    # Remove rows without SMILES
    missing_smiles = final_df['canonical_smiles'].isna().sum()
    log(f"Removing {missing_smiles:,} rows without SMILES...")
    final_df = final_df.dropna(subset=['canonical_smiles'])
    
    log(f"Final dataset shape: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
    log(f"Total data extraction time: {time.time() - start_time:.1f} seconds")
    
    return final_df

# Main execution
if __name__ == "__main__":
    try:
        overall_start = time.time()
        log("=== Enhanced ChEMBL Protein-Compound Pair Extraction ===")
        
        # Execute the main function
        dataset = get_all_protein_compound_pairs()
        
        # Calculate molecular properties using GPU acceleration
        dataset = calculate_molecular_properties(dataset)
        
        # Display summary statistics
        log("\n=== Dataset Summary ===")
        log(f"Total protein-compound pairs: {len(dataset):,}")
        log(f"Unique compounds: {dataset['molecule_chembl_id'].nunique():,}")
        log(f"Unique targets: {dataset['target_chembl_id'].nunique():,}")
        
        # Show affinity types
        affinity_counts = dataset['standard_type'].value_counts().to_dict()
        log("Affinity measurement types:")
        for affinity_type, count in affinity_counts.items():
            log(f"  - {affinity_type}: {count:,}")
        
        # Calculate memory usage
        memory_usage = dataset.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
        log(f"Dataset memory usage: {memory_usage:.2f} MB")
        
        # Show sample data
        log("\n=== Sample Data (first 5 rows) ===")
        print(dataset.head().to_string())
        
        # Calculate data quality metrics
        log("\n=== Data Quality Metrics ===")
        missing_values = dataset.isna().sum()
        log("Missing values per column:")
        for column, count in missing_values.items():
            if count > 0:
                log(f"  - {column}: {count:,} ({count/len(dataset)*100:.1f}%)")
        
        # Calculate molecular property statistics if available
        if 'qed' in dataset.columns:
            log("\n=== Molecular Property Statistics ===")
            log(f"QED (drug-likeness): {dataset['qed'].mean():.3f} mean, {dataset['qed'].median():.3f} median")
            log(f"LogP (lipophilicity): {dataset['logP'].mean():.2f} mean, {dataset['logP'].median():.2f} median")
            log(f"Molecular Weight: {dataset['molecular_weight'].mean():.2f} mean, {dataset['molecular_weight'].median():.2f} median")
        
        # Save the dataset with molecular properties
        output_file = "chembl_protein_compound_pairs_with_properties.csv.gz"
        log("\nSaving final dataset...")
        save_dataset(dataset, output_file)
        
        overall_time = time.time() - overall_start
        hours, remainder = divmod(overall_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        log(f"\n=== Process Complete! ===")
        log(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        log(f"Records processed: {len(dataset):,}")
        log(f"Processing rate: {len(dataset)/overall_time:.1f} records/second")
        
    except Exception as e:
        log(f"Error in main execution: {e}")
        import traceback
        log(traceback.format_exc())