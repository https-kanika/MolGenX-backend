###############################   IMPORTS   ##############################################
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, AllChem, Draw
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple, Optional, Set
import requests
import numpy as np
import re
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

###############################   RNN Molecule Gen   #########################################

def validate_molecule(smiles: str) -> bool:
    """
    Validate if the generated SMILES represents a valid molecule
    
    Args:
        smiles (str): SMILES string to validate
    
    Returns:
        bool: True if molecule is valid, False otherwise
    """
    # Remove the <PAD> token before validation
    smiles = smiles.replace("<PAD>", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and Chem.MolToSmiles(mol) == smiles  #this is a canonicalization check
    except:
        return False

class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNNGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)

def generate_diverse_molecules(
    model: nn.Module, 
    char_to_idx: dict, 
    idx_to_char: dict, 
    device: torch.device, 
    start_token: str = "C", 
    num_molecules: int = 10, 
    max_length: int = 100, 
    max_attempts: int = 100
) -> List[str]:
    """
    Generate multiple unique and valid molecules
    
    Args:
        model (nn.Module): Trained RNN model
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        device (torch.device): Device to run model on
        start_token (str): Starting token for molecule generation
        num_molecules (int): Number of unique molecules to generate
        max_length (int): Maximum length of molecule SMILES
        max_attempts (int): Maximum generation attempts
    
    Returns:
        List[str]: List of unique valid molecules
    """
    model.eval() #switched the model from training to evaluation mode
    unique_molecules: Set[str] = set()
    generation_attempts = 0
    
    while len(unique_molecules) < num_molecules and generation_attempts < max_attempts:
        # Reset sequence with start token
        sequence = [char_to_idx[start_token]]
        
        for _ in range(max_length):
            input_seq = torch.tensor([sequence]).to(device)
            
            # Use temperature-based sampling for diversity
            output_logits = model(input_seq)[:, -1, :]
            temperature = 0.7  # Adjust for more/less randomness
            scaled_logits = output_logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            
            # Sample from probability distribution
            output_idx = torch.multinomial(probabilities.squeeze(), 1).item()
            sequence.append(output_idx)

            # Stop conditions
            if output_idx == char_to_idx["<PAD>"]:
                break

        # Convert to SMILES and validate
        molecule_smiles = "".join(idx_to_char[idx] for idx in sequence)
        # Remove the <PAD> token
        molecule_smiles = molecule_smiles.replace("<PAD>", "")
        if validate_molecule(molecule_smiles) and molecule_smiles not in unique_molecules:
            unique_molecules.add(molecule_smiles)
        
        generation_attempts += 1
    
    return list(unique_molecules)
    
def create_vocabulary(smiles_data):
    """Create character vocabulary from SMILES strings"""
    # Include all possible SMILES characters and special tokens
    chars = set("".join(smiles_data))
    char_to_idx = {char: idx +1 for idx,char in enumerate(sorted(chars))}
    char_to_idx["<PAD>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char


###############################   Drug Optimizer   #########################################

class DrugOptimizer:
    """
    A comprehensive drug candidate optimization pipeline that integrates 
    multiple objectives, ESM-2 for protein interaction, and Google Cloud services.
    """
    
    def __init__(self, candidate_smiles: List[str], target_protein: Optional[str] = None):
        """
        Initialize the DrugOptimizer with a list of candidate SMILES and optionally a target protein.
        
        Args:
            candidate_smiles: List of SMILES strings representing drug candidates
            target_protein: Optional sequence of the target protein
        """
        self.candidates = candidate_smiles
        self.target_protein = target_protein
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in candidate_smiles if Chem.MolFromSmiles(smiles)]
        self.valid_indices = [i for i, mol in enumerate(self.mols) if mol is not None]
        self.valid_smiles = [candidate_smiles[i] for i in self.valid_indices]
        self.metrics_cache = {}
    
        # 1. ESM-2 for protein embeddings (you already have this)
            #self.esm_predictor = Predictor("esm2")
        
        # 2. DiffDock for protein-ligand binding prediction
            # self.diffdock_predictor = Predictor("diffdock")
        
        # 3. MegaMolBART for molecule generation and optimization
            # self.molbart_predictor = Predictor("megamolbart")
        
        # 4. MoLFormer-XL for molecular property prediction
            # self.molformer_predictor = Predictor("molformer")

    def predict_protein_structure(self):
        """Predict structure information for the target protein using ESM-2"""
        if not self.target_protein:
            return None
    
        try:
            print("Loading ESM-2 model...")
            # Use a much smaller model (650M parameters instead of 15B)
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
            # Process only the first 1000 amino acids to avoid memory issues
            protein_seq = self.target_protein[:1000]
            print(f"Processing protein sequence (length: {len(protein_seq)})")
        
            inputs = tokenizer(protein_seq, return_tensors="pt")
        
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
                # Get last hidden state
                embeddings = outputs.hidden_states[-1]
            
                # Mean pool over sequence length (excluding special tokens)
                sequence_embeddings = embeddings[:, 1:-1].mean(dim=1)
            
                print(f"Generated protein embeddings with shape: {sequence_embeddings.shape}")
            
            return sequence_embeddings
    
        except Exception as e:
            print(f"Error in protein structure prediction: {str(e)}")
            return None
    
    def calculate_druglikeness(self, mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness)"""
        return QED.qed(mol) if mol else 0.0

    def calculate_synthetic_accessibility(self,mol):
        """
        A more comprehensive synthetic accessibility estimation
        """
        if not mol:
            return 10.0
        
        # Multiple complexity factors
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = Descriptors.RingCount(mol)
        spiro_count = Descriptors.NumSpiroAtoms(mol)
        bridgehead_count = Descriptors.NumBridgeheadAtoms(mol)
        
        # Complexity calculation
        complexity_score = (
            rotatable_bonds * 0.3 +     # Rotatable bonds add complexity
            ring_count * 0.5 +          # Rings increase synthesis difficulty
            spiro_count * 1.0 +         # Spiro atoms significantly increase complexity
            bridgehead_count * 1.2      # Bridgehead atoms are challenging
        )
        
        return min(max(complexity_score, 1.0), 10.0)
    
    def calculate_lipinski_violations(self, mol) -> int:
        """Check how many Lipinski's Rule of Five violations exist"""
        violations = 0
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if h_donors > 5: violations += 1
            if h_acceptors > 10: violations += 1
        
        return violations
    
    def predict_toxicity(self, mol) -> float:
        """Predict toxicity using IBM's MoLFormer-XL model or fallback to structural alerts"""
        if not mol:
            return 1.0
        
        try:
            # Convert molecule to SMILES first (in case input is a mol object)
            smiles = Chem.MolToSmiles(mol)
            print(f"Analyzing toxicity for SMILES: {smiles}")
            
            # Load model and tokenizer (existing code)
            model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                            deterministic_eval=True, 
                                            trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                                    trust_remote_code=True)
            
            # Generate embeddings
            inputs = tokenizer(smiles, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract the embedding and convert to numpy array
            mol_embedding = outputs.pooler_output.squeeze().numpy()
            
            # Process the embedding to create a toxicity score
            # 1. Calculate simple statistics from the embedding
            embedding_norm = np.linalg.norm(mol_embedding)  # Magnitude of vector
            embedding_mean = np.mean(mol_embedding)         # Mean of features
            embedding_std = np.std(mol_embedding)           # Standard deviation
            
            # 2. Check for structural alerts (using your existing method)
            alerts = self._check_toxicity_alerts(mol)
            
            # 3. Check for PAINS patterns
            pains_score = self._check_pains_patterns(mol)
            
            # 4. Check for high aromatic ring count
            ring_info = mol.GetRingInfo()
            ring_count = ring_info.NumRings()
            aromatic_rings = 0
            for ring in ring_info.AtomRings():
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_rings += 1
            
            aromatic_score = min(aromatic_rings / 3.0, 1.0)  # Normalize, cap at 1.0
            
            # 5. Combine embedding features to create a toxicity indicator
            embedding_toxicity_contribution = (
                0.2 * abs(embedding_mean) +         # General intensity 
                0.3 * embedding_std +               # Feature diversity
                0.1 * (embedding_norm / 10.0)       # Overall magnitude (normalized)
            )
            
            # Apply sigmoid function to normalize to 0-1 range
            embedding_toxicity_score = 1.0 / (1.0 + np.exp(-embedding_toxicity_contribution))
            print(f"Embedding-based toxicity: {embedding_toxicity_score:.4f}")
            
            # 6. Combine all scores: structural alerts, PAINS, ring count, and embeddings
            toxicity_score = (
                0.4 * min(alerts / 3.0, 1.0) +      # Structural alerts contribution
                0.2 * pains_score +                 # PAINS patterns contribution
                0.1 * aromatic_score +              # Aromatic ring count contribution
                0.3 * embedding_toxicity_score      # Embedding-based contribution
            )
            
            print(f"Final toxicity score: {toxicity_score:.4f}")
            return min(1.0, max(0.0, toxicity_score))
            
        except Exception as e:
            print(f"Error in toxicity prediction with MoLFormer: {str(e)}")
            print("Falling back to structural alerts method")
            return self._fallback_toxicity_estimate(mol)

    def _check_pains_patterns(self, mol) -> float:
        """
        Check molecule for PAINS patterns
        Returns a score between 0 and 1 where higher indicates more problematic PAINS patterns
        """
        try:
            from rdkit.Chem import FilterCatalog
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)
            
            # Get matches
            entry = catalog.GetFirstMatch(mol)
            if entry:
                # Get all matches
                matches = catalog.GetMatches(mol)
                num_matches = len(matches)
                
                # Extract pattern IDs for reporting
                pattern_ids = [match.GetDescription() for match in matches]
                print(f"PAINS patterns found: {pattern_ids}")
                
                # Return normalized score - cap at 3 patterns
                return min(num_matches / 3.0, 1.0)
            else:
                return 0.0
        except Exception as e:
            print(f"Error checking PAINS patterns: {str(e)}")
            return 0.0
    
    def _check_toxicity_alerts(self, mol) -> int:
        """Check for structural patterns associated with toxicity"""
        alerts = 0
        toxicity_patterns = [
            # DNA/protein binding (mutagenic/carcinogenic)
            '[N+](=O)[O-]',  # Nitro groups
            'C(=O)Cl',       # Acid chlorides
            'N=[N+]=[N-]',   # Azides
            
            # Reactive functional groups
            'C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2=O',  # Anthraquinone
            'C(OC(=O)Cl)',    # Acyl chlorides
            'C=[N+]=[N-]',    # Diazo
            'C1(=O)OC1',      # Epoxides
            
            # Toxic elements
            '[Hg]', '[Cd]', '[As]', '[Se]', '[Pb]'
        ]
        
        for pattern in toxicity_patterns:
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    alerts += 1
            except:
                # Skip if pattern is invalid
                continue
        
        return alerts
        
    def _fallback_toxicity_estimate(self, mol) -> float:
        """Fallback method for toxicity estimation"""
        if not mol:
            return 1.0
            
    # Simple heuristic based on structural alerts
        alerts = 0
        smarts_patterns = [
            '[N+](=O)[O-]',  # Nitro group
            'C(=O)Cl',       # Acid chloride
            'C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2=O'  # Anthraquinone
        ]
    
        for pattern in smarts_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                alerts += 1
                
        return min(alerts / 3.0, 1.0)
    
    def estimate_binding_affinity(self, mol) -> float:
        """Estimate binding affinity using BioNeMo's DiffDock model"""
        if not mol or not self.target_protein or not hasattr(self, 'diffdock_predictor'):
            return self._fallback_binding_estimate(mol)
    
        try:
        # Convert molecule to SMILES
            smiles = Chem.MolToSmiles(mol)
        
        # Prepare input for DiffDock
            inputs = {
            "ligand": smiles,
            "protein_sequence": self.target_protein
            }
        
        # Run DiffDock prediction
            results = self.diffdock_predictor.predict(inputs)
        
        # Extract binding score - structure may vary based on BioNeMo implementation
        # Adjust according to actual DiffDock output format
            binding_score = results.get("binding_score", 0.0)
        
        # Convert to normalized score between 0 and 1
        # DiffDock typically returns lower values for better binding
            normalized_score = 1.0 / (1.0 + np.exp(binding_score))
        
            return normalized_score
        
        except Exception as e:
            print(f"Error using DiffDock for binding affinity: {e}")
            return self._fallback_binding_estimate(mol)
        
    def _fallback_binding_estimate(self, mol) -> float:
        """Fallback method for binding estimation when DiffDock fails"""
        if not mol:
            return 0.0
        
    # Simple estimation based on molecular properties
        mol_weight = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = CalcTPSA(mol)
        rotatable_bonds = CalcNumRotatableBonds(mol)
    
    # Simplified score
        score = (
            -0.1 * abs(mol_weight - 400) +
            -0.2 * abs(logp - 3) +
            -0.1 * abs(tpsa - 90) +
            -0.05 * rotatable_bonds
        )
    
        return 1.0 / (1.0 + np.exp(-score/10))
    
    def calculate_solubility(self, mol) -> float:
        """Estimate aqueous solubility (logS)"""
        if not mol:
            return -5.0  # Poor solubility as default
            
        # Use ESOL model to estimate logS
        # Based on: Delaney, J. S. (2004)
        def _esol_atom_contribution(mol, log_p, mw):
            rings = mol.GetRingInfo()
            rings_count = rings.NumRings()
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            
            result = 0.16 - 0.63 * log_p - 0.0062 * mw + 0.066 * aromatic_atoms - 0.74 * rings_count
            return result
            
        log_p = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        return _esol_atom_contribution(mol, log_p, mw)
    
    def calculate_all_metrics(self, mol) -> Dict:
        """Calculate all drug metrics for a molecule"""
        if mol is None:
            return {
                'druglikeness': 0.0,
                'synthetic_accessibility': 10.0,
                'lipinski_violations': 4,
                'toxicity': 1.0,
                'binding_affinity': 0.0,
                'solubility': -5.0
            }
            
        smiles = Chem.MolToSmiles(mol)
        
        # Check cache first
        if smiles in self.metrics_cache:
            return self.metrics_cache[smiles]
            
        # Calculate all metrics
        metrics = {
            'druglikeness': self.calculate_druglikeness(mol),
            'synthetic_accessibility': self.calculate_synthetic_accessibility(mol),
            'lipinski_violations': self.calculate_lipinski_violations(mol),
            'toxicity': self.predict_toxicity(mol),
            'binding_affinity': self.estimate_binding_affinity(mol),
            'solubility': self.calculate_solubility(mol)
        }
        
        # Cache results
        self.metrics_cache[smiles] = metrics
        print(metrics)
        return metrics
    
    def calculate_objective_score(self, mol, weights: Dict[str, float]) -> float:
        """
        Calculate weighted score based on multiple objectives
        
        Args:
            mol: RDKit molecule object
            weights: Dictionary of weights for each metric
                     e.g. {'druglikeness': 1.0, 'toxicity': -1.0, ...}
        
        Returns:
            float: Overall weighted score (higher is better)
        """
        if mol is None:
            return -float('inf')
            
        metrics = self.calculate_all_metrics(mol)
        
        # Normalize scores where needed (for metrics where lower is better)
        normalized_metrics = metrics.copy()
        normalized_metrics['synthetic_accessibility'] = 1.0 - (metrics['synthetic_accessibility'] / 10.0)
        normalized_metrics['lipinski_violations'] = 1.0 - (metrics['lipinski_violations'] / 4.0)
        normalized_metrics['toxicity'] = 1.0 - metrics['toxicity']
        normalized_metrics['solubility'] = 1.0 / (1.0 + np.exp(-metrics['solubility'])) # Sigmoid transform
        
        # Calculate weighted sum
        score = sum(weights.get(metric, 0.0) * normalized_metrics.get(metric, 0.0) 
                   for metric in normalized_metrics)
                   
        return score
    
    def optimize(self, optimization_parameters: Dict = None) -> List[Dict]:
        """
        Perform multi-objective optimization on the drug candidates
        
        Args:
            optimization_parameters: Dictionary with optimization parameters
                                    including weights for different objectives
        
        Returns:
            List of dictionaries with optimized molecules and their scores
        """
        if optimization_parameters is None:
            optimization_parameters = {
                'weights': {
                    'druglikeness': 1.0,
                    'synthetic_accessibility': 0.7,
                    'lipinski_violations': 0.8,
                    'toxicity': 1.0,
                    'binding_affinity': 1.0,
                    'solubility': 0.5
                },
                'top_n': 10
            }
        
        weights = optimization_parameters.get('weights', {})
        top_n = optimization_parameters.get('top_n', 10)
        
        results = []
        
        # Calculate scores for all molecules
        for i, mol in enumerate(self.mols):
            if mol is None:
                continue
                
            smiles = self.valid_smiles[self.valid_indices.index(i)]
            score = self.calculate_objective_score(mol, weights)
            
            # Get all metrics for reporting
            metrics = self.calculate_all_metrics(mol)
            
            results.append({
                'smiles': smiles,
                'molecule': mol,
                'score': score,
                'metrics': metrics
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        # print(results)
        # Take top_n results
        return results[:top_n]
    
    def filter_candidates(self, 
                         filters: Dict[str, Tuple[float, float]] = None,
                         compounds: List[Dict] = None) -> List[Dict]:
        """
        Filter candidates based on property ranges
        
        Args:
            filters: Dictionary of property filters with (min, max) tuple values
                    e.g. {'druglikeness': (0.5, 1.0), 'toxicity': (0, 0.3)}
            compounds: List of compounds to filter (if None, uses optimized compounds)
        
        Returns:
            List of filtered compounds
        """
        if filters is None:
            filters = {
                'druglikeness': (0.5, 1.0),
                'synthetic_accessibility': (0, 5.0),
                'lipinski_violations': (0, 1),
                'toxicity': (0, 0.3),
                'binding_affinity': (0.6, 1.0),
                'solubility': (-4.0, 0)
            }
        
        if compounds is None:
            # Optimize with default parameters if not already done
            compounds = self.optimize()
        
        filtered_results = []
        
        for compound in compounds:
            metrics = compound['metrics']
            passes_filters = True
            
            for prop, (min_val, max_val) in filters.items():
                if prop in metrics:
                    value = metrics[prop]
                    if value < min_val or value > max_val:
                        passes_filters = False
                        break
            
            if passes_filters:
                filtered_results.append(compound)
        
        return filtered_results
    
    def generate_molecular_modifications(self, smiles: str, num_variants: int = 50) -> List[str]:
        """Generate structural variations using MolGPT"""
        # if not hasattr(self, 'molbart_predictor'):
        #     return self._fallback_molecule_generation(smiles, num_variants)
    
        try:
            from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

            # Load tokenizer directly from Hugging Face
            tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
            tokenizer.pad_token = "<pad>"
            tokenizer.bos_token = "<bos>"
            tokenizer.eos_token = "<eos>"
            model = GPT2LMHeadModel.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
            # Load pretrained model
            inputs = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)  # Start with <bos> token
            temperature = 1.5
            outputs = model.generate(
                    input_ids=inputs,
                    max_length=128,
                    num_return_sequences=num_variants,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,  # Enable sampling for diverse outputs
                    temperature=temperature,  # Higher temp = more diverse outputs
                    return_dict_in_generate=True,
                )
    
            # Decode generated SMILES strings
            generated_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
        # Filter out invalid molecules
            valid_variants = []
            for variant in generated_smiles:
                mol = Chem.MolFromSmiles(variant)
                if mol and variant != smiles and variant not in valid_variants:
                    valid_variants.append(variant)
                
            return valid_variants[:num_variants]
        
        except Exception as e:
            print(f"Error using MolGPT for molecule generation: {e}")
            return self._fallback_molecule_generation(smiles, num_variants)

    def _fallback_molecule_generation(self, smiles: str, num_variants: int) -> List[str]:
        """
        Advanced fallback method for molecule generation using sophisticated RDKit modifications
        
        Strategies include:
        1. Functional group transformations
        2. Ring modifications
        3. Substituent additions
        4. Structural rearrangements
        5. Stereochemistry alterations
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        variants = []
        max_attempts = num_variants * 5  # Allow multiple attempts to generate unique variants
        
        while len(variants) < num_variants and max_attempts > 0:
            try:
                # Create a deep copy of the molecule
                new_mol = Chem.Mol(mol)
                
                # Randomly select a modification strategy
                modification_strategies = [
                    self._add_substituent,
                    self._replace_functional_group,
                    self._modify_ring_structure,
                    self._stereochemistry_modification,
                    self._structural_rearrangement
                ]
                
                strategy = random.choice(modification_strategies)
                modified_mol = strategy(new_mol)
                
                # Validate and process the modified molecule
                if modified_mol:
                    # Add hydrogens for proper structure, optimize, then remove hydrogens
                    modified_mol = AllChem.AddHs(modified_mol)
                    AllChem.EmbedMolecule(modified_mol, randomSeed=random.randint(1, 1000))
                    AllChem.MMFFOptimizeMolecule(modified_mol)
                    modified_mol = Chem.RemoveHs(modified_mol)
                    
                    # Convert to SMILES and validate
                    new_smiles = Chem.MolToSmiles(modified_mol)
                    
                    if (new_smiles != smiles and 
                        new_smiles not in variants and 
                        self._validate_molecule_properties(modified_mol)):
                        variants.append(new_smiles)
                
            except Exception as e:
                print(f"Molecule generation error: {e}")
            
            max_attempts -= 1
        
        return variants

    def _add_substituent(self, mol):
        """Add a substituent to a suitable atom"""
        substituents = [
            '[CH3]',   # Methyl
            '[OH]',    # Hydroxyl
            '[F]',     # Fluorine
            '[Cl]',    # Chlorine
            '[NH2]',   # Amine
            '[COOH]',  # Carboxylic acid
        ]
        
        # Find atoms with free valence
        atoms_with_valence = [
            atom.GetIdx() for atom in mol.GetAtoms() 
            if atom.GetDegree() < atom.GetTotalValence() and not atom.IsInRing()
        ]
        
        if not atoms_with_valence:
            return None
        
        atom_idx = random.choice(atoms_with_valence)
        substituent = Chem.MolFromSmiles(random.choice(substituents))
        
        try:
            new_mol = AllChem.ReplaceSubstructs(
                mol, 
                Chem.MolFromSmarts('[*]'), 
                substituent, 
                True, 
                atom_idx
            )[0]
            return new_mol
        except:
            return None

    def _replace_functional_group(self, mol):
        """Replace a functional group with another"""
        functional_groups = {
            'alcohol': ('[OH]', '[=O]'),   # Alcohol to ketone
            'amine': ('[NH2]', '[N]'),     # Primary to secondary amine
            'carboxylic_acid': ('[COOH]', '[COC]')  # Carboxylic acid to ester
        }
        
        # Convert to SMARTS for substructure matching
        try:
            for group, (old, new) in functional_groups.items():
                old_group = Chem.MolFromSmarts(old)
                new_group = Chem.MolFromSmiles(new)
                
                if mol.HasSubstructMatch(old_group):
                    new_mol = AllChem.ReplaceSubstructs(mol, old_group, new_group, True)[0]
                    return new_mol
        except:
            pass
        
        return None

    def _modify_ring_structure(self, mol):
        """Modify ring structure by expanding, contracting, or adding rings"""
        # Ring expansion/contraction
        if mol.GetRingInfo().NumRings() > 0:
            try:
                ring_info = mol.GetRingInfo()
                ring_atoms = [list(ring) for ring in ring_info.AtomRings()]
                
                if ring_atoms:
                    ring = random.choice(ring_atoms)
                    if len(ring) == 6:  # 6-membered ring modification
                        # Example: convert to 5-membered ring
                        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace('C1CCCCC1', 'C1CCC1'))
                        return new_mol
            except:
                pass
        
        return None

    def _stereochemistry_modification(self, mol):
        """Modify stereochemistry of the molecule"""
        try:
            # Find chiral centers
            chiral_centers = Chem.FindMolChiralCenters(mol)
            
            if chiral_centers:
                # Randomly select a chiral center
                center_idx, _ = random.choice(chiral_centers)
                
                # Create a new molecule with inverted stereochemistry
                new_mol = Chem.Mol(mol)
                new_mol.GetAtomWithIdx(center_idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                
                return new_mol
        except:
            pass
        
        return None

    def _structural_rearrangement(self, mol):
        """Perform structural rearrangement"""
        # Simple example of rearranging bond connections
        try:
            # Identify potential atoms for rearrangement
            rearrangeable_atoms = [
                atom.GetIdx() for atom in mol.GetAtoms() 
                if atom.GetDegree() > 1 and not atom.IsInRing()
            ]
            
            if len(rearrangeable_atoms) >= 2:
                atom1, atom2 = random.sample(rearrangeable_atoms, 2)
                
                # Modify bond connections
                new_mol = Chem.EditableMol(mol)
                new_mol.RemoveBond(atom1, atom2)
                new_mol.AddBond(atom1, atom2, Chem.BondType.SINGLE)
                
                return new_mol.GetMol()
        except:
            pass
        
        return None

    def _validate_molecule_properties(self, mol):
        """
        Validate generated molecule properties
        
        Checks:
        - Molecular weight
        - Valence rules
        - Simple drug-likeness criteria
        """
        try:
            # Basic property checks
            mol_weight = Descriptors.ExactMolWt(mol)
            num_rings = mol.GetRingInfo().NumRings()
            
            # Lipinski's Rule of Five (simplified)
            valid_mol_weight = 100 < mol_weight < 600
            valid_rings = num_rings <= 4
            
            return valid_mol_weight and valid_rings
        except:
            return False
    
    def visualize_molecules(self, compounds: List[Dict]) -> None:
        """
        Generate visualization for top compounds
        In a real implementation, this would connect to IDX for 3D visualization
        """
        # This is a placeholder for visualization logic
        print("Generating visualizations for top compounds...")
        
        # In a real implementation, this would:
        # 1. Generate 3D coordinates for molecules
        # 2. Connect to Google IDX for immersive visualization
        # 3. Show protein-ligand interactions if target protein available
        
        for i, compound in enumerate(compounds[:5]):
            mol = compound['molecule']
            smiles = compound['smiles']
            score = compound['score']
            
            print(f"Compound {i+1}: {smiles}")
            print(f"Score: {score:.4f}")
            print("---")
    
    def explain_results_with_gemini(self, compounds: List[Dict]) -> str:
        """
        Use Gemini API to explain the results in natural language
        """
        # Format compound data for Gemini
        compound_data = []
        for i, compound in enumerate(compounds[:3]):
            compound_data.append({
                "rank": i+1,
                "smiles": compound['smiles'],
                "score": compound['score'],
                "druglikeness": compound['metrics']['druglikeness'],
                "toxicity": compound['metrics']['toxicity'],
                "binding_affinity": compound['metrics']['binding_affinity'],
                "solubility": compound['metrics']['solubility'],
                "lipinski_violations": compound['metrics']['lipinski_violations'],
                "synthetic_accessibility": compound['metrics']['synthetic_accessibility']
            })

        url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCdvbELo9_WNP4ti4wogC5TAjDRL16PmFQ'
        # Define the headers
        headers = {
            'Content-Type': 'application/json'
        }
        # Define the data payload
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"""Explain the following optimized drug candidates in simple terms:
                                    {json.dumps(compound_data, indent=2)}
                                    
                                    Focus on:
                                    1. Why these compounds might be promising drug candidates
                                    2. Their key properties and how they relate to drug efficacy
                                    3. Potential next steps for validation"""
                        }
                    ]
                }
            ]
        }

        # Make the POST request
        response = requests.post(url, headers=headers, json=data)

        # Print the response
        # print(response.json())
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        # Assuming 'text' is a string variable containing the text with Markdown syntax
        text_with_new_lines = text.replace('\\n', '\n')

        # Remove common Markdown syntax
        cleaned_text = re.sub(r'(\\|##|#|_|[*])', '', text_with_new_lines)

        print(cleaned_text)
        return cleaned_text

    
    def export_results(self, compounds: List[Dict], filepath: str) -> None:
        """Export results to CSV file"""
        if not compounds:
            print("No compounds to export")
            return
            
        data = []
        for i, compound in enumerate(compounds):
            row = {
                'rank': i+1,
                'smiles': compound['smiles'],
                'score': compound['score']
            }
            # Add all metrics
            for metric, value in compound['metrics'].items():
                row[metric] = value
                
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")

    
def visualize_simple(compounds, show_protein=True):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    import py3Dmol
    from IPython.display import display
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "compound_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    # 1. Generate individual 2D images for each compound (as PNG)
    print("Visualizing top compounds:")
    
    for i, comp in enumerate(compounds):
        if comp['molecule'] is not None:
            mol = comp['molecule']
            
            # Generate 2D image of the molecule
            img = Draw.MolToImage(
                mol,
                size=(500, 500),
                legend=f"Compound {i+1}\nScore: {comp['score']:.2f}"
            )
            
            # Save individual image as PNG
            filename_2d = f"{output_dir}/compound_{i+1}_2D.png"
            img.save(filename_2d)
            print(f"2D visualization saved as '{filename_2d}'")
            
            # Generate 3D coordinates and save as PDB file (3D format)
            try:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                
                # Save as PDB file (standard 3D structure format)
                pdb_str = Chem.MolToPDBBlock(mol_3d)
                pdb_filename = f"{output_dir}/compound_{i+1}_3D.pdb"
                with open(pdb_filename, 'w') as f:
                    f.write(pdb_str)
                print(f"3D structure saved as PDB: '{pdb_filename}'")
                
                # Also save as SDF file (alternative 3D format with more chemical information)
                sdf_filename = f"{output_dir}/compound_{i+1}_3D.sdf"
                writer = Chem.SDWriter(sdf_filename)
                writer.write(mol_3d)
                writer.close()
                print(f"3D structure saved as SDF: '{sdf_filename}'")
                
                # Save HTML file with embedded 3D viewer
                html_filename = f"{output_dir}/compound_{i+1}_3D_viewer.html"
                
                # Create py3Dmol view for display in notebook
                view = py3Dmol.view(width=600, height=500)
                view.addModel(pdb_str, 'pdb')
                view.setStyle({'stick': {'radius': 0.2, 'colorscheme': 'cyanCarbon'}})
                view.addStyle({'atom': 'C'}, {'sphere': {'radius': 0.4, 'color': 'cyan'}})
                view.addStyle({'atom': 'O'}, {'sphere': {'radius': 0.4, 'color': 'red'}})
                view.addStyle({'atom': 'N'}, {'sphere': {'radius': 0.4, 'color': 'blue'}})
                view.addStyle({'atom': 'S'}, {'sphere': {'radius': 0.4, 'color': 'yellow'}})
                view.addStyle({'atom': 'Cl'}, {'sphere': {'radius': 0.4, 'color': 'green'}})
                view.setBackgroundColor('white')
                view.zoomTo()
                
                # Generate HTML with embedded viewer - carefully check string escaping
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Compound {i+1} - 3D Structure</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
</head>
<body>
    <h2>Compound {i+1}</h2>
    <p>SMILES: {comp['smiles']}</p>
    <p>Score: {comp['score']:.2f}</p>
    <div id="container" style="width: 600px; height: 500px; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById("container"));
        let pdbData = `{pdb_str}`;
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({{}}, {{"stick": {{"radius": 0.2, "colorscheme": "cyanCarbon"}}}});
        viewer.addStyle({{"atom": "C"}}, {{"sphere": {{"radius": 0.4, "color": "cyan"}}}});
        viewer.addStyle({{"atom": "O"}}, {{"sphere": {{"radius": 0.4, "color": "red"}}}});
        viewer.addStyle({{"atom": "N"}}, {{"sphere": {{"radius": 0.4, "color": "blue"}}}});
        viewer.addStyle({{"atom": "S"}}, {{"sphere": {{"radius": 0.4, "color": "yellow"}}}});
        viewer.addStyle({{"atom": "Cl"}}, {{"sphere": {{"radius": 0.4, "color": "green"}}}});
        viewer.setBackgroundColor("white");
        viewer.zoomTo();
        viewer.render();
    </script>
</body>
</html>
"""
                
                with open(html_filename, 'w') as f:
                    f.write(html_content)
                print(f"Interactive 3D viewer saved as HTML: '{html_filename}'")
                
                # Display in notebook if in interactive environment
                try:
                    display(img)
                    display(view)
                    print(f"Compound {i+1}: {comp['smiles']}")
                    print("Properties:")
                    for metric, value in comp['metrics'].items():
                        print(f"  {metric}: {value:.3f}")
                    print("-" * 50)
                except Exception as display_error:
                    print(f"Note: Could not display in interactive environment")
            
            except Exception as e:
                print(f"Error generating 3D visualization for compound {i+1}: {e}")
    
    # Also save a combined grid image for reference
    mols = [comp['molecule'] for comp in compounds if comp['molecule'] is not None]
    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(300, 300),
            legends=[f"Compound {i+1}\nScore: {compounds[i]['score']:.2f}" 
                    for i in range(len(mols))]
        )
        # Save the grid image
        grid_filename = f"{output_dir}/all_compounds_grid.png"
        img.save(grid_filename)
        print(f"Grid visualization saved as '{grid_filename}'")
    
    # Show the target protein (if requested)
    if show_protein and len(compounds) > 0:
        try:
            # Show EGFR structure from PDB
            pdb_id = "1M17"  # EGFR kinase domain
            
            # Fetch the PDB file
            import requests
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)
            
            if response.status_code == 200:
                # Save the PDB file
                protein_pdb_filename = f"{output_dir}/target_protein_EGFR.pdb"
                with open(protein_pdb_filename, 'w') as f:
                    f.write(response.text)
                print(f"Protein structure saved as PDB: '{protein_pdb_filename}'")
                
                # Create HTML viewer for protein
                protein_html_filename = f"{output_dir}/target_protein_EGFR_viewer.html"
                
                # Generate HTML - with very careful string handling
                protein_html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EGFR Kinase Domain - 3D Structure</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
</head>
<body>
    <h2>EGFR Kinase Domain (PDB ID: {pdb_id})</h2>
    <div id="container" style="width: 800px; height: 600px; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById("container"));
        viewer.addPDBURL("https://files.rcsb.org/download/{pdb_id}.pdb", function() {{
            viewer.setStyle({{}}, {{"cartoon": {{"colorscheme": "spectrum"}}}});
            viewer.setStyle({{"hetflag": true}}, {{"stick": {{"colorscheme": "greenCarbon"}}}});
            viewer.zoomTo();
            viewer.render();
        }});
    </script>
</body>
</html>
"""
                
                with open(protein_html_filename, 'w') as f:
                    f.write(protein_html_content)
                print(f"Interactive protein viewer saved as HTML: '{protein_html_filename}'")
                
                # Display in notebook
                view = py3Dmol.view(query=f'pdb:{pdb_id}', width=800, height=600)
                view.setStyle({'cartoon': {'colorscheme': 'spectrum'}})
                view.zoomTo()
                
                try:
                    display(view)
                    print(f"Target Protein: EGFR kinase domain (PDB ID: {pdb_id})")
                except:
                    pass
            else:
                print(f"Could not fetch protein structure: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error displaying protein: {str(e)}")

# Example usage
if __name__ == "__main__":
    pdb_id = "1M17"  # EGFR kinase domain
    try:
            clean_smiles = pd.read_csv("cleaned_smiles.csv")["smiles"].tolist() # Specify the column name
    except KeyError:
            clean_smiles = pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv').iloc[:, 0].values  # Use first column
        
        # Create vocabulary
    char_to_idx, idx_to_char = create_vocabulary(clean_smiles)

        # Tokenize SMILES
    max_length = max(len(smiles) for smiles in clean_smiles)

    model = RNNGenerator(vocab_size=len(char_to_idx), embed_dim=128, hidden_dim=256)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example usage in your existing script
    # Assuming you have already defined model, char_to_idx, idx_to_char, and device
    model = RNNGenerator(vocab_size=len(char_to_idx), embed_dim=128, hidden_dim=256)
    model.load_state_dict(torch.load("rnn_model.pth", map_location=device))
    model.to(device)
    diverse_molecules = generate_diverse_molecules(
        model, 
        char_to_idx, 
        idx_to_char, 
        device, 
        start_token="C", 
        num_molecules=5
    )

    print("Generated Molecules:")
    for mol in diverse_molecules:
        print(mol)
    # Example target protein (EGFR kinase domain fragment)
    example_protein ="MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLNDGGNYVLCDFGSVTKLPQKSGDVYSFGVVLLELLTGQPIFPGDEGDQLACMIELLGMPSQKLLDASKRAKNRNDIBKVSGGPNDISQSASNPKLARQPHYVQRESZAVRHGAFMKNL"
    
    # Initialize optimizer
    optimizer = DrugOptimizer(diverse_molecules, example_protein)
    ibuprofen_mol = Chem.MolFromSmiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')
    #optimizer.predict_toxicity(ibuprofen_mol)
    #optimizer.predict_protein_structure()
    
    # Calculate metrics for the ibuprofen molecule
    ibuprofen_metrics = optimizer.calculate_all_metrics(ibuprofen_mol)

    # Create the compound in the expected format (list of dictionaries)
    ibuprofen_compound = [{
        'smiles': Chem.MolToSmiles(ibuprofen_mol),
        'molecule': ibuprofen_mol,
        'score': 0.5,  # Sample score, you might calculate this properly
        'metrics': ibuprofen_metrics
    }]

    # Now call explain_results_with_gemini with properly formatted input
    #optimizer.explain_results_with_gemini(ibuprofen_compound)
    optimizer.visualize_molecules(ibuprofen_compound)
    # Define optimization parameters
    optimization_params = {
        'weights': {
            'druglikeness': 1.0,
            'synthetic_accessibility': 0.8,
            'lipinski_violations': 0.7,
            'toxicity': 1.2,
            'binding_affinity': 1.5,
            'solubility': 0.6
        },
        'top_n': 10
    }
    
    # Run optimization
    optimized_compounds = optimizer.optimize(optimization_params)

    
    # Apply filters
    filters = {
        'druglikeness': (0.6, 1.0),
        'toxicity': (0, 0.3),
        'binding_affinity': (0.0, 1.0)
    }
    filtered_compounds = optimizer.filter_candidates(filters, optimized_compounds)
    print("filtered_components : ",filtered_compounds)
    # Generate variations of top compound
    if filtered_compounds:
        top_compound = filtered_compounds[0]['smiles']
        
        variants = optimizer.generate_molecular_modifications(top_compound, 50)
        
        print(f"Generated {len(variants)} variants of top compound")
        # Create new optimizer instance for variants
        variant_optimizer = DrugOptimizer(variants, example_protein)
        
        # Optimize variants with same parameters
        optimized_variants = variant_optimizer.optimize(optimization_params)
        
        # Filter variants if needed
        filtered_variants = variant_optimizer.filter_candidates(filters, optimized_variants)
        
        # Sort by score and get top 10 (or all if less than 10)
        sorted_variants = sorted(filtered_variants, key=lambda x: x['score'], reverse=True)
        num_to_show = min(10, len(sorted_variants))
        
        print(f"\nTop {num_to_show} optimized variants:")
        print("=" * 80)
        for i, variant in enumerate(sorted_variants[:num_to_show], 1):
            print(f"\nVariant {i}:")
            print(f"SMILES: {variant['smiles']}")
            print(f"Score: {variant['score']:.4f}")
            print("Metrics:")
            for metric, value in variant['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        # Visualize top variants
        variant_optimizer.visualize_molecules(sorted_variants[:num_to_show])
        
        # Export results
        variant_optimizer.export_results(sorted_variants[:num_to_show], 
                                      "optimized_variants.csv")
        explanation = variant_optimizer.explain_results_with_gemini(sorted_variants[:num_to_show])
        print(explanation)
    # Export results
    # optimizer.export_results(filtered_compounds, "optimized_drug_candidates.csv")
    
    # # Visualize (placeholder in this example)
    # optimizer.visualize_molecules(filtered_compounds)
    
    # # Get explanation from Gemini (placeholder)
    # explanation = optimizer.explain_results_with_gemini(filtered_compounds)
    # print(explanation)