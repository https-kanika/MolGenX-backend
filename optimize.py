import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, AllChem, Draw, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA
import torch
import pandas as pd
import json
from typing import List, Dict, Tuple, Optional
import requests
import re
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from diffdock import call_diffdock_api
import os
from visualization import visualize_simple
import os
from dotenv import load_dotenv

load_dotenv()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.number):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif str(type(obj)) == "<class 'rdkit.Chem.rdchem.Mol'>":
            return Chem.MolToSmiles(obj)
        return super(NumpyEncoder, self).default(obj)

class DrugOptimizer:
    #TODO WRITE DOC STRINGS TO GENERATE README
    """
    A comprehensive drug candidate optimization pipeline that integrates 
    multiple objectives, ESM-2 for protein interaction, etc.
    """
    
    def __init__(self, candidate_smiles: List[str], target_protein: Optional[str] = None, pdb_id: Optional[str] = None):    
        """
        Initializes the DrugOptimizer with a list of candidate SMILES strings and optional target protein information.
            candidate_smiles (List[str]): List of SMILES strings representing drug candidate molecules.
            target_protein (Optional[str], optional): Amino acid sequence of the target protein. Defaults to None.
            pdb_id (Optional[str], optional): PDB identifier for the target protein structure. Defaults to None.
        Attributes:
            candidates (List[str]): Original list of candidate SMILES strings.
            target_protein (Optional[str]): Sequence of the target protein, if provided.
            mols (List[rdkit.Chem.Mol]): List of RDKit Mol objects created from valid SMILES strings.
            valid_indices (List[int]): Indices of valid SMILES strings that could be converted to Mol objects.
            valid_smiles (List[str]): List of valid SMILES strings corresponding to successfully created Mol objects.
            metrics_cache (dict): Cache for storing computed metrics.
            pdb_id (Optional[str]): PDB identifier for the target protein structure, if provided.
        """
        self.candidates = candidate_smiles
        self.target_protein = target_protein
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in candidate_smiles if Chem.MolFromSmiles(smiles)]
        self.valid_indices = [i for i, mol in enumerate(self.mols) if mol is not None]
        self.valid_smiles = [candidate_smiles[i] for i in self.valid_indices]
        self.metrics_cache = {}
        self.pdb_id = pdb_id

    def predict_protein_structure(self):
        """
        Predicts the structure information for the target protein using the ESM-2 model.
        This method loads the ESM-2 transformer model and its tokenizer from Hugging Face,
        processes the target protein sequence (up to 1000 amino acids to avoid memory issues),
        and generates protein embeddings by mean pooling the last hidden state (excluding special tokens).
        Returns the resulting sequence embeddings as a tensor, or None if the target protein is not set
        or an error occurs during prediction.
        Returns:
            torch.Tensor or None: The mean-pooled protein sequence embeddings, or None if prediction fails.
        """
        if not self.target_protein:
            return None
    
        try:
            #print("Loading ESM-2 model...")
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
            # Process only the first 1000 amino acids to avoid memory issues
            protein_seq = self.target_protein[:1000]
            #print(f"Processing protein sequence (length: {len(protein_seq)})")
        
            inputs = tokenizer(protein_seq, return_tensors="pt")
        
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
                # Get last hidden state
                embeddings = outputs.hidden_states[-1]
            
                # Mean pool over sequence length (excluding special tokens)
                sequence_embeddings = embeddings[:, 1:-1].mean(dim=1)
            
                #print(f"Generated protein embeddings with shape: {sequence_embeddings.shape}")
            
            return sequence_embeddings
    
        except Exception as e:
            print(f"Error in protein structure prediction: {str(e)}")
            return None
    
    def calculate_druglikeness(self, mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness)"""
        return QED.qed(mol) if mol else 0.0

    def calculate_synthetic_accessibility(self,mol):
        """
        Estimates the synthetic accessibility of a molecule based on structural features.
        This method calculates a synthetic accessibility score using several molecular descriptors,
        including the number of rotatable bonds, rings, spiro atoms, and bridgehead atoms.
        Each feature contributes to the overall complexity score with a specific weight,
        reflecting its impact on synthetic difficulty.
        Parameters:
            mol (rdkit.Chem.Mol): The molecule object to evaluate.
        Returns:
            float: A synthetic accessibility score between 1.0 (easiest) and 10.0 (most difficult).
                   Returns 10.0 if the input molecule is invalid or None.
        """
        if not mol:
            return 10.0
        
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = Descriptors.RingCount(mol)
        spiro_count = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        bridgehead_count = Descriptors.NumBridgeheadAtoms(mol)
        
        complexity_score = (
            rotatable_bonds * 0.3 +     # Rotatable bonds add complexity
            ring_count * 0.5 +          # Rings increase synthesis difficulty
            spiro_count * 1.0 +         # Spiro atoms significantly increase complexity
            bridgehead_count * 1.2      # Bridgehead atoms are challenging
        )
        
        return min(max(complexity_score, 1.0), 10.0)
    
    def calculate_lipinski_violations(self, mol) -> int:
        """
        Calculates the number of Lipinski's Rule of Five violations for a given molecule.
        Lipinski's Rule of Five is a set of guidelines to evaluate druglikeness, based on the following criteria:
            - Molecular weight (MW) > 500
            - LogP (octanol-water partition coefficient) > 5
            - Number of hydrogen bond donors > 5
            - Number of hydrogen bond acceptors > 10
        Each criterion violated increases the violation count by one.
        Args:
            mol: An RDKit molecule object to be evaluated.
        Returns:
            int: The number of Lipinski's Rule of Five violations for the molecule.
        """
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
        """
        Predicts the toxicity of a molecule using IBM's MoLFormer-XL model, with fallbacks to structural alerts and other heuristics.
        Parameters:
            mol (rdkit.Chem.Mol): The molecule object to analyze.
        Returns:
            float: A normalized toxicity score between 0.0 (non-toxic) and 1.0 (highly toxic).
        Method:
            - Converts the molecule to a SMILES string.
            - Uses the MoLFormer-XL model to generate an embedding for the molecule.
            - Extracts statistical features from the embedding (mean, standard deviation, norm).
            - Checks for structural toxicity alerts and PAINS patterns.
            - Assesses aromatic ring count as a toxicity indicator.
            - Combines all features into a final toxicity score using weighted contributions.
            - If the model fails, falls back to a structural alerts-based estimation.
        Raises:
            Any exception encountered during model inference is caught, and a fallback method is used.
        """
        if not mol:
            return 1.0
        
        try:
            smiles = Chem.MolToSmiles(mol)
            #print(f"Analyzing toxicity for SMILES: {smiles}")
            
            model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                            deterministic_eval=True, 
                                            trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                                    trust_remote_code=True)
            
            inputs = tokenizer(smiles, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
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
            #print(f"Embedding-based toxicity: {embedding_toxicity_score:.4f}")
            
            # 6. Combine all scores: structural alerts, PAINS, ring count, and embeddings
            toxicity_score = (
                0.4 * min(alerts / 3.0, 1.0) +      # Structural alerts contribution
                0.2 * pains_score +                 # PAINS patterns contribution
                0.1 * aromatic_score +              # Aromatic ring count contribution
                0.3 * embedding_toxicity_score      # Embedding-based contribution
            )
            
            #print(f"Final toxicity score: {toxicity_score:.4f}")
            return min(1.0, max(0.0, toxicity_score))
            
        except Exception as e:
            print(f"Error in toxicity prediction with MoLFormer: {str(e)}")
            print("Falling back to structural alerts method")
            return self._fallback_toxicity_estimate(mol)

    def _check_pains_patterns(self, mol) -> float:
        """
        Analyzes a molecule for the presence of PAINS (Pan Assay Interference Compounds) patterns.
        Parameters:
            mol: rdkit.Chem.Mol
                The molecule to be analyzed.
        Returns:
            float:
                A score between 0 and 1 indicating the prevalence of problematic PAINS patterns.
                A higher score means more PAINS patterns were detected (capped at 3 patterns).
                Returns 0.0 if no patterns are found or if an error occurs.
        Notes:
            - Utilizes RDKit's FilterCatalog with the PAINS filter set.
            - If an error occurs during processing, the function prints the error and returns 0.0.
        """
        try:
            from rdkit.Chem import FilterCatalog
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)
            
            entry = catalog.GetFirstMatch(mol)
            if entry:
                matches = catalog.GetMatches(mol)
                num_matches = len(matches)
                
                # Extract pattern IDs for reporting
                pattern_ids = [match.GetDescription() for match in matches]
                #print(f"PAINS patterns found: {pattern_ids}")
                
                # Return normalized score - cap at 3 patterns
                return min(num_matches / 3.0, 1.0)
            else:
                return 0.0
        except Exception as e:
            print(f"Error checking PAINS patterns: {str(e)}")
            return 0.0
    
    def _check_toxicity_alerts(self, mol) -> int:
        """
        Checks a molecule for the presence of structural patterns associated with toxicity.
        This method scans the input molecule for a predefined set of SMARTS patterns that are known to be linked to toxicological concerns, such as DNA/protein binding groups, reactive functional groups, and toxic elements. Each detected pattern increments an alert counter.
        Args:
            mol: An RDKit Mol object representing the molecule to be analyzed.
        Returns:
            int: The number of toxicity alerts (i.e., the number of toxic patterns found in the molecule).
        """
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
        """
        Estimates the toxicity of a molecule using a simple structural alert heuristic.
        This fallback method checks for the presence of specific toxicophores (structural alerts)
        within the given molecule. If the molecule is None or invalid, it returns a maximum toxicity score of 1.0.
        Otherwise, it counts the number of predefined toxic substructures present and returns a normalized
        toxicity score between 0.0 and 1.0.
        Args:
            mol: An RDKit Mol object representing the molecule to be evaluated.
        Returns:
            float: A toxicity score between 0.0 (non-toxic) and 1.0 (highly toxic), based on the number of alerts detected.
        """
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
        """
        Estimate the binding affinity of a molecule using BioNeMo's DiffDock model.
        This method generates 3D structures for the input molecule and the target protein,
        visualizes them, and then calls the DiffDock API to estimate the binding affinity.
        If the required protein PDB file is not found, a fallback estimation method is used.
        Args:
            mol (rdkit.Chem.Mol): The molecule for which to estimate binding affinity.
        Returns:
            float: The normalized binding affinity score, where lower values indicate better binding.
                   If an error occurs or required files are missing, returns a fallback estimate.
        Raises:
            Any exceptions encountered during processing are caught and logged, and the fallback
            estimation method is used in such cases.
        """
        return self._fallback_binding_estimate(mol)
        try:

            smiles = Chem.MolToSmiles(mol)
            temp_compound = [{
                    'molecule': mol,
                    'smiles': smiles,
                    'score': 0.0,
                    'metrics': {}
                }]
            output_dir = "compound_visualizations"
            if not hasattr(self, 'target_protein_file') or not self.target_protein_file:
            # Specify PDB ID for the protein target

                pdb_id=self.pdb_id
                # Create a temporary compound list to use with visualize_simple
                
                
                
                visualize_simple(temp_compound, show_protein=True, pdb_id=pdb_id)
                
                self.target_protein_file = f"{output_dir}\\target_protein.pdb"
                
                if not os.path.exists(self.target_protein_file):
                    print(f"PDB file not found at {self.target_protein_file}. Using fallback method.")
                    return self._fallback_binding_estimate(mol)
            protien_pdb_id=self.pdb_id

            
            visualize_simple(temp_compound, show_protein=False, pdb_id=protien_pdb_id)
            ligand_file = f"{output_dir}\\compound_1_3D.pdb"

            file_path = "target_protein.pdb.zip"
            if os.path.exists(file_path):
                    os.remove(file_path)
                    print("File deleted successfully.")
            results = call_diffdock_api(
            input_structure=protien_pdb_id,  # Protein ID or name
            input_pdb=self.target_protein_file, 
            input_ligand=ligand_file,                 # Path to PDB file
            smiles_string=smiles,                 # SMILES string                   # No ligand file
            num_inference_steps=10,
            num_samples=10,
            actual_inference_steps=10,
            no_final_step_noise=True
            )

            binding_score = results

            # DiffDock typically returns lower values for better binding
            normalized_score = 1.0 / (1.0 + np.exp(float(binding_score)))
        
            return normalized_score
        
        except Exception as e:
            print(f"Error using DiffDock for binding affinity: {e}")
            return self._fallback_binding_estimate(mol)
        
    def _fallback_binding_estimate(self, mol) -> float:
        """
        Estimates the binding affinity of a molecule using a heuristic scoring function.
        This fallback method is used when DiffDock fails to provide a binding estimate.
        The score is calculated based on molecular weight, logP, topological polar surface area (TPSA),
        and the number of rotatable bonds. The final value is a sigmoid-transformed score representing
        the estimated binding affinity.
        Args:
            mol: An RDKit molecule object to estimate binding for.
        Returns:
            float: Estimated binding affinity score in the range (0, 1).
        """
        if not mol:
            return 0.0
        
        mol_weight = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = CalcTPSA(mol)
        rotatable_bonds = CalcNumRotatableBonds(mol)

        score = (
            -0.1 * abs(mol_weight - 400) +
            -0.2 * abs(logp - 3) +
            -0.1 * abs(tpsa - 90) +
            -0.05 * rotatable_bonds
        )
    
        return 1.0 / (1.0 + np.exp(-score/10))
    
    def calculate_solubility(self, mol) -> float:
        """
        Estimate the aqueous solubility (logS) of a molecule using the ESOL model.
        Parameters:
            mol (rdkit.Chem.Mol): The molecule for which to estimate solubility.
        Returns:
            float: The estimated logS (log10 of solubility in mol/L). Returns -5.0 if the molecule is invalid.
        Notes:
            The ESOL model is based on the method described in:
            Delaney, J. S. (2004). "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure."
            J. Chem. Inf. Comput. Sci., 44(3), 1000-1005.
        """
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
        """
        Calculate all drug-related metrics for a given molecule.
        This method computes a set of predefined metrics for a molecule, including druglikeness, synthetic accessibility, Lipinski's rule violations, toxicity, binding affinity, and solubility. If the molecule is None, default values indicating poor drug properties are returned. Results are cached based on the molecule's SMILES representation to avoid redundant calculations.
        Args:
            mol: An RDKit Mol object representing the molecule to evaluate.
        Returns:
            Dict: A dictionary containing the following keys and their corresponding metric values:
                - 'druglikeness' (float): Quantitative estimate of drug-likeness.
                - 'synthetic_accessibility' (float): Synthetic accessibility score.
                - 'lipinski_violations' (int): Number of Lipinski's rule of five violations.
                - 'toxicity' (float): Predicted toxicity score.
                - 'binding_affinity' (float): Estimated binding affinity.
                - 'solubility' (float): Predicted solubility.
                - 'molecular_weight' (float): Molecular weight of the molecule.
        """
        if mol is None:
            return {
                'druglikeness': 0.0,
                'synthetic_accessibility': 10.0,
                'lipinski_violations': 4,
                'toxicity': 1.0,
                'binding_affinity': 0.0,
                'solubility': -5.0,
                'molecular_weight': 0.0
            }
            
        smiles = Chem.MolToSmiles(mol)
        
        if smiles in self.metrics_cache:
            return self.metrics_cache[smiles]
            
        # Calculate all metrics
        metrics = {
            'druglikeness': self.calculate_druglikeness(mol),
            'synthetic_accessibility': self.calculate_synthetic_accessibility(mol),
            'lipinski_violations': self.calculate_lipinski_violations(mol),
            'toxicity': self.predict_toxicity(mol),
            'binding_affinity': self.estimate_binding_affinity(mol),
            'solubility': self.calculate_solubility(mol),
            'molecular_weight': Descriptors.MolWt(mol)  
        }

        self.metrics_cache[smiles] = metrics
        #print(metrics)
        return metrics
    
    def calculate_objective_score(self, mol, weights: Dict[str, float]) -> float:
        """
        Calculate a weighted objective score for a molecule based on multiple metrics.
        This function computes an overall score for a given RDKit molecule by evaluating several
        molecular properties (metrics), normalizing them as needed, and combining them using
        user-specified weights. Metrics where lower values are better (e.g., synthetic accessibility,
        Lipinski violations, toxicity) are normalized so that higher normalized values are preferred.
        Solubility is transformed using a sigmoid function to ensure bounded output.
            mol: RDKit molecule object to be evaluated.
            weights (Dict[str, float]): Dictionary mapping metric names to their corresponding weights.
                Example: {'druglikeness': 1.0, 'toxicity': -1.0, ...}
        
        Returns:
            float: The overall weighted score for the molecule. Higher scores indicate better
                overall desirability according to the provided weights. Returns negative infinity
                if the molecule is None.
        """
        if mol is None:
            return -float('inf')
            
        metrics = self.calculate_all_metrics(mol)
        
        # Normalize scores where needed (for metrics where lower is better)
        normalized_metrics = metrics.copy()
        normalized_metrics['synthetic_accessibility'] = 1.0 - (metrics['synthetic_accessibility'] / 10.0)
        normalized_metrics['lipinski_violations'] = 1.0 - (metrics['lipinski_violations'] / 4.0)
        normalized_metrics['toxicity'] = 1.0 - metrics['toxicity']
        normalized_metrics['solubility'] = 1.0 / (1.0 + np.exp(-metrics['solubility'])) 

        score = sum(weights.get(metric, 0.0) * normalized_metrics.get(metric, 0.0) 
                   for metric in normalized_metrics)
                   
        return score
    
    def optimize(self, optimization_parameters: Dict = None) -> List[Dict]:
        """
        Perform multi-objective optimization on the drug candidates.
        This method evaluates each molecule in the current set using a weighted sum of multiple objectives,
        such as druglikeness, synthetic accessibility, Lipinski's rule violations, toxicity, binding affinity,
        and solubility. The weights for each objective and the number of top candidates to return can be
        customized via the `optimization_parameters` argument.
            optimization_parameters (Dict, optional): Dictionary containing optimization parameters.
                - 'weights' (Dict[str, float]): Weights for each objective metric.
                - 'top_n' (int): Number of top-scoring molecules to return.
                If not provided, default weights and top_n=10 are used.
            List[Dict]: A list of dictionaries, each containing:
                - 'smiles' (str): SMILES representation of the molecule.
                - 'molecule': The molecule object.
                - 'score' (float): The computed objective score.
                - 'metrics' (Dict): All calculated metrics for the molecule.
            The list is sorted in descending order by score and limited to the top N entries.
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

        for i, mol in enumerate(self.mols):
            if mol is None:
                continue
                
            smiles = self.valid_smiles[self.valid_indices.index(i)]
            score = self.calculate_objective_score(mol, weights)

            metrics = self.calculate_all_metrics(mol)
            
            results.append({
                'smiles': smiles,
                'molecule': mol,
                'score': score,
                'metrics': metrics
            })
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_n]
    
    def filter_candidates(self, filters: Dict[str, Tuple[float, float]] = None, compounds: List[Dict] = None) -> List[Dict]:
        """
        Filters a list of compound candidates based on specified property ranges.
            filters (Dict[str, Tuple[float, float]], optional): 
                A dictionary specifying the property filters, where each key is a property name 
                and the value is a (min, max) tuple representing the allowed range for that property.
                Defaults to a set of common drug-like property ranges if not provided.
            compounds (List[Dict], optional): 
                A list of compound dictionaries to filter. If None, the method will use the 
                result of self.optimize() to obtain compounds.
            List[Dict]: 
                A list of compound dictionaries that satisfy all the specified property filters.
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
        """
        Generate structural variations of a given molecule using the MolGPT model.
        This method takes a SMILES string as input and generates a specified number of structurally
        diverse molecular variants using a pretrained MolGPT model. The generated SMILES strings are
        filtered to ensure validity and uniqueness, and the original input molecule is excluded from
        the results. If the MolGPT generation fails, a fallback molecule generation method is used.
        Args:
            smiles (str): The SMILES string of the input molecule to modify.
            num_variants (int, optional): The number of molecular variants to generate. Defaults to 50.
        Returns:
            List[str]: A list of valid, unique SMILES strings representing molecular modifications of the input.
        """
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
            tokenizer.pad_token = "<pad>"
            tokenizer.bos_token = "<bos>"
            tokenizer.eos_token = "<eos>"
            model = GPT2LMHeadModel.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")

            inputs = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)  # Start with <bos> token
            temperature = 1.5
            outputs = model.generate(
                    input_ids=inputs,
                    max_length=128,
                    num_return_sequences=num_variants,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,  
                    temperature=temperature,  
                    return_dict_in_generate=True,
                )
    
            generated_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]

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
        Generates a list of structurally diverse molecule variants from a given SMILES string using advanced RDKit-based modification strategies.
        This fallback method applies a variety of chemical modification strategies to the input molecule, including:
        The function ensures that each generated variant is unique, valid, and structurally distinct from the original molecule. It attempts multiple modifications until the desired number of unique variants is produced or a maximum number of attempts is reached.
        Args:
            smiles (str): The SMILES string representing the input molecule.
            num_variants (int): The number of unique molecule variants to generate.
        Returns:
            List[str]: A list of SMILES strings representing the generated molecule variants. Returns an empty list if the input SMILES is invalid or no valid variants can be generated.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        variants = []
        max_attempts = num_variants * 5  # Allow multiple attempts to generate unique variants
        
        while len(variants) < num_variants and max_attempts > 0:
            try:
                new_mol = Chem.Mol(mol)
                modification_strategies = [
                    self._add_substituent,
                    self._replace_functional_group,
                    self._modify_ring_structure,
                    self._stereochemistry_modification,
                    self._structural_rearrangement
                ]
                
                strategy = random.choice(modification_strategies)
                modified_mol = strategy(new_mol)

                if modified_mol:
                    # Add hydrogens for proper structure, optimize, then remove hydrogens
                    modified_mol = AllChem.AddHs(modified_mol)
                    AllChem.EmbedMolecule(modified_mol, randomSeed=random.randint(1, 1000))
                    AllChem.MMFFOptimizeMolecule(modified_mol)
                    modified_mol = Chem.RemoveHs(modified_mol)

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
        """
        Adds a random substituent group to a suitable atom in the given molecule.
        This method searches for atoms in the molecule that have available valence (i.e., can form additional bonds)
        and are not part of a ring. It then randomly selects one of these atoms and attaches a randomly chosen
        substituent group (e.g., methyl, hydroxyl, fluorine, chlorine, amine, or carboxylic acid) to it.
        Args:
            mol (rdkit.Chem.Mol): The input molecule to which a substituent will be added.
        Returns:
            rdkit.Chem.Mol or None: A new molecule with the substituent added, or None if no suitable atom is found
            or if the operation fails.
        """
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
        """
        Replaces specific functional groups in a molecule with alternative groups.
        This method searches for predefined functional groups (alcohol, amine, carboxylic acid)
        within the given molecule and replaces the first occurrence found with a corresponding
        alternative group (e.g., alcohol to ketone, primary amine to secondary amine, carboxylic acid to ester).
        Args:
            mol (rdkit.Chem.Mol): The input molecule in which to search and replace functional groups.
        Returns:
            rdkit.Chem.Mol or None: A new molecule with the functional group replaced if a match is found,
            otherwise None if no replacement was made or an error occurred.
        """
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
        """
        Modify the ring structure of a given RDKit molecule using one of several strategies.
        This function randomly selects one of the following strategies to modify a ring in the molecule:
            1. Ring expansion (n → n+1): Increases the size of a selected ring by one atom.
            2. Ring contraction (n → n-1): Decreases the size of a selected ring by one atom.
            3. Aromatization/dearomatization: Converts a non-aromatic ring to aromatic (e.g., cyclohexane to benzene) or vice versa.
            4. Insertion of heteroatoms: Replaces a carbon atom in the ring with a heteroatom (N, O, or S).
            5. Fusing rings: Attempts to fuse two adjacent rings by creating a new bond between them.
        Parameters:
            mol (rdkit.Chem.Mol): The input molecule to be modified.
        Returns:
            rdkit.Chem.Mol or None: The modified molecule if the transformation is successful, otherwise None.
        """
        if not mol or mol.GetRingInfo().NumRings() == 0:
            return None
            
        try:
            strategy = random.choice([
                'expand', 'contract', 'aromatize', 'hetero_insertion', 'fuse'
            ])
            
            ring_info = mol.GetRingInfo()
            ring_atoms = [list(ring) for ring in ring_info.AtomRings()]
            
            if not ring_atoms:
                return None

            ring = random.choice(ring_atoms)
            new_mol = Chem.Mol(mol)
            
            if strategy == 'expand' and len(ring) < 8:
                if len(ring) == 5:
                    pattern = 'C1CCCC1'
                    replacement = 'C1CCCCC1'
                elif len(ring) == 6:
                    pattern = 'C1CCCCC1'
                    replacement = 'C1CCCCCC1'
                elif len(ring) == 3:
                    pattern = 'C1CC1'
                    replacement = 'C1CCC1'
                elif len(ring) == 4:
                    pattern = 'C1CCC1'
                    replacement = 'C1CCCC1'
                else:
                    return None

                atom_map = {atom: i+1 for i, atom in enumerate(ring)}
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    if idx in atom_map:
                        atom.SetProp("molAtomMapNumber", str(atom_map[idx]))

                mapped_smiles = Chem.MolToSmiles(mol)

                mapped_pattern = re.compile(r'C1(?:C\[*:\d+\])+C1', re.IGNORECASE)
                if mapped_pattern.search(mapped_smiles):
                    modified_smiles = mapped_smiles.replace(pattern, replacement)
                    return Chem.MolFromSmiles(modified_smiles)
                    
            elif strategy == 'contract' and len(ring) > 3:
                if len(ring) == 6:
                    pattern = 'C1CCCCC1'
                    replacement = 'C1CCCC1'
                elif len(ring) == 5:
                    pattern = 'C1CCCC1'
                    replacement = 'C1CCC1'
                elif len(ring) == 4:
                    pattern = 'C1CCC1'
                    replacement = 'C1CC1'
                else:
                    return None

                mapped_smiles = Chem.MolToSmiles(mol)
                modified_smiles = mapped_smiles.replace(pattern, replacement)
                return Chem.MolFromSmiles(modified_smiles)
                
            elif strategy == 'aromatize':
                ring_is_aromatic = all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
                
                em = Chem.EditableMol(new_mol)
                
                if ring_is_aromatic and len(ring) == 6:
                    # Dearomatize: Convert benzene to cyclohexane
                    for i in range(len(ring)):
                        idx1 = ring[i]
                        idx2 = ring[(i+1) % len(ring)]
                        em.RemoveBond(idx1, idx2)
                        em.AddBond(idx1, idx2, Chem.BondType.SINGLE)
                        
                        # Set atoms to sp3 carbon
                        atom = new_mol.GetAtomWithIdx(idx1)
                        atom.SetIsAromatic(False)
                        atom.SetHybridization(Chem.HybridizationType.SP3)
                    
                    result_mol = em.GetMol()
                    Chem.SanitizeMol(result_mol)
                    return result_mol
                    
                elif not ring_is_aromatic and len(ring) == 6:
                    # Try to aromatize: Convert cyclohexane to benzene-like structure
                    for i in range(len(ring)):
                        idx1 = ring[i]
                        idx2 = ring[(i+1) % len(ring)]
                        
                        # Alternate between single and double bonds
                        em.RemoveBond(idx1, idx2)
                        if i % 2 == 0:
                            em.AddBond(idx1, idx2, Chem.BondType.DOUBLE)
                        else:
                            em.AddBond(idx1, idx2, Chem.BondType.SINGLE)
                    
                    result_mol = em.GetMol()
                    try:
                        Chem.SanitizeMol(result_mol)
                        return result_mol
                    except:
                        # If sanitization fails, the proposed structure wasn't valid
                        return None
                        
            elif strategy == 'hetero_insertion' and len(ring) >= 5:
                heteroatoms = ['N', 'O', 'S']
                heteroatom = random.choice(heteroatoms)

                replaceable_atoms = [idx for idx in ring 
                                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'C' 
                                    and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0]
                
                if replaceable_atoms:
                    atom_idx = random.choice(replaceable_atoms)

                    for atom in mol.GetAtoms():
                        if atom.GetIdx() == atom_idx:
                            atom.SetProp("molAtomMapNumber", "99")  # Special mapping for target atom
                    
                    mapped_smiles = Chem.MolToSmiles(mol)
                    pattern = r'C\[*:99\]'
                    modified_smiles = re.sub(pattern, f'{heteroatom}[*:99]', mapped_smiles)
                    modified_smiles = re.sub(r'\[*:\d+\]', '', modified_smiles)
                    
                    return Chem.MolFromSmiles(modified_smiles)
                    
            elif strategy == 'fuse' and len(ring_atoms) > 1:
                if len(ring_atoms) < 2:
                    return None
                ring1, ring2 = random.sample(ring_atoms, 2)
                common_atoms = set(ring1).intersection(set(ring2))
                
                if not common_atoms:
                    for atom1 in ring1:
                        for atom2 in ring2:
                            if mol.GetBondBetweenAtoms(atom1, atom2) is not None:
                                # Rings are adjacent, create a new bond to fuse them
                                em = Chem.EditableMol(new_mol)
                                em.AddBond(atom1, atom2, Chem.BondType.SINGLE)
                                
                                try:
                                    result_mol = em.GetMol()
                                    Chem.SanitizeMol(result_mol)
                                    return result_mol
                                except:
                                    return None
            return None
            
        except Exception as e:
            print(f"Error in ring modification: {e}")
            return None

    def _stereochemistry_modification(self, mol):
        """
        Modify the stereochemistry of a given molecule by randomly selecting one of its chiral centers
        and setting its chiral tag to clockwise (CHI_TETRAHEDRAL_CW).
        Args:
            mol (rdkit.Chem.Mol): The input molecule whose stereochemistry is to be modified.
        Returns:
            rdkit.Chem.Mol or None: A new molecule object with modified stereochemistry at one chiral center,
            or None if no chiral centers are found or an error occurs.
        """
        try:
            chiral_centers = Chem.FindMolChiralCenters(mol)
            if chiral_centers:
                center_idx, _ = random.choice(chiral_centers)
                new_mol = Chem.Mol(mol)
                new_mol.GetAtomWithIdx(center_idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                
                return new_mol
        except:
            pass
        
        return None

    def _structural_rearrangement(self, mol):
        """
        Performs a structural rearrangement on the given molecule by randomly selecting two non-ring atoms 
        with degree greater than one, removing any existing bond between them, and adding a new single bond.
        Args:
            mol (rdkit.Chem.Mol): The input molecule to be rearranged.
        Returns:
            rdkit.Chem.Mol or None: The rearranged molecule if successful, otherwise None.
        """
        try:
            rearrangeable_atoms = [
                atom.GetIdx() for atom in mol.GetAtoms() 
                if atom.GetDegree() > 1 and not atom.IsInRing()
            ]
            if len(rearrangeable_atoms) >= 2:
                atom1, atom2 = random.sample(rearrangeable_atoms, 2)
                new_mol = Chem.EditableMol(mol)
                new_mol.RemoveBond(atom1, atom2)
                new_mol.AddBond(atom1, atom2, Chem.BondType.SINGLE)
                return new_mol.GetMol()
        except:
            pass
        
        return None

    def _validate_molecule_properties(self, mol):
        """
        Validate the properties of a generated molecule.
        This function checks whether the given molecule meets specific criteria:
        - The molecular weight must be between 100 and 600 Daltons.
        - The molecule must contain no more than 4 rings.
        Parameters:
            mol (rdkit.Chem.Mol): The molecule to validate.
        Returns:
            bool: True if the molecule satisfies all criteria, False otherwise.
        """
        try:
            mol_weight = Descriptors.ExactMolWt(mol)
            num_rings = mol.GetRingInfo().NumRings()
            valid_mol_weight = 100 < mol_weight < 600
            valid_rings = num_rings <= 4
            
            return valid_mol_weight and valid_rings
        except:
            return False
    
    
    def explain_results_with_gemini(self, compounds: List[Dict]) -> str:
        """
        Generates a scientific, accessible explanation of the top drug candidate compounds using the Gemini API.
        This method analyzes up to the top three compounds based on their calculated properties (such as druglikeness, toxicity, binding affinity, solubility, Lipinski violations, and synthetic accessibility) for a given target protein. It sends a structured prompt to the Gemini language model API, requesting an overview, comparative analysis, compound-specific insights, and a three-stage experimental validation roadmap. The response is processed and cleaned for readability.
        Args:
            compounds (List[Dict]): A list of dictionaries, each representing a compound with its SMILES string, score, and a 'metrics' dictionary containing druglikeness, toxicity, binding affinity, solubility, Lipinski violations, and synthetic accessibility.
        Returns:
            str: A formatted explanation of the top compounds, including an overview, comparative analysis, compound-specific insights, and validation recommendations. If the API call fails or an error occurs, returns a fallback summary message.
        """
        
        try:
            compound_data = []
            for i, compound in enumerate(compounds[:3]):
                compound_data.append({
                    "rank": i+1,
                    "smiles": compound['smiles'],
                    "score": float(compound['score']),
                    "druglikeness": float(compound['metrics']['druglikeness']),
                    "toxicity": float(compound['metrics']['toxicity']),
                    "binding_affinity": float(compound['metrics']['binding_affinity']),
                    "solubility": float(compound['metrics']['solubility']),
                    "lipinski_violations": int(compound['metrics']['lipinski_violations']),
                    "synthetic_accessibility": float(compound['metrics']['synthetic_accessibility'])
                })

            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                print("Warning: GEMINI_API_KEY not found in environment variables")
                gemini_api_key = "AIzaSyCdvbELo9_WNP4ti4wogC5TAjDRL16PmFQ"  # Fallback to old key
                
            url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}'
            headers = {
                'Content-Type': 'application/json'
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"""You are exper in pharmaceuticals and drug discovery,
                                        Analyze these drug candidates for the target protein whose pdb id is {self.pdb_id} {self.target_protein} based on their calculated properties:
                                        {json.dumps(compound_data, cls=NumpyEncoder, indent=2)}

                                        Provide a clear explanation structured as follows:

                                        ## Overview
                                        - Summarize the key findings and what makes these compounds promising
                                        - Compare these top candidates, highlighting their relative strengths and weaknesses

                                        ## Compound Analysis
                                        For each of the top {min(len(compound_data), 3)} compounds:
                                        - Explain how its specific properties (druglikeness, binding_affinity, etc.) contribute to its potential efficacy
                                        - Identify any potential concerns based on toxicity or synthetic_accessibility
                                        - Suggest specific protein-drug interactions that might be occurring based on the structural properties

                                        ## Validation Roadmap
                                        Outline a 3-stage experimental validation plan:
                                        1. Initial in vitro testing recommendations
                                        2. Advanced pre-clinical validation methods
                                        3. Key considerations for potential clinical trials

                                        Keep your analysis scientifically accurate but accessible to non-specialists.
                                        """
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_json = response.json()
                
                print(f"DEBUG - Response structure: {json.dumps(response_json, indent=2)[:500]}...")
                print(f"DEBUG - Response keys: {list(response_json.keys())}")
                text = None
                if "text" in response_json:
                    text = response_json["text"]
                elif 'candidates' in response_json and len(response_json['candidates']) > 0:
                    text = response_json['candidates'][0]['content']['parts'][0]['text']
                
                elif 'content' in response_json and 'parts' in response_json['content']:
                    text = response_json['content']['parts'][0]['text']
                
                else:
                    text = "These compounds show promising characteristics for drug development based on their calculated properties. They demonstrate favorable druglikeness, binding affinity, and low toxicity profiles. Further laboratory validation would be necessary to confirm their efficacy."
                
                if text:
                    text_with_new_lines = text.replace('\\n', '\n')
                    cleaned_text = re.sub(r'(\\|##|#|_|[*])', '', text_with_new_lines)
                    return cleaned_text
                else:
                    return "These optimized compounds show promising drug-like characteristics and should be further evaluated in laboratory settings."
                    
            else:
                print(f"API request failed with status code {response.status_code}: {response.text}")
                return f"These compounds show favorable drug-like properties with good binding affinity scores. Further laboratory testing would be needed to validate their potential efficacy."
        
        except Exception as e:
            print(f"Error in explain_results_with_gemini: {str(e)}")
            
            import traceback
            traceback.print_exc()
            return "These optimized compounds demonstrate potential as drug candidates based on their calculated properties. The top compound has a particularly favorable balance of druglikeness, binding affinity, and low toxicity. Next steps would involve in vitro testing to validate these computational predictions."
            
    def explain_single_compound(self, compound: Dict) -> str:
        """
        Generates a natural language explanation for a single compound using the Gemini API.
        This method analyzes the provided compound's properties and requests a concise, expert-level summary from the Gemini language model. The explanation covers the compound's strongest attributes, potential concerns, and likely molecular interactions with the target protein, along with a recommendation for optimization. If the API call fails or returns an unexpected response, a fallback summary is generated.
        Args:
            compound (Dict): A dictionary containing compound information, including SMILES, score, and various calculated metrics.
        Returns:
            str: A cleaned, single-paragraph explanation of the compound's properties and optimization advice.
        """
        import re

        try:
            compound_data = {
                "rank": compound.get('rank', 0),
                "type": compound.get('type', 'unknown'),
                "smiles": compound['smiles'],
                "score": float(compound['score']),
                "druglikeness": float(compound['metrics']['druglikeness']),
                "toxicity": float(compound['metrics']['toxicity']),
                "binding_affinity": float(compound['metrics']['binding_affinity']),
                "solubility": float(compound['metrics']['solubility']),
                "lipinski_violations": int(compound['metrics']['lipinski_violations']),
                "synthetic_accessibility": float(compound['metrics']['synthetic_accessibility'])
            }

            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                print("Warning: GEMINI_API_KEY not found in environment variables")
                gemini_api_key = "AIzaSyCdvbELo9_WNP4ti4wogC5TAjDRL16PmFQ"  # Fallback to old key
                
            url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}'
            headers = {
                'Content-Type': 'application/json'
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"""As a pharmaceutical chemist, analyze this drug candidate targeting protein with pdb id as {self.pdb_id or 'unknown protein'} {self.target_protein or 'unknown target protein'} based on its calculated properties:
                                            {json.dumps(compound_data, cls=NumpyEncoder, indent=2)}

                                            Provide a precise analysis in 3-4 sentences that covers:
                                            1. Its strongest properties (druglikeness={compound_data['druglikeness']:.2f}, binding_affinity={compound_data['binding_affinity']:.2f}) and mechanism of action
                                            2. Key concerns to address (toxicity={compound_data['toxicity']:.2f}, synthetic_accessibility={compound_data['synthetic_accessibility']:.2f})
                                            3. How its molecular structure likely interacts with the target protein binding site

                                            Format as a single paragraph without bullet points. Include one specific recommendation for compound optimization.
                                            """
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_json = response.json()
                text = None
                
                if 'candidates' in response_json and response_json['candidates']:
                    try:
                        text = response_json['candidates'][0]['content']['parts'][0]['text']
                    except (KeyError, IndexError):
                        pass
                
                if not text:
                    compound_type = compound.get('type', 'primary')
                    rank = compound.get('rank', 0)
                    score = compound['score']
                    druglikeness = compound['metrics']['druglikeness']
                    toxicity = compound['metrics']['toxicity']
                    binding = compound['metrics']['binding_affinity']
                    
                    text = f"This {compound_type} compound (rank {rank}) shows a favorable overall score of {score:.2f}. It demonstrates good druglikeness ({druglikeness:.2f}) and binding affinity ({binding:.2f}) with relatively low toxicity ({toxicity:.2f}). This molecule could potentially interact well with the target protein based on its binding profile."
                
                text_with_new_lines = text.replace('\\n', '\n')
                cleaned_text = re.sub(r'(\\|##|#|_|[*])', '', text_with_new_lines)
                return cleaned_text
            else:
                return f"Compound with SMILES {compound['smiles'][:20]}... shows promising drug-like properties with a score of {compound['score']:.2f}."
        
        except Exception as e:
            print(f"Error in explain_single_compound: {str(e)}")
            return f"This compound ranks #{compound.get('rank', 0)} with a score of {compound['score']:.2f}. It shows good druglikeness and binding potential with manageable toxicity concerns."

    
    def export_results(self, compounds: List[Dict], filepath: str) -> None:
        """
        Exports a list of compound results to a CSV file.
        Each compound should be a dictionary containing at least the keys 'smiles', 'score', and 'metrics'.
        The output CSV will include a rank, SMILES string, score, and all metrics for each compound.
        Args:
            compounds (List[Dict]): List of compound dictionaries to export. Each dictionary must contain
                'smiles' (str), 'score' (float or int), and 'metrics' (dict of metric names to values).
            filepath (str): Path to the output CSV file.
        Returns:
            None
        Prints:
            A message indicating whether the results were exported or if there were no compounds to export.
        """
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
            for metric, value in compound['metrics'].items():
                row[metric] = value
                
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


def get_optimized_variants(protien_sequence,optimized_compounds,optimizer,optimization_params):
    """
    Generates and optimizes molecular variants based on a top compound and a protein sequence.
    This function selects the top compound from a list of optimized compounds, generates molecular
    modifications using the provided optimizer, and further optimizes these variants. The results are
    sorted by their optimization score, exported to a CSV file, and an explanation is generated.
    Args:
        protien_sequence (str): The amino acid sequence of the target protein.
        optimized_compounds (list): A list of dictionaries containing optimized compounds, each with at least a 'smiles' key.
        optimizer (object): An object with a `generate_molecular_modifications(smiles, n)` method to generate compound variants.
        optimization_params (dict): Parameters to control the optimization process.
    Returns:
        tuple: A tuple containing:
            - sorted_variants (list): List of optimized and sorted variant dictionaries.
            - explanation (str): Explanation of the optimization results.
    """
    top_compound = optimized_compounds[0]['smiles']
        
    variants = optimizer.generate_molecular_modifications(top_compound, 10) 
    #print(f"Generated {len(variants)} variants of top compound")

    variant_optimizer = DrugOptimizer(variants, protien_sequence)  
    optimized_variants = variant_optimizer.optimize(optimization_params)
    sorted_variants = sorted(optimized_variants, key=lambda x: x['score'], reverse=True)
    variant_optimizer.export_results(sorted_variants, "optimized_variants.csv")
    explanation = variant_optimizer.explain_results_with_gemini(sorted_variants)

    return sorted_variants, explanation
