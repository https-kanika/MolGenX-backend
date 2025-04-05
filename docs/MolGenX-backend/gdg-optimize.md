Module MolGenX-backend.gdg-optimize
===================================

Functions
---------

`create_vocabulary(smiles_data)`
:   Create character vocabulary from SMILES strings

`generate_diverse_molecules(model: torch.nn.modules.module.Module, char_to_idx: dict, idx_to_char: dict, device: torch.device, start_token: str = 'C', num_molecules: int = 10, max_length: int = 100, max_attempts: int = 100) ‑> List[str]`
:   Generate multiple unique and valid molecules
    
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

`validate_molecule(smiles: str) ‑> bool`
:   Validate if the generated SMILES represents a valid molecule
    
    Args:
        smiles (str): SMILES string to validate
    
    Returns:
        bool: True if molecule is valid, False otherwise

`visualize_simple(compounds, show_protein=True)`
:   

Classes
-------

`DrugOptimizer(candidate_smiles: List[str], target_protein: str | None = None)`
:   A comprehensive drug candidate optimization pipeline that integrates 
    multiple objectives, ESM-2 for protein interaction, and Google Cloud services.
    
    Initialize the DrugOptimizer with a list of candidate SMILES and optionally a target protein.
    
    Args:
        candidate_smiles: List of SMILES strings representing drug candidates
        target_protein: Optional sequence of the target protein

    ### Methods

    `calculate_all_metrics(self, mol) ‑> Dict`
    :   Calculate all drug metrics for a molecule

    `calculate_druglikeness(self, mol) ‑> float`
    :   Calculate QED (Quantitative Estimate of Drug-likeness)

    `calculate_lipinski_violations(self, mol) ‑> int`
    :   Check how many Lipinski's Rule of Five violations exist

    `calculate_objective_score(self, mol, weights: Dict[str, float]) ‑> float`
    :   Calculate weighted score based on multiple objectives
        
        Args:
            mol: RDKit molecule object
            weights: Dictionary of weights for each metric
                     e.g. {'druglikeness': 1.0, 'toxicity': -1.0, ...}
        
        Returns:
            float: Overall weighted score (higher is better)

    `calculate_solubility(self, mol) ‑> float`
    :   Estimate aqueous solubility (logS)

    `calculate_synthetic_accessibility(self, mol)`
    :   A more comprehensive synthetic accessibility estimation

    `estimate_binding_affinity(self, mol) ‑> float`
    :   Estimate binding affinity using BioNeMo's DiffDock model

    `explain_results_with_gemini(self, compounds: List[Dict]) ‑> str`
    :   Use Gemini API to explain the results in natural language

    `export_results(self, compounds: List[Dict], filepath: str) ‑> None`
    :   Export results to CSV file

    `filter_candidates(self, filters: Dict[str, Tuple[float, float]] = None, compounds: List[Dict] = None) ‑> List[Dict]`
    :   Filter candidates based on property ranges
        
        Args:
            filters: Dictionary of property filters with (min, max) tuple values
                    e.g. {'druglikeness': (0.5, 1.0), 'toxicity': (0, 0.3)}
            compounds: List of compounds to filter (if None, uses optimized compounds)
        
        Returns:
            List of filtered compounds

    `generate_molecular_modifications(self, smiles: str, num_variants: int = 50) ‑> List[str]`
    :   Generate structural variations using MolGPT

    `optimize(self, optimization_parameters: Dict = None) ‑> List[Dict]`
    :   Perform multi-objective optimization on the drug candidates
        
        Args:
            optimization_parameters: Dictionary with optimization parameters
                                    including weights for different objectives
        
        Returns:
            List of dictionaries with optimized molecules and their scores

    `predict_protein_structure(self)`
    :   Predict structure information for the target protein using ESM-2

    `predict_toxicity(self, mol) ‑> float`
    :   Predict toxicity using IBM's MoLFormer-XL model or fallback to structural alerts

    `visualize_molecules(self, compounds: List[Dict]) ‑> None`
    :   Generate visualization for top compounds
        In a real implementation, this would connect to IDX for 3D visualization

`RNNGenerator(vocab_size, embed_dim, hidden_dim)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.