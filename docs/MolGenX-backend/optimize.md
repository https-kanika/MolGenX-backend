Module MolGenX-backend.optimize
===============================

Functions
---------

`get_optimized_variants(protien_sequence, optimized_compounds, optimizer, optimization_params)`
:   

Classes
-------

`DrugOptimizer(candidate_smiles: List[str], target_protein: str | None = None, pdb_id: str | None = None)`
:   A comprehensive drug candidate optimization pipeline that integrates 
    multiple objectives, ESM-2 for protein interaction, etc.
    
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

`NumpyEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)`
:   Extensible JSON <https://json.org> encoder for Python data structures.
    
    Supports the following objects and types by default:
    
    +-------------------+---------------+
    | Python            | JSON          |
    +===================+===============+
    | dict              | object        |
    +-------------------+---------------+
    | list, tuple       | array         |
    +-------------------+---------------+
    | str               | string        |
    +-------------------+---------------+
    | int, float        | number        |
    +-------------------+---------------+
    | True              | true          |
    +-------------------+---------------+
    | False             | false         |
    +-------------------+---------------+
    | None              | null          |
    +-------------------+---------------+
    
    To extend this to recognize other objects, subclass and implement a
    ``.default()`` method with another method that returns a serializable
    object for ``o`` if possible, otherwise it should call the superclass
    implementation (to raise ``TypeError``).
    
    Constructor for JSONEncoder, with sensible defaults.
    
    If skipkeys is false, then it is a TypeError to attempt
    encoding of keys that are not str, int, float or None.  If
    skipkeys is True, such items are simply skipped.
    
    If ensure_ascii is true, the output is guaranteed to be str
    objects with all incoming non-ASCII characters escaped.  If
    ensure_ascii is false, the output can contain non-ASCII characters.
    
    If check_circular is true, then lists, dicts, and custom encoded
    objects will be checked for circular references during encoding to
    prevent an infinite recursion (which would cause an RecursionError).
    Otherwise, no such check takes place.
    
    If allow_nan is true, then NaN, Infinity, and -Infinity will be
    encoded as such.  This behavior is not JSON specification compliant,
    but is consistent with most JavaScript based encoders and decoders.
    Otherwise, it will be a ValueError to encode such floats.
    
    If sort_keys is true, then the output of dictionaries will be
    sorted by key; this is useful for regression tests to ensure
    that JSON serializations can be compared on a day-to-day basis.
    
    If indent is a non-negative integer, then JSON array
    elements and object members will be pretty-printed with that
    indent level.  An indent level of 0 will only insert newlines.
    None is the most compact representation.
    
    If specified, separators should be an (item_separator, key_separator)
    tuple.  The default is (', ', ': ') if *indent* is ``None`` and
    (',', ': ') otherwise.  To get the most compact JSON representation,
    you should specify (',', ':') to eliminate whitespace.
    
    If specified, default is a function that gets called for objects
    that can't otherwise be serialized.  It should return a JSON encodable
    version of the object or raise a ``TypeError``.

    ### Ancestors (in MRO)

    * json.encoder.JSONEncoder

    ### Methods

    `default(self, obj)`
    :   Implement this method in a subclass such that it returns
        a serializable object for ``o``, or calls the base implementation
        (to raise a ``TypeError``).
        
        For example, to support arbitrary iterators, you could
        implement default like this::
        
            def default(self, o):
                try:
                    iterable = iter(o)
                except TypeError:
                    pass
                else:
                    return list(iterable)
                # Let the base class default method raise the TypeError
                return super().default(o)