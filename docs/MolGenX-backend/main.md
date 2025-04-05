Module MolGenX-backend.main
===========================

Functions
---------

`find_optimized_candidates()`
:   Endpoint to find optimized drug candidates based on input protein and SMILES data.
    Input: target protein (can be either amino acid sequence or PDB ID), weights for optimization metrics should be like 
    'weights': {
              'druglikeness': 1.0,
              'synthetic_accessibility': 0.8,
              'lipinski_violations': 0.7,
              'toxicity': 1.2,
              'binding_affinity': 1.5,
              'solubility': 0.6
          },