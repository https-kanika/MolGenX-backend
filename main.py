import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from optimize import DrugOptimizer, get_optimized_variants
import pandas as pd
import numpy as np
from RnnClass import RNNGenerator, generate_diverse_molecules
from utils import return_vocabulary
import torch
from visualization import visualize_simple
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": os.environ.get('CORS_ORIGINS', '*'),
    "methods": ["POST", "OPTIONS", "GET"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}})

MODEL_PATH = os.environ.get('MODEL_PATH', Path(__file__).parent / "rnn_model.pth")


@app.route("/api/optimize", methods=["POST"])
def find_optimized_candidates():
  """
  Endpoint to find optimized drug candidates based on input protein and SMILES data.
  Input: target protein (can be either amino acid sequence or PDB ID), weights for optimization metrics should be like 
  'weights': {
            'druglikeness': 1.0,
            'synthetic_accessibility': 0.8,
            'lipinski_violations': 0.7,
            'toxicity': 1.2,
            'binding_affinity': 1.5,
            'solubility': 0.6
        },
  """
  if not request.json or 'protein' not in request.json:
    return jsonify({"error": "Missing protein data"}), 400
  
  
  pdb_id= request.json.get('pdb_id')
  protein_input = request.json.get('protein', None)

  if protein_input== None:
    try:
      pdb_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
      pdb_response = requests.get(pdb_url)
      pdb_response.raise_for_status()
      
      sequence_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
      sequence_response = requests.get(sequence_url)
      sequence_response.raise_for_status()

      protein_sequence = sequence_response.json().get('entity_poly', {}).get('pdbx_seq_one_letter_code', '')
      
      if not protein_sequence:
        return jsonify({"error": f"Could not retrieve sequence for PDB ID: {protein_input}"}), 400
      protein_sequence = protein_sequence.replace(" ", "").replace("\n", "")
    except Exception as e:
      return jsonify({"error": f"Failed to fetch sequence for PDB ID {protein_input}: {str(e)}"}), 400
  else:
    protein_sequence = protein_input

  
  char_to_idx, idx_to_char = return_vocabulary()
  device = torch.device("cpu")
  model = RNNGenerator(vocab_size=len(char_to_idx), embed_dim=128, hidden_dim=256)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
  model.to(device)
  diverse_molecules = generate_diverse_molecules(
        model, 
        char_to_idx, 
        idx_to_char, 
        device, 
        start_token="C", 
        num_molecules=5
    )

  
  #print("Generated Molecules:")
  #for mol in diverse_molecules:
    #print(mol)
  
  optimizer = DrugOptimizer(diverse_molecules, protein_sequence, pdb_id)
  weights =request.json['weights']
  optimization_params = {
        'weights': weights,
        'top_n': 10
    }

  optimized_compounds = optimizer.optimize(optimization_params)
  #TODO filteration on basis of druglikeness, synthetic accessibility, etc to be done in frontend

  optimizer.export_results(optimized_compounds, "optimized_drug_candidates.csv")
    
  #TODO Call visualization function
  explanation = optimizer.explain_results_with_gemini(optimized_compounds)
  optimized_variants, variants_explanation = get_optimized_variants(protein_sequence,optimized_compounds,optimizer,optimization_params)
  
  # Read the exported CSV instead of using the objects directly
  serialized_compounds = pd.read_csv("optimized_drug_candidates.csv").to_json(orient="records")

# For optimized_variants, either export to CSV first or create a serializable version
  if optimized_variants:
    # Option 1: Export variants to CSV and read back
    optimizer.export_results(optimized_variants, "optimized_variants.csv")
    serialized_variants = pd.read_csv("optimized_variants.csv").to_json(orient="records")
  else:
    serialized_variants = []

  visualize_simple(optimized_compounds, show_protein=False, pdb_id=pdb_id)
  visualize_simple(optimized_variants, show_protein=False, pdb_id=pdb_id)
    

  return jsonify({
    "optimized_compounds": serialized_compounds,
    "explanation": explanation,
    "optimized_variants": serialized_variants,
    "variants_explanation": variants_explanation
})
  #TODO Call visualization function

@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def handle_404_error(error):
    return jsonify({"error": "Resource not found"}), 404

if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))