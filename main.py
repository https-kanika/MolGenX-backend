import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from optimize import DrugOptimizer, get_optimized_variants
import pandas as pd
import numpy as np
from RnnClass import RNNGenerator, generate_diverse_molecules
from utils import return_vocabulary, get_compound_files
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
  Input: 
    - target protein (can be either amino acid sequence or PDB ID)
    - weights for optimization metrics
    - generate_visualizations (optional boolean, default=True): whether to generate compound visualizations
  
  Example payload:
  {
    "pdb_id": "1M17",
    "protein": "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREA...",
    "weights": {
      "druglikeness": 1.0,
      "synthetic_accessibility": 0.8,
      "lipinski_violations": 0.7,
      "toxicity": 1.2,
      "binding_affinity": 1.5,
      "solubility": 0.6
    },
    "generate_visualizations": true
  }
  
  Output: optimized drug candidates and their properties.
  When visualizations are requested, the images are Base64 encoded and sent as data URLs,
  The PDB and SDF files are sent as plain text in the JSON response.
  """
  
  if not request.json or 'protein' not in request.json:
    return jsonify({"error": "Missing protein data"}), 400
  
  
  pdb_id= request.json.get('pdb_id')
  protein_input = request.json.get('protein', None)
  
  generate_visualizations = request.json.get('generate_visualizations', False)

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
        num_molecules=10  #modify this to generate more molecules, idealy should be in 1000s
    )

  optimizer = DrugOptimizer(diverse_molecules, protein_sequence, pdb_id)
  weights =request.json['weights']
  optimization_params = {
        'weights': weights,
        'top_n': 10
    }

  optimized_compounds = optimizer.optimize(optimization_params)

  optimizer.export_results(optimized_compounds, "optimized_drug_candidates.csv")
    
  explanation = optimizer.explain_results_with_gemini(optimized_compounds)
  optimized_variants, variants_explanation = get_optimized_variants(protein_sequence,optimized_compounds,optimizer,optimization_params)

  serialized_compounds = pd.read_csv("optimized_drug_candidates.csv").to_json(orient="records")

  if optimized_variants:
    optimizer.export_results(optimized_variants, "optimized_variants.csv")
    serialized_variants = pd.read_csv("optimized_variants.csv").to_json(orient="records")
  else:
    serialized_variants = []

  visualization_data = {}
  if generate_visualizations:
    all_compounds = optimized_compounds + optimized_variants
    visualize_simple(all_compounds, show_protein=True, pdb_id=pdb_id)
    visualization_data = get_compound_files("compound_visualizations")

  # Build response based on visualization parameter
  response = {
    "optimized_compounds": serialized_compounds,
    "explanation": explanation,
    "optimized_variants": serialized_variants,
    "variants_explanation": variants_explanation
  }
  
  # Only include visualization data if it was generated
  if generate_visualizations:
    response["compound_visualization"] = visualization_data

  return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    """
    Base endpoint returning a simple HTML welcome page
    """
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MolGenX Backend</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
                background-color: #f9f9f9;
            }
            .container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
                margin-top: 40px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .status {
                display: inline-block;
                background-color: #2ecc71;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            .endpoints {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-top: 25px;
            }
            code {
                background-color: #f1f1f1;
                padding: 3px 5px;
                border-radius: 3px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MolGenX Backend API</h1>
            <p>You've successfully landed at the MolGenX backend server.</p>
            <p>Status: <span class="status">Running</span></p>
            
            <div class="endpoints">
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><code>POST /api/optimize</code> - Optimize drug candidates based on protein targets</li>
                </ul>
            </div>
            
            <p>For more information, please refer to the <a href="https://github.com/https-kanika/MolGenX-backend">documentation</a>.</p>
        </div>
    </body>
    </html>
    """
    return html


@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def handle_404_error(error):
    return jsonify({"error": "Resource not found"}), 404

if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))