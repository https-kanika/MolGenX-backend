import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import pandas as pd
import torch
from pathlib import Path
from RNNModel.RnnClass import RNNGenerator, generate_diverse_molecules
from optimize import DrugOptimizer, get_optimized_variants
from RNNModel.utils import return_vocabulary, get_compound_files
from visualization import visualize_simple
from CondRNN.model_loading import load_model as load_cond_rnn_model
from CondRNN.conditionalRNN import generate_molecules

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": os.environ.get('CORS_ORIGINS', '*'),
    "methods": ["POST", "OPTIONS", "GET"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}})

MODEL_PATH = os.environ.get('MODEL_PATH', Path(__file__).parent / "rnn_model.pth")

@app.route("/api/protein/<pdb_id>", methods=["GET"])
def get_protein_visualization(pdb_id):
    """
    Endpoint to retrieve protein visualization data for a given PDB ID.
    Returns the PDB structure and HTML visualization for the specified protein.
    
    :param pdb_id: The 4-character PDB ID for the protein
    :return: JSON with protein data and visualization
    """
    if not pdb_id or len(pdb_id) != 4:
        return jsonify({"error": "Invalid PDB ID. Please provide a valid 4-character PDB ID"}), 400
    
    output_dir = "protein_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        from visualization import visualize_simple
        visualize_simple(
            compounds=[], 
            show_protein=True, 
            pdb_id=pdb_id
        )
        
        metadata_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        metadata_response = requests.get(metadata_url)
        title = f"Protein {pdb_id}"
        description = ""
        experimental_method = ""
        resolution = ""
        
        if metadata_response.status_code == 200:
            try:
                metadata = metadata_response.json()
                
                if isinstance(metadata, dict):
                    struct_data = metadata.get('struct', {})
                    exptl_data = metadata.get('exptl', [{}])[0] if isinstance(metadata.get('exptl'), list) else {}
                    
                    if isinstance(struct_data, dict):
                        title = struct_data.get('title', title)
                        description = struct_data.get('pdbx_descriptor', description)
                    
                    if isinstance(exptl_data, dict):
                        experimental_method = exptl_data.get('method', experimental_method)
                        resolution = exptl_data.get('resolution', resolution)
            except Exception as e:
                print(f"Error parsing metadata: {str(e)}")
        
        protein_pdb_filename = f"compound_visualizations/target_protein.pdb"
        protein_html_filename = f"compound_visualizations/target_protein.html"
        
        if not os.path.exists(protein_pdb_filename) or not os.path.exists(protein_html_filename):
            return jsonify({"error": "Failed to generate protein visualization"}), 500
        
        with open(protein_pdb_filename, 'r') as f:
            pdb_content = f.read()
            
        with open(protein_html_filename, 'r') as f:
            html_content = f.read()
        
        response_data = {
            "pdb_id": pdb_id,
            "title": title,
            "description": description,
            "experimental_method": experimental_method,
            "resolution": resolution,
            "pdb_content": pdb_content,
            "html_viewer": html_content,
            "visualization_url": f"https://3dmol.org/viewer.html?pdb={pdb_id}&style=cartoon",
            "download_url": f"https://files.rcsb.org/download/{pdb_id}.pdb"
        }
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing protein visualization: {str(e)}"}), 500


@app.route("/api/optimize", methods=["POST"])
def find_optimized_candidates():
  """
  """
  
  if not request.json:
    return jsonify({"error": "Missing input data"}), 400
  
  if 'pdb_id' not in request.json :
    return jsonify({"error": "Missing protein PDB ID"}), 400
  
  pdb_id = request.json.get('pdb_id')  
  generate_visualizations = request.json.get('generate_visualizations', False)

  # Get protein sequence from PDB ID 
  try:
      pdb_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
      pdb_response = requests.get(pdb_url)
      pdb_response.raise_for_status()
      
      sequence_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
      sequence_response = requests.get(sequence_url)
      sequence_response.raise_for_status()

      protein_sequence = sequence_response.json().get('entity_poly', {}).get('pdbx_seq_one_letter_code', '')
      
      if not protein_sequence:
        return jsonify({"error": f"Could not retrieve sequence for PDB ID: {pdb_id}"}), 400
      protein_sequence = protein_sequence.replace(" ", "").replace("\n", "")
  except Exception as e:
      return jsonify({"error": f"Failed to fetch sequence for PDB ID {pdb_id}: {str(e)}"}), 400

  # Load Conditional RNN model and generate molecules
  try:
        COND_RNN_MODEL_PATH = os.environ.get('COND_RNN_MODEL_PATH', 
                                           Path(__file__).parent / "CondRNN/models/best_model.pt")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, protein_encoder, vocab_data = load_cond_rnn_model(COND_RNN_MODEL_PATH, device)
        
        # Generate molecules with various affinity levels for diversity
        diverse_molecules = []
        for affinity in [0.5, 0.7, 0.9]:
            molecules = generate_molecules(
                model,
                protein_encoder,
                protein_sequence,
                vocab_data,
                affinity_value=affinity,
                # Generate enough molecules for optimization
                num_molecules=30,  # Generate 30 per affinity level = 90 total
                device=device,
                temperature=0.75,  # Slightly lower temperature for more valid structures
                max_attempts=5
            )
            diverse_molecules.extend(molecules)
            
        print(f"Generated {len(diverse_molecules)} initial molecules with Conditional RNN")
        
  except Exception as e:
    import traceback
    traceback.print_exc()
    # Fall back to the original RNN model if there's an error
    print(f"Error using Conditional RNN: {str(e)}. Falling back to original RNN generator.")
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
        num_molecules=90
        )

  optimizer = DrugOptimizer(diverse_molecules, protein_sequence, pdb_id)
  weights = request.json['weights']
  optimization_params = {
        'weights': weights,
        'top_n': 10
    }

  optimized_compounds = optimizer.optimize(optimization_params)
  optimized_variants, variants_explanation = get_optimized_variants(protein_sequence, optimized_compounds, optimizer, optimization_params)
  
  all_compounds = optimized_compounds + optimized_variants
  all_compounds.sort(key=lambda x: x['score'], reverse=True)
  top_compounds = all_compounds[:20]
  for i, compound in enumerate(top_compounds):
    compound['rank'] = i + 1
    compound['type'] = 'primary' if compound in optimized_compounds else 'variant'
  
  optimizer.export_results(top_compounds, "top_compounds.csv")
  
  compound_explanations = {}
  for i, compound in enumerate(top_compounds):
    explanation = optimizer.explain_single_compound(compound)
    compound_explanations[f"compound_{i+1}"] = explanation
  
  overall_explanation = optimizer.explain_results_with_gemini(top_compounds[:3])
  
  df = pd.read_csv("top_compounds.csv")
  serialized_compounds = df.to_json(orient="records")

  visualization_data = {}
  if generate_visualizations:
    visualize_simple(top_compounds, show_protein=True, pdb_id=pdb_id)
    visualization_data = get_compound_files("compound_visualizations")
  
  response = {
    "optimized_compounds": serialized_compounds,
    "explanation": overall_explanation,
    "compound_explanations": compound_explanations,
  }
  
  if generate_visualizations:
    response["compound_visualization"] = visualization_data
  
  return jsonify(response)


@app.route('/', methods=['GET'])
def index():
    """
    Base endpoint returning a simple HTML welcome page
    """
    return render_template('index.html')

@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def handle_404_error(error):
    return jsonify({"error": "Resource not found"}), 404

if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))