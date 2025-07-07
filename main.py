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
#from CondRNN.model_loading import load_model as load_cond_rnn_model
#from CondRNN.conditionalRNN import generate_molecules
#vae
from VAEwithCondRNN.cycle_loading import load_model as load_vae_model
from VAEwithCondRNN.cycle import generate_molecules as generate_vae_molecules


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
  Endpoint to find optimized drug candidates based on input protein and parameters.
  
  Accepts:
    - pdb_id: PDB ID of the target protein
    - weights: Weights for different optimization properties
      druglikeness, synthetic_accessibility, lipinski_violations, toxicity, binding_affinity, solubility
    - num_compounds: Number of compounds to return (default: 20)
    - binding_affinity: Target binding affinity level (default: 0.7)
      note: should be between 0.1 and 1.0 as this is log normalized IC50 value
    - generate_visualizations: Whether to generate visualizations (default: False)
  
  Returns:
    JSON with optimized compounds, explanations, and optional visualizations
  """
  
  if not request.json:
    return jsonify({"error": "Missing input data"}), 400
  
  if 'pdb_id' not in request.json :
    return jsonify({"error": "Missing protein PDB ID"}), 400
  
  pdb_id = request.json.get('pdb_id')  
  generate_visualizations = request.json.get('generate_visualizations', False)
  num_compounds = min(50, max(1, int(request.json.get('num_compounds', 20))))  
  binding_affinity = min(1.0, max(0.1, float(request.json.get('binding_affinity', 0.7))))  
  molecules_per_level = max(40, num_compounds)
  
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

  # Load the new VAE + Conditional RNN model
  try:
        # Use the trained model path with exact parameters
        VAE_MODEL_PATH = os.path.join(
            Path(__file__).parent, 
            "VAEwithCondRNN", 
            "final_model", 
            "best_model.pt"
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading VAE model from: {VAE_MODEL_PATH}")
        
        # Load the model with training parameters
        model, protein_encoder, vae_encoder, vocab_data = load_vae_model(
            VAE_MODEL_PATH, 
            device, 
            use_affinity=True,  # Model was trained with affinity
            embed_dim=192,      # Match training parameters
            hidden_dim=384,
            output_dim=384,
            num_layers=4,
            latent_dim=64
        )
        
        # Generate molecules at the requested affinity only
        diverse_molecules = []
        
        print(f"Generating {molecules_per_level} molecules at affinity {binding_affinity}")
        
        molecules = generate_vae_molecules(
            model,
            protein_encoder,
            vae_encoder,
            protein_sequence,
            vocab_data,
            affinity_value=binding_affinity,
            num_molecules=molecules_per_level,
            device=device,
            temperature=0.8,
            max_attempts=3,
            latent_noise=0.3
        )
        diverse_molecules.extend(molecules)
        
        print(f"Generated {len(diverse_molecules)} initial molecules with VAE + Conditional RNN")
        
        # If we didn't get enough molecules, generate more with higher temperature
        if len(diverse_molecules) < molecules_per_level * 0.8:
            additional_needed = molecules_per_level - len(diverse_molecules)
            print(f"Generating {additional_needed} additional molecules with higher temperature...")
            additional_molecules = generate_vae_molecules(
                model,
                protein_encoder,
                vae_encoder,
                protein_sequence,
                vocab_data,
                affinity_value=binding_affinity,
                num_molecules=additional_needed,
                device=device,
                temperature=1.1,  # Higher temperature for more diversity
                max_attempts=5,
                latent_noise=0.5
            )
            diverse_molecules.extend(additional_molecules)
        
  except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error using VAE + Conditional RNN: {str(e)}. Falling back to basic RNN generator.")
    
    # Final fallback to basic RNN model
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
        num_molecules=molecules_per_level
    )
    # Fall back to the original Conditional RNN model
    """try:
        COND_RNN_MODEL_PATH = os.environ.get('COND_RNN_MODEL_PATH', 
                                           Path(__file__).parent / "CondRNN/models_550k/best_model.pt")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, protein_encoder, vocab_data = load_cond_rnn_model(COND_RNN_MODEL_PATH, device)
        
        # Generate molecules with the original model
        diverse_molecules = []
        
        # Primary generation at requested affinity (60% of molecules)
        molecules = generate_molecules(
            model,
            protein_encoder,
            protein_sequence,
            vocab_data,
            affinity_value=binding_affinity,
            num_molecules=int(molecules_per_level * 0.6),
            device=device,
            temperature=0.75,
            max_attempts=5
        )
        diverse_molecules.extend(molecules)
        
        # Generate some molecules at higher and lower affinities for diversity (20% each)
        lower_affinity = max(0.1, binding_affinity - 0.2)
        higher_affinity = min(0.95, binding_affinity + 0.2)
        
        for affinity in [lower_affinity, higher_affinity]:
            molecules = generate_molecules(
                model,
                protein_encoder,
                protein_sequence,
                vocab_data,
                affinity_value=affinity,
                num_molecules=int(molecules_per_level * 0.2),
                device=device,
                temperature=0.75,
                max_attempts=5
            )
            diverse_molecules.extend(molecules)
            
        print(f"Generated {len(diverse_molecules)} initial molecules with original Conditional RNN")
        
    except Exception as e2:
        import traceback
        traceback.print_exc()
        # Final fallback to basic RNN model
        print(f"Error using Conditional RNN: {str(e2)}. Falling back to basic RNN generator.")
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
            num_molecules=molecules_per_level
        )"""

  optimizer = DrugOptimizer(diverse_molecules, protein_sequence, pdb_id)
  weights = request.json['weights']
  
  # Process all generated molecules instead of limiting to top_n
  optimization_params = {
        'weights': weights,
        'top_n': len(diverse_molecules)  # Process all molecules
    }
  print(f"Optimizing {len(diverse_molecules)} molecules...")
  optimized_compounds = optimizer.optimize(optimization_params)
  
  # Generate variants for ALL optimized compounds, not just the top ones
  print(f"Generating variants for {len(optimized_compounds)} compounds...")
  all_compounds_with_variants = []
  
  for i, compound in enumerate(optimized_compounds):
    # Add the original compound
    all_compounds_with_variants.append({
        **compound,
        'type': 'primary',
        'parent_index': i,
        'variant_of': None
    })
    
    # Generate variants for this specific compound
    try:
        variants, _ = get_optimized_variants(
            protein_sequence, 
            [compound],  # Pass single compound
            optimizer, 
            optimization_params
        )
        
        # Add all variants with proper tracking
        for variant in variants:
            all_compounds_with_variants.append({
                **variant,
                'type': 'variant',
                'parent_index': i,
                'variant_of': compound['smiles']
            })
            
    except Exception as e:
        print(f"Error generating variants for compound {i}: {str(e)}")
        continue
  
  print(f"Total compounds with variants: {len(all_compounds_with_variants)}")
  
  # Sort ALL compounds (primary + variants) by their overall score
  all_compounds_with_variants.sort(key=lambda x: x['score'], reverse=True)
  
  # Select only the top N compounds as requested
  top_compounds = all_compounds_with_variants[:num_compounds]
  
  # Add ranking and requested affinity to final selection
  for i, compound in enumerate(top_compounds):
    compound['rank'] = i + 1
    compound['requested_affinity'] = binding_affinity
    
    # Add some analytics
    compound['total_candidates_evaluated'] = len(all_compounds_with_variants)
    compound['selection_percentile'] = round((1 - i / len(all_compounds_with_variants)) * 100, 2)
  
  print(f"Selected top {len(top_compounds)} compounds from {len(all_compounds_with_variants)} total candidates")
  
  # Export results
  optimizer.export_results(top_compounds, "top_compounds.csv")
  
  # Generate explanations for the selected compounds
  compound_explanations = {}
  for i, compound in enumerate(top_compounds):
    explanation = optimizer.explain_single_compound(compound)
    compound_explanations[f"compound_{i+1}"] = explanation
  
  overall_explanation = optimizer.explain_results_with_gemini(top_compounds[:min(3, len(top_compounds))])
  
  df = pd.read_csv("top_compounds.csv")
  serialized_compounds = df.to_json(orient="records")

  visualization_data = {}
  if generate_visualizations:
    visualize_simple(top_compounds, show_protein=True, pdb_id=pdb_id)
    visualization_data = get_compound_files("compound_visualizations")
  
  # Enhanced response with more analytics
  response = {
    "optimized_compounds": serialized_compounds,
    "explanation": overall_explanation,
    "compound_explanations": compound_explanations,
    "requested_parameters": {
        "num_compounds": num_compounds,
        "binding_affinity": binding_affinity
    },
    "optimization_stats": {
        "total_molecules_generated": len(diverse_molecules),
        "total_candidates_evaluated": len(all_compounds_with_variants),
        "primary_compounds": len([c for c in all_compounds_with_variants if c['type'] == 'primary']),
        "variant_compounds": len([c for c in all_compounds_with_variants if c['type'] == 'variant']),
        "final_selection": len(top_compounds),
        "selection_ratio": round(len(top_compounds) / len(all_compounds_with_variants) * 100, 2)
    }
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