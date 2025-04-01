import os
from flask import Flask
from optimize import DrugOptimizer
import pandas as pd
from RnnClass import RNNGenerator, generate_diverse_molecules
from utils import create_vocabulary,validate_molecule
import torch

app = Flask(__name__)

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"

@app.route("/api/optimize")
def find_optimized_candidates(request):
  #TODO Create utils function for getting the vocabulary and loading the model
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
  
  protein=request.data
  optimizer = DrugOptimizer(diverse_molecules, protein)
  #ibuprofen_mol = Chem.MolFromSmiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')
  #optimizer.predict_toxicity(ibuprofen_mol)
  #optimizer.predict_protein_structure()
    
  # Calculate metrics for the ibuprofen molecule
  #ibuprofen_metrics = optimizer.calculate_all_metrics(ibuprofen_mol)

  ## Create the compound in the expected format (list of dictionaries)
  #ibuprofen_compound = [{'smiles': Chem.MolToSmiles(ibuprofen_mol),molecule': ibuprofen_mol,'score': 0.5,  # Sample score, you might calculate this properly'metrics': ibuprofen_metrics}]

  # Now call explain_results_with_gemini with properly formatted input
  #optimizer.explain_results_with_gemini(ibuprofen_compound)
  #optimizer.visualize_molecules(ibuprofen_compound)
  # Define optimization parameters

  #TODO Accept this from request
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
  #TODO filteration on basis of druglikeness, synthetic accessibility, etc to be done in frontend

  # Export results
  optimizer.export_results(optimized_compounds, "optimized_drug_candidates.csv")
    
  #TODO Call visualization function
  #TODO call variants of optimized compounds
  explanation = optimizer.explain_results_with_gemini(optimized_compounds)
  return


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))