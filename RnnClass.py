import torch
import torch.nn as nn
from typing import List, Set
from utils import validate_molecule

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