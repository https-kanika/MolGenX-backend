import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from RnnClass import RNNGenerator
from utils import return_vocabulary
from data_cleaning import preprocess_smiles
import pandas as pd
import numpy as np

clean_smiles=[]
smiles=pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv')
smiles.drop(columns=['logP','qed','SAS'], inplace=True)
smiles_list = smiles['smiles'].tolist()

# Preprocess the SMILES strings
clean_smiles = preprocess_smiles(smiles_list)
print(clean_smiles[:5])
pd.DataFrame(clean_smiles, columns=["smiles"]).to_csv("cleaned_smiles.csv", index=False)

char_to_idx, idx_to_char = return_vocabulary()
max_length = max(len(smiles) for smiles in clean_smiles)

def smiles_to_sequence(smiles):
    """
    Converts a SMILES string into a sequence of integer indices, padding the result to a fixed maximum length.

    Args:
        smiles (str): The SMILES string to be converted.

    Returns:
        list[int]: A list of integer indices representing the SMILES string, padded with zeros to match max_length.

    Raises:
        KeyError: If a character in the SMILES string is not found in char_to_idx.
    """
    return [char_to_idx[char] for char in smiles] + [0] * (max_length - len(smiles))

sequences = np.array([smiles_to_sequence(smi) for smi in clean_smiles])


model = RNNGenerator(vocab_size=len(char_to_idx), embed_dim=128, hidden_dim=256)

train_data = torch.tensor(sequences, dtype=torch.long)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    total_loss = 0
    for i in tqdm(range(0, len(train_data), batch_size)):
        batch = train_data[i : i + batch_size].to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]  # Shifted sequence
        outputs = model(inputs)
        
        loss = criterion(outputs.view(-1, len(char_to_idx)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_data)}")

# Save model
torch.save(model.state_dict(), "rnn_model.pth")
