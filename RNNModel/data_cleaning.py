from rdkit import Chem
from rdkit.Chem import MolStandardize
import numpy as np
import pandas as pd

def preprocess_smiles(smiles_list):
    clean_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            clean_smiles.append(Chem.MolToSmiles(mol))
    return clean_smiles

clean_smiles=[]
smiles=pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv')
smiles.drop(columns=['logP','qed','SAS'], inplace=True)
smiles_list = smiles['smiles'].tolist()

# Preprocess the SMILES strings
clean_smiles = preprocess_smiles(smiles_list)
print(clean_smiles[:5])