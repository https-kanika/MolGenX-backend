import torch
from rdkit import Chem
import os
import argparse

def debug_generation(model_path, protein_sequence, device='cuda'):
    """Debug molecule generation with a trained model"""
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract vocabulary data
    vocab_data = checkpoint['vocab_data']
    
    # Print raw generation (skip model conditioning)
    start_token_idx = vocab_data['smiles_char_to_idx']['<START>']
    pad_token_idx = vocab_data['smiles_char_to_idx']['<PAD>']
    
    print("\nTesting raw string generation:")
    for i in range(5):
        # Generate a random string from the vocabulary
        generated = '<START>'
        for _ in range(50):  # Generate 50 characters
            # Randomly select the next character from vocabulary
            next_char_idx = torch.randint(len(vocab_data['smiles_char_to_idx']), (1,)).item()
            next_char = vocab_data['smiles_idx_to_char'].get(next_char_idx, '<PAD>')
            if next_char == '<PAD>':
                break
            generated += next_char
        
        # Check if it's a valid molecule
        smiles = generated.replace('<START>', '').replace('<PAD>', '')
        mol = Chem.MolFromSmiles(smiles)
        valid = mol is not None
        
        print(f"String {i+1}: {smiles}")
        print(f"Valid molecule: {valid}")
        if valid:
            print("SUCCESS! Valid molecule found.")
            
    print("\nCommon SMILES patterns test:")
    test_smiles = ['C', 'CC', 'CCO', 'c1ccccc1', 'CC(=O)O']
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        valid = mol is not None
        print(f"SMILES: {smiles}, Valid: {valid}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug molecule generation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    args = parser.parse_args()
    
    # Use a simple protein sequence
    protein_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    
    debug_generation(args.model_path, protein_seq, args.device)