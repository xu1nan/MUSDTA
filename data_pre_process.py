import torch
import os
import json
import numpy as np
from transformers import EsmTokenizer, EsmModel
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from transformers import LongformerTokenizer, LongformerModel
from collections import OrderedDict



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model and tokenizer
target_model_name = "Pertrainmodel/ESM1b_t33_650M_UR50S"
target_model = EsmModel.from_pretrained(target_model_name).to(device)
target_tokenizer = EsmTokenizer.from_pretrained(target_model_name)
target_model.eval()

# target_model_name = "Pertrainmodel/ProteinBERTa"
# target_model = BertModel.from_pretrained(target_model_name).to(device)
# target_tokenizer = BertTokenizer.from_pretrained(target_model_name)
# target_model.eval()

# target_model_name = "Pertrainmodel/RoBERTa-base"
# target_model = RobertaModel.from_pretrained(target_model_name).to(device)
# target_tokenizer = RobertaTokenizerFast.from_pretrained(target_model_name)
# target_model.eval()

drug_model_name = "Pertrainmodel/RoBERTa-base"
drug_model = RobertaModel.from_pretrained(drug_model_name).to(device)
drug_tokenizer = RobertaTokenizerFast.from_pretrained(drug_model_name)
drug_model.eval()

# drug_model_name = "Pertrainmodel/ChemBERTa"
# drug_model = RobertaModel.from_pretrained(drug_model_name).to(device)
# drug_tokenizer = RobertaTokenizerFast.from_pretrained(drug_model_name)
# drug_model.eval()

# drug_model_name = "Pertrainmodel/MolFormer-XL"
# drug_model = AutoModel.from_pretrained(drug_model_name).to(device)
# drug_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
# drug_model.eval()


def get_target_features(proteins, feature_type="sequence"):
    target_sequence = OrderedDict()

    for protein_id, sequence in proteins.items():
        if feature_type == "sequence":
            size, embedding = target_to_sequence_feature_esm1b_esm1v_robreta(sequence)
            file_path = os.path.join(target_output_dir, f"{protein_id}_sequence_features.pt")
            torch.save(embedding, file_path)
            print(f"Saved sequence embedding for {protein_id} to {file_path}")
            target_sequence[protein_id] = (size, file_path)

        elif feature_type == "residue":
            residue_embedding = target_to_residue_features_esm1b_esm1v_robreta(sequence)
            file_path = os.path.join(residue_output_dir, f"{protein_id}_residue_features.npy")
            np.save(file_path, residue_embedding)
            print(f"Saved residue features for {protein_id} to {file_path}")
            target_sequence[protein_id] = file_path

    return target_sequence

window_size = 510
overlap_size = 256

def format_protein_sequence(seq):
    return " ".join(list(seq))

def sliding_window_sequence(sequence, window_size=1022, overlap_size=512):
    step_size = window_size - overlap_size
    windows = []
    for i in range(0, len(sequence), step_size):
        end = min(i + window_size, len(sequence))
        windows.append(sequence[i:end])
    return windows

def target_to_sequence_feature_esm1b_esm1v_robreta(pro_seq):
    windows = sliding_window_sequence(pro_seq, window_size=window_size, overlap_size=overlap_size)
    print(len(pro_seq))
    embeddings = []
    for window in windows:
        inputs = target_tokenizer(window, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = target_model(**inputs)
            window_embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze()
            embeddings.append(window_embedding)

    combined_embedding = torch.mean(torch.stack(embeddings), dim=0)
    size = len(pro_seq)
    print(combined_embedding.shape)
    return size, combined_embedding

def target_to_residue_features_esm1b_esm1v_robreta(pro_seq):
    windows = sliding_window_sequence(pro_seq, window_size=window_size, overlap_size=overlap_size)
    all_residue_features = []
    window_indices = []
    print(len(pro_seq))
    for i, window in enumerate(windows):
        inputs = target_tokenizer(window, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = target_model(**inputs)
            token_representations = outputs.last_hidden_state
            residue_features = token_representations[0, 1:-1].cpu().numpy()
            all_residue_features.append(residue_features)

        start = i * (window_size - overlap_size)
        end = start + len(window)
        window_indices.append((start, end))

    combined_residue_features = np.zeros((len(pro_seq), residue_features.shape[1]))
    for i, (window, (start, end)) in enumerate(zip(all_residue_features, window_indices)):
        window_length = end - start
        if window_length != len(window):
            if len(window) > window_length:
                window = window[:window_length]
            else:
                window = np.pad(window, ((0, window_length - len(window)), (0, 0)), mode='constant')
        combined_residue_features[start:end] = window
    print(combined_residue_features.shape)
    return combined_residue_features

def target_to_sequence_feature(pro_seq):
    # formatted_seq = format_protein_sequence(pro_seq) # protbert
    inputs = target_tokenizer(pro_seq, return_tensors="pt", add_special_tokens=True).to(device)

    size = len(pro_seq)
    with torch.no_grad():
        outputs = target_model(**inputs)
    sequence_embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze()

    return size, sequence_embedding

def target_to_residue_features(pro_seq):
    # formatted_seq = format_protein_sequence(pro_seq) # protbert
    inputs = target_tokenizer(pro_seq, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = target_model(**inputs)
        token_representations = outputs.last_hidden_state
    residue_features = token_representations[0, 1:-1].cpu().numpy()

    return residue_features


def get_drug_features(smiles):
    drug_features = OrderedDict()

    for drug_id, sequence in smiles.items():
        size, embedding = drug_to_sequence_feature(sequence)
        file_path = os.path.join(drug_output_dir, f"{drug_id}_sequence_features.pt")
        torch.save(embedding, file_path)
        print(f"Saved sequence embedding for {drug_id} to {file_path}")
        drug_features[drug_id] = (size, file_path)

    return drug_features

def drug_to_sequence_feature(smile_seq):
    inputs = drug_tokenizer(smile_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    size = len(smile_seq)
    with torch.no_grad():
        outputs = drug_model(**inputs)
    sequence_embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze()

    return size, sequence_embedding




if __name__ == '__main__':
    with open('data/davis/targets_pdb.txt', 'r') as file:
        target_data = file.read()
    target_dict = json.loads(target_data)

    with open('data/davis/drugs.txt', 'r') as file:
        drug_data = file.read()
    drug_dict = json.loads(drug_data)

    target_output_dir = "data/davis/target_sequence_embedding"
    residue_output_dir = "data/davis/target_residue_embedding"
    drug_output_dir = "data/davis/drug_sequence_embedding"
    os.makedirs(target_output_dir, exist_ok=True)
    os.makedirs(residue_output_dir, exist_ok=True)
    os.makedirs(drug_output_dir, exist_ok=True)

    residue_targets = get_target_features(target_dict, feature_type="residue")
    sequence_targets = get_target_features(target_dict, feature_type="sequence")
    # drug_features = get_drug_features(drug_dict)