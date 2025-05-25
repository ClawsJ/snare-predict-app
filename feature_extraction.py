import numpy as np

# 假設 one-hot encoding，或替換為你的特徵邏輯
def extract_features(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    max_len = 100

    features = np.zeros((max_len, len(amino_acids)))
    for i, aa in enumerate(sequence[:max_len]):
        if aa in aa_to_idx:
            features[i][aa_to_idx[aa]] = 1
    return features
