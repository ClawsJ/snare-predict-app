import numpy as np

def extract_features(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    seq_len = len(sequence)

    # 建立實際長度的特徵矩陣
    features = np.zeros((seq_len, len(amino_acids)))

    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            features[i][aa_to_idx[aa]] = 1

    return features
