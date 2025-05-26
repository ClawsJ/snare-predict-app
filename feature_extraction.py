import numpy as np

def extract_features(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    max_len = 500

    features = np.zeros((max_len, len(amino_acids)))

    # 截斷序列長度，避免超過 max_len
    seq = sequence[:max_len]

    # 將序列逐字 one-hot 編碼
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            features[i][aa_to_idx[aa]] = 1

    # 如果序列長度不足 max_len，剩下的部分已是 0，代表 padding
    return features
