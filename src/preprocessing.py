import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def create_sequences(data, seq_len):
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)


def preprocess_zone_data(data_dir, seq_len=10):

    zone_data = {}
    labels = None

    zones = [0, 1, 2, 3]

    for z in zones:

        df = pd.read_csv(f"{data_dir}/zone{z}.csv")

        df = df.select_dtypes(include=[np.number])

        if labels is None:
            labels = (df["ATTACK_FLAG"] == 1).astype(int).values

        feature_cols = [c for c in df.columns if c != "ATT_FLAG"]

        data = df[feature_cols].values

        # 🔥 NORMALIZATION
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        seqs = create_sequences(data, seq_len)

        zone_data[z] = torch.tensor(seqs, dtype=torch.float32)

    labels = labels[seq_len:]

    print("\nZone shapes:")
    for z in zones:
        print(f"Zone {z}: {zone_data[z].shape}")

    print("Labels:", np.unique(labels))

    return zone_data, labels, zones