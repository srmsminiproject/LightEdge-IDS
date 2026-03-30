import torch
import torch.nn as nn
from models import ZoneAdapter
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_adapters(zone_data, labels, zones, epochs=20, batch_size=64):

    os.makedirs("models", exist_ok=True)

    adapters = {}

    y = torch.tensor(labels, dtype=torch.float32).to(device)

    # 🔥 class imbalance
    pos = sum(labels)
    neg = len(labels) - pos
    pos_weight = torch.tensor(neg / pos).to(device)

    print(f"Adapter Pos weight: {pos_weight.item():.2f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for z in zones:

        model = ZoneAdapter(zone_data[z].shape[-1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        X = zone_data[z]

        for ep in range(epochs):

            total = 0

            for i in range(0, len(X), batch_size):

                xb = X[i:i+batch_size].to(device)
                yb = y[i:i+batch_size]

                logits, _ = model(xb)

                loss = loss_fn(logits.squeeze(), yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total += loss.item()

            avg_loss = total / (len(X) // batch_size)
            print(f"Zone {z} Epoch {ep}: Avg Loss={avg_loss:.4f}")

        torch.save(model.state_dict(), f"../outputs/models/adapter_{z}.pth")
        adapters[z] = model

    return adapters