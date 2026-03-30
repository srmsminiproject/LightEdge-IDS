import torch
import torch.nn as nn
from models import GlobalModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_global(zone_data, labels, zones, adapters, epochs=20, batch_size=64):

    os.makedirs("models", exist_ok=True)

    for z in zones:
        adapters[z].eval()
        for p in adapters[z].parameters():
            p.requires_grad = False

    model = GlobalModel(num_zones=len(zones)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 🔥 imbalance
    pos = sum(labels)
    neg = len(labels) - pos
    pos_weight = torch.tensor(neg / pos).to(device)

    print(f"Global Pos weight: {pos_weight.item():.2f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    y = torch.tensor(labels, dtype=torch.float32).to(device)

    for ep in range(epochs):

        total = 0

        for i in range(0, len(y), batch_size):

            embs_batch = []

            for z in zones:
                xb = zone_data[z][i:i+batch_size].to(device)

                with torch.no_grad():
                    _, emb_seq = adapters[z](xb)   # [B, T, F]

                embs_batch.append(emb_seq)

            xg = torch.stack(embs_batch, dim=2)  # [B, T, Z, F]

            yb = y[i:i+batch_size]

            logits = model(xg)

            loss = loss_fn(logits.squeeze(), yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg_loss = total / (len(y) // batch_size)
        print(f"Global Epoch {ep}: Avg Loss={avg_loss:.4f}")

    torch.save(model.state_dict(), "../outputs/models/global.pth")

    return model