import torch
import torch.nn as nn


class ZoneAdapter(nn.Module):
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()

        # 🔥 Increased capacity
        self.gru = nn.GRU(input_dim, 128, batch_first=True)

        self.fc_embed = nn.Linear(128, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, T, F]

        out, _ = self.gru(x)              # [B, T, 128]

        emb_seq = self.fc_embed(out)      # [B, T, embed_dim]

        logits = self.classifier(emb_seq[:, -1])

        return logits, emb_seq


class SimpleGNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class GlobalModel(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64, num_zones=4):
        super().__init__()

        self.gnn = SimpleGNN(embed_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * num_zones, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, Z, F]

        B, T, Z, F = x.shape

        out_seq = []
        for t in range(T):
            xt = x[:, t]              # [B, Z, F]
            xt = self.gnn(xt)         # [B, Z, H]
            out_seq.append(xt)

        x = torch.stack(out_seq, dim=1)   # [B, T, Z, H]
        x = x.view(B, T, -1)

        out, _ = self.gru(x)
        out = out[:, -1]

        return self.fc(out)