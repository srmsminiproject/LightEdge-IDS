import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from models import ZoneAdapter, GlobalModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "../outputs/analysis_results"


def save_confusion_matrix(cm, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title(name)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig(f"{OUTPUT_DIR}/{name}.png")
    plt.close()


def evaluate(zone_data, labels, zones, model_dir="../outputs/models"):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------
    # LOAD MODELS
    # -----------------------
    adapters = {}
    for z in zones:
        m = ZoneAdapter(zone_data[z].shape[-1]).to(device)
        m.load_state_dict(torch.load(f"{model_dir}/adapter_{z}.pth"))
        m.eval()
        adapters[z] = m

    global_model = GlobalModel(num_zones=len(zones)).to(device)
    global_model.load_state_dict(torch.load(f"{model_dir}/global.pth"))
    global_model.eval()

    # -----------------------
    # STORAGE
    # -----------------------
    y_true = []
    global_preds = []
    zone_preds = []

    # -----------------------
    # INFERENCE LOOP
    # -----------------------
    for i in range(len(labels)):

        embs = []
        zone_probs = []

        for z in zones:
            x = zone_data[z][i].unsqueeze(0).to(device)  # [1, T, F]

            with torch.no_grad():
                logits, emb = adapters[z](x)

            prob = torch.sigmoid(logits).item()
            zone_probs.append(prob)

            # 🔥 FIX: restore time dimension
            T = x.shape[1]
            logits, emb_seq = adapters[z](x)
            embs.append(emb_seq)
            
        # 🔥 stack → [1, T, Z, F]
        xg = torch.stack(embs, dim=2)

        with torch.no_grad():
            g_prob = torch.sigmoid(global_model(xg)).item()

        y_true.append(int(labels[i]))
        global_preds.append(int(g_prob > 0.5))
        zone_preds.append(int(np.mean(zone_probs) > 0.5))

    # -----------------------
    # METRICS
    # -----------------------
    cm_global = confusion_matrix(y_true, global_preds)
    cm_zone = confusion_matrix(y_true, zone_preds)

    tn_g, fp_g, fn_g, tp_g = cm_global.ravel()
    tn_z, fp_z, fn_z, tp_z = cm_zone.ravel()

    results = {
        "global": {
            "accuracy": round(accuracy_score(y_true, global_preds), 3),
            "precision": round(precision_score(y_true, global_preds, zero_division=0), 3),
            "recall": round(recall_score(y_true, global_preds), 3),
            "f1": round(f1_score(y_true, global_preds), 3),
            "TP": int(tp_g),
            "TN": int(tn_g),
            "FP": int(fp_g),
            "FN": int(fn_g)
        },
        "zone_avg": {
            "accuracy": round(accuracy_score(y_true, zone_preds), 3),
            "precision": round(precision_score(y_true, zone_preds, zero_division=0), 3),
            "recall": round(recall_score(y_true, zone_preds), 3),
            "f1": round(f1_score(y_true, zone_preds), 3),
            "TP": int(tp_z),
            "TN": int(tn_z),
            "FP": int(fp_z),
            "FN": int(fn_z)
        }
    }

    # -----------------------
    # PRINT
    # -----------------------
    print("\nFinal Results:")
    print(results)

    print("\nConfusion Matrix (Global):")
    print(cm_global)

    print("\nConfusion Matrix (Zone Avg):")
    print(cm_zone)

    # -----------------------
    # SAVE
    # -----------------------
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    save_confusion_matrix(cm_global, "global_confusion_matrix")
    save_confusion_matrix(cm_zone, "zone_confusion_matrix")