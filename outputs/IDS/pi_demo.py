"""
pi_demo.py  —  LightEdge IDS  —  Raspberry Pi edge inference

GPIO outputs (both LEDs, no buzzer)
-------------------------------------
  LED_ZONE   (pin 17, YELLOW) : Zone 1 alert  — active zone classifier fired
  LED_GLOBAL (pin 27, RED)    : Global alert  — temporal GRU confirmed attack

Two-LED convention
------------------
  YELLOW on, RED off  → Zone 1 suspicious, global model not yet confirmed
  YELLOW on, RED on   → Confirmed attack (both levels agree)
  YELLOW off, RED on  → Global model triggered without zone-level precursor
                        (rare; possible on first SEQ window boundary)
  Both off            → Normal operation

Files needed on Pi
------------------
    outputs/ts/zone_{0..3}_ts.pt
    outputs/ts/global_model_ts.pt
    outputs/scalers/scaler_zone{0..3}.pkl
    outputs/zone_{0..3}_meta.pt
    data/zone{0..3}.csv
"""

import os, time
import numpy as np
import pandas as pd
import torch
import joblib
from collections import deque

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] RPi.GPIO not available — simulation mode")

# ── GPIO pin assignments ──────────────────────────────────────────────────────
LED_ZONE   = 17   # Yellow LED — Zone 1 (active zone) alert
LED_GLOBAL = 27   # Red LED   — Global model alert

N_ZONES       = 4
SEQ           = 10
GLOBAL_THRESH = 0.35
STEP_MS       = 100

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    for pin in [LED_ZONE, LED_GLOBAL]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

# ── Metadata loader ───────────────────────────────────────────────────────────
def load_meta(z):
    p = f"outputs/zone_{z}_meta.pt"
    if os.path.exists(p):
        return torch.load(p, weights_only=False)
    return {"label_col": "ATTACK_FLAG", "continuous_cols": [], "binary_cols": [],
            "threshold": 0.999, "in_dim": 1, "is_active": False}

metas = [load_meta(z) for z in range(N_ZONES)]

print("Zone roles:")
for z in range(N_ZONES):
    m   = metas[z]
    tag = f"ACTIVE  thresh={m['threshold']:.2f}" if m["is_active"] else "CONTEXT"
    print(f"  Zone {z}: {tag}  features={m['in_dim']}")

print("\nGPIO mapping:")
print(f"  Pin {LED_ZONE:2d} (Yellow LED) → Zone 1 alert")
print(f"  Pin {LED_GLOBAL:2d} (Red LED)    → Global alert")

# ── Load TorchScript models + scalers ─────────────────────────────────────────
zone_models, zone_scalers = [], []
for z in range(N_ZONES):
    zm = torch.jit.load(f"outputs/ts/zone_{z}_ts.pt")
    zm.eval()
    zone_models.append(zm)
    zone_scalers.append(joblib.load(f"outputs/scalers/scaler_zone{z}.pkl"))

global_model = torch.jit.load("outputs/ts/global_model_ts.pt")
global_model.eval()

# ── Load CSVs ─────────────────────────────────────────────────────────────────
dfs = [pd.read_csv(f"data/zone{z}.csv", sep=None, engine="python")
       for z in range(N_ZONES)]
n_steps = min(len(df) for df in dfs)

buffer: deque = deque(maxlen=SEQ)
stats = dict(total=0, tp=0, fp=0, fn=0, tn=0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def led_set(pin, state):
    if GPIO_AVAILABLE:
        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)

def preprocess(row, scaler, cont_cols, bin_cols):
    cont = row[cont_cols].values.reshape(1, -1).astype(np.float32) if cont_cols else None
    binr = row[bin_cols].values.reshape(1, -1).astype(np.float32)  if bin_cols  else None
    if cont is not None:
        cont = scaler.transform(cont)
    parts = [p for p in [cont, binr] if p is not None]
    x = np.hstack(parts) if parts else np.zeros((1, 1), np.float32)
    return torch.tensor(x, dtype=torch.float32)

# ── Inference loop ────────────────────────────────────────────────────────────
print(f"\nLightEdge IDS running — {n_steps:,} timesteps")
print("-" * 60)

try:
    for t in range(n_steps):
        z1_prob  = 0.0
        z1_alert = False
        embs_t   = []

        # Run all zone models — all 4 produce embeddings for global model
        for z in range(N_ZONES):
            m = metas[z]
            x = preprocess(dfs[z].iloc[t], zone_scalers[z],
                           m["continuous_cols"], m["binary_cols"])
            with torch.no_grad():
                z_logit, emb, _ = zone_models[z](x)
            z_prob = torch.sigmoid(z_logit).item()

            # Only Zone 1 (is_active) raises a zone-level alert
            if m["is_active"]:
                z1_prob  = z_prob
                z1_alert = z_prob > m["threshold"]

            embs_t.append(emb.squeeze(0))

        # Yellow LED — Zone 1 alert
        led_set(LED_ZONE, z1_alert)

        # Update rolling buffer with all 4 zone embeddings
        buffer.append(torch.stack(embs_t, dim=0))   # [N_ZONES, EMB_DIM]

        # Global model inference (once buffer is full)
        g_alert = False
        g_prob  = 0.0
        if len(buffer) == SEQ:
            seq_t = torch.stack(list(buffer), dim=0).unsqueeze(0)  # [1,SEQ,Z,D]
            with torch.no_grad():
                g_prob = torch.sigmoid(global_model(seq_t)).item()
            g_alert = g_prob > GLOBAL_THRESH

        # Red LED — Global alert
        led_set(LED_GLOBAL, g_alert)

        # Ground truth + metrics
        gt   = int(any(int(dfs[z].iloc[t][metas[z]["label_col"]]) == 1
                       for z in range(N_ZONES)))
        pred = int(g_alert) if len(buffer) == SEQ else int(z1_alert)

        stats["total"] += 1
        if   gt == 1 and pred == 1: stats["tp"] += 1
        elif gt == 0 and pred == 1: stats["fp"] += 1
        elif gt == 1 and pred == 0: stats["fn"] += 1
        else:                        stats["tn"] += 1

        # LED state description for console
        led_state = ("BOTH" if (z1_alert and g_alert) else
                     "YELLOW" if z1_alert else
                     "RED"    if g_alert  else
                     "off")

        print(f"[t={t:6d}] Z1={'ATK' if z1_alert else 'NRM'}({z1_prob:.3f}) || "
              f"Global={'ATTACK' if g_alert else 'NORMAL'} "
              f"(p={g_prob:.3f}) | LEDs={led_state} | GT={'ATK' if gt else 'NRM'}")

        if (t + 1) % 100 == 0:
            tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            print(f"\n  t={t+1} — Prec={p:.3f}  Rec={r:.3f}  F1={f1:.3f}\n")

        time.sleep(STEP_MS / 1000.0)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    print(f"\nFINAL — Prec={p:.3f}  Rec={r:.3f}  F1={f1:.3f}  "
          f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Turn off both LEDs on exit
    if GPIO_AVAILABLE:
        led_set(LED_ZONE,   False)
        led_set(LED_GLOBAL, False)
        GPIO.cleanup()