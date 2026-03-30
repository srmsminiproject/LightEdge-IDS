import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
MASTER_XLSX = "../inputs/BATADAL_test_dataset_labelled.xlsx"
ZONES_CSV   = "../outputs/zone_div/zones.csv"

SENSOR_COL = "Node"
ZONE_COL   = "Zone"
LABEL_COL  = "ATTACK_FLAG"

OUT_DIR = Path("../outputs/zone_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------

# Load data
df = pd.read_excel(MASTER_XLSX)
zones = pd.read_csv(ZONES_CSV)

print("zones.csv columns:", zones.columns.tolist())
print("master Excel columns:", df.columns.tolist())

# ---------------- LABEL FIX ----------------
# -999 → 0 (normal), everything else → 1 (attack)
df[LABEL_COL] = (df[LABEL_COL] != 0).astype(int)

# Sanity check
print("Label distribution after conversion:")
print(df[LABEL_COL].value_counts())
# -------------------------------------------

# Build logical sensor → real column mapping
sensor_to_columns = {}
for col in df.columns:
    if col in [LABEL_COL, "DATETIME"]:
        continue
    if "_" in col:
        logical = col.split("_", 1)[1]
        sensor_to_columns.setdefault(logical, []).append(col)

# Generate zone-wise CSVs
for zid in sorted(zones[ZONE_COL].unique()):
    logical_sensors = zones.loc[zones[ZONE_COL] == zid, SENSOR_COL].tolist()

    real_cols = []
    for s in logical_sensors:
        if s in sensor_to_columns:
            real_cols.extend(sensor_to_columns[s])
        else:
            print(f"⚠ Zone {zid}: sensor '{s}' not found")

    real_cols = sorted(set(real_cols))
    if not real_cols:
        print(f"❌ Zone {zid}: no matching columns, skipping")
        continue

    real_cols.append(LABEL_COL)

    zone_df = df[real_cols]
    out_file = OUT_DIR / f"zone{int(zid)}.csv"
    zone_df.to_csv(out_file, index=False)

    print(f"✅ Generated {out_file} with {len(real_cols)-1} features")
