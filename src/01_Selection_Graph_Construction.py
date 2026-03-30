#!/usr/bin/env python3

import os
import re
import json
import difflib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# INPUT AND OUTPUT
INPUT_DIR = "../inputs"
INP_FILE = os.path.join(INPUT_DIR, "CTOWN.inp")
INPUT_CSV = os.path.join(INPUT_DIR, "BATADAL_dataset04.csv")
OUTPUT_DIR = "../outputs/subgraph_constr"

TIME_COL = "DATETIME"
LABEL_COL = "ATT_FLAG"
MA_WINDOW = 3

# Helpers 
def ensure_dirs():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_name(s: str) -> str:      #Remove spaces/special chars and uppercase.
    return re.sub(r"\W+", "", str(s)).upper()

    '''Parse EPANET INP and return networkx graph + coordinates.
       Pumps/valves are nodes; coords missing are midpoint of endpoints.
    '''
def parse_inp_with_coords(inp_path: str):
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"INP not found: {inp_path}")

    G = nx.Graph()
    coords = {}
    section = None
    mid_calc_nodes = []

    with open(inp_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(";") or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue

            parts = re.split(r"\s+", line)
            node = parts[0].strip()

            if section == "junctions":
                G.add_node(normalize_name(node), type="junction")

            elif section in ("reservoirs", "reservoir"):
                G.add_node(normalize_name(node), type="reservoir")

            elif section in ("tanks", "tank"):
                G.add_node(normalize_name(node), type="tank")

            elif section in ("pipes", "pipe"):
                if len(parts) >= 3:
                    n1, n2 = normalize_name(parts[1]), normalize_name(parts[2])
                    G.add_node(n1)
                    G.add_node(n2)
                    G.add_edge(n1, n2)

            elif section in ("pumps", "pump", "valves", "valve"):
                if len(parts) >= 3:
                    pump_name = normalize_name(parts[0])
                    n1, n2 = normalize_name(parts[1]), normalize_name(parts[2])

                    G.add_node(n1)
                    G.add_node(n2)
                    G.add_node(pump_name, type=section.rstrip('s'))

                    G.add_edge(n1, pump_name)
                    G.add_edge(pump_name, n2)

                    mid_calc_nodes.append((pump_name, n1, n2))

            elif "coord" in (section or ""):
                if len(parts) >= 3:
                    try:
                        coords[normalize_name(parts[0])] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue

    # Compute midpoints for pumps/valves without coords
    for pump_name, n1, n2 in mid_calc_nodes:
        if pump_name not in coords:
            if n1 in coords and n2 in coords:
                x_mid = (coords[n1][0] + coords[n2][0]) / 2
                y_mid = (coords[n1][1] + coords[n2][1]) / 2
                coords[pump_name] = (x_mid, y_mid)

    return G, coords

#  Load CSV file and process them.
def load_sensor_csv(csv_path: str, time_col: str = TIME_COL):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [normalize_name(c) for c in df.columns]

    time_col_norm = normalize_name(time_col)
    if time_col_norm not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found after normalization.")

    df[time_col_norm] = pd.to_datetime(
        df[time_col_norm],
        format="%m/%d/%y %H",
        errors="coerce"
    )

    df = df.dropna(subset=[time_col_norm]).set_index(time_col_norm).sort_index()
    return df

# Infer frequency of data
def infer_resample_freq(index: pd.DatetimeIndex, default="1min"):
    if len(index) < 2:
        return default
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return default
    median_s = float(diffs.median())
    if median_s < 60:
        return f"{int(round(median_s))}S"
    mins = median_s / 60.0
    if abs(round(mins) - mins) < 0.25:
        return f"{int(round(mins))}min"
    return default

# Mapping sensor columns to nodes.
def map_sensor_columns_to_nodes(sensor_cols, node_names):
    mapping = {}
    unmatched = []
    nodes_set = set(node_names)
    nodes_upper_map = {n.upper(): n for n in node_names}

    for col in sensor_cols:
        if col in nodes_set:
            mapping[col] = col
        elif col.upper() in nodes_upper_map:
            mapping[col] = nodes_upper_map[col.upper()]
        else:
            cand = difflib.get_close_matches(col.upper(), [n.upper() for n in node_names], n=1, cutoff=0.5)
            if cand:
                mapping[col] = nodes_upper_map[cand[0]]
            else:
                unmatched.append(col)
    return mapping, unmatched

#  Condensed subgraph (deterministic) 
def construct_condensed_subgraph(G_full, sensor_nodes):
    sensor_set = set(sensor_nodes)
    sensor_present = sorted([n for n in sensor_nodes if n in G_full.nodes()])  # sorted in order for determinism
    C = nx.Graph()
    C.add_nodes_from(sensor_present)

    # Precompute all shortest paths between sensors
    for i, src in enumerate(sensor_present):
        for dst in sensor_present[i+1:]:
            try:
                path = nx.shortest_path(G_full, source=src, target=dst)
            except nx.NetworkXNoPath:
                continue

            intermediates = path[1:-1]
            if all(node not in sensor_set for node in intermediates):
                # enforce canonical orientation: always src < dst
                if src > dst:
                    src, dst = dst, src
                    path = list(reversed(path))
                C.add_edge(src, dst, hop_length=len(path)-1, path=";".join(path))
    return C

# Save condensed subgraph description (deterministic)
def save_subgraph_description(C, out_path):
    desc = {
        "nodes": sorted(list(C.nodes())),  # sorted for determinism
        "edges": [
            {
                "src": min(u, v),
                "dst": max(u, v),
                "hop_length": d.get("hop_length"),
                "path": d.get("path")
            }
            for u, v, d in sorted(C.edges(data=True), key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))
        ]
    }
    with open(out_path, "w") as f:
        json.dump(desc, f, indent=2)

# Features
def preprocess_features(df_values, window_ma=MA_WINDOW):
    imputer = SimpleImputer(strategy="mean")    #skicit learn
    arr = imputer.fit_transform(df_values.values)
    df_raw = pd.DataFrame(arr, index=df_values.index, columns=df_values.columns)
    ma = df_raw.rolling(window=window_ma, min_periods=1).mean().values
    return np.stack([arr, ma], axis=-1)

#Plotting graphs
def plot_full_network(G, coords, out_path):
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    nx.draw(G, pos=coords, node_size=20, linewidths=0.1, edge_color="#999999", ax=ax)
    ax.set_title("Full Water Network")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_condensed_graph(C, coords_full, out_path):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    pos = {n: coords_full.get(n, None) for n in C.nodes if n in coords_full}
    missing = [n for n in C.nodes if n not in pos]
    if missing:
        pos_new = nx.spring_layout(C, seed=42)
        for m in missing:
            pos[m] = pos_new[m]

    nx.draw_networkx_nodes(C, pos=pos, node_size=120, node_color="orange", ax=ax)
    nx.draw_networkx_labels(C, pos=pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(C, pos=pos, width=1.2, ax=ax)
    ax.set_title("Condensed Sensor Graph")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------------- Main ----------------
def main():
    ensure_dirs()
    print("INPUT INP:", INP_FILE)
    print("INPUT CSV:", INPUT_CSV)

    G_full, coords = parse_inp_with_coords(INP_FILE)
    print(f"Parsed INP: nodes={G_full.number_of_nodes()}, edges={G_full.number_of_edges()}")

    pos_full = coords if coords else nx.spring_layout(G_full, seed=42)

    df_raw = load_sensor_csv(INPUT_CSV, time_col=TIME_COL)
    print("Loaded CSV shape:", df_raw.shape)

    sensor_cols = [c for c in df_raw.columns if c != normalize_name(LABEL_COL)]
    print(f"Sensor columns detected: {len(sensor_cols)}")

    freq = infer_resample_freq(df_raw.index)
    print("Inferred sampling freq:", freq)
    df_aligned = df_raw.resample(freq).mean().interpolate(method="time", limit_direction="both")
    df_aligned.to_csv(os.path.join(OUTPUT_DIR, "cleaned_aligned.csv"))

    mapping, unmatched = map_sensor_columns_to_nodes(sensor_cols, list(G_full.nodes()))
    print(f"Mapping results: mapped={len(mapping)}, unmatched={len(unmatched)}")
    json.dump(mapping, open(os.path.join(OUTPUT_DIR, "mapping.json"), "w"), indent=2)

    mapped_node_ids = list(set(mapping.values()))
    C = construct_condensed_subgraph(G_full, mapped_node_ids)
    print(f"Condensed graph: nodes={C.number_of_nodes()}, edges={C.number_of_edges()}")

    subgraph_desc_file = os.path.join(OUTPUT_DIR, "subgraph_description.json")
    save_subgraph_description(C, subgraph_desc_file)
    print("Subgraph description saved:", subgraph_desc_file)

    features_df = df_aligned.drop(columns=[normalize_name(LABEL_COL)], errors="ignore")
    features_tensor = preprocess_features(features_df)
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), features_tensor)

    plot_full_network(G_full, pos_full, os.path.join(OUTPUT_DIR, "full_network.png"))
    plot_condensed_graph(C, pos_full, os.path.join(OUTPUT_DIR, "condensed_graph.png"))

    print("Done. Outputs saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()


'''
Phase 1 pipeline for LightEdge-IDS (CTOWN + BATADAL).
- INP parsing with pumps/valves as nodes (midpoint coordinates if missing)
- Sensor CSV preprocessing
- Mapping sensor columns to INP nodes
- Full graph + condensed subgraph plotting
- Feature extraction
'''