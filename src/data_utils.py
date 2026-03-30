# data_utils.py
import os
import json
import numpy as np
import pandas as pd
import torch

BASE_SUB = os.path.join("..", "outputs", "subgraph_constr")
BASE_ZONE = os.path.join("..", "outputs", "zone_div")
DEFAULT_FEATURES = os.path.join(BASE_SUB, "cleaned_aligned.csv")
DEFAULT_SUBGRAPH = os.path.join(BASE_SUB, "subgraph_description.json")
DEFAULT_MAPPING = os.path.join(BASE_SUB, "mapping.json")
DEFAULT_ZONES = os.path.join(BASE_ZONE, "zones.json")


def load_raw_df(path_csv: str | None = None):
    """Return the full dataframe (including ATT_FLAG)."""
    if path_csv is None:
        path_csv = DEFAULT_FEATURES
    print(f"[data_utils] load_raw_df: {path_csv}")
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Features CSV not found: {path_csv}")
    df = pd.read_csv(path_csv, index_col=0)
    return df


def load_labels(path_csv: str | None = None):
    """
    Return binary labels array aligned with df rows:
      -999 -> 0 (normal)
      any other value -> 1 (attack)
    Also returns counts printed for debugging.
    """
    df = load_raw_df(path_csv)
    if "ATT_FLAG" not in df.columns and "ATTFLAG" not in df.columns:
        raise KeyError("Label column 'ATT_FLAG' or 'ATTFLAG' not found in CSV")
    label_col = "ATT_FLAG" if "ATT_FLAG" in df.columns else "ATTFLAG"
    y = df[label_col].values
    # convert to binary
    y_bin = np.where(y == -999, 0, 1).astype(np.int64)
    n_norm = int((y_bin == 0).sum())
    n_att = int((y_bin == 1).sum())
    print(f"[data_utils] Labels converted: {n_norm} normal, {n_att} attack")
    return y_bin


def load_features_df(path_csv: str | None = None):
    """Return features DataFrame (without label column)."""
    df = load_raw_df(path_csv)
    if "ATT_FLAG" in df.columns:
        df = df.drop(columns=["ATT_FLAG"])
    elif "ATTFLAG" in df.columns:
        df = df.drop(columns=["ATTFLAG"])
    return df


def load_mapping(path_json: str | None = None):
    if path_json is None:
        path_json = DEFAULT_MAPPING
    print(f"[data_utils] load_mapping: {path_json}")
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"Mapping JSON not found: {path_json}")
    with open(path_json, "r") as fh:
        mapping = json.load(fh)
    # mapping expected: sensor_column -> node_name
    return mapping


def load_subgraph(path_json: str | None = None):
    """Return nodes (list), node_index (dict), edge_index (torch.LongTensor 2xE), edge_weight (torch.FloatTensor E)."""
    if path_json is None:
        path_json = DEFAULT_SUBGRAPH
    print(f"[data_utils] load_subgraph: {path_json}")
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"Subgraph JSON not found: {path_json}")
    with open(path_json, "r") as fh:
        data = json.load(fh)

    nodes = list(data["nodes"])
    node_index = {n: i for i, n in enumerate(nodes)}

    edge_pairs = []
    weights = []
    for e in data.get("edges", []):
        u, v = e["src"], e["dst"]
        if u not in node_index or v not in node_index:
            continue
        ui, vi = node_index[u], node_index[v]
        hop = max(1, int(e.get("hop_length", 1)))
        w = 1.0 / hop
        # add both directions to be safe for undirected behavior
        edge_pairs.append([ui, vi]); weights.append(w)
        edge_pairs.append([vi, ui]); weights.append(w)

    if len(edge_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    return nodes, node_index, edge_index, edge_weight


def load_zones(path_json: str | None = None):
    if path_json is None:
        path_json = DEFAULT_ZONES
    print(f"[data_utils] load_zones: {path_json}")
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"Zones file not found: {path_json}")
    with open(path_json, "r") as fh:
        zones = json.load(fh)
    # zones: node -> zone_id
    # build zone_id -> [node_names] mapping
    zones_map = {}
    for node, zid in zones.items():
        zones_map.setdefault(int(zid), []).append(node)
    return zones_map


def build_feature_tensor(df: pd.DataFrame, mapping: dict, node_order: list, ma_window: int = 3):
    """
    Build feature tensor X shape [T, N, F] where:
      - T = number of timestamps (rows of df)
      - N = number of nodes (len(node_order))
      - F = 2 (raw mean across mapped columns, moving-average)
    mapping: dict column_name -> node_name (as produced earlier)
    node_order: list of node names (consistent with subgraph_description)
    """
    T = len(df)
    N = len(node_order)
    F = 2
    X = np.zeros((T, N, F), dtype=np.float32)

    # build reverse mapping: node -> list of columns that map to it
    cols_by_node = {}
    # mapping might contain keys that match df columns in case-sensitive or not
    df_cols = list(df.columns)
    for col, node in mapping.items():
        # prefer exact match, fallback to case-insensitive
        if col in df_cols:
            cols_by_node.setdefault(node, []).append(col)
        else:
            # case-insensitive match
            matched = None
            cl = col.lower()
            for c in df_cols:
                if c.lower() == cl:
                    matched = c
                    break
            if matched:
                cols_by_node.setdefault(node, []).append(matched)

    for j, node in enumerate(node_order):
        cols = cols_by_node.get(node, [])
        if len(cols) == 0:
            # leave zeros if node has no mapped sensor columns
            continue
        raw = df[cols].mean(axis=1).astype(float).fillna(0.0).values  # shape (T,)
        ma = pd.Series(raw).rolling(window=ma_window, min_periods=1).mean().values
        # normalize per-node (z-score) to stabilize training
        mu, sd = raw.mean(), raw.std() + 1e-9
        mu2, sd2 = ma.mean(), ma.std() + 1e-9
        X[:, j, 0] = ((raw - mu) / sd).astype(np.float32)
        X[:, j, 1] = ((ma - mu2) / sd2).astype(np.float32)

    return X, df.index.values
