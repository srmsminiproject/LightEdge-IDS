import os
import json
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig
import leidenalg
from cdlib.classes.node_clustering import NodeClustering

# Loading condensed graph

# ---------------- PATH RESOLUTION ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "subgraph_constr" / "subgraph_description.json"
ZONE_DIR = PROJECT_ROOT / "outputs" / "zone_div"
ZONE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_JSON = ZONE_DIR / "zones.json"
DEFAULT_CSV  = ZONE_DIR / "zones.csv"
DEFAULT_PNG  = ZONE_DIR / "zones.png"


def load_condensed_graph(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.Graph()
    for n in data["nodes"]:
        if isinstance(n, dict):
            G.add_node(n["id"], **n.get("attrs", {}))
        else:
            G.add_node(n)
    for e in data["edges"]:
        src, dst = e["src"], e["dst"]
        attrs = {k: v for k, v in e.items() if k not in ["src", "dst"]}
        G.add_edge(src, dst, **attrs)

    return G


# Leiden algorithm runner


def run_leiden(G, resolution=0.8, seed=42):
    '''Run Leiden algorithm with fixed seed and resolution.
        Ensures all nodes (even isolated) are assigned to a community.
    '''
    mapping = {n: i for i, n in enumerate(G.nodes())}       
    inv_mapping = {i: n for n, i in mapping.items()}

    g = ig.Graph()
    g.add_vertices(len(G.nodes()))
    g.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])

    part = leidenalg.find_partition(     
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed
    )

    communities = []
    for comm in part:
        communities.append([inv_mapping[i] for i in comm])

   
    assigned = set(n for comm in communities for n in comm)
    unassigned = set(G.nodes()) - assigned
    for n in unassigned:
        communities.append([n])  

    return NodeClustering(communities, G, "leiden", None)



# Save results

def save_results(G, comms, out_json, out_csv, out_png):
    zones = {}
    for i, comm in enumerate(comms.communities):
        for n in comm:
            zones[n] = i

    # Guarantee every node is present
    zones = {n: zones.get(n, -1) for n in G.nodes()}

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(zones, f, indent=2)

    # Save CSV
    pd.DataFrame(list(zones.items()), columns=["Node", "Zone"]).to_csv(out_csv, index=False)

    # Save plot
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    pos = nx.spring_layout(G, seed=42)
    colors = [zones[n] for n in G.nodes()]

    nx.draw(
        G, pos,
        node_color=colors,
        with_labels=True,
        cmap=plt.cm.tab20,
        node_size=600,
        font_size=10,
        ax=ax
    )
    ax.set_title("Zone Division (Leiden)")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=0.8,
                        help="Resolution parameter (higher = more clusters, lower = fewer clusters)")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--out_json", type=str, default=str(DEFAULT_JSON))
    parser.add_argument("--out_csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--out_png", type=str, default=str(DEFAULT_PNG))

    

    args = parser.parse_args()

    print(f"Loading condensed graph from: {args.input}")
    G = load_condensed_graph(args.input)
    print(f"Graph loaded: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    print(f"Running Leiden with resolution={args.resolution}, seed=42...")
    comms = run_leiden(G, resolution=args.resolution, seed=42)
    print(f"Detected {len(comms.communities)} zones.")

    save_results(G, comms, args.out_json, args.out_csv, args.out_png)
    print(f"Zones saved to: {args.out_json}, {args.out_csv}, {args.out_png}")


if __name__ == "__main__":
    main()

#Constructs the zones from the subgraph
'''def save_results(G, comms, out_json, out_csv, out_png):
    zones = {}
    for i, comm in enumerate(comms.communities):
        for n in comm:
            zones[n] = i

    # Guarantee every node is present
    zones = {n: zones.get(n, -1) for n in G.nodes()}

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(zones, f, indent=2)

    # Save CSV
    pd.DataFrame(list(zones.items()), columns=["Node", "Zone"]).to_csv(out_csv, index=False)

    # Save plot
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    colors = [zones[n] for n in G.nodes()]
    nx.draw(G, pos, node_color=colors, with_labels=True,
            cmap=plt.cm.tab20, node_size=500, font_size=8)
    plt.title("Zone Division (Leiden)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

'''