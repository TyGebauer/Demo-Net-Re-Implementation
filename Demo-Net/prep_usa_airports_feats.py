import os
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx

# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_DIR = "data/usa_airports"
EDGE_PATH = os.path.join(RAW_DIR, "usa-airports.edgelist")
LABEL_PATH = os.path.join(RAW_DIR, "labels-usa-airports.txt")

OUT_DIR = "data/usa_airports_degX_feats4"   # new folder
OUT_PATH = os.path.join(OUT_DIR, "processed.pt")


def load_labels(path):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # skip header row (e.g., "node label")
            if not parts[0].isdigit():
                continue

            node_id, lab = parts
            labels[int(node_id)] = int(lab)
    return labels


def preprocess_usa_feats():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load graph from edgelist
    edges = np.loadtxt(EDGE_PATH, dtype=int)
    G = nx.Graph()
    G.add_edges_from(edges)

    # 2. Load labels (activity classes 0–3)
    labels = load_labels(LABEL_PATH)

    # 3. Keep only labeled nodes
    nodes = sorted(set(G.nodes()).intersection(labels.keys()))
    id_map = {n: i for i, n in enumerate(nodes)}
    G = G.subgraph(nodes).copy()

    # 4. Build edge_index (using remapped IDs)
    edge_index = torch.tensor(
        [[id_map[u], id_map[v]] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    # 5. Graph-based features for each airport
    #    (a) degree
    deg = np.array([G.degree(n) for n in nodes], dtype=float)

    #    (b) clustering coefficient
    clust_dict = nx.clustering(G, nodes)
    clust = np.array([clust_dict[n] for n in nodes], dtype=float)

    #    (c) betweenness centrality
    #        (graph is small enough for exact computation)
    bet_dict = nx.betweenness_centrality(G, normalized=True)
    bet = np.array([bet_dict[n] for n in nodes], dtype=float)

    #    (d) closeness centrality
    close_dict = nx.closeness_centrality(G)
    close = np.array([close_dict[n] for n in nodes], dtype=float)

    # Stack features: [deg, clustering, betweenness, closeness]
    feats = np.stack([deg, clust, bet, close], axis=1)

    # Simple standardization per feature (mean 0, std 1)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    feats = (feats - mean) / std

    x = torch.tensor(feats, dtype=torch.float32)

    # 6. Labels vector (activity class for each node in same order as `nodes`)
    y = torch.tensor([labels[n] for n in nodes], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    torch.save(data, OUT_PATH)

    print(
        f"[done] Saved {OUT_PATH} | "
        f"nodes={len(nodes)} | edges={G.number_of_edges()} | "
        f"classes={len(set(y.tolist()))} | feat_dim={x.size(1)}"
    )


if __name__ == "__main__":
    preprocess_usa_feats()
