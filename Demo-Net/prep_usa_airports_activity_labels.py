import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import os

def preprocess_usa_activity():
    edge_path = "data/usa_airports/usa-airports.edgelist"
    save_dir = "data/usa_airports_degX_binsY"
    os.makedirs(save_dir, exist_ok=True)

    # Load edges
    edges = np.loadtxt(edge_path, dtype=int)
    G = nx.Graph()
    G.add_edges_from(edges)

    # Remap nodes to consecutive 0..N-1
    nodes = sorted(G.nodes())
    id_map = {n: i for i, n in enumerate(nodes)}
    G = nx.relabel_nodes(G, id_map)

    print(f"[info] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # -------------------------
    # 1. Compute activity features
    # -------------------------
    deg = np.array([G.degree(n) for n in G.nodes()])
    pagerank = nx.pagerank(G)
    bet = nx.betweenness_centrality(G, normalized=True)

    pr = np.array([pagerank[n] for n in G.nodes()])
    bt = np.array([bet[n] for n in G.nodes()])

    # Combine into activity score
    # scaled to similar ranges
    deg_norm = deg / deg.max()
    pr_norm = pr / pr.max()
    bt_norm = bt / bt.max()

    activity = 0.6 * deg_norm + 0.3 * pr_norm + 0.1 * bt_norm

    # -------------------------
    # 2. Bin into 4 quartiles (labels)
    # -------------------------
    num_bins = 4
    bins = np.quantile(activity, [0.25, 0.5, 0.75])
    
    labels = np.zeros(len(activity), dtype=int)
    labels[activity > bins[0]] = 1
    labels[activity > bins[1]] = 2
    labels[activity > bins[2]] = 3

    print("[info] Label distribution:", np.bincount(labels))

    # -------------------------
    # 3. Build PyG Data object
    # -------------------------
    edge_index = torch.tensor(np.array(list(G.edges())).T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)  # undirected

    # Features = degree one-hot
    max_deg = deg.max()
    x = np.eye(max_deg + 1)[deg]
    x = torch.tensor(x, dtype=torch.float)

    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    save_path = f"{save_dir}/processed.pt"
    torch.save(data, save_path)

    print(f"[done] Saved {save_path}")
    print(f"       nodes={len(nodes)} | edges={G.number_of_edges()} | classes=4")


if __name__ == "__main__":
    preprocess_usa_activity()
