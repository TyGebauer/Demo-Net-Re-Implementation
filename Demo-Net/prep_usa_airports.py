# prep_usa_airports.py
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from pathlib import Path


def preprocess_usa():
    edge_path = "data/usa_airports/usa-airports.edgelist"
    label_path = "data/usa_airports/labels-usa-airports.txt"

    out_dir = Path("data/usa_airports_degX")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed.pt"

    # ----------------- Load edge list -----------------
    edges = np.loadtxt(edge_path, dtype=int)
    G = nx.Graph()
    G.add_edges_from(edges)

    # ----------------- Load labels -----------------
    labels = {}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # skip header or malformed lines
            if not parts[0].isdigit():
                continue
            n, lab = parts
            labels[int(n)] = int(lab)

    # ----------------- Keep only labeled nodes -----------------
    nodes = sorted(set(G.nodes()).intersection(labels.keys()))
    id_map = {n: i for i, n in enumerate(nodes)}
    G = G.subgraph(nodes)

    # ----------------- edge_index -----------------
    edge_index = torch.tensor(
        [[id_map[u], id_map[v]] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    # ----------------- Features (constant) -----------------
    # Everyone gets the SAME 1-D feature [1.0]
    # This is intentionally "bare" so DEMO-Net has to exploit structure.
    x = torch.ones(len(nodes), 1, dtype=torch.float)

    # ----------------- Labels -----------------
    y = torch.tensor([labels[n] for n in nodes], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    torch.save(data, out_path)

    print(f"[done] Saved {out_path} | nodes={len(nodes)} | edges={G.number_of_edges()} | classes={len(set(y.tolist()))}")


if __name__ == "__main__":
    preprocess_usa()

