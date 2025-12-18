# prep_facebook_degree_labels.py
import os, argparse, json
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce, degree

def load_edge_list(path: str) -> Data:
    u, v = [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            a, b = map(int, s.split())
            u.append(a); v.append(b)
    edge_index = torch.tensor([u, v], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    num_nodes = int(edge_index.max()) + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes)
    return Data(edge_index=edge_index, num_nodes=num_nodes)

def make_onehot_degree_features(edge_index: torch.Tensor, num_nodes: int, cap: int | None = None):
    deg = degree(edge_index[0], num_nodes=num_nodes).to(torch.long)
    if cap is not None:
        deg = torch.clamp(deg, max=cap)  # optionally cap very large degrees
        dim = cap + 1
    else:
        dim = int(deg.max().item()) + 1
    x = torch.zeros((num_nodes, dim), dtype=torch.float32)
    x[torch.arange(num_nodes), deg] = 1.0
    return x, deg

def degree_quartile_labels(deg: torch.Tensor, n_classes: int = 4):
    # Map degrees to 4 bins by quartiles (paper: 4 classes derived from degree)
    d = deg.to(torch.float32).numpy()
    qs = np.quantile(d, [0.25, 0.5, 0.75])
    y = np.digitize(d, qs, right=True)  # 0..3
    return torch.from_numpy(y).long()

def masks(n, val_ratio=0.1, test_ratio=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_ratio); n_test = int(n * test_ratio)
    val = perm[:n_val]; test = perm[n_val:n_val+n_test]; train = perm[n_val+n_test:]
    m_tr = torch.zeros(n, dtype=torch.bool); m_tr[train] = True
    m_va = torch.zeros(n, dtype=torch.bool); m_va[val] = True
    m_te = torch.zeros(n, dtype=torch.bool); m_te[test] = True
    return m_tr, m_va, m_te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_path", default="facebook_combined.txt")
    ap.add_argument("--out_dir", default="data/facebook_paper")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap_degree", type=int, default=None, help="optional cap for one-hot dim")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_edge_list(args.edge_path)
    print(f"Loaded graph: nodes={data.num_nodes} | edges={data.num_edges}")

    # Features: one-hot degree (as in paper)
    x, deg = make_onehot_degree_features(data.edge_index, data.num_nodes, cap=args.cap_degree)

    # Labels: 4 degree-induced classes (quartiles)
    y = degree_quartile_labels(deg, n_classes=4)

    tr, va, te = masks(data.num_nodes, args.val_ratio, args.test_ratio, seed=args.seed)

    payload = {"x": x, "y": y, "edge_index": data.edge_index,
               "train_mask": tr, "val_mask": va, "test_mask": te}
    out_path = os.path.join(args.out_dir, "processed.pt")
    torch.save(payload, out_path)

    meta = {
        "nodes": int(data.num_nodes), "edges": int(data.num_edges),
        "x_dim": int(x.size(-1)), "n_classes": 4,
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "seed": args.seed
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[done] saved:", out_path)
    print(meta)

if __name__ == "__main__":
    main()
