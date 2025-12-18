# prep_facebook_binned_degree.py
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
            if not s or s.startswith("#"): 
                continue
            a, b = map(int, s.split())
            u.append(a); v.append(b)
    edge_index = torch.tensor([u, v], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    num_nodes = int(edge_index.max()) + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes)
    return Data(edge_index=edge_index, num_nodes=num_nodes)

def bin_edges_from_quantiles(deg: torch.Tensor, k: int = 4):
    """Return k-1 quantile cut points for k bins (e.g., k=4 -> quartiles)."""
    q = np.linspace(0, 1, num=k+1)[1:-1]  # exclude 0 and 1
    cuts = np.quantile(deg.cpu().numpy().astype(np.float32), q)
    return cuts.astype(np.int64)

def bucketize(deg: torch.Tensor, cuts: np.ndarray):
    """Map integer degrees -> bucket ids 0..K-1 using provided cut points."""
    # np.digitize with right=True puts values == cut into left bin
    y = np.digitize(deg.cpu().numpy(), cuts, right=True)
    return torch.from_numpy(y).long()

def one_hot_buckets(bucket_ids: torch.Tensor, k: int):
    x = torch.zeros((bucket_ids.numel(), k), dtype=torch.float32)
    x[torch.arange(bucket_ids.numel()), bucket_ids] = 1.0
    return x

def make_masks(n, val_ratio=0.1, test_ratio=0.1, seed=0):
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
    ap.add_argument("--out_dir", default="data/facebook_paper_binned")
    ap.add_argument("--bins", type=str, default="", 
                    help="Comma-separated degree cut points for 4 classes, e.g. '5,15,40'. "
                         "If omitted, use quartiles.")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_edge_list(args.edge_path)
    print(f"Loaded graph: nodes={data.num_nodes} | edges={data.num_edges}")

    # integer degrees
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)

    # choose cut points -> 4 buckets
    if args.bins.strip():
        cuts = np.array([int(s) for s in args.bins.split(",")], dtype=np.int64)
        if cuts.size != 3:
            raise ValueError("Provide exactly three cut points for 4 bins, e.g. --bins 5,15,40")
        cuts.sort()
        print(f"[bins] using custom cuts: {cuts.tolist()}")
    else:
        cuts = bin_edges_from_quantiles(deg, k=4)
        print(f"[bins] using quartile cuts: {cuts.tolist()}")

    # labels: degree-induced 4 classes
    y = bucketize(deg, cuts)  # 0..3

    # features: one-hot of the same 4 buckets
    x = one_hot_buckets(y, k=4)

    # sanity: class distribution
    unique, counts = np.unique(y.numpy(), return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"[labels] class distribution (0..3): {dist}")

    # masks
    tr, va, te = make_masks(data.num_nodes, args.val_ratio, args.test_ratio, seed=args.seed)

    # save as plain dict (no pickled classes)
    payload = {"x": x, "y": y, "edge_index": data.edge_index,
               "train_mask": tr, "val_mask": va, "test_mask": te}
    out_path = os.path.join(args.out_dir, "processed.pt")
    torch.save(payload, out_path)

    meta = {
        "nodes": int(data.num_nodes), "edges": int(data.num_edges),
        "x_dim": int(x.size(-1)), "n_classes": 4,
        "cuts": cuts.tolist(), "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "seed": args.seed
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[done] saved:", out_path)
    print(meta)

if __name__ == "__main__":
    main()
