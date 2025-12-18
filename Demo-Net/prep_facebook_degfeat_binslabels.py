
# prep_facebook_degfeat_binslabels.py
# Build node-classification tensors from facebook_combined.txt (UNDIRECTED).
# Stores ONE edge per undirected pair; computes degrees on a mirrored temp graph;
# Features: degree one-hot (capped) ; Labels: 4 degree bins (quartiles by default).

import os, argparse, json, numpy as np, torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, degree, to_undirected

def read_edgelist(path: str):
    u, v = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("%"):
                continue
            a, b = map(int, s.split())
            u.append(a); v.append(b)
    return u, v

def degree_onehot(deg_long: torch.Tensor, cap: int | None):
    if cap is not None and cap >= 0:
        deg_long = torch.clamp(deg_long, max=cap)
        dim = cap + 1
    else:
        dim = int(deg_long.max().item()) + 1
    x = torch.zeros((deg_long.numel(), dim), dtype=torch.float32)
    x[torch.arange(deg_long.numel()), deg_long] = 1.0
    return x

def quantile_cuts(deg_long: torch.Tensor, k: int = 4):
    q = np.linspace(0, 1, num=k+1)[1:-1]  # .25, .5, .75
    cuts = np.quantile(deg_long.cpu().numpy().astype(np.float32), q)
    return cuts.astype(np.int64)

def bucketize(deg_long: torch.Tensor, cuts: np.ndarray):
    return torch.from_numpy(np.digitize(deg_long.cpu().numpy(), cuts, right=True)).long()

def make_random_masks(n, val_ratio=0.1, test_ratio=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_ratio); n_test = int(n * test_ratio)
    val = perm[:n_val]; test = perm[n_val:n_val+n_test]; train = perm[n_val+n_test:]
    m_tr = torch.zeros(n, dtype=torch.bool); m_tr[train] = True
    m_va = torch.zeros(n, dtype=torch.bool); m_va[val]   = True
    m_te = torch.zeros(n, dtype=torch.bool); m_te[test]  = True
    return m_tr, m_va, m_te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_path", default="facebook_combined.txt")
    ap.add_argument("--out_dir",   default="data/facebook_paper_degX_binsY")
    ap.add_argument("--deg_cap",   type=int, default=20, help="cap for degree one-hot (dim = cap+1). Use -1 for no cap.")
    ap.add_argument("--bins",      type=str, default="", help="explicit 3 cuts (e.g. '5,15,40'); default=quartiles")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio",type=float, default=0.1)
    ap.add_argument("--seed",      type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) read
    u_raw, v_raw = read_edgelist(args.edge_path)
    num_nodes = max(max(u_raw), max(v_raw)) + 1

    # 2) store unique undirected edges (one per pair)
    ei = torch.tensor([u_raw, v_raw], dtype=torch.long)
    a = torch.minimum(ei[0], ei[1])
    b = torch.maximum(ei[0], ei[1])
    edge_index = torch.stack([a, b], dim=0)
    mask = edge_index[0] != edge_index[1]         # drop self loops
    edge_index = edge_index[:, mask]
    edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes)
    num_nodes = int(max(edge_index.max().item() + 1, num_nodes))
    print(f"Loaded graph: nodes={num_nodes} | edges={edge_index.size(1)} (unique undirected stored)")

    # 3) compute degrees on a temporary mirrored graph
    edge_index_for_deg = to_undirected(edge_index, num_nodes=num_nodes)
    deg = degree(edge_index_for_deg[0], num_nodes=num_nodes).to(torch.long)

    # 4) features & labels
    cap = args.deg_cap if args.deg_cap >= 0 else None
    x = degree_onehot(deg, cap)
    if args.bins.strip():
        cuts = np.array([int(s) for s in args.bins.split(",")], dtype=np.int64)
        if cuts.size != 3: raise ValueError("Provide exactly 3 cuts for 4 classes, e.g. --bins 5,15,40")
        cuts.sort(); print(f"[labels] using explicit cuts: {cuts.tolist()}")
    else:
        cuts = quantile_cuts(deg, k=4); print(f"[labels] using quartile cuts: {cuts.tolist()}")
    y = bucketize(deg, cuts)
    uniq, cnt = np.unique(y.numpy(), return_counts=True)
    print(f"[labels] class distribution: " + ", ".join(f"{int(k)}:{int(v)}" for k, v in zip(uniq, cnt)))
    print(f"[features] degree one-hot dim = {x.size(-1)} (cap={cap})")

    # 5) simple random masks (trainer can override)
    tr, va, te = make_random_masks(num_nodes, args.val_ratio, args.test_ratio, seed=args.seed)

    # 6) save (plain dict)
    out_path = os.path.join(args.out_dir, "processed.pt")
    torch.save({"x": x, "y": y, "edge_index": edge_index,
                "train_mask": tr, "val_mask": va, "test_mask": te}, out_path)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "nodes": int(num_nodes),
            "edges": int(edge_index.size(1)),
            "x_dim": int(x.size(-1)),
            "n_classes": 4,
            "cuts": [int(c) for c in cuts.tolist()],
            "deg_cap": cap,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed
        }, f, indent=2)

    print("[done] saved:", out_path)

if __name__ == "__main__":
    main()
