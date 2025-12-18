# USA_train_nodeclf.py
import argparse, json, math, random, csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv


# ---------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: str):
    # allow loading full Data object (PyG-style)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, Data):
        return obj
    return Data(**obj)


def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def stratified_ratio_split(y, train_ratio=0.10, val_ratio=0.20, seed=0):
    """Per-class 10/20/rest split (like paper’s node classification regime)."""
    set_seed(seed)
    y = y.cpu()
    n = y.numel()
    idx = torch.arange(n)
    classes = y.unique().tolist()

    train = torch.zeros(n, dtype=torch.bool)
    val   = torch.zeros(n, dtype=torch.bool)
    test  = torch.zeros(n, dtype=torch.bool)

    for c in classes:
        mask = (y == c)
        ids = idx[mask]
        ids = ids[torch.randperm(ids.numel())]

        n_train = max(1, int(math.floor(train_ratio * ids.numel())))
        n_val   = max(1, int(math.floor(val_ratio * ids.numel())))

        train_ids = ids[:n_train]
        val_ids   = ids[n_train:n_train+n_val]
        test_ids  = ids[n_train+n_val:]

        train[train_ids] = True
        val[val_ids]     = True
        test[test_ids]   = True

    return train, val, test


# ---------------------------------------------------------
# Build DEMO-Net degree structures
# ---------------------------------------------------------
def build_degree_tasks(data: Data):
    """
    degreeTasks: list of (deg, [node_ids])
    neighbor_list: flattened neighbors aligned with degreeTasks
    """
    n = data.num_nodes
    deg = degree(data.edge_index[0], num_nodes=n).to(torch.long)

    # neighbors per node
    neighbors = [[] for _ in range(n)]
    s, t = data.edge_index[0].tolist(), data.edge_index[1].tolist()
    for u, v in zip(s, t):
        neighbors[u].append(v)

    buckets = {}
    for nid in range(n):
        d = int(deg[nid].item())
        buckets.setdefault(d, []).append(nid)

    degreeTasks = []
    neighbor_list = []

    for d in sorted(buckets.keys()):
        ids = buckets[d]
        degreeTasks.append((d, ids))

        if d == 0:
            neighbor_list.append([])
            continue

        flat = []
        for nid in ids:
            nbrs = neighbors[nid]
            if len(nbrs) >= d:
                flat.extend(nbrs[:d])
            else:
                # pad by repeating neighbors if needed
                need = d - len(nbrs)
                if len(nbrs) == 0:
                    # pathological isolate with positive degree bucket; just repeat self
                    nbrs = [nid]
                flat.extend(nbrs + (nbrs * (need // len(nbrs) + 1))[:need])

        neighbor_list.append(flat)

    return degreeTasks, neighbor_list


# ---------------------------------------------------------
# Baseline Models
# ---------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge):
        x = F.relu(self.c1(x, edge))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, edge)


class ChebyNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, K=3, dropout=0.5):
        super().__init__()
        self.c1 = ChebConv(in_dim, hidden, K)
        self.c2 = ChebConv(hidden, out_dim, K)
        self.dropout = dropout

    def forward(self, x, edge):
        x = F.relu(self.c1(x, edge))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, edge)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.s1 = SAGEConv(in_dim, hidden)
        self.s2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge):
        x = F.relu(self.s1(x, edge))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.s2(x, edge)


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, h1=8, f1=8, h2=8, dropout=0.6):
        super().__init__()
        self.g1 = GATConv(in_dim, f1, heads=h1, dropout=dropout)
        self.g2 = GATConv(h1 * f1, out_dim, heads=h2, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.g1(x, edge))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.g2(x, edge)


# ---------------------------------------------------------
# DEMO-Net Weight (clean, degree-specific)
# ---------------------------------------------------------
class DEMOWeightLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        # global 1x1 conv (shared across all nodes)
        self.global_lin = nn.Linear(in_dim, out_dim, bias=False)
        # local 1x1 conv on neighbors
        self.local_lin  = nn.Linear(in_dim, out_dim, bias=False)
        # self transform
        self.self_lin   = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout

    def forward(self, x, edge, degreeTasks, neighbor_list):
        # x is [N, Fin] in ORIGINAL node order
        global_out = self.global_lin(x)              # [N, Fout]

        # We'll fill this in original node order as well.
        out_all = torch.zeros_like(global_out)

        for (deg, ids), neigh_flat in zip(degreeTasks, neighbor_list):
            if len(ids) == 0:
                continue

            ids_t = torch.tensor(ids, device=x.device)

            if deg == 0:
                # No neighbors: only global (and bias later)
                group_out = global_out[ids_t]
            else:
                neigh = torch.tensor(neigh_flat, device=x.device)
                # [len(ids), deg, Fin] -> mean over neighbors -> [len(ids), Fin]
                neigh_x = x[neigh].view(len(ids), deg, -1).mean(dim=1)

                local_out = self.local_lin(neigh_x)      # [len(ids), Fout]
                self_out  = self.self_lin(x[ids_t])      # [len(ids), Fout]
                g_out     = global_out[ids_t]            # [len(ids), Fout]

                group_out = local_out + self_out + g_out

            # Write back into the global tensor at ORIGINAL indices
            out_all[ids_t] = group_out

        out_all = out_all + self.bias
        out_all = F.dropout(out_all, p=self.dropout, training=self.training)
        return F.elu(out_all)



class DEMONetWeight(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.6,
                 degreeTasks=None, neighbor_list=None):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * num_layers

        for i in range(num_layers):
            self.layers.append(DEMOWeightLayer(dims[i], dims[i+1], dropout))

        self.out_lin = DEMOWeightLayer(hidden, out_dim, dropout)
        self.degreeTasks = degreeTasks
        self.neighbor_list = neighbor_list

    def ensure_structs(self, data):
        if self.degreeTasks is None or self.neighbor_list is None:
            self.degreeTasks, self.neighbor_list = build_degree_tasks(data)

    def forward(self, x, edge, data):
        self.ensure_structs(data)
        h = x
        for layer in self.layers:
            h = layer(h, edge, self.degreeTasks, self.neighbor_list)
        return self.out_lin(h, edge, self.degreeTasks, self.neighbor_list)


# ---------------------------------------------------------
# DEMO-Net Hash (fixed hashing)
# ---------------------------------------------------------
def make_hash_matrix(in_dim, hash_dim, device):
    col = torch.randint(0, hash_dim, (in_dim,), device=device)
    sign = (torch.randint(0, 2, (in_dim,), device=device) * 2 - 1).float()
    W = torch.zeros(in_dim, hash_dim, device=device)
    W[torch.arange(in_dim), col] = sign
    return W


class DEMOHashLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_hash, hash_dim, dropout):
        super().__init__()
        self.num_hash = num_hash
        self.hash_dim = hash_dim

        # Linear after concatenated hashed features
        self.post = nn.Linear(num_hash * hash_dim, out_dim, bias=False)
        # Self transform
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout

        # Will hold fixed hash matrices once built
        self.hash_mats = None

    def ensure_hash(self, in_dim, device):
        if self.hash_mats is None:
            self.hash_mats = [
                make_hash_matrix(in_dim, self.hash_dim, device)
                for _ in range(self.num_hash)
            ]

    def forward(self, x, edge, degreeTasks, neighbor_list):
        device = x.device
        in_dim = x.size(1)
        self.ensure_hash(in_dim, device)

        # We'll build hashed features in ORIGINAL node order
        N = x.size(0)
        hashed_all = torch.zeros(
            N, self.num_hash * self.hash_dim, device=device
        )

        for (deg, ids), flat in zip(degreeTasks, neighbor_list):
            if len(ids) == 0:
                continue

            ids_t = torch.tensor(ids, device=device)

            if deg == 0:
                # No neighbors, just use self features as base
                base = x[ids_t]                  # [B, Fin]
            else:
                neigh = torch.tensor(flat, device=device)
                # [B, deg, Fin] -> mean over neighbors -> [B, Fin]
                base = x[neigh].view(len(ids), deg, in_dim).mean(dim=1)

            # Apply all hash matrices and concat
            hashed_list = [base @ H for H in self.hash_mats]   # each [B, Hdim]
            group_hashed = torch.cat(hashed_list, dim=1)       # [B, num_hash*Hdim]

            # Write back to the positions of these nodes in ORIGINAL index space
            hashed_all[ids_t] = group_hashed

        # Now hashed_all is [N, num_hash*hash_dim] aligned with node indices 0..N-1
        neigh_part = self.post(hashed_all)  # [N, out_dim]
        self_part  = self.self_lin(x)       # [N, out_dim]

        out = neigh_part + self_part + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)
        return F.elu(out)



class DEMONetHash(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.6,
                 num_hash=4, hash_dim=128,
                 degreeTasks=None, neighbor_list=None):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * num_layers

        for i in range(num_layers):
            self.layers.append(
                DEMOHashLayer(dims[i], dims[i+1],
                              num_hash=num_hash, hash_dim=hash_dim,
                              dropout=dropout)
            )

        self.out_lin = DEMOHashLayer(hidden, out_dim,
                                     num_hash=num_hash, hash_dim=hash_dim,
                                     dropout=dropout)

        self.degreeTasks = degreeTasks
        self.neighbor_list = neighbor_list

    def ensure_structs(self, data):
        if self.degreeTasks is None or self.neighbor_list is None:
            self.degreeTasks, self.neighbor_list = build_degree_tasks(data)

    def forward(self, x, edge, data):
        self.ensure_structs(data)
        h = x
        for layer in self.layers:
            h = layer(h, edge, self.degreeTasks, self.neighbor_list)
        return self.out_lin(h, edge, self.degreeTasks, self.neighbor_list)


# ---------------------------------------------------------
# Model configs (paper defaults + Union/Intersection)
# ---------------------------------------------------------
CONFIGS = {
    "gcn": dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=5e-4,
                epochs=200, patience=10),

    # “Union” and “Intersection” use the same bare 2-layer GCN architecture
    # and hyperparameters as GCN (they differ in label expansion in the
    # original paper, which we’re not doing here).
    "union": dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=5e-4,
                  epochs=200, patience=10),
    "intersection": dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=5e-4,
                         epochs=200, patience=10),

    "cheby": dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=0.0,
                  epochs=200, patience=10),

    "sage": dict(hidden=128, dropout=0.0, lr=0.001, weight_decay=0.0,
                 epochs=50,  patience=10),

    "gat": dict(dropout=0.6, lr=0.005, weight_decay=0.0005,
                epochs=10000, patience=100, h1=8, f1=8, h2=8),

    # DEMO-Net (paper)
    "demo_weight": dict(hidden=64, dropout=0.6, lr=0.005,
                        weight_decay=0.0005, epochs=500,
                        patience=100, num_layers=2),

    "demo_hash":   dict(hidden=64, dropout=0.6, lr=0.005,
                        weight_decay=0.0005, epochs=500,
                        patience=100, num_layers=2,
                        num_hash=4, hash_dim=128),
}

MODEL_NAMES = [
    "gcn", "cheby", "sage", "gat",
    "union", "intersection",
    "demo_weight", "demo_hash", "all"
]


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def run_one_seed(data, model_name, cfg, seed, device):
    set_seed(seed)

    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item()) + 1

    # Model selection
    if model_name in ["gcn", "union", "intersection"]:
        net = GCN(in_dim, cfg["hidden"], out_dim, dropout=cfg["dropout"]).to(device)
    elif model_name == "cheby":
        net = ChebyNet(in_dim, cfg["hidden"], out_dim, K=3,
                       dropout=cfg["dropout"]).to(device)
    elif model_name == "sage":
        net = GraphSAGE(in_dim, cfg["hidden"], out_dim,
                        dropout=cfg["dropout"]).to(device)
    elif model_name == "gat":
        net = GAT(in_dim, out_dim, h1=cfg["h1"], f1=cfg["f1"],
                  h2=cfg["h2"], dropout=cfg["dropout"]).to(device)
    elif model_name == "demo_weight":
        net = DEMONetWeight(in_dim, cfg["hidden"], out_dim,
                            num_layers=cfg["num_layers"],
                            dropout=cfg["dropout"]).to(device)
    elif model_name == "demo_hash":
        net = DEMONetHash(in_dim, cfg["hidden"], out_dim,
                          num_layers=cfg["num_layers"],
                          dropout=cfg["dropout"],
                          num_hash=cfg["num_hash"],
                          hash_dim=cfg["hash_dim"]).to(device)
    else:
        raise ValueError(model_name)

    opt = Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.CrossEntropyLoss()

    best_val, best_test = -1.0, 0.0
    no_improve = 0

    x = data.x.to(device)
    y = data.y.to(device)
    ei = data.edge_index.to(device)

    tr = data.train_mask
    va = data.val_mask
    te = data.test_mask

    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        opt.zero_grad()

        if model_name.startswith("demo"):
            logits = net(x, ei, data)
        else:
            logits = net(x, ei)

        loss = crit(logits[tr], y[tr])
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            if model_name.startswith("demo"):
                logits = net(x, ei, data)
            else:
                logits = net(x, ei)

            tr_acc = accuracy(logits[tr], y[tr])
            va_acc = accuracy(logits[va], y[va])
            te_acc = accuracy(logits[te], y[te])

        if ep == 1 or ep % 10 == 0:
            print(f"[seed {seed}] {model_name:12s} | ep{ep:04d} "
                  f"loss {loss.item():.4f} | tr {tr_acc:.4f} | va {va_acc:.4f} | te {te_acc:.4f}")

        if va_acc > best_val:
            best_val, best_test = va_acc, te_acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg["patience"]:
            print(f"[seed {seed}] early stop | best VAL {best_val:.4f} | TEST@best {best_test:.4f}")
            break

    if no_improve < cfg["patience"]:
        print(f"[seed {seed}] finished max epochs | best VAL {best_val:.4f} | TEST@best {best_test:.4f}")

    return best_val, best_test


def write_csv(path, model_name, mv, sv, mt, st):
    first = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["model", "val_mean", "val_std", "test_mean", "test_std"])
        w.writerow([model_name, f"{mv:.4f}", f"{sv:.4f}", f"{mt:.4f}", f"{st:.4f}"])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=MODEL_NAMES, required=True)
    ap.add_argument("--data_path", type=str, required=True)

    ap.add_argument("--override_split", type=str,
                    choices=["ratios", "none"],
                    default="ratios")

    ap.add_argument("--train_ratio", type=float, default=0.10)
    ap.add_argument("--val_ratio", type=float,  default=0.20)

    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0,1,2,3,4,5,6,7,8,9])
    ap.add_argument("--csv_out", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data_path)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # apply new split per seed
    def apply_split(seed):
        if args.override_split == "ratios":
            tr, va, te = stratified_ratio_split(
                data.y,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=seed
            )
            data.train_mask = tr
            data.val_mask   = va
            data.test_mask  = te
            print(f"[split] ratios (seed={seed}) -> "
                  f"{tr.sum().item()} train | {va.sum().item()} val | {te.sum().item()} test")
        else:
            print(f"[split] existing masks used (seed={seed})")

    def run_model(name):
        cfg = CONFIGS[name]
        vals, tests = [], []
        print(f"\n>>> Running {name} with config {json.dumps(cfg)}")

        for s in args.seeds:
            apply_split(s)
            bv, bt = run_one_seed(data, name, cfg, s, device)
            vals.append(bv)
            tests.append(bt)

        mv = sum(vals) / len(vals)
        mt = sum(tests) / len(tests)
        sv = (sum((v - mv) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        st = (sum((t - mt) ** 2 for t in tests) / max(1, len(tests) - 1)) ** 0.5

        print(f"\n{name:12s} VAL {mv:.3f} ± {sv:.3f} | TEST {mt:.3f} ± {st:.3f}")
        if args.csv_out:
            write_csv(args.csv_out, name, mv, sv, mt, st)

    if args.model == "all":
        for m in ["gcn", "cheby", "sage", "gat",
                  "union", "intersection",
                  "demo_weight", "demo_hash"]:
            run_model(m)
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()