import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool


# ---------------------------------------------------------
# Degree-based neighbor structures (per batch)
# ---------------------------------------------------------
def build_degree_tasks(data: Data):
    """
    Build DEMO-Net degree groups and neighbor lists for a (possibly batched) graph.

    Returns:
      degreeTasks: list of (deg, [node_ids])
      neighbor_list: list of flattened neighbor ids aligned with degreeTasks.
    """
    n = data.num_nodes
    edge_index = data.edge_index
    deg = degree(edge_index[0], num_nodes=n).to(torch.long)

    # neighbors per node
    neighbors = [[] for _ in range(n)]
    s = edge_index[0].tolist()
    t = edge_index[1].tolist()
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
                need = d - len(nbrs)
                if len(nbrs) == 0:
                    # isolated: just use self
                    nbrs = [nid]
                flat.extend(nbrs + (nbrs * (need // len(nbrs) + 1))[:need])

        neighbor_list.append(flat)

    return degreeTasks, neighbor_list


# ---------------------------------------------------------
# DEMO-Net Weight Layers (graph-level)
# ---------------------------------------------------------
class DEMOWeightLayer(nn.Module):
    """
    One DEMO-Net weight-based layer, operating on node features.

    It uses:
      - global_lin: global 1x1 transform on all nodes
      - local_lin:  1x1 transform on mean(neighbor features per degree group)
      - self_lin:   1x1 transform on self-features
    """
    def __init__(self, in_dim, out_dim, dropout: float = 0.6):
        super().__init__()
        self.global_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.local_lin  = nn.Linear(in_dim, out_dim, bias=False)
        self.self_lin   = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, degreeTasks, neighbor_list):
        # x: [N, Fin]
        device = x.device
        global_out = self.global_lin(x)          # [N, Fout]
        out_all = torch.zeros_like(global_out)   # [N, Fout]

        for (deg, ids), flat in zip(degreeTasks, neighbor_list):
            if len(ids) == 0:
                continue

            ids_t = torch.tensor(ids, device=device, dtype=torch.long)

            if deg == 0:
                # only global + bias
                group_out = global_out[ids_t]
            else:
                neigh = torch.tensor(flat, device=device, dtype=torch.long)
                # [len(ids), deg, Fin] -> mean over neighbors -> [len(ids), Fin]
                neigh_x = x[neigh].view(len(ids), deg, -1).mean(dim=1)

                local_out = self.local_lin(neigh_x)
                self_out  = self.self_lin(x[ids_t])
                g_out     = global_out[ids_t]

                group_out = local_out + self_out + g_out

            out_all[ids_t] = group_out

        out_all = out_all + self.bias
        out_all = F.dropout(out_all, p=self.dropout, training=self.training)
        return F.elu(out_all)


class DEMONetWeightGraph(nn.Module):
    """
    DEMO-Net (weight) for graph-level classification.

    Architecture:
      - L DEMOWeightLayers on nodes
      - global mean pool -> graph embedding
      - linear classifier to num_classes
    """
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_layers: int = 2, dropout: float = 0.6):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DEMOWeightLayer(dims[i], dims[i+1], dropout))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # rebuild degreeTasks for each batch
        degreeTasks, neighbor_list = build_degree_tasks(data)

        h = x
        for layer in self.layers:
            h = layer(h, edge_index, degreeTasks, neighbor_list)

        # graph-level readout
        g = global_mean_pool(h, batch)      # [num_graphs, hidden_dim]
        g = F.dropout(g, p=self.dropout, training=self.training)
        out = self.classifier(g)            # [num_graphs, num_classes]
        return out


# ---------------------------------------------------------
# DEMO-Net Hash Layers (graph-level)
# ---------------------------------------------------------
def make_hash_matrix(in_dim, hash_dim, device):
    col = torch.randint(0, hash_dim, (in_dim,), device=device)
    sign = (torch.randint(0, 2, (in_dim,), device=device) * 2 - 1).float()
    W = torch.zeros(in_dim, hash_dim, device=device)
    W[torch.arange(in_dim), col] = sign
    return W


class DEMOHashLayer(nn.Module):
    """
    One DEMO-Net hash-based layer, operating on node features.

    It:
      - hashes neighbor-aggregated features with multiple random hash matrices
      - concatenates them, applies a linear projection
      - adds self-transform and bias
    """
    def __init__(self, in_dim, out_dim,
                 num_hash: int = 4, hash_dim: int = 128,
                 dropout: float = 0.6):
        super().__init__()
        self.num_hash = num_hash
        self.hash_dim = hash_dim

        self.post = nn.Linear(num_hash * hash_dim, out_dim, bias=False)
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout

        self.hash_mats = None  # List[Tensor], created on first forward

    def ensure_hash(self, in_dim, device):
        if self.hash_mats is None:
            self.hash_mats = [
                make_hash_matrix(in_dim, self.hash_dim, device)
                for _ in range(self.num_hash)
            ]

    def forward(self, x, edge_index, degreeTasks, neighbor_list):
        device = x.device
        in_dim = x.size(1)
        self.ensure_hash(in_dim, device)

        N = x.size(0)
        hashed_all = torch.zeros(
            (N, self.num_hash * self.hash_dim), device=device
        )

        for (deg, ids), flat in zip(degreeTasks, neighbor_list):
            if len(ids) == 0:
                continue

            ids_t = torch.tensor(ids, device=device, dtype=torch.long)

            if deg == 0:
                base = x[ids_t]  # [B, Fin]
            else:
                neigh = torch.tensor(flat, device=device, dtype=torch.long)
                base = x[neigh].view(len(ids), deg, in_dim).mean(dim=1)

            # apply hash matrices and concatenate
            hashed_list = [base @ H for H in self.hash_mats]   # each [B, Hdim]
            group_hashed = torch.cat(hashed_list, dim=1)       # [B, num_hash*Hdim]

            hashed_all[ids_t] = group_hashed

        neigh_part = self.post(hashed_all)
        self_part  = self.self_lin(x)

        out = neigh_part + self_part + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)
        return F.elu(out)


class DEMONetHashGraph(nn.Module):
    """
    DEMO-Net (hash) for graph-level classification.

    Architecture:
      - L DEMOHashLayers on nodes
      - global mean pool -> graph embedding
      - linear classifier
    """
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_layers: int = 2, dropout: float = 0.6,
                 num_hash: int = 4, hash_dim: int = 128):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DEMOHashLayer(dims[i], dims[i+1],
                              num_hash=num_hash, hash_dim=hash_dim,
                              dropout=dropout)
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        degreeTasks, neighbor_list = build_degree_tasks(data)

        h = x
        for layer in self.layers:
            h = layer(h, edge_index, degreeTasks, neighbor_list)

        g = global_mean_pool(h, batch)
        g = F.dropout(g, p=self.dropout, training=self.training)
        out = self.classifier(g)
        return out
