# models.py  (PyTorch version)
# DEMO-Net (weight) and DEMO-Net (hash) implemented without TensorFlow.
# Works with train_nodeclf.py that does: from models import DEMONetWeight, DEMONetHash

from __future__ import annotations
from typing import List, Sequence, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _scatter_to_nodes(num_nodes: int, ids_in_concat_order: List[int], group_outputs: List[Tensor]) -> Tensor:
    """
    Utility: take a list of per-group outputs (each [len(group_ids), C]) produced in the
    same order as we concatenated the group node ids, and scatter them back to [N, C]
    aligned to original node ids.
    """
    device = group_outputs[0].device
    out_dim = group_outputs[0].size(-1)
    out = torch.empty(num_nodes, out_dim, device=device)
    concat = torch.cat(group_outputs, dim=0)  # [sum |group_i|, C]
    idx = torch.tensor(ids_in_concat_order, device=device, dtype=torch.long)
    out[idx] = concat
    return out


class _Act(nn.Module):
    def __init__(self, name: str = "elu"):
        super().__init__()
        self.name = name.lower()

    def forward(self, x: Tensor) -> Tensor:
        if self.name == "relu":
            return F.relu(x)
        if self.name == "gelu":
            return F.gelu(x)
        # default: elu
        return F.elu(x)


class _WeightBlock(nn.Module):
    """
    One DEMO-Net "weight-based multi-task" block:
      - global 1x1 mapping on all nodes (Linear)
      - per degree group: average neighbor features (local 1x1 on neighbor features + global map on neighbors)
      - sum self (global on self) + aggregated-neigh
    """
    def __init__(self, in_dim: int, out_dim: int, act: str = "elu", dropout: float = 0.0):
        super().__init__()
        self.global_fc = nn.Linear(in_dim, out_dim, bias=False)
        self.local_fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.act = _Act(act)
        self.dropout = dropout

    @torch.no_grad()
    def reset_parameters(self):
        for m in (self.global_fc, self.local_fc):
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        x: Tensor,                                  # [N, F]
        degree_tasks: Sequence[Tuple[int, List[int]]],
        neighbor_list: Sequence[List[int]],
    ) -> Tensor:
        N = x.size(0)
        # Global map for all nodes
        global_maps = self.global_fc(x)             # [N, out]
        group_outs: List[Tensor] = []
        ids_in_concat_order: List[int] = []

        for (deg, node_ids), neigh_ids in zip(degree_tasks, neighbor_list):
            ids_in_concat_order.extend(node_ids)
            if deg == 0:
                # just take the global map of those nodes
                group_outs.append(global_maps[node_ids])  # [len(node_ids), out]
                continue

            # neigh features -> local 1x1
            neigh_x = x[neigh_ids]                              # [len(node_ids)*deg, F]
            local_neigh = self.local_fc(neigh_x)                # [len(node_ids)*deg, out]
            # global maps of neighbors
            global_neigh = global_maps[neigh_ids]               # [len(node_ids)*deg, out]
            mixed = local_neigh + global_neigh
            mixed = mixed.view(len(node_ids), deg, -1).mean(dim=1)  # [len(node_ids), out]
            group_outs.append(mixed)

        neigh_agg = _scatter_to_nodes(N, ids_in_concat_order, group_outs)  # [N, out]
        # self (global on self)
        self_map = global_maps
        # dropout
        if self.training and self.dropout > 0:
            neigh_agg = F.dropout(neigh_agg, p=self.dropout, inplace=False)
            self_map = F.dropout(self_map, p=self.dropout, inplace=False)

        out = neigh_agg + self_map + self.bias
        return self.act(out)


class _FixedHasher(nn.Module):
    """
    Fixed (non-trainable) sign hashing: x @ R, where R in {+1,-1}^{F x H}.
    We implement as a fixed weight Linear with requires_grad=False.
    """
    def __init__(self, in_dim: int, hash_dim: int, seed: Optional[int] = None):
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        # R in {-1, +1}
        R = (torch.rand(in_dim, hash_dim, generator=gen) > 0.5).float() * 2 - 1
        self.proj = nn.Linear(in_dim, hash_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.data = R.t()  # Linear expects [out, in]
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _HashBlock(nn.Module):
    """
    One DEMO-Net "hash-based multi-task" block:
      - For each of num_hash independent hashers:
          * hash self features and neighbor features (global + local), average by degree group
      - Concatenate all hashed streams and map with 1x1 Linear
    """
    def __init__(self, in_dim: int, out_dim: int, hash_dim: int, num_hash: int, act: str = "elu", dropout: float = 0.0, layer_id: int = 0):
        super().__init__()
        self.hashers = nn.ModuleList([_FixedHasher(in_dim, hash_dim, seed=layer_id*1000 + k) for k in range(num_hash)])
        concat_dim = num_hash * hash_dim
        self.self_fc = nn.Linear(concat_dim, out_dim, bias=False)
        self.neigh_fc = nn.Linear(concat_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.act = _Act(act)
        self.dropout = dropout

        nn.init.xavier_uniform_(self.self_fc.weight)
        nn.init.xavier_uniform_(self.neigh_fc.weight)

    def forward(
        self,
        x: Tensor,                                  # [N, F]
        degree_tasks: Sequence[Tuple[int, List[int]]],
        neighbor_list: Sequence[List[int]],
    ) -> Tensor:
        N, Fdim = x.size()
        # Build hashed features for self, and for neighbors (global + local share same hash since it's linear)
        # For TF parity we produce, per hash, a neighbor-aggregated tensor aligned to original node order.
        ids_in_concat_order: List[int] = []
        hashed_neigh_streams: List[Tensor] = []
        hashed_self_streams: List[Tensor] = []

        for hasher in self.hashers:
            group_outs: List[Tensor] = []
            ids_in_concat_order.clear()
            for (deg, node_ids), neigh_ids in zip(degree_tasks, neighbor_list):
                ids_in_concat_order.extend(node_ids)
                if deg == 0:
                    # use hashed self features for those nodes
                    self_feats = hasher(x[node_ids])                   # [len(node_ids), H]
                    # combine with a separate hashed_self path later via linear sum
                    group_outs.append(self_feats)
                else:
                    neigh_x = x[neigh_ids]                              # [len(node_ids)*deg, F]
                    hashed_neigh = hasher(neigh_x)                      # [len(node_ids)*deg, H]
                    h = hashed_neigh.view(len(node_ids), deg, -1).mean(dim=1)  # [len(node_ids), H]
                    group_outs.append(h)
            neigh_hashed = _scatter_to_nodes(N, ids_in_concat_order, group_outs)  # [N, H]
            self_hashed = hasher(x)                                     # [N, H]
            hashed_neigh_streams.append(neigh_hashed)
            hashed_self_streams.append(self_hashed)

        # Concatenate across hash streams
        neigh_concat = torch.cat(hashed_neigh_streams, dim=-1)          # [N, num_hash*H]
        self_concat  = torch.cat(hashed_self_streams,  dim=-1)          # [N, num_hash*H]

        if self.training and self.dropout > 0:
            neigh_concat = F.dropout(neigh_concat, p=self.dropout, inplace=False)
            self_concat  = F.dropout(self_concat,  p=self.dropout, inplace=False)

        neigh_mapped = self.neigh_fc(neigh_concat)                      # [N, out]
        self_mapped  = self.self_fc(self_concat)                        # [N, out]
        out = neigh_mapped + self_mapped + self.bias
        return self.act(out)


class DEMONetWeight(nn.Module):
    """
    DEMO-Net with weight-based multi-task function.
    Args (match train_nodeclf.py):
        in_dim:        input feature size
        hidden:        hidden size (paper: 64)
        out_dim:       number of classes
        num_layers:    number of hidden blocks (paper used 2)
        dropout:       dropout prob
        degree_groups: list of tuples (degree, [node_ids]) covering all nodes
        neighbor_list: list of flattened neighbor id lists aligned with degree_groups
        act:           'elu' (default)
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.6,
        degree_groups: Sequence[Tuple[int, List[int]]] = (),
        neighbor_list: Sequence[List[int]] = (),
        act: str = "elu",
    ):
        super().__init__()
        self.degree_groups = list(degree_groups)
        self.neighbor_list = list(neighbor_list)
        self.dropout = dropout

        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden] * num_layers
        for li in range(num_layers):
            layers.append(_WeightBlock(dims[li], dims[li+1], act=act, dropout=dropout))
        self.blocks = nn.ModuleList(layers)
        self.head = _WeightBlock(hidden, out_dim, act="identity", dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h, self.degree_groups, self.neighbor_list)
        logits = self.head(h, self.degree_groups, self.neighbor_list)
        return logits


class DEMONetHash(nn.Module):
    """
    DEMO-Net with hash-based multi-task function (sign hashing).
    Args additionally:
        hash_dim:   dimension per hash (H)
        num_hash:   number of independent hashes (K)
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.6,
        degree_groups: Sequence[Tuple[int, List[int]]] = (),
        neighbor_list: Sequence[List[int]] = (),
        hash_dim: int = 32,
        num_hash: int = 4,
        act: str = "elu",
    ):
        super().__init__()
        self.degree_groups = list(degree_groups)
        self.neighbor_list = list(neighbor_list)
        self.dropout = dropout

        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden] * num_layers
        for li in range(num_layers):
            layers.append(_HashBlock(dims[li], dims[li+1], hash_dim=hash_dim, num_hash=num_hash, act=act, dropout=dropout, layer_id=li))
        self.blocks = nn.ModuleList(layers)
        self.head = _HashBlock(hidden, out_dim, hash_dim=hash_dim, num_hash=num_hash, act="identity", dropout=dropout, layer_id=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h, self.degree_groups, self.neighbor_list)
        logits = self.head(h, self.degree_groups, self.neighbor_list)
        return logits
