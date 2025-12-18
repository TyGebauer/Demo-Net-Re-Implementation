import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class PSCNModel(nn.Module):
    """
    Simplified PATCHY-SAN baseline (no canonical ordering).
    Instead uses a small MLP on pooled node features.
    """
    def __init__(self, in_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = data.x
        batch = data.batch
        g = global_mean_pool(x, batch)
        h = F.relu(self.lin1(g))
        h = F.relu(self.lin2(h))
        return self.classifier(h)
