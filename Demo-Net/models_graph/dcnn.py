import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class DCNNModel(nn.Module):
    """
    A small DCNN-like baseline (simplified).
    """
    def __init__(self, in_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.theta = nn.Linear(in_dim, hidden_dim)
        self.phi = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Very simplified diffusion step:
        x1 = self.theta(x)
        x2 = self.phi(x)
        x = F.relu(x1 + x2)

        g = global_mean_pool(x, batch)
        return self.classifier(g)
