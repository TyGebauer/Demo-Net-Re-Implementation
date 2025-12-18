import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GraphConv

class DiffPoolModel(nn.Module):
    """
    Very small GraphConv → pooling → classifier baseline
    similar to how DEMO-Net authors compare.
    """
    def __init__(self, in_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        g = global_mean_pool(h, batch)
        return self.classifier(g)
