import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class DeepWLModel(nn.Module):
    """
    Simplified DeepWL baseline used in many GNN benchmark repos.
    2-layer MLP on graph-level WL embeddings.
    """
    def __init__(self, in_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        # Graph-level mean pooling over raw node features
        x = data.x
        batch = data.batch
        g = global_mean_pool(x, batch)
        h = F.relu(self.lin1(g))
        h = F.relu(self.lin2(h))
        return self.classifier(h)
