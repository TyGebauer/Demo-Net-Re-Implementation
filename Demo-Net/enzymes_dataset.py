# enzymes_dataset.py
import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset


class EnzymesDGK(InMemoryDataset):
    """
    ENZYMES graph classification dataset in DGK format (Dortmund).
    Expects files:
      ENZYMES_A.txt
      ENZYMES_graph_indicator.txt
      ENZYMES_graph_labels.txt
      ENZYMES_node_labels.txt
      ENZYMES_node_attributes.txt
    in root/raw_dir.
    """

    def __init__(self, root="data/ENZYMES_raw", transform=None,
                 pre_transform=None, use_node_attributes=True):
        self.use_node_attributes = use_node_attributes
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [
            "ENZYMES_A.txt",
            "ENZYMES_graph_indicator.txt",
            "ENZYMES_graph_labels.txt",
            "ENZYMES_node_labels.txt",
            "ENZYMES_node_attributes.txt",
        ]

    @property
    def processed_file_names(self):
        return ["enzymes.pt"]

    def download(self):
        # We already placed files manually.
        pass

    def process(self):
        raw_dir = self.root

        # --- 1. Load files ---
        # Edges (global node indexing, 1-based)
        edge_index_list = []
        with open(os.path.join(raw_dir, "ENZYMES_A.txt"), "r") as f:
            for line in f:
                u, v = line.strip().split(",")
                edge_index_list.append([int(u) - 1, int(v) - 1])  # 0-based

        edge_index_np = np.array(edge_index_list, dtype=np.int64).T  # [2, m]

        # Graph indicator: which graph each node belongs to
        graph_indicator = []
        with open(os.path.join(raw_dir, "ENZYMES_graph_indicator.txt"), "r") as f:
            for line in f:
                graph_indicator.append(int(line.strip()))
        graph_indicator = np.array(graph_indicator, dtype=np.int64)  # 1..N graphs

        # Graph labels (targets)
        graph_labels = []
        with open(os.path.join(raw_dir, "ENZYMES_graph_labels.txt"), "r") as f:
            for line in f:
                graph_labels.append(int(line.strip()))
        graph_labels = np.array(graph_labels, dtype=np.int64)

        # Node labels (categorical, we’ll one-hot encode)
        node_labels = []
        with open(os.path.join(raw_dir, "ENZYMES_node_labels.txt"), "r") as f:
            for line in f:
                node_labels.append(int(line.strip()))
        node_labels = np.array(node_labels, dtype=np.int64)

        # Node attributes (continuous)
        node_attr = np.loadtxt(
            os.path.join(raw_dir, "ENZYMES_node_attributes.txt"),
            delimiter=","
        )
        if node_attr.ndim == 1:
            node_attr = node_attr.reshape(-1, 1)

        # One-hot encode node labels
        num_node_label_classes = node_labels.max()
        node_label_onehot = np.eye(num_node_label_classes)[node_labels - 1]

        if self.use_node_attributes:
            x_all = np.concatenate([node_attr, node_label_onehot], axis=1)
        else:
            x_all = node_label_onehot

        x_all = torch.tensor(x_all, dtype=torch.float)

        # --- 2. Split into individual graphs ---
        num_nodes = graph_indicator.shape[0]
        graph_ids = np.unique(graph_indicator)
        data_list = []

        edge_index_global = torch.tensor(edge_index_np, dtype=torch.long)

        for g_id in graph_ids:
            # nodes for this graph
            node_mask = (graph_indicator == g_id)
            node_idx = np.nonzero(node_mask)[0]  # 0-based indices

            # mapping global -> local
            global_to_local = {int(n): i for i, n in enumerate(node_idx)}

            # select node features
            x = x_all[node_idx]

            # select edges where both endpoints are in this graph
            mask_edges = np.isin(edge_index_np[0], node_idx) & np.isin(edge_index_np[1], node_idx)
            edges_sub = edge_index_np[:, mask_edges]

            # remap edge indices to local
            src = [global_to_local[int(u)] for u in edges_sub[0]]
            dst = [global_to_local[int(v)] for v in edges_sub[1]]
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            y = torch.tensor([graph_labels[g_id - 1] - 1], dtype=torch.long)  # 0-based class

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # --- 3. Save as InMemoryDataset ---
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    # quick sanity check
    dataset = EnzymesDGK(root="data/ENZYMES_raw")
    print(dataset)
    print("Num graphs:", len(dataset))
    print("Num classes:", dataset.num_classes if hasattr(dataset, "num_classes") else "n/a")
    print("First graph:", dataset[0])
