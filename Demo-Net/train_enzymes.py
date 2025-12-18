import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold

# --- Dataset ---
from enzymes_dataset import EnzymesDGK

# --- Baseline Models ---
from models_graph.deepwl import DeepWLModel
from models_graph.dcnn import DCNNModel
from models_graph.patchy_san import PSCNModel
from models_graph.diffpool import DiffPoolModel

# --- DEMO-Net Models ---
from models_graph.demo_net_graph import DEMONetWeightGraph, DEMONetHashGraph


# =====================================================================
# Unified model builder
# =====================================================================

def build_model(model_name, in_dim, num_classes):

    if model_name == "deepwl":
        return DeepWLModel(in_dim=in_dim, num_classes=num_classes)

    elif model_name == "dcnn":
        return DCNNModel(in_dim=in_dim, num_classes=num_classes)

    elif model_name == "pscn":
        return PSCNModel(in_dim=in_dim, num_classes=num_classes)

    elif model_name == "diffpool":
        return DiffPoolModel(in_dim=in_dim, num_classes=num_classes)

    elif model_name == "demo_hash":
        return DEMONetHashGraph(
            in_dim=in_dim,
            hidden_dim=64,          # <-- correct arg name
            num_classes=num_classes,
            num_layers=2,
            dropout=0.6,
            num_hash=4,
            hash_dim=128
        )

    elif model_name == "demo_weight":
        return DEMONetWeightGraph(
            in_dim=in_dim,
            hidden_dim=64,          # <-- correct arg name
            num_classes=num_classes,
            num_layers=2,
            dropout=0.6
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =====================================================================
# Training + Evaluation
# =====================================================================

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=-1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    return correct / total if total > 0 else 0.0


# =====================================================================
# 10-Fold Cross Validation
# =====================================================================

def run_cross_validation(model_name):

    # ---- Load dataset ----
    dataset = EnzymesDGK(root="data/ENZYMES_raw")
    labels = np.array([int(d.y.item()) for d in dataset])
    in_dim = dataset[0].x.size(1)
    num_classes = len(set(labels))

    print(f"\nDataset Loaded: ENZYMES")
    print(f"Graphs = {len(dataset)} | Classes = {num_classes} | Node feat dim = {in_dim}")
    print(f"Running model: {model_name}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- CV ----
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_acc = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels), start=1):

        print(f"\n========== Fold {fold}/10 ==========")

        train_ds = [dataset[i] for i in train_idx]
        test_ds = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # ---- Build model ----
        model = build_model(model_name, in_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_acc = 0

        # ---- Train 100 epochs ----
        for epoch in range(1, 101):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            acc = eval_accuracy(model, test_loader, device)

            if acc > best_acc:
                best_acc = acc

            if epoch % 10 == 0:
                print(f"Fold {fold} | Epoch {epoch} | Loss {loss:.4f} | Acc {acc:.4f} | Best {best_acc:.4f}")

        fold_acc.append(best_acc)

    # ---- Summary ----
    mean_acc = float(np.mean(fold_acc))
    std_acc = float(np.std(fold_acc))

    print("\n=================== FINAL 10-FOLD RESULTS ===================")
    for i, a in enumerate(fold_acc, start=1):
        print(f"Fold {i:02d}: {a:.4f}")
    print(f"\nModel: {model_name}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print("==============================================================")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["deepwl", "dcnn", "pscn", "diffpool", "demo_hash", "demo_weight"],
        help="Choose a model from the DEMO-Net paper baselines"
    )

    args = parser.parse_args()
    run_cross_validation(args.model)
