"""
train_gnn.py
────────────
Phase 4 Training Script

Constructs a multi-resource heterogeneous security graph,
trains the SecurityHeteroGNN model, and evaluates graph-based
detection metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import (
    LABELED_DIR, GNN_MODEL_PATH, GNN_NUM_EPOCHS,
    GNN_LEARNING_RATE, GNN_WEIGHT_DECAY
)
from src.stage2_graph.graph_builder import build_hetero_security_graph
from src.stage2_graph.gnn_model import SecurityHeteroGNN
from src.utils.logger import get_logger

log = get_logger("train_gnn")


def create_train_test_masks(num_nodes: int, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_nodes)
    split = int(num_nodes * (1 - test_ratio))
    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[indices[:split]] = True
    test_mask[indices[split:]] = True
    return torch.tensor(train_mask), torch.tensor(test_mask)


def main():
    feat_npz = LABELED_DIR / "features_dataset.npz"
    meta_csv = LABELED_DIR / "features_metadata.csv"

    if not feat_npz.exists() or not meta_csv.exists():
        log.error("Extracted features not found. Run scripts/train_rf.py first.")
        sys.exit(1)

    data = np.load(feat_npz)
    X, y = data["X"], data["y"]
    df_meta = pd.read_csv(meta_csv)

    # 1. Build Heterogeneous Graph
    graph = build_hetero_security_graph(X, y, df_meta)

    # Convert to PyTorch tensors
    x_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in graph.node_features.items()}
    y_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in graph.node_labels.items()}
    edge_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in graph.edge_index.items()}

    # Train / Test Masks per node type
    train_masks, test_masks = {}, {}
    for k, v in graph.node_features.items():
        tr_m, te_m = create_train_test_masks(len(v), test_ratio=0.20)
        train_masks[k] = tr_m
        test_masks[k] = te_m

    # 2. Model, Optimizer, Loss
    model = SecurityHeteroGNN(in_dim=X.shape[1], hidden_dim=128, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    log.info(f"Training Heterogeneous GNN for {GNN_NUM_EPOCHS} epochs...")
    best_f1 = 0.0

    for epoch in range(1, GNN_NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, edge_dict)

        total_loss = 0.0
        for ntype in out:
            mask = train_masks[ntype]
            total_loss += criterion(out[ntype][mask], y_dict[ntype][mask])

        total_loss.backward()
        optimizer.step()

        # Evaluate every 20 epochs
        if epoch % 20 == 0 or epoch == GNN_NUM_EPOCHS:
            model.eval()
            with torch.no_grad():
                val_out = model(x_dict, edge_dict)
                all_preds, all_trues = [], []
                for ntype in val_out:
                    mask = test_masks[ntype]
                    preds = val_out[ntype][mask].argmax(dim=-1).cpu().numpy()
                    trues = y_dict[ntype][mask].cpu().numpy()
                    all_preds.extend(preds)
                    all_trues.extend(trues)

                macro_f1 = f1_score(all_trues, all_preds, average="macro")
                log.info(f"Epoch {epoch:03d}/{GNN_NUM_EPOCHS} | Loss: {total_loss.item():.4f} | Test Macro F1: {macro_f1:.4f}")

                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    GNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), GNN_MODEL_PATH)

    log.info(f"Best GNN model saved to {GNN_MODEL_PATH} (Test Macro F1: {best_f1:.4f})")

    # 3. Final Per-Node-Type Evaluation
    model.eval()
    print("\n" + "="*55)
    print("GNN GRAPH NODE CLASSIFICATION RESULTS")
    print("="*55)
    with torch.no_grad():
        final_out = model(x_dict, edge_dict)
        for ntype in final_out:
            mask = test_masks[ntype]
            preds = final_out[ntype][mask].argmax(dim=-1).cpu().numpy()
            trues = y_dict[ntype][mask].cpu().numpy()
            print(f"\n--- Node Type: {ntype} ({len(trues)} test nodes) ---")
            print(classification_report(trues, preds, target_names=["COMPLIANT", "MISCONFIGURED"], digits=4))
    print("="*55)


if __name__ == "__main__":
    main()
