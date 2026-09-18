"""
gnn_model.py
────────────
Phase 4 — Heterogeneous Graph Neural Network (GNN)

Implements a relational graph convolution network in pure PyTorch:
  - Supports multiple node types (Pod, IAM_Storage, Lambda)
  - Message passing across heterogeneous cross-resource edges:
      (Pod -> IAM_Storage), (Lambda -> IAM_Storage)
  - Combines self-node representation with aggregated neighbor context
    to detect chained cross-file misconfigurations.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import FEATURE_VECTOR_DIM, GNN_HIDDEN_DIM


class RelationalMessagePassingLayer(nn.Module):
    """
    Message passing layer that transforms source node features and
    aggregates messages across relational edge indices.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.src_proj = nn.Linear(in_dim, out_dim)
        self.self_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x_src: (N_src, D), x_dst: (N_dst, D), edge_index: (2, E)
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        messages = self.src_proj(x_src[src_nodes])  # (E, out_dim)

        # Scatter add aggregation into destination nodes
        out = self.self_proj(x_dst)  # (N_dst, out_dim)
        out = out.index_add(0, dst_nodes, messages)
        return F.relu(out)


class SecurityHeteroGNN(nn.Module):
    """
    Heterogeneous GNN architecture for cross-domain cloud configuration auditing.
    """
    def __init__(self, in_dim: int = FEATURE_VECTOR_DIM,
                 hidden_dim: int = GNN_HIDDEN_DIM,
                 num_classes: int = 2):
        super().__init__()
        # Node specific initial projections
        self.node_projections = nn.ModuleDict({
            "Pod": nn.Linear(in_dim, hidden_dim),
            "IAM_Storage": nn.Linear(in_dim, hidden_dim),
            "Lambda": nn.Linear(in_dim, hidden_dim)
        })

        # Relational message passing
        self.conv_pod_to_iam = RelationalMessagePassingLayer(hidden_dim, hidden_dim)
        self.conv_lam_to_iam = RelationalMessagePassingLayer(hidden_dim, hidden_dim)
        self.conv_iam_to_pod = RelationalMessagePassingLayer(hidden_dim, hidden_dim)

        # Node classifiers
        self.classifier_pod = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.classifier_iam = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.classifier_lambda = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x_dict: dict[str, torch.Tensor],
                edge_dict: dict[tuple, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 1. Initial node projections
        h = {k: F.relu(self.node_projections[k](v)) for k, v in x_dict.items()}

        # 2. Relational message passing across edge types
        # Pod -> IAM_Storage
        if ("Pod", "accesses", "IAM_Storage") in edge_dict:
            e_idx = edge_dict[("Pod", "accesses", "IAM_Storage")]
            h["IAM_Storage"] = self.conv_pod_to_iam(h["Pod"], h["IAM_Storage"], e_idx)

            # Reverse flow: context from IAM_Storage flows back to Pod
            rev_idx = torch.stack([e_idx[1], e_idx[0]], dim=0)
            h["Pod"] = self.conv_iam_to_pod(h["IAM_Storage"], h["Pod"], rev_idx)

        # Lambda -> IAM_Storage
        if ("Lambda", "assumes", "IAM_Storage") in edge_dict:
            e_idx = edge_dict[("Lambda", "assumes", "IAM_Storage")]
            h["IAM_Storage"] = self.conv_lam_to_iam(h["Lambda"], h["IAM_Storage"], e_idx)

        # 3. Classify each node type with its enriched context
        out = {
            "Pod": self.classifier_pod(h["Pod"]),
            "IAM_Storage": self.classifier_iam(h["IAM_Storage"]),
            "Lambda": self.classifier_lambda(h["Lambda"]),
        }
        return out
