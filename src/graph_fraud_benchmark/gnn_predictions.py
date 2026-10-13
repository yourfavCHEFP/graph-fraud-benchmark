"""Export leakage-safe test predictions from graph-fraud-ai artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.nn import SAGEConv

GRAPH_COLUMNS = [
    "degree",
    "transaction_entity_count",
    "log_transaction_amount",
    "card_degree",
    "email_degree",
    "device_degree",
    "address_degree",
    "node_type_id",
]

FEATURE_NAMES = [
    "transaction_amount",
    "log_card_degree",
    "log_email_degree",
    "log_device_degree",
    "log_address_degree",
    "entity_degree_mean",
    "entity_degree_max",
    "entity_degree_min",
    "entity_degree_std",
    "card_ratio",
    "email_ratio",
    "device_ratio",
    "address_ratio",
    "entity_concentration",
    "degree_imbalance",
    "hub_entity_count",
]


class _FraudGraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        node_features = self.dropout(torch.relu(self.conv1(node_features, edge_index)))
        node_features = self.dropout(torch.relu(self.conv2(node_features, edge_index)))
        return self.classifier(node_features)


def build_graph_features(graph) -> torch.Tensor:
    """Reproduce graph-fraud-ai's named, 16-column production feature transform."""
    if graph.x.ndim != 2 or graph.x.shape[1] != len(GRAPH_COLUMNS):
        raise ValueError(
            f"Expected graph.x with {len(GRAPH_COLUMNS)} columns, got {tuple(graph.x.shape)}"
        )

    columns = {name: index for index, name in enumerate(GRAPH_COLUMNS)}
    node_features = graph.x.float()
    amount = node_features[:, columns["log_transaction_amount"]]
    degrees = [
        node_features[:, columns["card_degree"]],
        node_features[:, columns["email_degree"]],
        node_features[:, columns["device_degree"]],
        node_features[:, columns["address_degree"]],
    ]
    logged_degrees = [torch.log1p(degree) for degree in degrees]
    entity_stack = torch.stack(logged_degrees, dim=1)
    degree_mean = entity_stack.mean(dim=1)
    degree_max = entity_stack.max(dim=1).values
    degree_min = entity_stack.min(dim=1).values
    degree_std = entity_stack.std(dim=1)
    total_degree = sum(logged_degrees) + 1e-6
    ratios = [degree / total_degree for degree in logged_degrees]
    hub_count = sum(
        (degree >= cap).float() for degree, cap in zip(degrees, (100, 1000, 1000, 1000))
    )

    return torch.stack(
        [
            amount,
            *logged_degrees,
            degree_mean,
            degree_max,
            degree_min,
            degree_std,
            *ratios,
            degree_max / total_degree,
            degree_std / (degree_mean + 1e-6),
            hub_count,
        ],
        dim=1,
    )


def chronological_split_indices(
    timestamps,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(torch.as_tensor(timestamps, dtype=torch.float64))
    train_end = int(0.70 * len(order))
    validation_end = train_end + int(0.15 * len(order))
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def run(
    graph_path: Path,
    checkpoint_path: Path,
    node_features_path: Path,
    transactions_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = pd.read_csv(
        transactions_csv,
        usecols=["TransactionID", "TransactionDT", "isFraud"],
    )
    node_ids = pd.read_parquet(node_features_path, columns=["node_id"])["node_id"]

    transaction_count = len(raw)
    expected_ids = [f"transaction_{value}" for value in raw["TransactionID"]]
    if (
        len(node_ids) != graph.num_nodes
        or node_ids.iloc[:transaction_count].tolist() != expected_ids
    ):
        raise ValueError("Graph node order does not match the raw TransactionID order")
    if not bool(graph.transaction_mask[:transaction_count].all()) or bool(
        graph.transaction_mask[transaction_count:].any()
    ):
        raise ValueError("Expected transaction nodes first, followed by entity nodes")
    if not torch.equal(
        graph.y[:transaction_count].cpu(),
        torch.tensor(raw["isFraud"].to_numpy(), dtype=torch.long),
    ):
        raise ValueError("Graph labels do not match train_transaction.csv")
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise ValueError(
            "Checkpoint feature names do not match the production feature contract"
        )

    features = build_graph_features(graph)
    train_indices, _, test_indices = chronological_split_indices(raw["TransactionDT"])
    mean = features[train_indices].mean(dim=0, keepdim=True)
    std = features[train_indices].std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    if not torch.allclose(mean, checkpoint["normalization_mean"], atol=1e-5):
        raise ValueError(
            "Checkpoint normalization mean does not match chronological train features"
        )
    if not torch.allclose(std, checkpoint["normalization_std"], atol=1e-5):
        raise ValueError(
            "Checkpoint normalization std does not match chronological train features"
        )

    model = _FraudGraphSAGE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalized = (features - checkpoint["normalization_mean"]) / checkpoint[
        "normalization_std"
    ]
    with torch.no_grad():
        probabilities = torch.softmax(model(normalized, graph.edge_index), dim=1)[:, 1]

    test_rows = test_indices.numpy()
    labels = raw["isFraud"].to_numpy(dtype="int64")[test_rows]
    scores = probabilities[test_indices].numpy()
    threshold = float(checkpoint["validation_threshold"])
    result = pd.DataFrame(
        {
            "TransactionID": raw["TransactionID"].to_numpy()[test_rows],
            "isFraud": labels,
            "gnn_pred": scores,
            "threshold": threshold,
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_csv.with_suffix(out_csv.suffix + ".tmp")
    result.to_csv(temporary_path, index=False)
    temporary_path.replace(out_csv)

    predicted = scores >= threshold
    print(
        f"[gnn_predictions] wrote {len(result)} chronological test predictions to {out_csv}"
    )
    print(
        f"[gnn_predictions] ROC-AUC={roc_auc_score(labels, scores):.4f} PR-AUC={average_precision_score(labels, scores):.4f} precision={precision_score(labels, predicted, zero_division=0):.4f} "
        f"recall={recall_score(labels, predicted, zero_division=0):.4f} F1={f1_score(labels, predicted, zero_division=0):.4f} threshold={threshold:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=Path("data/gnn/fraud_graph_ready.pt")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/gnn/graphsage_improved.pt")
    )
    parser.add_argument(
        "--node-features", type=Path, default=Path("data/gnn/node_features.parquet")
    )
    parser.add_argument(
        "--transactions-csv", type=Path, default=Path("data/raw/train_transaction.csv")
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("data/gnn/test_predictions.csv")
    )
    args = parser.parse_args()
    run(
        args.graph,
        args.checkpoint,
        args.node_features,
        args.transactions_csv,
        args.out_csv,
    )


if __name__ == "__main__":
    main()
