from types import SimpleNamespace

import torch

from src.graph_fraud_benchmark.gnn_predictions import (
    build_graph_features,
    chronological_split_indices,
)


def test_chronological_split_is_disjoint_and_ordered():
    timestamps = [4, 1, 5, 3, 2, 0, 7, 6, 8, 9]
    train, validation, test = chronological_split_indices(timestamps)

    assert len(train) == 7
    assert len(validation) == 1
    assert len(test) == 2
    assert not set(train.tolist()) & set(validation.tolist())
    assert not set(train.tolist()) & set(test.tolist())
    assert not set(validation.tolist()) & set(test.tolist())
    assert timestamps[test[0]] <= timestamps[test[1]]


def test_feature_transform_reads_amount_and_degrees_by_name():
    graph = SimpleNamespace(
        x=torch.tensor([[4, 4, 2.3, 9, 99, 0, 3, 0]], dtype=torch.float32)
    )

    features = build_graph_features(graph)

    assert features.shape == (1, 16)
    assert torch.isclose(features[0, 0], torch.tensor(2.3))
    assert torch.isclose(features[0, 1], torch.log1p(torch.tensor(9.0)))
