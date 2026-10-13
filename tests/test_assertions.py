from src.graph_fraud_benchmark.assertions import (
    graph_awareness,
    score_classification,
    score_ring_root,
)


# --- score_classification ------------------------------------------------

def test_correct_fraud_call():
    s = score_classification(predicted_fraud=True, parse_confidence="clear", true_label=1)
    assert s.correct is True


def test_wrong_fraud_call():
    s = score_classification(predicted_fraud=False, parse_confidence="clear", true_label=1)
    assert s.correct is False


def test_ambiguous_parse_is_unscorable_not_wrong():
    s = score_classification(predicted_fraud=True, parse_confidence="ambiguous", true_label=1)
    assert s.correct is None


# --- graph_awareness -------------------------------------------------------

def test_aware_when_neighbor_mentioned():
    assert graph_awareness(
        "This looks fraudulent because of transaction 1133 sharing the same card.",
        seed_txn_id=1342, neighbor_txn_ids=[1133, 1027],
    ) is True


def test_not_aware_when_only_seed_mentioned():
    assert graph_awareness(
        "Transaction 1342 itself seems fine.",
        seed_txn_id=1342, neighbor_txn_ids=[1133, 1027],
    ) is False


def test_not_aware_when_nothing_mentioned():
    assert graph_awareness(
        "This transaction amount looks unusually high.",
        seed_txn_id=1342, neighbor_txn_ids=[1133, 1027],
    ) is False


# --- score_ring_root (uses the proxy ground truth -- see tasks.py docstring) --

def test_legit_seed_saying_none_is_correct():
    assert score_ring_root(predicted_txn_id=None, seed_true_label=0, node_is_fraud={}).correct is True


def test_legit_seed_naming_a_node_is_wrong():
    assert score_ring_root(predicted_txn_id=111, seed_true_label=0, node_is_fraud={111: 0}).correct is False


def test_fraud_seed_saying_none_is_wrong():
    assert score_ring_root(predicted_txn_id=None, seed_true_label=1, node_is_fraud={}).correct is False


def test_fraud_seed_pointing_at_fraud_neighbor_is_correct():
    assert score_ring_root(predicted_txn_id=111, seed_true_label=1, node_is_fraud={111: 1}).correct is True


def test_fraud_seed_pointing_at_legit_neighbor_is_wrong():
    assert score_ring_root(predicted_txn_id=111, seed_true_label=1, node_is_fraud={111: 0}).correct is False


def test_graph_awareness_tolerates_nan_blank_and_none():
    from src.graph_fraud_benchmark.assertions import graph_awareness

    # A blank results-CSV cell arrives as float('nan'); the ID regex used to
    # raise TypeError on it.
    assert graph_awareness(float("nan"), 101, [105, 106]) is False
    assert graph_awareness("", 101, [105, 106]) is False
    assert graph_awareness(None, 101, [105, 106]) is False
    assert graph_awareness("Transaction 105 is odd", 101, [105, 106]) is True
