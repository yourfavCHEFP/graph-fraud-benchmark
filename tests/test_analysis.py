import pandas as pd

from src.graph_fraud_benchmark.analysis import (
    build_report_card,
    score_classification_rows,
    score_ring_root_rows,
)

GNN_THRESHOLD = 0.1980

# Realistic multi-digit IDs on purpose: the ring-root ID regex requires
# 3+ digits (real TransactionIDs always are), so small single-digit
# fixture IDs would silently fail to match and give false results.
VALID_TXN_IDS = {101: [101, 105, 106], 102: [102, 107, 108], 103: [103, 109], 104: [104, 110]}
NODE_IS_FRAUD = {101: 1, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0}

LABELS = pd.DataFrame(
    {
        "seed_txn_id": [101, 102, 103, 104],
        "label": [1, 1, 0, 0],
        "gnn_pred": [0.50, 0.05, 0.10, 0.30],
    }
).set_index("seed_txn_id")

CLASSIFY_ROWS = pd.DataFrame(
    [
        {"model_name": "m1", "seed_txn_id": 101, "raw_response": "FRAUD, transaction 105 was related."},
        {"model_name": "m1", "seed_txn_id": 102, "raw_response": "LEGIT"},
        {"model_name": "m1", "seed_txn_id": 103, "raw_response": "LEGIT"},
        {"model_name": "m1", "seed_txn_id": 104, "raw_response": "NOT FRAUD"},
    ]
)

RING_ROOT_ROWS = pd.DataFrame(
    [
        {"model_name": "m1", "seed_txn_id": 101, "raw_response": "Transaction 105 did it."},
        {"model_name": "m1", "seed_txn_id": 102, "raw_response": "NONE"},
        {"model_name": "m1", "seed_txn_id": 103, "raw_response": "NONE"},
        {"model_name": "m1", "seed_txn_id": 104, "raw_response": "Transaction 110 looks weird."},
    ]
)


def test_classification_scoring_matches_hand_computed_values():
    scored = score_classification_rows(CLASSIFY_ROWS, LABELS, VALID_TXN_IDS)
    correct_by_seed = dict(zip(scored.seed_txn_id, scored.correct))
    assert correct_by_seed == {101: True, 102: False, 103: True, 104: True}

    aware_by_seed = dict(zip(scored.seed_txn_id, scored.graph_aware))
    assert aware_by_seed == {101: True, 102: False, 103: False, 104: False}


def test_ring_root_scoring_matches_hand_computed_values():
    scored = score_ring_root_rows(RING_ROOT_ROWS, LABELS, VALID_TXN_IDS, NODE_IS_FRAUD)
    correct_by_seed = dict(zip(scored.seed_txn_id, scored.correct))
    assert correct_by_seed == {101: False, 102: False, 103: True, 104: False}


def test_report_card_accuracy_and_gnn_agreement_buckets():
    classification_scores = score_classification_rows(CLASSIFY_ROWS, LABELS, VALID_TXN_IDS)
    ring_root_scores = score_ring_root_rows(RING_ROOT_ROWS, LABELS, VALID_TXN_IDS, NODE_IS_FRAUD)
    report = build_report_card(classification_scores, ring_root_scores, LABELS, GNN_THRESHOLD)

    classify_row = report[report.task == "classify_transaction"].iloc[0]
    assert classify_row.n_samples == 4
    assert classify_row.accuracy == 0.75
    assert classify_row.ambiguous_rate == 0.0
    assert classify_row.graph_aware_rate == 0.25

    assert classify_row["gnn_agreement__both_correct"] == 2
    assert classify_row["gnn_agreement__model_caught_what_gnn_missed"] == 1
    assert classify_row["gnn_agreement__model_missed_what_gnn_caught"] == 0
    assert classify_row["gnn_agreement__both_missed"] == 1
    assert classify_row["gnn_agreement__unscorable"] == 0

    ring_row = report[report.task == "identify_ring_root"].iloc[0]
    assert ring_row.n_samples == 4
    assert ring_row.accuracy == 0.25


# --- Blank responses are unanswered samples, not results ------------------
#
# Regression tests for the run that produced a CSV of blank raw_response
# cells: pandas read them as float('nan'), _classify_logic called .upper()
# on them, and analysis died with
# AttributeError: 'float' object has no attribute 'upper'.

BLANK_CLASSIFY_ROWS = pd.DataFrame(
    [
        {"model_name": "m1", "seed_txn_id": 101, "raw_response": "FRAUD"},
        {"model_name": "m1", "seed_txn_id": 102, "raw_response": float("nan")},
        {"model_name": "m1", "seed_txn_id": 103, "raw_response": "   "},
        {"model_name": "m1", "seed_txn_id": 104, "raw_response": None},
    ]
)

BLANK_RING_ROOT_ROWS = pd.DataFrame(
    [
        {"model_name": "m1", "seed_txn_id": 101, "raw_response": "Transaction 105 did it."},
        # seed 103 is LEGIT, so "NONE" would be correct -- a blank must not
        # be scored as if the model had answered NONE.
        {"model_name": "m1", "seed_txn_id": 103, "raw_response": ""},
    ]
)


def test_blank_classification_responses_are_unscorable_not_crashes():
    scored = score_classification_rows(BLANK_CLASSIFY_ROWS, LABELS, VALID_TXN_IDS)
    correct_by_seed = dict(zip(scored.seed_txn_id, scored.correct))
    # Seed 101 answered "FRAUD" and is a fraud seed, so it still scores.
    # The three blanks are unanswered samples, not wrong answers.
    assert correct_by_seed == {101: True, 102: None, 103: None, 104: None}
    confidence_by_seed = dict(zip(scored.seed_txn_id, scored.parse_confidence))
    assert confidence_by_seed[101] == "clear"
    for seed in (102, 103, 104):
        assert confidence_by_seed[seed] == "empty_response"


def test_blank_ring_root_response_is_never_scored_correct():
    scored = score_ring_root_rows(BLANK_RING_ROOT_ROWS, LABELS, VALID_TXN_IDS, NODE_IS_FRAUD)
    correct_by_seed = dict(zip(scored.seed_txn_id, scored.correct))
    assert correct_by_seed[101] is False
    # The regression: seed 103 is legit, so a silent/blank response parsed to
    # None and used to be scored True.
    assert correct_by_seed[103] is None


def test_nan_response_does_not_score_as_graph_aware():
    scored = score_classification_rows(BLANK_CLASSIFY_ROWS, LABELS, VALID_TXN_IDS)
    aware_by_seed = dict(zip(scored.seed_txn_id, scored.graph_aware))
    # "FRAUD" names no other transaction, and the blanks name nothing either.
    assert aware_by_seed[101] is False
    assert aware_by_seed[102] is False
