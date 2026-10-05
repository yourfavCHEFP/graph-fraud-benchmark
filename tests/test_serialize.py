from src.graph_fraud_benchmark.serialize import build_prompt


def _subgraph_with_label(label_value):
    """A subgraph dict shaped exactly like sample_subgraphs.py produces,
    including the 'label' key that build_prompt must never read."""
    return {
        "seed_txn_id": 1342,
        "label": label_value,
        "nodes": {
            "1342": {"TransactionAmt": 100.0, "ProductCD": "W", "card4": "visa",
                     "card6": "credit", "P_emaildomain": "gmail.com",
                     "DeviceType": "mobile", "TransactionDT": 0},
            "1133": {"TransactionAmt": 50.0, "ProductCD": "W", "card4": "visa",
                     "card6": "credit", "P_emaildomain": "gmail.com",
                     "DeviceType": "mobile", "TransactionDT": 86400},
        },
        "edges": [{"a": 1342, "b": 1133, "shared_via": "card1=C1"}],
    }


def test_label_never_appears_in_prompt_text():
    for label in (0, 1):
        prompt = build_prompt(_subgraph_with_label(label))
        assert "label" not in prompt.lower()
        assert "fraud" not in prompt.lower()  # the data itself should never say the word
        assert "isfraud" not in prompt.lower()


def test_seed_is_called_out_and_neighbor_is_present():
    prompt = build_prompt(_subgraph_with_label(1))
    assert "THE TRANSACTION IN QUESTION" in prompt
    assert "1342" in prompt
    assert "1133" in prompt
    assert "share card1=C1" in prompt


def test_no_edges_produces_explicit_none_found_message():
    sub = _subgraph_with_label(0)
    sub["edges"] = []
    sub["nodes"] = {"1342": sub["nodes"]["1342"]}
    prompt = build_prompt(sub)
    assert "none found" in prompt.lower()
