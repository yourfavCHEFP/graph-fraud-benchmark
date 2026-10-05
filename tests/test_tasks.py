"""
Tests for tasks.py's parsing logic. These need no Kaggle access, no
real dataset, and no kbench install -- that's the whole point of
keeping the logic in plain functions separate from the @task wrappers.
"""

from src.graph_fraud_benchmark.tasks import (
    _classify_logic,
    _ring_root_logic,
    classify_transaction,
    identify_ring_root,
)


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def prompt(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_classification_task_uses_kaggle_prompt_api_and_scores_answer():
    llm = _FakeLLM("FRAUD")

    result = classify_transaction(llm, "seed transaction description", true_label=1)

    assert result is True
    assert len(llm.prompts) == 1
    assert "seed transaction description" in llm.prompts[0]
    assert "true_label" not in llm.prompts[0]


def test_classification_task_marks_ambiguous_answer_incorrect():
    assert (
        classify_transaction(_FakeLLM("unclear"), "description", true_label=1) is False
    )


def test_ring_root_task_scores_other_fraud_node_proxy():
    llm = _FakeLLM("Transaction 1133")

    result = identify_ring_root(
        llm,
        "network description",
        valid_txn_ids=[9999, 1133],
        seed_txn_id=9999,
        seed_true_label=1,
        fraud_txn_ids=[1133],
    )

    assert result is True
    assert "fraud_txn_ids" not in llm.prompts[0]


def test_ring_root_task_never_accepts_seed_as_its_own_root():
    llm = _FakeLLM("Transaction 9999")

    result = identify_ring_root(
        llm,
        "network description",
        valid_txn_ids=[9999, 1133],
        seed_txn_id=9999,
        seed_true_label=1,
        fraud_txn_ids=[9999],
    )

    assert result is False


# --- classify_transaction ---------------------------------------------


def test_clear_fraud():
    r = _classify_logic("FRAUD")
    assert r.predicted_fraud is True
    assert r.parse_confidence == "clear"


def test_clear_legit():
    r = _classify_logic("LEGIT")
    assert r.predicted_fraud is False
    assert r.parse_confidence == "clear"


def test_negated_fraud_is_clear_legit():
    """Regression: 'NOT FRAUD' contains the substring 'FRAUD' and was
    once misparsed as ambiguous."""
    for text in ["NOT FRAUD", "not-fraud", "This is NOT FRAUDULENT activity."]:
        r = _classify_logic(text)
        assert r.predicted_fraud is False
        assert r.parse_confidence == "clear", f"failed on: {text!r}"


def test_ambiguous_when_both_keywords_present():
    r = _classify_logic("I am not sure, could be fraud or legit.")
    assert r.parse_confidence == "ambiguous"


def test_ambiguous_when_neither_keyword_present():
    r = _classify_logic("The weather is nice today.")
    assert r.parse_confidence == "ambiguous"


# --- identify_ring_root -------------------------------------------------


def test_ring_root_accepts_valid_other_transaction():
    assert (
        _ring_root_logic(
            "Transaction 1133 is the root.", [1133, 1027], seed_txn_id=9999
        )
        == 1133
    )


def test_ring_root_rejects_the_seed_itself():
    """The bug: valid_txn_ids as built by serialize.py includes the
    seed. A model naming the seed as its own ring root must NOT be
    accepted -- the prompt explicitly asks for an OTHER transaction."""
    seed_id = 123
    valid_ids = [123, 456]  # seed included, exactly as serialize.py produces
    assert _ring_root_logic("Transaction 123 is the root", valid_ids, seed_id) is None


def test_ring_root_rejects_hallucinated_id():
    assert (
        _ring_root_logic(
            "Transaction 9999999 looks suspicious.", [111, 222], seed_txn_id=333
        )
        is None
    )


def test_ring_root_none_is_whole_word_only():
    """Regression: a naive substring check for 'NONE' would false-match
    inside an unrelated word like 'nonetheless'."""
    assert _ring_root_logic("NONE", [111, 222], seed_txn_id=333) is None
    # "nonetheless" contains "NONE" as a substring but is not the word NONE
    result = _ring_root_logic(
        "Nonetheless, transaction 111 looks responsible.", [111, 222], seed_txn_id=333
    )
    assert (
        result == 111
    ), "a real candidate ID should still be found, not swallowed by the substring match"


def test_ring_root_no_candidates_left_after_excluding_seed():
    """If the seed is the only node in valid_txn_ids (no real neighbors),
    there is nothing valid left to point at -- any mentioned ID must be
    rejected, not just the seed's own ID."""
    assert _ring_root_logic("Transaction 123 did it.", [123], seed_txn_id=123) is None


# --- Blank / non-string responses must not crash the parsers --------------

def test_classify_logic_tolerates_nan_blank_and_none():
    # pandas hands back float('nan') for an empty raw_response cell.
    assert _classify_logic(float("nan")).parse_confidence == "ambiguous"
    assert _classify_logic("").parse_confidence == "ambiguous"
    assert _classify_logic("   ").parse_confidence == "ambiguous"
    assert _classify_logic(None).parse_confidence == "ambiguous"
    assert _classify_logic(float("nan")).predicted_fraud is False


def test_classify_logic_still_parses_real_text_unchanged():
    assert _classify_logic("FRAUD").parse_confidence == "clear"
    assert _classify_logic("  legit  ").predicted_fraud is False
    assert _classify_logic("NOT FRAUD").parse_confidence == "clear"


def test_ring_root_logic_tolerates_nan_blank_and_none():
    assert _ring_root_logic(float("nan"), [101, 105], 101) is None
    assert _ring_root_logic("", [101, 105], 101) is None
    assert _ring_root_logic(None, [101, 105], 101) is None
    assert _ring_root_logic("Transaction 105", [101, 105], 101) == 105
