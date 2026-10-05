"""
Tests for responses.py -- the "did the model actually answer?" rules.

These need no Kaggle access, no dataset, and no model credentials. Every test
here pins down a failure that actually happened during the Gemma run: a
Kaggle proxy that returned HTTP 200 with `choices[0].message is None`, a CSV
whose blank raw_response cells came back from pandas as float('nan'), and a
resume that treated those blank rows as finished work.
"""

import math

import pandas as pd
import pytest

from src.graph_fraud_benchmark.responses import (
    EmptyModelResponseError,
    TASKS,
    completed_keys_from_frame,
    completion_key,
    extract_response_text,
    is_complete_response,
    load_response_cache,
    normalize_response,
    read_responses_csv,
    request_response,
    validate_response_run,
    write_responses_csv,
)

MODEL = "Gemma 4 31B"


# --- fakes shaped like the real proxy's return values ---------------------


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, message):
        self.message = message


class _ChoiceWithNoneMessage:
    """The exact shape reported from Kaggle: choices exist, message does not."""

    message = None


class _OpenAIResponse:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _OpenAIResponseWithoutMessage:
    def __init__(self):
        self.choices = [_ChoiceWithNoneMessage()]


class _TextResponse:
    def __init__(self, text):
        self.text = text


# --- normalize_response / is_complete_response -----------------------------


def test_blank_like_values_never_normalize_to_text():
    # The load-bearing case: str(float('nan')) == "nan", which is truthy and
    # would sail through a naive bool(str(value).strip()) check.
    assert normalize_response(float("nan")) == ""
    assert not is_complete_response(float("nan"))
    assert not is_complete_response(math.nan)


def test_none_empty_and_whitespace_are_incomplete():
    for value in (None, "", "   ", "\n\t ", float("nan")):
        assert normalize_response(value) == "", value
        assert is_complete_response(value) is False, value


def test_real_text_is_stripped_and_complete():
    assert normalize_response("  FRAUD  ") == "FRAUD"
    assert normalize_response("\nTransaction 12345 did it.\n") == "Transaction 12345 did it."
    assert is_complete_response("  FRAUD ") is True


def test_pandas_na_is_incomplete():
    assert normalize_response(pd.NA) == ""
    assert is_complete_response(pd.NA) is False


def test_nonnan_numbers_are_real_content_not_blanks():
    # A response of "0" or a numeric id is content, not a missing value.
    assert normalize_response(12345) == "12345"
    assert is_complete_response(12345) is True


# --- extract_response_text ------------------------------------------------


def test_extracts_plain_string():
    assert extract_response_text("  LEGIT  ") == "LEGIT"


def test_extracts_openai_style_content():
    assert extract_response_text(_OpenAIResponse(_Message("FRAUD"))) == "FRAUD"


def test_extracts_text_attribute():
    assert extract_response_text(_TextResponse(" NONE ")) == "NONE"


def test_none_message_is_treated_as_no_answer():
    # The reported Kaggle failure. Must yield "" so the caller rejects it --
    # NOT fall through and find something else to pass off as a reply.
    assert extract_response_text(_OpenAIResponseWithoutMessage()) == ""


def test_empty_and_none_content_are_no_answer():
    assert extract_response_text(_OpenAIResponse(_Message(""))) == ""
    assert extract_response_text(_OpenAIResponse(_Message(None))) == ""
    assert extract_response_text(_OpenAIResponse(_Message("   "))) == ""
    assert extract_response_text(None) == ""


# --- request_response: retry once, then stop loudly -----------------------


def test_returns_text_on_first_attempt():
    calls = []

    def call(prompt):
        calls.append(prompt)
        return "FRAUD"

    assert request_response("p", call, "classify_transaction", 101) == "FRAUD"
    assert len(calls) == 1


def test_retries_once_on_empty_then_succeeds():
    responses = [_OpenAIResponseWithoutMessage(), "FRAUD"]
    calls = []

    def call(prompt):
        calls.append(prompt)
        return responses[len(calls) - 1]

    assert request_response("p", call, "classify_transaction", 101) == "FRAUD"
    assert len(calls) == 2, "an empty first response must trigger a fresh request"


def test_raises_after_one_retry_and_names_the_failure():
    calls = []

    def call(prompt):
        calls.append(prompt)
        return _OpenAIResponseWithoutMessage()

    with pytest.raises(EmptyModelResponseError) as excinfo:
        request_response("p", call, "classify_transaction", 101)
    assert len(calls) == 2, "exactly one retry, never more"
    assert "choices[0].message is None" in str(excinfo.value)
    assert "101" in str(excinfo.value)


def test_retries_a_transport_error_then_succeeds():
    attempts = []

    def call(prompt):
        attempts.append(prompt)
        if len(attempts) == 1:
            raise RuntimeError("502 upstream")
        return "LEGIT"

    assert request_response("p", call, "identify_ring_root", 102) == "LEGIT"
    assert len(attempts) == 2


def test_raises_rather_than_returning_empty_string():
    # Returning "" here is exactly how a fake completed row gets written.
    with pytest.raises(EmptyModelResponseError):
        request_response("p", lambda prompt: "", "classify_transaction", 103)


# --- completion key --------------------------------------------------------


def test_completion_key_is_model_task_and_seed():
    assert completion_key(MODEL, "classify_transaction", "101") == (
        MODEL,
        "classify_transaction",
        101,
    )


# --- CSV round trip: blanks stay blanks -----------------------------------


def _rows(n_samples, model=MODEL, response="FRAUD"):
    return [
        {
            "model_name": model,
            "task": task,
            "seed_txn_id": seed,
            "raw_response": response,
        }
        for seed in range(n_samples)
        for task in TASKS
    ]


def test_blank_cells_read_back_as_empty_string_not_nan(tmp_path):
    path = tmp_path / "raw_responses.csv"
    rows = _rows(2)
    rows[0]["raw_response"] = ""      # genuinely blank cell
    rows[1]["raw_response"] = "   "   # whitespace only
    pd.DataFrame(rows).to_csv(path, index=False)

    frame = read_responses_csv(path)
    assert frame["raw_response"].tolist() == ["", "", "FRAUD", "FRAUD"]
    assert not frame["response_complete"].iloc[0]
    assert not frame["response_complete"].iloc[1]
    assert frame["response_complete"].tolist() == [False, False, True, True]
    # seed_txn_id is still an int, so it still joins against labels.csv
    assert frame["seed_txn_id"].tolist() == [0, 0, 1, 1]


def test_missing_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"model_name": MODEL, "task": "classify_transaction"}]).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        read_responses_csv(path)


# --- resume: only real text counts as done --------------------------------


def test_resume_skips_completed_keys(tmp_path):
    path = tmp_path / "model3_pilot_raw_responses.csv"
    write_responses_csv(_rows(2), path)

    cache = load_response_cache(path, expected_model=MODEL)
    assert len(cache) == 4  # 2 samples x 2 tasks
    assert cache[completion_key(MODEL, "classify_transaction", 0)]["raw_response"] == "FRAUD"


def test_blank_cached_rows_are_not_treated_as_completed(tmp_path):
    path = tmp_path / "model3_pilot_raw_responses.csv"
    rows = _rows(2)
    rows[0]["raw_response"] = ""
    rows[1]["raw_response"] = "   "
    write_responses_csv(rows, path)

    cache = load_response_cache(path, expected_model=MODEL)
    assert len(cache) == 2, "only the two real responses count as done"
    assert completion_key(MODEL, "classify_transaction", 0) not in cache
    assert completion_key(MODEL, "identify_ring_root", 0) not in cache
    # ...and the blank rows are still on disk, not destroyed
    assert len(read_responses_csv(path)) == 4


def test_a_resumed_blank_row_is_recoverable(tmp_path):
    """A blank row stays in the file and can be replaced by a real answer."""
    path = tmp_path / "model3_pilot_raw_responses.csv"
    rows = _rows(1)
    rows[0]["raw_response"] = ""
    write_responses_csv(rows, path)

    cache = load_response_cache(path, expected_model=MODEL)
    key = completion_key(MODEL, "classify_transaction", 0)
    assert key not in cache

    cache[key] = {
        "model_name": MODEL,
        "task": "classify_transaction",
        "seed_txn_id": 0,
        "raw_response": "LEGIT",
    }
    cached_incomplete = [
        r for r in read_responses_csv(path).to_dict("records")
        if not r["response_complete"] and completion_key(
            r["model_name"], r["task"], r["seed_txn_id"]
        ) not in cache
    ]
    write_responses_csv(list(cache.values()) + cached_incomplete, path)

    frame = read_responses_csv(path)
    assert len(frame) == 2
    assert frame["response_complete"].all()
    assert frame.set_index(["task"]).loc["classify_transaction", "raw_response"] == "LEGIT"


def test_mixing_another_models_cache_raises(tmp_path):
    path = tmp_path / "model3_pilot_raw_responses.csv"
    rows = _rows(1) + _rows(1, model="Gemini 2.5 Pro")
    write_responses_csv(rows, path)

    with pytest.raises(ValueError, match="other models"):
        load_response_cache(path, expected_model=MODEL)


def test_missing_file_is_an_empty_cache_not_an_error(tmp_path):
    assert load_response_cache(tmp_path / "nope.csv", expected_model=MODEL) == {}


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "model3_pilot_raw_responses.csv"
    write_responses_csv(_rows(2), path)
    assert path.is_file()
    assert not (tmp_path / "model3_pilot_raw_responses.csv.tmp").exists()
    assert len(read_responses_csv(path)) == 4


# --- run-level validation --------------------------------------------------


def test_pilot_run_of_ten_responses_validates():
    frame = read_responses_csv_from_rows(_rows(5))
    summary = validate_response_run(frame, MODEL, range(5))
    assert summary["non_empty_responses"] == 10
    assert summary["by_task"] == {t: 5 for t in TASKS}


def test_full_run_of_two_hundred_responses_validates():
    frame = read_responses_csv_from_rows(_rows(100))
    summary = validate_response_run(frame, MODEL, range(100))
    assert summary["non_empty_responses"] == 200


def test_two_hundred_blank_rows_is_not_a_successful_run():
    # The exact failure this repair exists to catch: right row count, no content.
    frame = read_responses_csv_from_rows(_rows(100, response=""))
    with pytest.raises(ValueError, match="not complete") as excinfo:
        validate_response_run(frame, MODEL, range(100))
    message = str(excinfo.value)
    assert "blank raw_response" in message
    assert "expected 200 non-empty responses, found 0" in message


def test_partially_blank_run_is_rejected():
    rows = _rows(5)
    rows[0]["raw_response"] = ""
    rows[3]["raw_response"] = "  "
    frame = read_responses_csv_from_rows(rows)
    with pytest.raises(ValueError, match="not complete"):
        validate_response_run(frame, MODEL, range(5))


def test_run_for_another_model_is_rejected():
    frame = read_responses_csv_from_rows(_rows(5, model="Gemini 2.5 Pro"))
    with pytest.raises(ValueError, match="other models"):
        validate_response_run(frame, MODEL, range(5))


def test_missing_sample_is_named():
    rows = [r for r in _rows(5) if r["seed_txn_id"] != 3]
    frame = read_responses_csv_from_rows(rows)
    with pytest.raises(ValueError, match="missing seeds: 3"):
        validate_response_run(frame, MODEL, range(5))


def read_responses_csv_from_rows(rows):
    """In-memory equivalent of reading a written CSV back."""
    frame = pd.DataFrame(rows)
    frame["model_name"] = frame["model_name"].map(lambda v: str(v).strip())
    frame["task"] = frame["task"].map(lambda v: str(v).strip())
    frame["seed_txn_id"] = frame["seed_txn_id"].astype("int64")
    frame["raw_response"] = frame["raw_response"].map(normalize_response)
    frame["response_complete"] = frame["raw_response"].ne("")
    return frame


# --- Both attempts are reported, not just the last ------------------------
#
# The first real Gemma pilot failure reported only "attempt 2 returned no
# text", hiding whether attempt 1 had raised (rate limit / proxy error) or
# had also come back empty. Those point at different causes, so both are kept.


def test_error_reports_every_attempt_not_just_the_last():
    calls = []

    def call(prompt):
        calls.append(prompt)
        return ""

    with pytest.raises(EmptyModelResponseError) as excinfo:
        request_response("p", call, "classify_transaction", 101)

    message = str(excinfo.value)
    assert "attempt 1 returned no text" in message
    assert "attempt 2 returned no text" in message
    assert len(calls) == 2, "still exactly two attempts -- this changes no gate"


def test_error_distinguishes_a_raising_attempt_from_an_empty_one():
    calls = []

    def call(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("429 rate limited")
        return ""

    with pytest.raises(EmptyModelResponseError) as excinfo:
        request_response("p", call, "classify_transaction", 101)

    message = str(excinfo.value)
    assert "attempt 1 raised RuntimeError: 429 rate limited" in message
    assert "attempt 2 returned no text" in message


def test_retry_cap_is_not_relaxed():
    """The empty-response retry budget stays at exactly two attempts."""
    for max_attempts in (2,):
        calls = []

        def call(prompt):
            calls.append(prompt)
            return ""

        with pytest.raises(EmptyModelResponseError):
            request_response("p", call, "identify_ring_root", 7, max_attempts=max_attempts)
        assert len(calls) == max_attempts
