"""
responses.py
------------
One place for every rule about "did the model actually answer?"

The failure this file exists to prevent: an LLM call that returns HTTP 200
but no usable text gets written to the results CSV as if it were a finished
row. A file of 200 such rows is indistinguishable, by row count, from a
completed benchmark run -- and the analysis step then scores blank values and
reports nonsense instead of saying "the model never answered." So the rule is
enforced at the point of capture, not inferred later from row counts.

Three separate questions are answered here, and nothing else in the repo
re-implements them:

1. Is this value non-empty text?        normalize_response / is_complete_response
2. Does this returned object hold text? extract_response_text
3. Did the model actually answer?       request_response (one retry, then raise)

COMPLETION KEY
==============
A benchmark result is identified by ``(model_name, task, seed_txn_id)`` -- see
completion_key. Nothing here is keyed by list position, so resuming a partly
finished run cannot silently pair a response with the wrong sample, and a
response for one task can never satisfy the other task's key.

BLANKS ARE NOT ANSWERS
======================
A blank ``raw_response`` means the call produced nothing. It is not a wrong
answer (which would silently penalize the model) and it is not a parse
failure to be scored as ambiguous. It means the key is *not done* and must be
re-requested. is_complete_response is the single predicate that decides this.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

TASKS = ("classify_transaction", "identify_ring_root")

RESPONSE_COLUMNS = ["model_name", "task", "seed_txn_id", "raw_response"]


class EmptyModelResponseError(RuntimeError):
    """A model call came back with no usable text, even after one retry.

    Raised instead of returning "" so that an empty answer can never be
    mistaken for a finished row at a call site.
    """


# --- 1. Is this value non-empty text? -------------------------------------


def normalize_response(value) -> str:
    """Return a model's response as stripped text, or "" if there isn't any.

    Order matters here. A blank ``raw_response`` cell read back by pandas is
    ``float('nan')``, and ``str(nan) == "nan"`` -- truthy, and it would sail
    straight through a naive ``bool(str(value).strip())`` check as though the
    model had answered the word "nan". So missing values are rejected BEFORE
    any string conversion happens.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        return "" if math.isnan(value) else str(value).strip()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        # array-likes report missingness element-wise; fall through to str()
        pass
    return str(value).strip()


def is_complete_response(value) -> bool:
    """True only for a real, non-blank answer.

    ``NaN``, ``None``, ``""``, and whitespace-only strings are all incomplete.
    """
    return bool(normalize_response(value))


# --- 2. Does this returned object hold text? ------------------------------


def describe_response_object(response) -> str:
    """A short description of a returned object, for failure diagnostics."""
    if response is None:
        return "None"
    choices = getattr(response, "choices", None)
    if choices is not None:
        try:
            first = choices[0]
        except (IndexError, TypeError, KeyError):
            return f"<{type(response).__name__} with no usable choices[0]>"
        message = getattr(first, "message", None)
        if message is None:
            return f"<{type(response).__name__} choices[0].message is None>"
        content = getattr(message, "content", None)
        return f"<{type(response).__name__} choices[0].message.content={content!r}>"
    return f"<{type(response).__name__} {response!r}>"


def extract_response_text(response) -> str:
    """Pull the answer out of whatever the LLM proxy returned, or return "".

    The shapes this has to survive:

    * a plain string -- what ``llm.prompt()`` normally returns.
    * an OpenAI-style response, ``.choices[0].message.content``.
    * an object exposing ``.text`` or ``.content``.
    * an OpenAI-style response whose ``.choices[0].message`` is None -- the
      empty proxy response. That is treated as "no answer" and returns ""
      immediately, rather than falling through to look for text elsewhere: a
      message-less response has no content hiding behind it, and searching
      past it would let a broken envelope masquerade as a real answer.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()

    choices = getattr(response, "choices", None)
    if choices is not None:
        try:
            message = choices[0].message
        except (AttributeError, IndexError, TypeError, KeyError):
            message = None
        if message is None:
            return ""
        text = normalize_response(getattr(message, "content", None))
        if text:
            return text

    for attribute in ("text", "content"):
        text = normalize_response(getattr(response, attribute, None))
        if text:
            return text
    return ""


# --- 3. Did the model actually answer? ------------------------------------


def request_response(
    prompt: str,
    call,
    task_name: str,
    seed_txn_id,
    max_attempts: int = 2,
) -> str:
    """Send ``prompt`` and return non-empty text, retrying at most once.

    ``call`` must issue a FRESH request every time it is invoked -- a new
    chat, a new HTTP call. Retrying by re-inspecting the same returned object
    would just rediscover the same emptiness, which is not a retry.

    Raises :class:`EmptyModelResponseError` rather than returning "" so the
    caller cannot accidentally checkpoint an empty string as a result.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    failure = "no attempt was made"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call(prompt)
        except Exception as exc:  # transport/HTTP failures are retryable too
            failure = f"attempt {attempt} raised {type(exc).__name__}: {exc}"
            continue
        text = extract_response_text(raw)
        if text:
            return text
        failure = f"attempt {attempt} returned no text ({describe_response_object(raw)})"

    raise EmptyModelResponseError(
        f"{task_name} for seed {seed_txn_id} produced no usable text after "
        f"{max_attempts} attempt(s): {failure}. Nothing was checkpointed for "
        "this key, so re-running resumes it cleanly."
    )


# --- Identity, storage, and resume ----------------------------------------


def completion_key(model_name, task, seed_txn_id) -> tuple[str, str, int]:
    """The logical identity of one benchmark result."""
    return (str(model_name).strip(), str(task).strip(), int(seed_txn_id))


def read_responses_csv(path: Path) -> pd.DataFrame:
    """Read a results CSV without letting blank cells become NaN floats.

    ``keep_default_na=False`` is the load-bearing argument: with pandas'
    defaults an empty ``raw_response`` cell parses as ``float('nan')``, and the
    first ``.upper()`` applied to it downstream raises
    ``AttributeError: 'float' object has no attribute 'upper'``. Reading every
    column as text and converting ``seed_txn_id`` back to int explicitly keeps
    responses as text and still lets them join against ``labels.csv``.
    """
    frame = pd.read_csv(path, keep_default_na=False, na_filter=False, dtype=str)
    missing = set(RESPONSE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["model_name"] = frame["model_name"].map(lambda v: str(v).strip())
    frame["task"] = frame["task"].map(lambda v: str(v).strip())
    frame["seed_txn_id"] = pd.to_numeric(
        frame["seed_txn_id"], errors="raise"
    ).astype("int64")
    frame["raw_response"] = frame["raw_response"].map(normalize_response)
    frame["response_complete"] = frame["raw_response"].ne("")
    return frame


def completed_keys_from_frame(
    frame: pd.DataFrame, expected_model: str | None = None
) -> dict[tuple[str, str, int], dict]:
    """Index a results frame by completion key, keeping only real responses.

    Rows whose response is blank are dropped from the result: that key is not
    done and stays eligible to be re-requested. The rows themselves are left
    untouched on disk, so nothing is destroyed -- they simply don't count as
    completed work.

    ``expected_model`` makes cross-model cache mixing impossible: a row
    recorded for any other model raises rather than being silently adopted.
    """
    expected = str(expected_model).strip() if expected_model is not None else None
    completed: dict[tuple[str, str, int], dict] = {}
    foreign: set[str] = set()

    for row in frame.to_dict("records"):
        key = completion_key(row["model_name"], row["task"], row["seed_txn_id"])
        if expected is not None and key[0] != expected:
            foreign.add(key[0])
            continue
        if not row.get("response_complete", is_complete_response(row.get("raw_response"))):
            continue
        completed[key] = {
            "model_name": key[0],
            "task": key[1],
            "seed_txn_id": key[2],
            "raw_response": normalize_response(row["raw_response"]),
        }

    if foreign:
        raise ValueError(
            f"results contain rows for other models: {sorted(foreign)}. This "
            f"run is pinned to {expected!r}; refusing to mix caches from "
            "different models."
        )
    return completed


def load_response_cache(
    path: Path, expected_model: str | None = None
) -> dict[tuple[str, str, int], dict]:
    """Read a results CSV into a completion-keyed dict of valid responses."""
    if not Path(path).is_file():
        return {}
    return completed_keys_from_frame(read_responses_csv(Path(path)), expected_model)


def write_responses_csv(rows, path: Path) -> None:
    """Write result rows to ``path`` atomically, via a temp file and replace.

    A crash mid-write leaves the previous complete file intact rather than a
    truncated one, so an interrupted run never destroys earlier results.
    """
    frame = pd.DataFrame(
        [
            {
                "model_name": str(row["model_name"]).strip(),
                "task": str(row["task"]).strip(),
                "seed_txn_id": int(row["seed_txn_id"]),
                "raw_response": normalize_response(row["raw_response"]),
            }
            for row in rows
        ],
        columns=RESPONSE_COLUMNS,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


# --- Run-level validation --------------------------------------------------


def validate_response_run(
    frame: pd.DataFrame,
    expected_model: str,
    expected_samples,
    expected_tasks=TASKS,
) -> dict:
    """Check a results frame against what the run was supposed to produce.

    Counts NON-EMPTY responses only. A CSV holding 200 rows whose
    ``raw_response`` is blank is not a 200-response run, and row count must
    never stand in for content.

    Returns a summary dict on success; raises :class:`ValueError` listing every
    discrepancy otherwise.
    """
    expected_model = str(expected_model).strip()
    expected_seeds = sorted({int(seed) for seed in expected_samples})
    expected_tasks = tuple(expected_tasks)
    problems: list[str] = []

    complete = frame[frame["response_complete"]]
    incomplete = frame[~frame["response_complete"]]

    foreign = sorted(set(complete["model_name"]) - {expected_model})
    if foreign:
        problems.append(
            f"responses recorded for other models: {foreign} "
            f"(this run is pinned to {expected_model!r})"
        )

    unknown_tasks = sorted(set(complete["task"]) - set(expected_tasks))
    if unknown_tasks:
        problems.append(f"responses for unknown tasks: {unknown_tasks}")

    per_task: dict[str, int] = {}
    for task in expected_tasks:
        task_seeds = {int(seed) for seed in complete.loc[complete["task"] == task, "seed_txn_id"]}
        per_task[task] = len(task_seeds)
        missing = sorted(set(expected_seeds) - task_seeds)
        extra = sorted(task_seeds - set(expected_seeds))
        if missing:
            shown = ", ".join(str(seed) for seed in missing[:10])
            more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
            problems.append(
                f"{task}: {len(task_seeds)} of {len(expected_seeds)} samples have a "
                f"non-empty response; missing seeds: {shown}{more}"
            )
        if extra:
            problems.append(f"{task}: {len(extra)} responses for out-of-scope seeds: {extra[:10]}")

    if not incomplete.empty:
        blanks = ", ".join(
            f"({row.model_name}, {row.task}, {row.seed_txn_id})"
            for row in incomplete.head(10).itertuples()
        )
        more = "" if len(incomplete) <= 10 else f" (+{len(incomplete) - 10} more)"
        problems.append(
            f"{len(incomplete)} row(s) have a blank raw_response and do not count "
            f"as completed: {blanks}{more}"
        )

    total = int(complete.shape[0])
    expected_total = len(expected_seeds) * len(expected_tasks)
    if total != expected_total:
        problems.append(f"expected {expected_total} non-empty responses, found {total}")

    if problems:
        raise ValueError(
            f"{expected_model} run is not complete:\n  - " + "\n  - ".join(problems)
        )

    return {
        "model_name": expected_model,
        "samples": len(expected_seeds),
        "non_empty_responses": total,
        "by_task": per_task,
    }
