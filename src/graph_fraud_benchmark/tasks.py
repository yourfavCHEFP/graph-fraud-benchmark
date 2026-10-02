"""
tasks.py
--------
The exam questions: hands each AI model a serialized transaction
neighborhood (from serialize.py) and asks it a real question, using
Kaggle's `@kbench.task` decorator so these register as benchmark tasks
once this runs inside notebooks/kaggle_benchmark_task.ipynb on Kaggle.

TWO QUESTIONS PER SAMPLE
========================
1. classify_transaction -- is this transaction fraud or legit?
   The baseline. Answerable in principle just from the seed
   transaction's own features, without reading the network at all.

2. identify_ring_root -- which OTHER transaction in the neighborhood
   looks most responsible for the suspicious pattern?
   NOT answerable without actually reading the edges. This is the
   question that specifically tests graph reasoning, not just "does
   this one transaction look weird in isolation" -- which is the
   whole point of the benchmark.

WHY THE LOGIC IS SPLIT FROM THE DECORATED FUNCTIONS
====================================================
kbench's exact task-calling contract -- what keyword arguments it
passes in, whether it wants a return value or an assertion -- can only
be confirmed by hand inside a live Kaggle notebook; that's not
something verifiable from here. So every task below is a thin wrapper
around a plain function (_classify_logic, _ring_root_logic) that is
fully testable on its own, with zero kbench dependency. If kbench's
real signature turns out to differ once you're on Kaggle, only the
thin @task wrapper needs adjusting -- the parsing logic underneath,
already tested below, does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import kbench
except ImportError:  # kbench is only installed in the Kaggle runtime
    kbench = None


def _noop_task_decorator(fn):
    """Stand-in for @kbench.task so this file still imports and is
    testable outside Kaggle, where the real package isn't installed."""
    return fn


task = kbench.task if kbench is not None else _noop_task_decorator


# --- Question 1: classify_transaction ---------------------------------

FRAUD_KEYWORDS = ("FRAUD", "FRAUDULENT")
LEGIT_KEYWORDS = ("LEGIT", "LEGITIMATE")
# "NOT FRAUD" contains the substring "FRAUD" -- without this, a plain
# substring check would flag it as both fraud and legit at once and
# mark it ambiguous, when a human reader would call it a clear LEGIT.
_NEGATED_FRAUD_RE = re.compile(r"\bNOT[\s-]*FRAUD(ULENT)?\b")


@dataclass
class ClassificationResult:
    predicted_fraud: bool
    raw_response: str
    parse_confidence: str  # "clear" | "ambiguous"


def build_classification_prompt(serialized_subgraph: str) -> str:
    return (
        "You are a fraud analyst. Below is a transaction and its network "
        "of related transactions (linked by shared card, address, or "
        "device information).\n\n"
        f"{serialized_subgraph}\n\n"
        "Based on this transaction and its network, is THE TRANSACTION IN "
        "QUESTION fraudulent?\n"
        "Answer with exactly one word: FRAUD or LEGIT."
    )


def _classify_logic(response_text: str) -> ClassificationResult:
    """
    Parse a model's free-text answer into a fraud/legit verdict.

    Deliberately conservative: only a clean, unambiguous match counts
    as "clear" -- a response matching both keyword sets, or neither,
    is flagged "ambiguous" so assertions.py can treat it as unscorable
    rather than silently guessing what the model meant.
    """
    upper = response_text.upper()
    negated_fraud = _NEGATED_FRAUD_RE.search(upper) is not None
    # check the plain FRAUD keyword only against text with "NOT FRAUD"
    # phrases removed, so that phrase can't also register as a FRAUD hit
    stripped = _NEGATED_FRAUD_RE.sub("", upper)
    has_fraud = any(kw in stripped for kw in FRAUD_KEYWORDS)
    has_legit = negated_fraud or any(kw in upper for kw in LEGIT_KEYWORDS)

    if has_fraud and not has_legit:
        return ClassificationResult(True, response_text, "clear")
    if has_legit and not has_fraud:
        return ClassificationResult(False, response_text, "clear")
    return ClassificationResult(has_fraud, response_text, "ambiguous")


@task
def classify_transaction(model, serialized_subgraph: str) -> bool:
    """Kaggle Benchmarks entry point for question 1.
    Real logic lives in _classify_logic, tested independently of kbench."""
    prompt = build_classification_prompt(serialized_subgraph)
    response = model.generate(prompt)
    return _classify_logic(response.text).predicted_fraud


# --- Question 2: identify_ring_root ------------------------------------

_TXN_ID_RE = re.compile(r"\b\d{3,}\b")


def build_ring_root_prompt(serialized_subgraph: str) -> str:
    return (
        "You are a fraud analyst. Below is a transaction and its network "
        "of related transactions.\n\n"
        f"{serialized_subgraph}\n\n"
        "If this network shows signs of a fraud ring, which single "
        "transaction ID looks most responsible for connecting the "
        "suspicious activity? Reply with exactly one transaction ID and "
        "nothing else. If you see no evidence of a ring, reply with NONE."
    )


def _ring_root_logic(response_text: str, valid_txn_ids: list[int]) -> int | None:
    """
    Pull a transaction ID out of the model's answer -- but only if it's
    actually a node that existed in this subgraph. Models do hallucinate
    IDs they were never shown, and a hallucinated ID is a wrong answer,
    not a lucky guess that happens to parse.
    """
    if "NONE" in response_text.upper():
        return None
    for match in _TXN_ID_RE.findall(response_text):
        candidate = int(match)
        if candidate in valid_txn_ids:
            return candidate
    return None  # mentioned something, but never an ID that was in the prompt


@task
def identify_ring_root(model, serialized_subgraph: str, valid_txn_ids: list[int]):
    """Kaggle Benchmarks entry point for question 2.
    Real logic lives in _ring_root_logic, tested independently of kbench."""
    prompt = build_ring_root_prompt(serialized_subgraph)
    response = model.generate(prompt)
    return _ring_root_logic(response.text, valid_txn_ids)
