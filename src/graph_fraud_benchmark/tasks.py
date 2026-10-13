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

   DECIDED SCOPE: IEEE-CIS has no ground-truth "ring leader" label, so
   this task does NOT measure "did the model find the true root cause."
   It measures a narrower, code-verifiable proxy instead: did the model
   point at some OTHER transaction that is itself independently labeled
   fraudulent (see assertions.score_ring_root). That's a real signal --
   pointing at a known-fraud neighbor beats pointing at a known-legit
   one or hallucinating an ID -- but report it in the write-up as
   exactly that proxy, not as "ring leader detection accuracy."

WHY THE LOGIC IS SPLIT FROM THE DECORATED FUNCTIONS
====================================================
Kaggle's LLM transport and task registration use the documented
`llm.prompt()` and `@kbench.task` APIs. Parsing remains in plain
functions (_classify_logic, _ring_root_logic), so it can be tested
without a Kaggle runtime or model credentials.
"""

import re
from dataclasses import dataclass

from .responses import is_complete_response, normalize_response

try:
    import kaggle_benchmarks as kbench
except ImportError:  # kaggle-benchmarks is preinstalled in Kaggle task notebooks
    kbench = None


def _noop_task_decorator(fn=None, **_kwargs):
    """Stand-in for @kbench.task so logic remains testable outside Kaggle."""
    if fn is None:
        return lambda decorated: decorated
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

    response_text is normalized first: a blank cell read back from a results
    CSV arrives as float('nan'), and calling .upper() on it raises
    AttributeError. A blank response normalizes to "", matches neither
    keyword set, and therefore lands on "ambiguous" -- unscorable, never
    silently wrong.
    """
    response_text = normalize_response(response_text)
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


@task(name="classify_transaction")
def classify_transaction(llm, serialized_subgraph: str, true_label: int) -> bool:
    """Ask Kaggle's LLM to classify a transaction and score its verdict."""
    prompt = build_classification_prompt(serialized_subgraph)
    parsed = _classify_logic(llm.prompt(prompt))
    if parsed.parse_confidence == "ambiguous":
        return False
    return parsed.predicted_fraud == bool(true_label)


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


def _ring_root_logic(
    response_text: str, valid_txn_ids: list[int], seed_txn_id: int
) -> int | None:
    """
    Pull a transaction ID out of the model's answer -- but only if it's
    actually a node that existed in this subgraph, AND it isn't the
    seed transaction itself. Models do hallucinate IDs they were never
    shown (a hallucinated ID is a wrong answer, not a lucky guess that
    happens to parse) -- and the prompt explicitly asks which OTHER
    transaction looks responsible, so the seed answering for itself is
    just as wrong, not a trivially-true match.

    seed_txn_id is required, not optional, specifically so this
    exclusion can't be forgotten at a call site -- it was, once:
    valid_txn_ids as built by serialize.py includes the seed, and an
    earlier version of this function didn't filter it out, so a fraud
    seed could be scored as correctly identifying itself as the ring
    root.

    An empty response returns None, but callers must not read that as a
    deliberate "NONE" answer -- for a legit seed, None is the *expected*
    reply and would otherwise be scored correct. Use
    is_complete_response() to tell a blank apart from a real "NONE";
    analysis.py and the notebook both gate on it before scoring.
    """
    response_text = normalize_response(response_text)
    candidates = [t for t in valid_txn_ids if t != seed_txn_id]
    if re.search(r"\bNONE\b", response_text.upper()):
        return None
    for match in _TXN_ID_RE.findall(response_text):
        candidate = int(match)
        if candidate in candidates:
            return candidate
    return None  # mentioned something, but never a valid OTHER transaction ID


@task(name="identify_ring_root")
def identify_ring_root(
    llm,
    serialized_subgraph: str,
    valid_txn_ids: list[int],
    seed_txn_id: int,
    seed_true_label: int,
    fraud_txn_ids: list[int],
) -> bool:
    """Score the disclosed proxy: selecting another known-fraud node."""
    prompt = build_ring_root_prompt(serialized_subgraph)
    predicted_id = _ring_root_logic(llm.prompt(prompt), valid_txn_ids, seed_txn_id)
    node_is_fraud = {int(txn_id): 1 for txn_id in fraud_txn_ids}
    if seed_true_label == 0:
        return predicted_id is None
    return predicted_id is not None and node_is_fraud.get(predicted_id) == 1
