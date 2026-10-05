"""
assertions.py
-------------
The rubric: decides whether an answer counts as right, and -- for the
classification task -- whether the model's reasoning actually touched
the graph, or just looked at the seed transaction alone.

A REAL GAP, FLAGGED RATHER THAN HIDDEN
=======================================
classify_transaction has clean ground truth: isFraud is a real label
in the dataset.

identify_ring_root does NOT. IEEE-CIS labels individual transactions
as fraud/legit -- it never labels one specific node as "the ring
leader" of a connected group. There is no ground truth to grade
against directly.

The proxy used here: a ring-root guess is scored correct if the node
the model pointed to is ITSELF a transaction independently labeled
fraudulent in the raw dataset. That's a defensible signal (the model
at least pointed at another real fraud case, not a random neighbor)
but it is explicitly a proxy, not a verified "this was the actual
ringleader." Say so plainly in the write-up -- don't let the
leaderboard number imply more precision than this task actually has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .responses import normalize_response


@dataclass
class ClassificationScore:
    correct: bool | None   # None = unscorable (model's answer was ambiguous)
    graph_aware: bool      # did the response reference another node, not just the seed?


def score_classification(
    predicted_fraud: bool,
    parse_confidence: str,
    true_label: int,
) -> ClassificationScore:
    """
    Grade a classify_transaction answer. parse_confidence comes straight
    from tasks._classify_logic -- an "ambiguous" parse is scored as
    unscorable (None), not silently counted as wrong, so a model isn't
    penalized for a parsing failure that wasn't really its fault.
    """
    if parse_confidence == "ambiguous":
        return ClassificationScore(correct=None, graph_aware=False)
    correct = predicted_fraud == bool(true_label)
    return ClassificationScore(correct=correct, graph_aware=False)


_TXN_ID_RE = re.compile(r"\b\d{3,}\b")


def graph_awareness(response_text: str, seed_txn_id: int, neighbor_txn_ids: list[int]) -> bool:
    """
    Heuristic: did the model's own explanation mention at least one
    OTHER transaction ID from the subgraph, not just the seed? This is
    the cheapest possible signal that the reasoning used the network
    rather than judging the seed transaction in isolation -- it's not
    proof of genuine graph reasoning, just evidence the model engaged
    with it at all. Treat it as a weak, directional signal in the
    write-up, not a precise metric.
    """
    # Normalized first: a blank results-CSV cell arrives as float('nan'),
    # which the regex would otherwise reject with a TypeError.
    response_text = normalize_response(response_text)
    mentioned = {int(m) for m in _TXN_ID_RE.findall(response_text)}
    mentioned.discard(seed_txn_id)
    return any(n in mentioned for n in neighbor_txn_ids)


@dataclass
class RingRootScore:
    correct: bool | None   # None = seed was legit, so "NONE" is the expected answer

SCORE_UNSCORABLE = None  # re-exported for clarity at call sites


def score_ring_root(
    predicted_txn_id: int | None,
    seed_true_label: int,
    node_is_fraud: dict[int, int],
) -> RingRootScore:
    """
    Grade an identify_ring_root answer using the proxy described in the
    module docstring. node_is_fraud must be built by analysis.py from
    the ORIGINAL raw transaction table (every transaction has a real
    isFraud label there, not just the seeds) -- it is not something
    sample_subgraphs.py or serialize.py produce, since that label must
    never appear in anything the model is shown.
    """
    if seed_true_label == 0:
        # a legit seed has no known ring to find -- NONE is the
        # expected answer; anything else is scored wrong, not ambiguous
        return RingRootScore(correct=predicted_txn_id is None)

    if predicted_txn_id is None:
        return RingRootScore(correct=False)  # seed was fraud; model found nothing

    pointed_at_fraud = node_is_fraud.get(predicted_txn_id) == 1
    return RingRootScore(correct=pointed_at_fraud)
