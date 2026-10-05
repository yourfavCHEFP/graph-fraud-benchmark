"""
analysis.py
-----------
The report card: takes the raw model responses collected during the
actual Kaggle Benchmarks run and turns them into the comparison table
the whole submission's insight depends on -- not "which model scored
highest" so much as "did each model's mistakes line up with where the
GNN also struggled, or somewhere else entirely."

INPUT CONTRACT
==============
This file does no parsing of its own -- every parsing/scoring rule
already lives in tasks.py and assertions.py, and analysis.py just
calls into them, so each rule is defined in exactly one place. It
expects a results file (one row per model x task x sample) holding
ONLY the raw text a model actually returned:

    model_name, task, seed_txn_id, raw_response

A blank raw_response is an UNANSWERED sample, not a result. It is
excluded from scoring and, unless require_complete=False, rejects the
whole file -- because a CSV of the right number of rows with nothing
in them is precisely the artifact that makes a failed run look like a
finished leaderboard.

Everything else -- labels.csv (ground truth + gnn_pred, once that
column is filled in from the real predictions), the serialized
prompts (for each sample's valid_txn_ids), and the raw transaction
table (every node's own isFraud, for ring-root scoring) -- is loaded
and joined here.

GNN COMPARISON THRESHOLD
=========================
labels.csv's gnn_pred column (once filled in) holds a fraud
probability, not a 0/1 call. Binarizing it uses graph-fraud-ai's own
production decision threshold (0.1980, from its saved evaluation
report) by default, so "did the GNN catch this" means the same thing
here as it does in that project -- not an arbitrary 0.5 cutoff picked
for this benchmark alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .assertions import graph_awareness, score_classification, score_ring_root
from .responses import is_complete_response, read_responses_csv
from .tasks import _classify_logic, _ring_root_logic

GNN_DECISION_THRESHOLD = 0.1980  # graph-fraud-ai's own production threshold


def load_results(results_csv: Path) -> pd.DataFrame:
    """Load a results CSV, marking which rows are real answers.

    Reading goes through responses.read_responses_csv, so a blank
    raw_response stays the empty string instead of becoming float('nan') --
    which is what previously made scoring a partly-empty file die with
    AttributeError: 'float' object has no attribute 'upper'. Rows are not
    dropped here; the response_complete column lets callers decide.
    """
    return read_responses_csv(results_csv)


def load_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df.set_index("seed_txn_id")
    return df


def load_valid_txn_ids(serialized_jsonl: Path) -> dict[int, list[int]]:
    lookup = {}
    with serialized_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["seed_txn_id"]] = rec["valid_txn_ids"]
    return lookup


def load_node_is_fraud(raw_transactions_csv: Path) -> dict[int, int]:
    raw = pd.read_csv(raw_transactions_csv, usecols=["TransactionID", "isFraud"])
    return dict(zip(raw["TransactionID"], raw["isFraud"]))


def score_classification_rows(
    rows: pd.DataFrame, labels: pd.DataFrame, valid_txn_ids: dict[int, list[int]]
) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        seed_id = row["seed_txn_id"]
        true_label = int(labels.loc[seed_id, "label"])
        neighbors = [t for t in valid_txn_ids.get(seed_id, []) if t != seed_id]
        if not is_complete_response(row["raw_response"]):
            # A blank response is not a wrong answer and not an ambiguous
            # parse -- the model never answered. Scored unscorable, and
            # reported so it can't be mistaken for a completed row.
            records.append(
                {
                    "model_name": row["model_name"],
                    "seed_txn_id": seed_id,
                    "true_label": true_label,
                    "predicted_fraud": None,
                    "parse_confidence": "empty_response",
                    "correct": None,
                    "graph_aware": False,
                }
            )
            continue
        parsed = _classify_logic(row["raw_response"])
        score = score_classification(parsed.predicted_fraud, parsed.parse_confidence, true_label)
        aware = graph_awareness(row["raw_response"], seed_id, neighbors)
        records.append(
            {
                "model_name": row["model_name"],
                "seed_txn_id": seed_id,
                "true_label": true_label,
                "predicted_fraud": parsed.predicted_fraud,
                "parse_confidence": parsed.parse_confidence,
                "correct": score.correct,  # None = unscorable, not auto-wrong
                "graph_aware": aware,
            }
        )
    return pd.DataFrame(records)


def score_ring_root_rows(
    rows: pd.DataFrame,
    labels: pd.DataFrame,
    valid_txn_ids: dict[int, list[int]],
    node_is_fraud: dict[int, int],
) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        seed_id = row["seed_txn_id"]
        true_label = int(labels.loc[seed_id, "label"])
        candidates = valid_txn_ids.get(seed_id, [])
        if not is_complete_response(row["raw_response"]):
            # Guarded explicitly. A blank response parses to no transaction
            # ID, which is indistinguishable from a deliberate "NONE" -- and
            # for a legit seed "NONE" is the expected answer, scored correct.
            # Without this check an unanswered sample would be recorded as a
            # successful answer purely because the model said nothing.
            records.append(
                {
                    "model_name": row["model_name"],
                    "seed_txn_id": seed_id,
                    "true_label": true_label,
                    "predicted_txn_id": None,
                    "correct": None,
                }
            )
            continue
        predicted_id = _ring_root_logic(row["raw_response"], candidates, seed_id)
        score = score_ring_root(predicted_id, true_label, node_is_fraud)
        records.append(
            {
                "model_name": row["model_name"],
                "seed_txn_id": seed_id,
                "true_label": true_label,
                "predicted_txn_id": predicted_id,
                "correct": score.correct,
            }
        )
    return pd.DataFrame(records)


def gnn_agreement_label(true_label: int, model_correct: bool | None, gnn_correct: bool | None) -> str:
    """
    The headline comparison: did the model and the GNN land in the same
    place? Returns one of four buckets, or "unscorable" if either side
    has no verdict to compare (an ambiguous model parse, or a seed with
    no gnn_pred filled in yet).
    """
    if model_correct is None or gnn_correct is None:
        return "unscorable"
    if model_correct and gnn_correct:
        return "both_correct"
    if model_correct and not gnn_correct:
        return "model_caught_what_gnn_missed"
    if not model_correct and gnn_correct:
        return "model_missed_what_gnn_caught"
    return "both_missed"


def build_report_card(
    classification_scores: pd.DataFrame,
    ring_root_scores: pd.DataFrame,
    labels: pd.DataFrame,
    gnn_threshold: float,
) -> pd.DataFrame:
    has_gnn_pred = "gnn_pred" in labels.columns and labels["gnn_pred"].notna().any()

    rows = []
    for model_name, group in classification_scores.groupby("model_name"):
        scored = group[group["correct"].notna()]
        n_total = len(group)
        n_scorable = len(scored)
        accuracy = scored["correct"].mean() if n_scorable else None
        ambiguous_rate = 1 - (n_scorable / n_total) if n_total else None
        graph_aware_rate = group["graph_aware"].mean() if n_total else None

        row = {
            "model_name": model_name,
            "task": "classify_transaction",
            "n_samples": n_total,
            "accuracy": accuracy,
            "ambiguous_rate": ambiguous_rate,
            "graph_aware_rate": graph_aware_rate,
        }

        if has_gnn_pred:
            gnn_correct_map = {
                seed_id: (
                    (labels.loc[seed_id, "gnn_pred"] >= gnn_threshold)
                    == bool(labels.loc[seed_id, "label"])
                    if pd.notna(labels.loc[seed_id, "gnn_pred"])
                    and labels.loc[seed_id, "gnn_pred"] != ""
                    else None
                )
                for seed_id in group["seed_txn_id"]
            }
            agreement = group.apply(
                lambda r: gnn_agreement_label(
                    r["true_label"], r["correct"], gnn_correct_map.get(r["seed_txn_id"])
                ),
                axis=1,
            )
            for bucket in [
                "both_correct",
                "model_caught_what_gnn_missed",
                "model_missed_what_gnn_caught",
                "both_missed",
                "unscorable",
            ]:
                row[f"gnn_agreement__{bucket}"] = (agreement == bucket).sum()

        rows.append(row)

    for model_name, group in ring_root_scores.groupby("model_name"):
        scored = group[group["correct"].notna()]
        rows.append(
            {
                "model_name": model_name,
                "task": "identify_ring_root",
                "n_samples": len(group),
                "accuracy": scored["correct"].mean() if len(scored) else None,
                "ambiguous_rate": None,
                "graph_aware_rate": None,
            }
        )

    return pd.DataFrame(rows)


def run(
    results_csv: Path,
    labels_csv: Path,
    serialized_jsonl: Path,
    raw_transactions_csv: Path,
    out_path: Path,
    gnn_threshold: float,
    require_complete: bool = True,
) -> None:
    """Build the report card from a results CSV.

    With require_complete (the default), a file containing any blank
    raw_response is rejected before scoring: those rows are unanswered
    samples, not wrong answers, and quietly scoring them as unscorable would
    publish a leaderboard that looks finished but isn't. Set it False only to
    inspect a deliberately partial run.
    """
    results = load_results(results_csv)
    incomplete = results[~results["response_complete"]]
    if not incomplete.empty:
        detail = ", ".join(
            f"({row.model_name}, {row.task}, {row.seed_txn_id})"
            for row in incomplete.head(10).itertuples()
        )
        message = (
            f"{results_csv} holds {len(incomplete)} row(s) with a blank "
            f"raw_response, which are unanswered samples rather than results: "
            f"{detail}{'' if len(incomplete) <= 10 else ' (+more)'}. "
            "Re-run the benchmark cell to recover them, or pass "
            "require_complete=False to score only the completed rows."
        )
        if require_complete:
            raise ValueError(message)
        print(f"[analysis] WARNING: {message}")

    results = results[results["response_complete"]]
    labels = load_labels(labels_csv)
    valid_txn_ids = load_valid_txn_ids(serialized_jsonl)
    node_is_fraud = load_node_is_fraud(raw_transactions_csv)

    classify_rows = results[results["task"] == "classify_transaction"]
    ring_root_rows = results[results["task"] == "identify_ring_root"]

    classification_scores = score_classification_rows(classify_rows, labels, valid_txn_ids)
    ring_root_scores = score_ring_root_rows(ring_root_rows, labels, valid_txn_ids, node_is_fraud)

    report_card = build_report_card(classification_scores, ring_root_scores, labels, gnn_threshold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_card.to_json(out_path, orient="records", indent=2)

    print(report_card.to_string(index=False))
    print(f"\n[analysis] scored {len(results)} non-empty responses")
    print(f"[analysis] wrote full report card to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=Path("results/raw_responses.csv"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--serialized-jsonl", type=Path, default=Path("data/serialized_subgraphs.jsonl"))
    parser.add_argument("--raw-transactions-csv", type=Path, default=Path("data/raw/train_transaction.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("results/leaderboard_export.json"))
    parser.add_argument("--gnn-threshold", type=float, default=GNN_DECISION_THRESHOLD)
    args = parser.parse_args()
    run(
        args.results_csv,
        args.labels_csv,
        args.serialized_jsonl,
        args.raw_transactions_csv,
        args.out_path,
        args.gnn_threshold,
    )


if __name__ == "__main__":
    main()
