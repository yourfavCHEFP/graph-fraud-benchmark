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
from .tasks import _classify_logic, _ring_root_logic

GNN_DECISION_THRESHOLD = 0.1980  # graph-fraud-ai's own production threshold


def load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = {"model_name", "task", "seed_txn_id", "raw_response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results file is missing columns: {missing}")
    return df


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
        parsed = _classify_logic(row["raw_response"])
        score = score_classification(parsed.predicted_fraud, parsed.parse_confidence, true_label)
        neighbors = [t for t in valid_txn_ids.get(seed_id, []) if t != seed_id]
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
        predicted_id = _ring_root_logic(row["raw_response"], candidates)
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
) -> None:
    results = load_results(results_csv)
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
    print(f"\n[analysis] wrote full report card to {out_path}")


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
