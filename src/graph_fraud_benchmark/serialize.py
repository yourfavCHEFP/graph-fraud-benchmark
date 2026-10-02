"""
serialize.py
------------
The translator: turns each sampled subgraph (produced by
sample_subgraphs.py) into a clean, readable text description an LLM
can actually read -- with the fraud/legit label stripped out, since
that's the one thing nothing downstream is allowed to see.

FORMAT
======
The seed transaction is always introduced first and called out
explicitly ("THE TRANSACTION IN QUESTION"), since that's the one the
model is actually being asked about. Every other node in the
neighborhood is described along with *why* it's connected (which
shared entity pulled it in) -- the relationship is the whole point of
this benchmark, so it's never left implicit for the model to infer.

LABEL SAFETY
============
build_prompt() only ever reads subgraph["seed_txn_id"], ["nodes"], and
["edges"]. It never touches subgraph.get("label"), even though that
key exists in the source JSON (sample_subgraphs.py writes it in).
That's deliberate: the label simply isn't in scope for this function,
so there's no path by which it could leak into a prompt by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_amount(amt) -> str:
    return f"${amt:,.2f}" if amt is not None else "unknown amount"


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "unknown time"
    days, rem = divmod(int(seconds), 86400)
    hours = rem // 3600
    return f"day {days}, hour {hours}"


def _fmt_node_line(txn_id: int, feats: dict, is_seed: bool) -> str:
    amt = _fmt_amount(feats.get("TransactionAmt"))
    time_str = _fmt_time(feats.get("TransactionDT"))
    product = feats.get("ProductCD") or "unknown product"
    card = f"{feats.get('card4') or 'unknown'} {feats.get('card6') or 'card'}"
    email = feats.get("P_emaildomain") or "unknown email domain"
    device = feats.get("DeviceType") or "unknown device"

    tag = "THE TRANSACTION IN QUESTION" if is_seed else "Related transaction"
    return (
        f"- [{tag}] Transaction {txn_id}: {amt}, product '{product}', "
        f"paid with {card}, email domain '{email}', device type '{device}', "
        f"occurred at {time_str}."
    )


def build_prompt(subgraph: dict) -> str:
    """Build the full text prompt for one subgraph. See module docstring
    for why this function can never leak the label."""
    seed_id = subgraph["seed_txn_id"]
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]

    lines = ["TRANSACTION NETWORK:", ""]
    lines.append(_fmt_node_line(seed_id, nodes[str(seed_id)], is_seed=True))
    for txn_id_str, feats in nodes.items():
        if int(txn_id_str) == seed_id:
            continue
        lines.append(_fmt_node_line(int(txn_id_str), feats, is_seed=False))

    if edges:
        lines.append("")
        lines.append("CONNECTIONS:")
        for edge in edges:
            lines.append(
                f"- Transaction {edge['a']} and Transaction {edge['b']} "
                f"share {edge['shared_via']}."
            )
    else:
        lines.append("")
        lines.append(
            "CONNECTIONS: none found -- this transaction has no linked "
            "neighbors in the sampled network."
        )

    return "\n".join(lines)


def run(subgraphs_dir: Path, out_jsonl: Path) -> None:
    subgraph_paths = sorted(subgraphs_dir.glob("*.json"))
    if not subgraph_paths:
        raise FileNotFoundError(
            f"No subgraph JSON files found in {subgraphs_dir} -- "
            "run sample_subgraphs.py first."
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_jsonl.open("w") as f:
        for path in subgraph_paths:
            subgraph = json.loads(path.read_text())
            prompt = build_prompt(subgraph)
            valid_txn_ids = [int(k) for k in subgraph["nodes"].keys()]
            record = {
                "seed_txn_id": subgraph["seed_txn_id"],
                "prompt": prompt,
                "valid_txn_ids": valid_txn_ids,
                # label is intentionally NOT included here -- it lives only
                # in labels.csv, joined back in at scoring time (analysis.py)
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"[serialize] wrote {written} serialized prompts to {out_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subgraphs-dir", type=Path, default=Path("data/subgraphs"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/serialized_subgraphs.jsonl"))
    args = parser.parse_args()
    run(args.subgraphs_dir, args.out_jsonl)


if __name__ == "__main__":
    main()
