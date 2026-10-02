"""
sample_subgraphs.py
--------------------
The net: scoops balanced little neighborhoods of transactions out of the
full IEEE-CIS fraud dataset, so the benchmark has manageable, labeled
cases to hand an LLM one at a time.

WHY THIS FILE IS THE HARD ONE
==============================
Everything downstream (serialize.py, tasks.py, assertions.py, analysis.py)
just assumes good subgraphs already exist. This file is where they
actually get made, and two decisions here are not obvious:

1. How do you even build a "graph" out of flat transaction rows?
   IEEE-CIS ships one row per transaction, not an edge list. The
   standard trick (used across most published fraud-GNN work on this
   dataset) is to treat shared identifiers as entities: two
   transactions are "connected" if they share a card number, an email
   domain, a device fingerprint, or a billing address. That turns the
   flat table into a bipartite transaction<->entity graph, which is
   exactly the kind of heterogeneous structure a GNN (and this
   benchmark) cares about.

2. Some entities are hubs, not signal.
   "gmail.com" might be shared by tens of thousands of transactions.
   If you let sampling walk through it, a 2-hop neighborhood around
   any transaction balloons into a meaningless chunk of the whole
   dataset. This file caps entity degree (`hub_degree_cap`) so only
   genuinely distinctive shared attributes (a specific device
   fingerprint, an unusual card/address combo) are allowed to connect
   the sample. That's the difference between a subgraph that actually
   tests graph reasoning and one that's just noise.

Everything else in the pipeline is comparatively mechanical once this
file's output (data/subgraphs/*.json + data/labels.csv) exists.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Columns used to link transactions into entities. Keep this list short
# and high-signal on purpose -- more columns means more edges means more
# hub problems. Confirmed against the real graph-fraud-ai repo's
# src/graph/graph_schema.py RELATIONSHIP_COLUMNS -- same four columns,
# same entity types (Card, Address, Email, Device).
ENTITY_COLUMNS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]

# Per-entity-type hub thresholds, taken directly from graph-fraud-ai's
# src/features/graph_features.py (the hub_card/hub_email/hub_device/
# hub_address definition baked into the production GraphSAGE model's own
# features). A card shared by >=100 transactions, or an email/device/
# address shared by >=1000, counts as a hub there -- so this sampler
# uses the same cutoffs rather than one flat guessed number, to stay
# consistent with what the production model itself already treats as
# "too common to be informative."
HUB_DEGREE_CAPS = {
    "card1": 100,
    "addr1": 1000,
    "P_emaildomain": 1000,
    "DeviceInfo": 1000,
}

# Kept on each transaction node for later serialization. Deliberately a
# small, readable subset -- this is what an LLM will actually see, not
# a 400-column dump.
FEATURE_COLUMNS = [
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "DeviceType",
    "TransactionDT",
]


@dataclass
class SubgraphSample:
    seed_txn_id: int
    label: int  # ground-truth isFraud for the seed transaction
    nodes: dict = field(default_factory=dict)   # txn_id -> feature dict
    edges: list = field(default_factory=list)   # [(txn_id_a, txn_id_b, shared_entity)]

    def to_json_dict(self) -> dict:
        return {
            "seed_txn_id": self.seed_txn_id,
            "label": self.label,
            "num_nodes": len(self.nodes),
            "nodes": self.nodes,
            "edges": [
                {"a": a, "b": b, "shared_via": via} for a, b, via in self.edges
            ],
        }


def load_transactions(transactions_csv: Path, identity_csv: Path | None) -> pd.DataFrame:
    """Load the flat transaction table and, if given, merge identity features."""
    df = pd.read_csv(transactions_csv)
    if identity_csv is not None and identity_csv.exists():
        identity = pd.read_csv(identity_csv)
        df = df.merge(identity, on="TransactionID", how="left")
    missing = [c for c in ENTITY_COLUMNS + FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found in the input data: {missing}. "
            "Check that transactions_csv / identity_csv match the IEEE-CIS schema."
        )
    return df


def build_entity_index(
    df: pd.DataFrame, hub_degree_caps: dict[str, int] = HUB_DEGREE_CAPS
) -> dict[str, list[int]]:
    """
    Build entity -> [txn_id, txn_id, ...] lookups for each entity column,
    dropping any entity value whose degree exceeds that column's hub cap.

    This is the hub-filtering step described in the module docstring --
    it's what keeps "gmail.com" from connecting half the dataset. Caps
    are per-entity-type (see HUB_DEGREE_CAPS) rather than one flat
    number, matching how the production GNN itself defines a hub.
    """
    entity_index: dict[str, list[int]] = {}
    dropped_hubs = 0

    for col in ENTITY_COLUMNS:
        cap = hub_degree_caps.get(col, 200)  # fallback if a new column is added later
        grouped = df.groupby(col)["TransactionID"].apply(lambda s: s.tolist())
        for value, txn_ids in grouped.items():
            if pd.isna(value):
                continue
            if len(txn_ids) < 2:
                continue  # no edges to make from a value only one txn has
            if len(txn_ids) > cap:
                dropped_hubs += 1
                continue
            entity_index[f"{col}={value}"] = txn_ids

    print(
        f"[sample_subgraphs] built {len(entity_index)} usable entity keys, "
        f"dropped {dropped_hubs} hub entities (per-column caps: {hub_degree_caps})"
    )
    return entity_index


def build_txn_to_entities(entity_index: dict[str, list[int]]) -> dict[int, list[str]]:
    """Invert entity_index into txn_id -> [entity_key, ...] for fast neighbor lookup."""
    txn_to_entities: dict[int, list[str]] = {}
    for entity_key, txn_ids in entity_index.items():
        for t in txn_ids:
            txn_to_entities.setdefault(t, []).append(entity_key)
    return txn_to_entities


def _to_native(value):
    """
    pandas/numpy hand back numpy scalar types (int64, float64, bool_)
    that json.dumps doesn't know how to serialize. Convert to plain
    Python types, and normalize NaN to None while we're at it.
    """
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        return value.item()
    return value


def txn_features(row: pd.Series) -> dict:
    feats = {c: row[c] for c in FEATURE_COLUMNS if c in row}
    return {k: _to_native(v) for k, v in feats.items()}


def sample_one_subgraph(
    seed_txn_id: int,
    df_by_id: pd.DataFrame,
    entity_index: dict[str, list[int]],
    txn_to_entities: dict[int, list[str]],
    hops: int,
    max_neighbors_per_entity: int,
    max_subgraph_size: int,
    rng: random.Random,
) -> SubgraphSample:
    """
    BFS out from a seed transaction through shared entities, up to `hops`
    hops, capping how many neighbor transactions each entity is allowed
    to contribute (so one shared attribute can't dominate the sample)
    and capping total subgraph size (so the LLM prompt stays readable).
    """
    seed_row = df_by_id.loc[seed_txn_id]
    sample = SubgraphSample(seed_txn_id=seed_txn_id, label=int(seed_row["isFraud"]))
    sample.nodes[seed_txn_id] = txn_features(seed_row)

    visited = {seed_txn_id}
    frontier = deque([(seed_txn_id, 0)])

    while frontier and len(sample.nodes) < max_subgraph_size:
        current_id, depth = frontier.popleft()
        if depth >= hops:
            continue

        for entity_key in txn_to_entities.get(current_id, []):
            neighbors = entity_index[entity_key]
            # Cap how many neighbors any single entity can contribute,
            # and randomize which ones, so a moderately-sized entity
            # still doesn't dominate the sample deterministically.
            candidates = [n for n in neighbors if n != current_id and n not in visited]
            rng.shuffle(candidates)
            candidates = candidates[:max_neighbors_per_entity]

            for neighbor_id in candidates:
                if len(sample.nodes) >= max_subgraph_size:
                    break
                visited.add(neighbor_id)
                sample.nodes[neighbor_id] = txn_features(df_by_id.loc[neighbor_id])
                sample.edges.append((current_id, neighbor_id, entity_key))
                frontier.append((neighbor_id, depth + 1))

    return sample


def pick_balanced_seeds(df: pd.DataFrame, n_per_class: int, rng: random.Random) -> list[int]:
    fraud_ids = df.loc[df["isFraud"] == 1, "TransactionID"].tolist()
    legit_ids = df.loc[df["isFraud"] == 0, "TransactionID"].tolist()
    rng.shuffle(fraud_ids)
    rng.shuffle(legit_ids)
    n_fraud = min(n_per_class, len(fraud_ids))
    n_legit = min(n_per_class, len(legit_ids))
    if n_fraud < n_per_class or n_legit < n_per_class:
        print(
            f"[sample_subgraphs] warning: only found {n_fraud} fraud / {n_legit} legit "
            f"seeds available (asked for {n_per_class} each)"
        )
    return fraud_ids[:n_fraud] + legit_ids[:n_legit]


def run(
    transactions_csv: Path,
    identity_csv: Path | None,
    out_dir: Path,
    n_seeds_per_class: int,
    hops: int,
    max_neighbors_per_entity: int,
    max_subgraph_size: int,
    hub_degree_caps: dict[str, int],
    seed: int,
) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    subgraphs_dir = out_dir / "subgraphs"
    subgraphs_dir.mkdir(parents=True, exist_ok=True)

    df = load_transactions(transactions_csv, identity_csv)
    df_by_id = df.set_index("TransactionID", drop=False)

    entity_index = build_entity_index(df, hub_degree_caps=hub_degree_caps)
    txn_to_entities = build_txn_to_entities(entity_index)

    seeds = pick_balanced_seeds(df, n_seeds_per_class, rng)
    print(f"[sample_subgraphs] sampling {len(seeds)} seed transactions "
          f"({n_seeds_per_class} fraud + {n_seeds_per_class} legit target)")

    labels_rows = []
    for seed_txn_id in seeds:
        sample = sample_one_subgraph(
            seed_txn_id=seed_txn_id,
            df_by_id=df_by_id,
            entity_index=entity_index,
            txn_to_entities=txn_to_entities,
            hops=hops,
            max_neighbors_per_entity=max_neighbors_per_entity,
            max_subgraph_size=max_subgraph_size,
            rng=rng,
        )

        out_path = subgraphs_dir / f"{seed_txn_id}.json"
        out_path.write_text(json.dumps(sample.to_json_dict(), indent=2))

        labels_rows.append(
            {
                "seed_txn_id": seed_txn_id,
                "label": sample.label,
                "subgraph_size": len(sample.nodes),
                # filled in later, once the trained GNN scores these same
                # subgraphs -- left blank here on purpose so serialize.py
                # and analysis.py can tell "not yet scored" apart from 0.0
                "gnn_pred": "",
            }
        )

    labels_path = out_dir / "labels.csv"
    pd.DataFrame(labels_rows).to_csv(labels_path, index=False)
    print(f"[sample_subgraphs] wrote {len(labels_rows)} subgraphs to {subgraphs_dir}")
    print(f"[sample_subgraphs] wrote ground-truth labels to {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions-csv", type=Path, default=Path("data/raw/train_transaction.csv"))
    parser.add_argument("--identity-csv", type=Path, default=Path("data/raw/train_identity.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-seeds-per-class", type=int, default=50,
                         help="How many fraud seeds and how many legit seeds to sample (balanced).")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-neighbors-per-entity", type=int, default=4,
                         help="Cap on neighbors pulled in through any single shared entity.")
    parser.add_argument("--max-subgraph-size", type=int, default=12,
                         help="Hard cap on total nodes per subgraph, so prompts stay readable.")
    parser.add_argument(
        "--hub-degree-cap-override", type=int, default=None,
        help=(
            "If set, overrides ALL per-entity hub caps with one flat number "
            "(mainly for quick experiments). Default: use HUB_DEGREE_CAPS, "
            "the real per-entity-type thresholds from graph-fraud-ai's "
            "production model (card1=100, addr1/P_emaildomain/DeviceInfo=1000)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hub_degree_caps = (
        {col: args.hub_degree_cap_override for col in ENTITY_COLUMNS}
        if args.hub_degree_cap_override is not None
        else HUB_DEGREE_CAPS
    )

    identity_csv = args.identity_csv if args.identity_csv.exists() else None
    run(
        transactions_csv=args.transactions_csv,
        identity_csv=identity_csv,
        out_dir=args.out_dir,
        n_seeds_per_class=args.n_seeds_per_class,
        hops=args.hops,
        max_neighbors_per_entity=args.max_neighbors_per_entity,
        max_subgraph_size=args.max_subgraph_size,
        hub_degree_caps=hub_degree_caps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
