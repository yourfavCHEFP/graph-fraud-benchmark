import random

import pandas as pd
import pytest

from src.graph_fraud_benchmark.sample_subgraphs import (
    build_entity_index,
    pick_balanced_seeds,
    run,
)


def _tiny_df(n=20):
    """A small, fully synthetic transaction table -- no real IEEE-CIS
    data needed for these tests."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "TransactionID": 1000 + i,
                "isFraud": 1 if i % 5 == 0 else 0,
                "TransactionAmt": 10.0 + i,
                "ProductCD": "W",
                "card1": "C1" if i < 15 else "C2",  # C1 shared by 15 txns
                "card4": "visa",
                "card6": "credit",
                "addr1": f"A{i}",  # every address unique -> no edges from addr1
                "P_emaildomain": "gmail.com",  # shared by ALL -> should be a hub
                "DeviceType": "mobile",
                "DeviceInfo": f"D{i}",  # every device unique -> no edges from device
                "TransactionDT": 86400 * i,
            }
        )
    return pd.DataFrame(rows)


def test_hub_cap_drops_entities_above_threshold():
    df = _tiny_df(n=20)
    # card1 cap of 10: "C1" (shared by 15) should be dropped as a hub;
    # email cap of 1000: "gmail.com" (shared by 20) should survive here
    index = build_entity_index(
        df,
        hub_degree_caps={
            "card1": 10,
            "addr1": 1000,
            "P_emaildomain": 1000,
            "DeviceInfo": 1000,
        },
    )
    assert (
        "card1=C1" not in index
    ), "card1=C1 (shared by 15 > cap of 10) should have been dropped as a hub"
    assert (
        "card1=C2" in index
    ), "card1=C2 (shared by only 5, under the cap) should survive"
    assert any(
        k.startswith("P_emaildomain=") for k in index
    ), "email below its own cap should survive"


def test_hub_cap_is_per_entity_type_not_global():
    df = _tiny_df(n=20)
    # Same data, but now cap email tightly too -- it should now also drop
    index = build_entity_index(
        df,
        hub_degree_caps={
            "card1": 10,
            "addr1": 1000,
            "P_emaildomain": 5,
            "DeviceInfo": 1000,
        },
    )
    assert not any(
        k.startswith("P_emaildomain=") for k in index
    ), "email should be dropped once ITS OWN cap is tight"


def test_balanced_seeds_respect_eligible_heldout_ids():
    df = _tiny_df(n=20)
    eligible = {1000, 1001, 1005, 1006, 1007, 1008}
    seeds = pick_balanced_seeds(
        df, n_per_class=2, rng=random.Random(1), eligible_seed_ids=eligible
    )

    labels = df.set_index("TransactionID")["isFraud"]
    assert set(seeds) <= eligible
    assert sum(labels.loc[seed] == 1 for seed in seeds) == 2
    assert sum(labels.loc[seed] == 0 for seed in seeds) == 2


def test_rerun_with_fewer_seeds_clears_stale_subgraphs(tmp_path):
    df = _tiny_df(n=20)
    raw_csv = tmp_path / "train_transaction.csv"
    df.to_csv(raw_csv, index=False)

    out_dir = tmp_path / "data"
    common_kwargs = dict(
        transactions_csv=raw_csv,
        identity_csv=None,
        out_dir=out_dir,
        hops=2,
        max_neighbors_per_entity=3,
        max_subgraph_size=10,
        hub_degree_caps={
            "card1": 100,
            "addr1": 1000,
            "P_emaildomain": 1000,
            "DeviceInfo": 1000,
        },
    )

    run(n_seeds_per_class=3, seed=1, **common_kwargs)
    first_run_files = set((out_dir / "subgraphs").glob("*.json"))
    assert len(first_run_files) == 6  # 3 fraud + 3 legit

    run(n_seeds_per_class=2, seed=2, **common_kwargs)
    second_run_files = set((out_dir / "subgraphs").glob("*.json"))

    assert (
        len(second_run_files) == 4
    ), "stale files from the first run must not survive a smaller rerun"
    labels = pd.read_csv(out_dir / "labels.csv")
    assert (
        len(labels) == 4
    ), "labels.csv must match the subgraphs actually on disk, not the previous run's count"


def test_invalid_input_preserves_previous_outputs(tmp_path):
    out_dir = tmp_path / "data"
    subgraphs_dir = out_dir / "subgraphs"
    subgraphs_dir.mkdir(parents=True)
    old_subgraph = subgraphs_dir / "previous.json"
    old_subgraph.write_text('{"seed_txn_id": 1234}')
    labels_path = out_dir / "labels.csv"
    labels_path.write_text("seed_txn_id,label\n1234,1\n")
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"wrong_column": [1]}).to_csv(invalid_csv, index=False)

    common_kwargs = dict(
        identity_csv=None,
        out_dir=out_dir,
        n_seeds_per_class=1,
        hops=1,
        max_neighbors_per_entity=1,
        max_subgraph_size=2,
        hub_degree_caps={
            "card1": 100,
            "addr1": 1000,
            "P_emaildomain": 1000,
            "DeviceInfo": 1000,
        },
        seed=1,
    )
    try:
        run(transactions_csv=invalid_csv, **common_kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input should fail validation")

    assert old_subgraph.read_text() == '{"seed_txn_id": 1234}'
    assert labels_path.read_text() == "seed_txn_id,label\n1234,1\n"


# --- The run() API the notebook's Step 1 calls ----------------------------
#
# Regression cover for `TypeError: run() got an unexpected keyword argument
# 'eligible_seed_ids'`, which is what a stale sample_subgraphs.py produces
# against the notebook's call. These pin the signature the notebook depends
# on, the GNN seed filtering, and determinism.


def _common_kwargs(raw_csv, out_dir):
    return dict(
        transactions_csv=raw_csv,
        identity_csv=None,
        out_dir=out_dir,
        hops=2,
        max_neighbors_per_entity=3,
        max_subgraph_size=10,
        hub_degree_caps={
            "card1": 100,
            "addr1": 1000,
            "P_emaildomain": 1000,
            "DeviceInfo": 1000,
        },
    )


def test_run_accepts_the_notebook_step1_signature(tmp_path):
    """The exact keyword set notebooks/kaggle_benchmark_task.ipynb passes."""
    import inspect

    parameters = inspect.signature(run).parameters
    for name in (
        "transactions_csv",
        "identity_csv",
        "out_dir",
        "n_seeds_per_class",
        "hops",
        "max_neighbors_per_entity",
        "max_subgraph_size",
        "hub_degree_caps",
        "seed",
        "eligible_seed_ids",
        "gnn_predictions",
    ):
        assert name in parameters, f"run() must accept {name}"
        assert parameters[name].kind is not inspect.Parameter.VAR_KEYWORD


def test_eligible_seed_ids_and_gnn_predictions_are_optional():
    """Existing callers that predate GNN filtering keep working unchanged."""
    import inspect

    parameters = inspect.signature(run).parameters
    assert parameters["eligible_seed_ids"].default is None
    assert parameters["gnn_predictions"].default is None


def test_run_filters_seeds_to_the_gnn_eligible_ids(tmp_path):
    df = _tiny_df(n=20)
    raw_csv = tmp_path / "train_transaction.csv"
    df.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "data"

    # 2 fraud (1000, 1005) and 2 legit (1001, 1006) inside the eligible set.
    eligible = {1000, 1005, 1001, 1006}
    run(
        n_seeds_per_class=2,
        seed=1,
        eligible_seed_ids=eligible,
        gnn_predictions={txn_id: 0.01 for txn_id in eligible},
        **_common_kwargs(raw_csv, out_dir),
    )

    labels = pd.read_csv(out_dir / "labels.csv")
    assert set(labels["seed_txn_id"]) <= eligible
    assert len(labels) == 4
    assert set(labels["label"]) == {0, 1}


def test_gnn_predictions_land_in_labels_csv_gnn_pred(tmp_path):
    df = _tiny_df(n=20)
    raw_csv = tmp_path / "train_transaction.csv"
    df.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "data"

    predictions = {1000: 0.9, 1005: 0.4, 1001: 0.2, 1006: 0.7}
    run(
        n_seeds_per_class=2,
        seed=1,
        eligible_seed_ids=set(predictions),
        gnn_predictions=predictions,
        **_common_kwargs(raw_csv, out_dir),
    )

    labels = pd.read_csv(out_dir / "labels.csv")
    assert "gnn_pred" in labels.columns
    for txn_id, probability in predictions.items():
        row = labels[labels["seed_txn_id"] == txn_id]
        assert len(row) == 1, f"seed {txn_id} missing from labels.csv"
        assert row["gnn_pred"].iloc[0] == pytest.approx(probability)


def test_missing_gnn_prediction_leaves_gnn_pred_blank_not_zero(tmp_path):
    """A seed with no prediction must stay blank -- 0.0 would read as a
    confident legitimate score in the report card."""
    df = _tiny_df(n=20)
    raw_csv = tmp_path / "train_transaction.csv"
    df.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "data"

    run(
        n_seeds_per_class=2,
        seed=1,
        eligible_seed_ids={1000, 1005, 1001, 1006},
        gnn_predictions={1000: 0.9},  # deliberately incomplete
        **_common_kwargs(raw_csv, out_dir),
    )

    labels = pd.read_csv(out_dir / "labels.csv")
    assert labels["gnn_pred"].isna().any()
    assert (labels.loc[labels["seed_txn_id"] == 1000, "gnn_pred"].iloc[0] == pytest.approx(0.9))


def test_run_is_deterministic_for_a_fixed_seed(tmp_path):
    df = _tiny_df(n=20)
    raw_csv = tmp_path / "train_transaction.csv"
    df.to_csv(raw_csv, index=False)

    seeds_by_run = []
    for n in (1, 2):
        out_dir = tmp_path / f"data{n}"
        run(
            n_seeds_per_class=2,
            seed=42,
            eligible_seed_ids=set(range(1000, 1020)),
            gnn_predictions={},
            **_common_kwargs(raw_csv, out_dir),
        )
        labels = pd.read_csv(out_dir / "labels.csv")
        seeds_by_run.append(sorted(labels["seed_txn_id"]))

    assert seeds_by_run[0] == seeds_by_run[1], "same seed must give the same sample set"


def test_eligible_ids_that_match_nothing_warns_and_yields_no_seeds(tmp_path, capsys):
    """An eligible set matching no transactions must not pass silently.

    run() warns and writes an empty sample rather than raising, which is the
    existing documented behaviour and is deliberately left unchanged here.
    What matters is that the shortfall is announced and that the notebook's
    own sample-count guard catches the resulting empty run.
    """
    df = _tiny_df(n=20)
    raw_csv = tmp_path / "train_transaction.csv"
    df.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "data"

    run(
        n_seeds_per_class=2,
        seed=1,
        eligible_seed_ids={999_999},
        gnn_predictions={},
        **_common_kwargs(raw_csv, out_dir),
    )

    printed = capsys.readouterr().out
    assert "only found 0 fraud / 0 legit seeds available" in printed
    # No seeds means labels.csv is written with no header and no rows at all,
    # so it is checked as raw text rather than parsed.
    assert (out_dir / "labels.csv").read_text().strip() == ""
    assert list((out_dir / "subgraphs").glob("*.json")) == []
