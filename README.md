# graph-fraud-benchmark

> Can an LLM out-reason a Graph Neural Network on fraud?

[![kaggle](https://img.shields.io/badge/Kaggle-Benchmarking%20Challenge-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/benchmarks)
[![deadline](https://img.shields.io/badge/deadline-Oct%2011%202026-red)](https://www.kaggle.com/benchmarks)
[![built on](https://img.shields.io/badge/built%20on-graph--fraud--ai-8B5CF6)](https://github.com/yourfavCHEFP/graph-fraud-ai)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](./LICENSE)

## What this tests

A trained heterogeneous GNN catches fraud by reading the *shape* of a transaction network — shared devices, chained transfers, ring structures — not just a single transaction's features. This benchmark asks: **can a general-purpose LLM catch the same thing, if you just describe the graph to it in words?**

Every test case is a real subgraph pulled from the IEEE-CIS fraud dataset, serialized into plain text, with the label stripped out. Each held-out seed also carries the GNN's fraud probability, so LLM errors can be compared with GNN errors. The ring-root task is scored using a proxy: whether the model selects another transaction independently labeled fraudulent, not whether it finds a verified ring leader.

## How it works

1. Sample k-hop neighborhoods around fraud and legit transactions.
2. Serialize each neighborhood into a clean text/JSON description.
3. Ask 3–4 LLMs to classify the transaction and name the anomaly's source node.
4. Score against ground truth *and* against the GNN's own predictions.
5. Report where the reasoning gap actually lives.

## Repo structure

See `docs/diagrams/kaggle-file-sequence.svg` for the build order, or the project root tree in the architecture doc.

## Quick start

```bash
git clone https://github.com/yourfavCHEFP/graph-fraud-benchmark.git
cd graph-fraud-benchmark
pip install -r requirements.txt

# Place the IEEE-CIS training CSVs in data/raw/ and matching GNN artifacts in data/gnn/.
python -m src.graph_fraud_benchmark.gnn_predictions
python -m src.graph_fraud_benchmark.sample_subgraphs \
  --gnn-predictions-csv data/gnn/test_predictions.csv
python -m src.graph_fraud_benchmark.serialize
```

Then open `notebooks/kaggle_benchmark_task.ipynb` on Kaggle and attach the IEEE-CIS dataset and matching GNN artifacts. Raw data, model artifacts, generated prompts, and mock outputs are local-only and ignored by Git. The notebook uses mock models by default for pipeline checks; set `USE_MOCK = False` in Kaggle to query the model lineup and produce the real results file.

## Model lineup

| Model | Why it's here |
| --- | --- |
| Gemini 2.5 Pro | Ceiling check — does more reasoning budget close the gap? |
| Gemini 2.5 Flash | Baseline — what does an "average" model catch? |
| Gemma 4 31B | Is this a frontier-only capability or architecture-agnostic? |

## Results

*Filled in after the benchmark run — see `results/leaderboard_export.json` and `writeup/devto_submission.md`.*

## Links

- Kaggle notebook: *add link once published*
- dev.to write-up: *add link once published*

## Author

**Olumide "CHEF_P" Oladosu** — [@yourfavCHEFP](https://github.com/yourfavCHEFP)

## License

Apache-2.0
