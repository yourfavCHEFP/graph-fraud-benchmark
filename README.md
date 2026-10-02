# graph-fraud-benchmark

> Can an LLM out-reason a Graph Neural Network on fraud?

[![kaggle](https://img.shields.io/badge/Kaggle-Benchmarking%20Challenge-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/benchmarks)
[![deadline](https://img.shields.io/badge/deadline-Oct%2011%202026-red)]()
[![built on](https://img.shields.io/badge/built%20on-graph--fraud--ai-8B5CF6)](https://github.com/yourfavCHEFP/graph-fraud-ai)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](./LICENSE)

## What this tests

A trained heterogeneous GNN catches fraud by reading the *shape* of a transaction network — shared devices, chained transfers, ring structures — not just a single transaction's features. This benchmark asks: **can a general-purpose LLM catch the same thing, if you just describe the graph to it in words?**

Every test case is a real subgraph pulled from the IEEE-CIS fraud dataset, serialized into plain text, with the label stripped out. Every case also carries the GNN's own prediction, so a wrong LLM answer can be split into two very different stories: *"missed what the GNN also missed"* vs. *"missed what the GNN caught by actually using graph structure."*

## How it works

1. Sample k-hop neighborhoods around fraud and legit transactions.
2. Serialize each neighborhood into a clean text/JSON description.
3. Ask 3–4 LLMs to classify the transaction and name the anomaly's source node.
4. Score against ground truth *and* against the GNN's own predictions.
5. Report where the reasoning gap actually lives.

## Repo structure

See `docs/diagrams/kaggle-file-sequence.svg` for the build order, or the project root tree in the architecture doc.

## Quick start

\`\`\`bash
git clone https://github.com/yourfavCHEFP/graph-fraud-benchmark.git
cd graph-fraud-benchmark
pip install -r requirements.txt

python -m src.graph_fraud_benchmark.sample_subgraphs
python -m src.graph_fraud_benchmark.serialize
\`\`\`

Then open `notebooks/kaggle_benchmark_task.ipynb` on Kaggle to run the actual benchmark against the model lineup.

## Model lineup

| Model | Why it's here |
|---|---|
| *(frontier reasoning model)* | Ceiling check — does more reasoning budget close the gap? |
| *(fast/cheap general model)* | Baseline — what does an "average" model catch? |
| *(open-weight model, e.g. DeepSeek)* | Is this a frontier-only capability or architecture-agnostic? |

## Results

_Filled in after the benchmark run — see `results/leaderboard_export.json` and `writeup/devto_submission.md`._

## Links

- Kaggle notebook: _add link once published_
- dev.to write-up: _add link once published_

## Author

**Olumide "CHEF_P" Oladosu** — [@yourfavCHEFP](https://github.com/yourfavCHEFP)

## License

Apache-2.0
