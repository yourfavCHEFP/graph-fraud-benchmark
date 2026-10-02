"""
models.py
---------
The guest list: who's actually sitting this test.

Deliberately short. Three models, each invited for a different reason,
so the leaderboard answers a real question ("is this frontier-only, or
architecture-agnostic?") instead of just ranking an arbitrary pile of
names.

CAVEAT -- VERIFY BEFORE RUNNING ON KAGGLE
==========================================
`kaggle_model_id` below is a placeholder slug, not a confirmed string
from Kaggle's live model catalog. Kaggle Benchmarks' exact model
identifiers aren't something I can verify without hands-on access to
the current catalog inside a notebook -- check
https://kaggle.com/benchmarks (or the model picker inside your
notebook) and correct these three strings before the real run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    display_name: str
    kaggle_model_id: str  # PLACEHOLDER -- confirm against Kaggle's catalog
    role: str             # why this model is in the lineup, for the write-up


MODEL_LINEUP: list[ModelSpec] = [
    ModelSpec(
        display_name="Frontier reasoning model",
        kaggle_model_id="VERIFY_ME/frontier-reasoning",
        role=(
            "Ceiling check -- if a model with a large reasoning budget "
            "still misses the graph signal, the gap isn't about effort."
        ),
    ),
    ModelSpec(
        display_name="Fast / cheap general model",
        kaggle_model_id="VERIFY_ME/fast-general",
        role=(
            "Baseline -- what an 'average' model catches without any "
            "special reasoning effort."
        ),
    ),
    ModelSpec(
        display_name="Open-weight model (e.g. DeepSeek)",
        kaggle_model_id="VERIFY_ME/open-weight",
        role=(
            "Architecture check -- is the gap frontier-only, or does it "
            "show up outside closed commercial models too?"
        ),
    ),
]


def get_model_lineup() -> list[ModelSpec]:
    """Single source of truth for which models the benchmark runs
    against -- the notebook should import this rather than hardcoding
    model names a second time."""
    return MODEL_LINEUP
