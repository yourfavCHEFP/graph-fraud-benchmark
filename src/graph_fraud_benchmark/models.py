"""
models.py
---------
The guest list: who's actually sitting this test.

Deliberately short. Three models, each invited for a different reason,
so the leaderboard answers a real question ("is this frontier-only, or
architecture-agnostic?") instead of just ranking an arbitrary pile of
names.

The model identifiers below follow the `kbench.llms["provider/model"]`
format shown in Kaggle Benchmarks' current SDK documentation. Availability
can vary by notebook/runtime; confirm each one in Kaggle's model picker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    display_name: str
    kaggle_model_id: str
    role: str  # why this model is in the lineup, for the write-up


MODEL_LINEUP: list[ModelSpec] = [
    ModelSpec(
        display_name="Gemini 2.5 Pro",
        kaggle_model_id="google/gemini-2.5-pro",
        role=(
            "Ceiling check -- if a model with a large reasoning budget "
            "still misses the graph signal, the gap isn't about effort."
        ),
    ),
    ModelSpec(
        display_name="Gemini 2.5 Flash",
        kaggle_model_id="google/gemini-2.5-flash",
        role=(
            "Baseline -- what an 'average' model catches without any "
            "special reasoning effort."
        ),
    ),
    ModelSpec(
        display_name="Gemma 4 31B",
        kaggle_model_id="google/gemma-4-31b",
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
