"""
Loader for the hand-written goldens used by DeepEval tests.

Hand-writing 12 goldens once is cheaper, more accurate, and more
predictable than running DeepEval's Synthesizer on every test session.
The Synthesizer wrapper is still here (`synthesize_more`) but is only
used by `make synth-goldens` for the @local big-dataset matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


GOLDENS_PATH = Path(__file__).parent / "goldens.json"


@dataclass(slots=True)
class Golden:
    id: str
    input: str
    expected_output: str
    must_mention: list[str]
    rag_filter: dict[str, str]
    smoke: bool


def load() -> list[Golden]:
    raw = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    return [Golden(**g) for g in raw]


def smoke_only() -> list[Golden]:
    """Subset used by smoke tests — small to keep CI cost predictable."""
    return [g for g in load() if g.smoke]
