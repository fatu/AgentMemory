"""dc-frozen: a fixed cheatsheet injected unchanged from position 0 — no learning.

The control that separates "good advice in context" from "learning on this stream":
    dc − dc-frozen  = within-stream learning
    dc-frozen − none = advice / cross-seed transfer
Point FROZEN_SHEET_PATH at another run's memory_state.json (a different seed's final
DC sheet) or at a plain text file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .base import CAP, Memory, count_tokens


class FrozenSheet(Memory):
    name = "dc-frozen"

    def __init__(self, path: str | Path | None = None, cap: int = CAP):
        path = Path(path or os.environ["FROZEN_SHEET_PATH"])
        text = path.read_text()
        if path.suffix == ".json":
            text = json.loads(text)["sheet"]
        self.sheet = text.strip()
        self.source = str(path)
        assert count_tokens(self.sheet) + 24 <= cap, "frozen sheet exceeds the injection cap"

    def retrieve(self, question: str) -> str:
        return f"## Memory (fixed cheatsheet, source: {Path(self.source).parent.name})\n{self.sheet}"

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        pass

    def size_tokens(self) -> int:
        return count_tokens(self.sheet)
