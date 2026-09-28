"""ExpRAG (ours): the matched storage-and-retrieval control (blog §2.4 / §5.1).

Stores raw (question, model answer, correct?) triples — successes AND failures — and
injects the most similar past triples, whole entries, until the shared CAP (top_k is only
an upper bound on candidates — the cap is what binds, so the budget matches the other systems). No LLM
calls (tokens_mem = 0); embeddings via the shared Embedder. Any gain of a consolidating
system over this row is attributable to consolidation, not to seeing more history.
"""
from __future__ import annotations

import numpy as np

from .base import CAP, Memory, count_tokens, fit_entries


class ExpRAG(Memory):
    name = "exprag"

    def __init__(self, embedder, top_k: int = 64, cap: int = CAP):
        self.embedder = embedder
        self.top_k = top_k
        self.cap = cap
        self._entries: list[tuple[str, str, bool]] = []
        self._vecs: list[np.ndarray] = []

    @staticmethod
    def _format(i: int, q: str, a: str, correct: bool) -> str:
        return (f"[{i}] Past question:\n{q}\n"
                f"Your earlier answer: {a}\n"
                f"Result: {'correct' if correct else 'wrong'}")

    def retrieve(self, question: str) -> str:
        if not self._entries:
            return ""
        qv = self.embedder.encode([question])[0]
        sims = np.stack(self._vecs) @ qv                     # vectors are L2-normalised
        order = np.argsort(-sims)[: self.top_k]
        entries = [self._format(rank + 1, *self._entries[j]) for rank, j in enumerate(order)]
        body = fit_entries(entries, cap=self.cap - 32)       # header allowance
        return "## Past experience (most similar first)\n" + body if body else ""

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        self._entries.append((question, answer, correct))
        self._vecs.append(self.embedder.encode([question])[0])

    def size_tokens(self) -> int:
        return sum(count_tokens(self._format(0, *e)) for e in self._entries)
