"""Part A systems for LoCoMo: full-context, BM25, Mem0 — each = ingest(conv) once, then
context_for(question) per question. Only the CONTEXT differs; prompt and scoring are shared.
The instrumented client and the shared Embedder / Mem0 adapter come from stream_runner/.
"""
from __future__ import annotations

import sys
from pathlib import Path

STREAM_RUNNER = Path(__file__).resolve().parents[2] / "stream_runner"
sys.path.insert(0, str(STREAM_RUNNER))

from memory.base import CAP, count_tokens, fit_entries          # noqa: E402
from locomo_data import history_text, turns_as_documents, session_ids, session_text  # noqa: E402


class FullContext:
    name = "full"
    uses_cache = True                    # the history is the cached system block

    def ingest(self, conv):
        self.history = history_text(conv)

    def context_for(self, question):
        return self.history

    def ingest_calls(self):
        return 0


class BM25:
    name = "bm25"
    uses_cache = False

    def __init__(self, cap: int = CAP):
        from rank_bm25 import BM25Okapi
        self._BM25 = BM25Okapi
        self.cap = cap

    def ingest(self, conv):
        self.docs = turns_as_documents(conv)
        self.bm25 = self._BM25([d["text"].lower().split() for d in self.docs])

    def context_for(self, question):
        scores = self.bm25.get_scores(question.lower().split())
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])
        picked = [self.docs[i]["text"] for i in order[:64]]
        return fit_entries(picked, cap=self.cap, sep="\n")

    def ingest_calls(self):
        return 0


class Mem0:
    """Mem0 as shipped (step-7 adapter), ingesting each session once as a conversation."""
    name = "mem0"
    uses_cache = False

    def __init__(self, client, embedder, state_dir: Path, cap: int = CAP):
        from memory.mem0_adapter import Mem0Memory
        self.m = Mem0Memory(client, embedder, state_dir=state_dir)
        self.client = client
        self.cap = cap

    def ingest(self, conv):
        if self.m.load():                                   # resume: sessions already ingested
            return
        speaker_a = conv["conversation"]["speaker_a"]
        for i in session_ids(conv):
            self.client.current_row_id = f"ingest:s{i}"
            msgs = [{"role": "user" if d["speaker"] == speaker_a else "assistant",
                     "content": f'{d["speaker"]}: {d["text"]}' + (f' [shared {d["blip_caption"]}]' if d.get("blip_caption") else "")}
                    for d in conv["conversation"][f"session_{i}"]]
            msgs.insert(0, {"role": "user", "content": f"(Session on {conv['conversation'][f'session_{i}_date_time']})"})
            self.m.m.add(msgs, user_id=self.m.user_id)
            self.m.n_added += 1
        (self.m.state_dir / "n_added").write_text(str(self.m.n_added))

    def context_for(self, question):
        return self.m.retrieve(question)

    def ingest_calls(self):
        return -1                                           # read from calls.jsonl (tag=memory, row ingest:*)


def make_system(name: str, *, client=None, embedder=None, state_dir: Path | None = None):
    if name == "full":
        return FullContext()
    if name == "bm25":
        return BM25()
    if name == "mem0":
        assert client is not None and embedder is not None and state_dir is not None
        return Mem0(client, embedder, state_dir)
    raise KeyError(name)
