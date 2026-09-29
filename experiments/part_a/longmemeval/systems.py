"""Part A systems for LongMemEval: full-context, BM25, Mem0, A-MEM — each = ingest(entry)
once per QUESTION (every question has its own haystack → a fresh store per question), then
context_for(question). Only the context differs; prompt, answer call and judge are shared.
The instrumented client, the shared Embedder and the step-6/7 adapters come from stream_runner/.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

STREAM_RUNNER = Path(__file__).resolve().parents[2] / "stream_runner"
sys.path.insert(0, str(STREAM_RUNNER))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # hindsight_store.py (part_a/)

from memory.base import CAP, count_tokens                                   # noqa: E402
from lme_data import (full_history, history_string, rounds, sessions, session_plain,   # noqa: E402
                      wrap, GEN_MAX_TOKENS)

TRUNCATION_RESERVE = 1000          # run_generation.py:344  max_retrieval_length = max_len - gen_length - 1000
QWEN_TOKEN_FACTOR = 1.08           # o200k → Qwen tokenizer safety margin (LoCoMo: 17.8K Qwen vs 16.9K o200k)


class FullContext:
    """`full-history-session`: the whole haystack. Official truncation rule when it does not
    fit: keep the FIRST max_retrieval_length tokens (run_generation.py:268-272)."""
    name = "full"
    uses_cache = False               # unique haystack per question — nothing to share
    ingest_unit = "none"

    def __init__(self, max_context: int | None, backbone: str):
        self.budget = None
        if max_context:
            budget = max_context - GEN_MAX_TOKENS - TRUNCATION_RESERVE
            self.budget = int(budget / QWEN_TOKEN_FACTOR) if backbone == "qwen27b" else budget
        self.truncated = False

    def ingest(self, entry):
        from memory.base import _ENC
        self.history = full_history(entry)
        self.truncated = False
        if self.budget is not None:
            toks = _ENC.encode(self.history, disallowed_special=())
            if len(toks) > self.budget:
                self.history = _ENC.decode(toks[: self.budget])
                self.truncated = True

    def context_for(self, question):
        return self.history


class BM25:
    """Round-level BM25 (one user turn + its reply = one document), top hits fitted under the
    4,096-token cap in rank order, then date-sorted and wrapped exactly as the official
    flat-turn path does."""
    name = "bm25"
    uses_cache = False
    ingest_unit = "round"

    def __init__(self, cap: int = CAP):
        from rank_bm25 import BM25Okapi
        self._BM25 = BM25Okapi
        self.cap = cap
        self.truncated = False

    def ingest(self, entry):
        self.docs = rounds(entry)
        self.bm25 = self._BM25([d["text"].lower().split() for d in self.docs])

    def context_for(self, question):
        scores = self.bm25.get_scores(question.lower().split())
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])
        picked, used = [], 0
        for i in order[:64]:
            d = self.docs[i]
            n = count_tokens(wrap(99, d["date"], d["chunk"]))
            if used + n > self.cap:
                break
            picked.append((d["date"], d["chunk"])); used += n
        return history_string(picked)


class _PerQuestionStore:
    """Shared plumbing for the LLM-ingesting systems: a fresh store per question under
    runs/<run>/<system>/<qid>, sessions ingested one at a time with row_id ingest:<qid>:<k>."""
    uses_cache = False
    ingest_unit = "session"

    def __init__(self, client, embedder, state_root: Path, cap: int = CAP):
        self.client, self.embedder, self.state_root, self.cap = client, embedder, Path(state_root), cap
        self.truncated = False

    def _fresh_dir(self, qid: str) -> Path:
        d = self.state_root / qid
        if d.exists():
            shutil.rmtree(d)                     # a half-ingested store is worse than none
        d.mkdir(parents=True, exist_ok=True)
        return d


class Mem0(_PerQuestionStore):
    """Mem0 as shipped (step-7 adapter): each session added once as a role-tagged
    conversation; Mem0 decides which facts to keep."""
    name = "mem0"

    def ingest(self, entry):
        from memory.mem0_adapter import Mem0Memory
        qid = entry["question_id"]
        self.m = Mem0Memory(self.client, self.embedder, state_dir=self._fresh_dir(qid), cap=self.cap)
        for k, (date, sid, turns) in enumerate(sessions(entry)):
            self.client.current_row_id = f"ingest:{qid}:{k}"
            msgs = [{"role": "user", "content": f"(Session on {date})"}] + turns
            self.m.m.add(msgs, user_id=self.m.user_id)
            self.m.n_added += 1
        (self.m.state_dir / "n_added").write_text(str(self.m.n_added))

    def context_for(self, question):
        return self.m.retrieve(question)


class AMem(_PerQuestionStore):
    """A-MEM as shipped (step-6 adapter): one note per session (plain text with its date);
    A-MEM extracts keywords/context/tags and links neighbours."""
    name = "amem"

    def ingest(self, entry):
        from memory.amem_adapter import AMemMemory
        qid = entry["question_id"]
        self.m = AMemMemory(self.client, self.embedder, state_path=self._fresh_dir(qid) / "amem.pkl", cap=self.cap)
        for k, (date, sid, turns) in enumerate(sessions(entry)):
            self.client.current_row_id = f"ingest:{qid}:{k}"
            self.m.ams.add_note(session_plain(date, turns), time=date)
        self.m.dump()

    def context_for(self, question):
        return self.m.retrieve(question)


class Hindsight:
    """Hindsight (step 9b): one bank per QUESTION (each question has its own haystack), one
    retain() per session with the session date as the event timestamp; recall() fitted under
    the cap → our prompt (`hindsight`), or Hindsight's own reader (`hindsight-reflect`)."""
    name = "hindsight"
    uses_cache = False
    ingest_unit = "session"

    def __init__(self, client, backbone: str, reflect: bool = False, cap: int = CAP):
        from hindsight_store import HindsightStore, parse_ts
        self._Store, self._ts, self.client, self.backbone, self.cap = HindsightStore, parse_ts, client, backbone, cap
        self.kind = "reader" if reflect else "context"
        self.truncated = False
        if reflect:
            self.name = "hindsight-reflect"

    def ingest(self, entry):
        qid = entry["question_id"]
        self.store = self._Store(f"lme-{qid}-{self.backbone}", self.client, backbone=self.backbone)
        self.store.fresh()
        for k, (date, sid, turns) in enumerate(sessions(entry)):
            self.client.current_row_id = f"ingest:{qid}:{k}"
            self.store.retain(session_plain(date, turns), timestamp=self._ts(date), context=f"session {sid}",
                              document_id=sid, row_id=f"ingest:{qid}:{k}")
        self.question_date = entry.get("question_date")

    def context_for(self, question):
        return self.store.recall(question, query_timestamp=_iso(self._ts(self.question_date)), cap=self.cap)

    def answer(self, question, row_id):
        return self.store.reflect(question, row_id=row_id)


def _iso(dt):
    return dt.isoformat() if dt else None


def make_system(name: str, *, client=None, embedder=None, state_root: Path | None = None,
                max_context: int | None = None, backbone: str = ""):
    if name == "full":
        return FullContext(max_context, backbone)
    if name == "bm25":
        return BM25()
    if name in ("mem0", "amem"):
        assert client is not None and embedder is not None and state_root is not None
        return (Mem0 if name == "mem0" else AMem)(client, embedder, state_root)
    if name in ("hindsight", "hindsight-reflect"):
        assert client is not None
        return Hindsight(client, backbone, reflect=(name == "hindsight-reflect"))
    raise KeyError(name)
