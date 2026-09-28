"""Part A systems for BEAM: full-context (official long-context), BM25 over official pair
chunks, Mem0, A-MEM — each = ingest(chat) once per CONVERSATION (20 questions share it),
then context_for(question). The answer call differs by kind:
  full → history = the chat as real turns + the official NOTE user message
  others → the official RAG prompt with <context> = the retrieved block (≤ 4,096 tokens)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

STREAM_RUNNER = Path(__file__).resolve().parents[2] / "stream_runner"
sys.path.insert(0, str(STREAM_RUNNER))

from memory.base import CAP, count_tokens, fit_entries                      # noqa: E402
from beam_data import messages, pairs, turns                                 # noqa: E402

# prune_from_tail (long_term_memory_methods.py:245-265): prune when total > max − 10,000,
# keep the most recent messages up to max − 2,000, drop a leading message if the count is odd
PRUNE_SLACK, PRUNE_KEEP = 10_000, 2_000
TOKEN_FACTOR = {"sonnet5": 1.6, "qwen27b": 1.08}      # o200k → backbone tokenizer, measured in 9.1


class FullContext:
    name = "full"
    kind = "history"
    uses_cache = True                    # 20 questions share the turns → cache breakpoint on the last one
    ingest_unit = "none"

    def __init__(self, max_context: int | None, backbone: str):
        self.max_context, self.factor = max_context, TOKEN_FACTOR.get(backbone, 1.1)
        self.truncated = False

    def ingest(self, chat):
        msgs = messages(chat)
        self.truncated = False
        if self.max_context:
            est = [count_tokens(m["content"]) * self.factor + 4 for m in msgs]
            if sum(est) > self.max_context - PRUNE_SLACK:
                keep, count = [], 0.0
                for m, n in zip(reversed(msgs), reversed(est)):
                    count += n
                    if count > self.max_context - PRUNE_KEEP:
                        break
                    keep.append(m)
                msgs = list(reversed(keep))
                if len(msgs) % 2 == 1:
                    msgs = msgs[1:]
                self.truncated = True
        self.history = msgs

    def context_for(self, question):
        return self.history                      # the runner passes it as `history`


class BM25:
    """Official pair_chunk documents, rank_bm25, hits concatenated in rank order (as the
    official RAG path does) until the 4,096-token cap."""
    name = "bm25"
    kind = "context"
    uses_cache = False
    ingest_unit = "pair"

    def __init__(self, cap: int = CAP):
        from rank_bm25 import BM25Okapi
        self._BM25, self.cap, self.truncated = BM25Okapi, cap, False

    def ingest(self, chat):
        self.docs = pairs(chat)
        self.bm25 = self._BM25([d["search"].lower().split() for d in self.docs])

    def context_for(self, question):
        scores = self.bm25.get_scores(question.lower().split())
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])
        return fit_entries([self.docs[i]["text"] for i in order[:64]], cap=self.cap, sep="")


class _PerConversationStore:
    kind = "context"
    uses_cache = False
    ingest_unit = "turn"

    def __init__(self, client, embedder, state_dir: Path, cap: int = CAP):
        self.client, self.embedder, self.state_dir, self.cap = client, embedder, Path(state_dir), cap
        self.truncated = False

    def _fresh(self) -> Path:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir


class Mem0(_PerConversationStore):
    """Mem0 as shipped (step-7 adapter): one `add(messages)` per turn."""
    name = "mem0"

    def ingest(self, chat, conv_id: str = "conv", resume: bool = True):
        from memory.mem0_adapter import Mem0Memory
        if resume and self.state_dir.exists():
            self.m = Mem0Memory(self.client, self.embedder, state_dir=self.state_dir, cap=self.cap)
            if self.m.load():
                return False                                # store already built
        self.m = Mem0Memory(self.client, self.embedder, state_dir=self._fresh(), cap=self.cap)
        for k, t in enumerate(turns(chat)):
            self.client.current_row_id = f"ingest:{conv_id}:{k}"
            msgs = list(t["messages"])
            if t["time_anchor"]:
                msgs.insert(0, {"role": "user", "content": f"(Time anchor: {t['time_anchor']})"})
            self.m.m.add(msgs, user_id=self.m.user_id)
            self.m.n_added += 1
        (self.m.state_dir / "n_added").write_text(str(self.m.n_added))
        return True

    def context_for(self, question):
        return self.m.retrieve(question)


class AMem(_PerConversationStore):
    """A-MEM as shipped (step-6 adapter): one note per turn (official turn_chunk text)."""
    name = "amem"

    def ingest(self, chat, conv_id: str = "conv", resume: bool = True):
        from memory.amem_adapter import AMemMemory
        state_path = self.state_dir / "amem.pkl"
        if resume and state_path.exists():
            self.m = AMemMemory(self.client, self.embedder, state_path=state_path, cap=self.cap)
            if self.m.load():
                return False
        self._fresh()
        self.m = AMemMemory(self.client, self.embedder, state_path=state_path, cap=self.cap)
        for k, t in enumerate(turns(chat)):
            self.client.current_row_id = f"ingest:{conv_id}:{k}"
            self.m.ams.add_note(t["text"], time=t["time_anchor"] or f"batch{t['batch']}-turn{t['turn']}")
        self.m.dump()
        return True

    def context_for(self, question):
        return self.m.retrieve(question)


def make_system(name: str, *, client=None, embedder=None, state_dir: Path | None = None,
                max_context: int | None = None, backbone: str = ""):
    if name == "full":
        return FullContext(max_context, backbone)
    if name == "bm25":
        return BM25()
    if name in ("mem0", "amem"):
        assert client is not None and embedder is not None and state_dir is not None
        return (Mem0 if name == "mem0" else AMem)(client, embedder, state_dir)
    raise KeyError(name)
