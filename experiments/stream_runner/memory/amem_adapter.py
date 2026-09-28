"""A-MEM (Agentic Memory, Xu et al. 2025; `WujiangXu/AgenticMemory`, MIT) behind the
Memory interface.

As shipped: `AgenticMemorySystem` — each episode becomes a MemoryNote (LLM extracts
keywords / context / tags), neighbours are retrieved, and an evolution call may link
notes and rewrite neighbours' context/tags. Wired to the study: LLM calls through our
client (tag=memory, no thinking), the shared Qwen3-Embedding-0.6B as the retriever
model, injected block capped at CAP tokens.

Reference checkout: projects/AgentMemory/code-repo/A-mem (memory_layer.py); set
AMEM_PATH if it lives elsewhere.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

from .base import CAP, Memory, count_tokens, fit_entries
from .llm_shim import AmemLLM

TOP_K = 5
_DEFAULT_AMEM = Path(__file__).resolve().parents[3] / "code-repo" / "A-mem"


class _STLike:
    """Minimal SentenceTransformer look-alike: A-MEM's retriever only calls `.encode(list)`.
    `get_config_dict()` is what `consolidate_memories()` reads to rebuild the retriever."""

    def __init__(self, embedder):
        self.embedder = embedder

    def encode(self, texts, **_):
        return self.embedder.encode(list(texts))

    def get_config_dict(self):
        return {"model_name": self.embedder.model_name}


class _QwenRetriever:
    """A-MEM's SimpleEmbeddingRetriever with the study's Embedder in place of a
    SentenceTransformer. Subclass-free on purpose: `__init__` is replaced so that NO
    SentenceTransformer is ever constructed — not at start-up, not when A-MEM's
    `consolidate_memories()` rebuilds the retriever (it re-instantiates the class by
    model name; we bind the class to our embedder before A-MEM sees it)."""

    @staticmethod
    def install(SimpleEmbeddingRetriever, embedder):
        def __init__(self, model_name: str = embedder.model_name):
            self.model = _STLike(embedder)
            self.corpus = []
            self.embeddings = None
            self.document_ids = {}
        SimpleEmbeddingRetriever.__init__ = __init__


class AMemMemory(Memory):
    name = "amem"

    def __init__(self, client, embedder, state_path: str | Path, cap: int = CAP):
        amem_dir = Path(os.environ.get("AMEM_PATH", _DEFAULT_AMEM))
        if str(amem_dir) not in sys.path:
            sys.path.insert(0, str(amem_dir))
        from memory_layer import AgenticMemorySystem, SimpleEmbeddingRetriever
        self._SR = SimpleEmbeddingRetriever
        self.cap = cap
        self.state_path = Path(state_path)
        self.embed_model_name = embedder.model_name
        # Bind A-MEM's retriever class to the study's Embedder BEFORE constructing the system:
        # every SimpleEmbeddingRetriever A-MEM ever creates (init or consolidate) embeds with
        # Qwen3-Embedding-0.6B and nothing else is downloaded or loaded.
        _QwenRetriever.install(SimpleEmbeddingRetriever, embedder)
        os.environ.setdefault("OPENAI_API_KEY", "x")     # llm_backend "openai" builds offline; replaced below
        self.ams = AgenticMemorySystem(model_name=self.embed_model_name, llm_backend="openai",
                                       llm_model="placeholder", evo_threshold=100)
        self.ams.llm_controller.llm = AmemLLM(client)    # every LLM call → our client, tag=memory
        assert isinstance(self.ams.retriever.model, _STLike)
        self.position = 0

    # ------------------------------------------------------------ interface
    def retrieve(self, question: str) -> str:
        if not self.ams.memories:
            return ""
        text = self.ams.find_related_memories_raw(question, k=TOP_K)
        lines = [l for l in text.splitlines() if l.strip()]
        body = fit_entries(lines, cap=self.cap - 24, sep="\n")
        return "## Memory (related notes)\n" + body if body else ""

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        self.position += 1
        content = (f"Question: {question}\nFinal answer given: {answer or '(none)'}\n"
                   f"Result: {'correct' if correct else 'wrong'}")
        self.ams.add_note(content, time=f"Q{self.position}")
        self.dump()

    def size_tokens(self) -> int:
        return sum(count_tokens(m.content + " " + m.context + " " + " ".join(m.keywords) + " " + " ".join(m.tags))
                   for m in self.ams.memories.values())

    # ------------------------------------------------------------ resume
    def dump(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("wb") as f:
            pickle.dump({"memories": self.ams.memories, "position": self.position}, f)

    def load(self) -> bool:
        if not self.state_path.exists():
            return False
        with self.state_path.open("rb") as f:
            st = pickle.load(f)
        self.ams.memories, self.position = st["memories"], st["position"]
        # rebuild the retriever from the notes with the shared model (no second model load)
        r = self.ams.retriever
        r.corpus, r.embeddings, r.document_ids = [], None, {}
        docs = [f"{m.content} , {m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                for m in self.ams.memories.values()]
        if docs:
            r.add_documents(docs)
        return True
