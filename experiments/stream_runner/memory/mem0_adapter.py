"""Mem0 (mem0ai package) behind the Memory interface.

As shipped: default fact-extraction and update prompts (ADD / UPDATE / DELETE / NOOP),
vector config (no graph). Wired to the study: LLM calls through our client (tag=memory,
no thinking), embeddings from the shared Qwen3-Embedding-0.6B, FAISS store on disk per
run, injected block capped at CAP tokens.

    pip install mem0ai faiss-cpu

Each episode is added as one user/assistant exchange: the question, then the model's
final answer with its binary result. Mem0 decides what (if anything) to keep.
"""
from __future__ import annotations

from pathlib import Path

from .base import CAP, Memory, count_tokens, fit_entries
from .llm_shim import Mem0LLM, Mem0Embedder

TOP_K = 10


class Mem0Memory(Memory):
    name = "mem0"

    def __init__(self, client, embedder, state_dir: str | Path, cap: int = CAP):
        from mem0 import Memory as _Mem0
        self.cap = cap
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # mem0 validates provider NAMES against a whitelist, so our Embedder cannot be named in
        # the config. When the study's embedder is served remotely (EMBED_BASE_URL), mem0's
        # "openai" provider is pointed straight at that endpoint — the config then literally
        # says Qwen3-Embedding-0.6B. Either way the embedder object is replaced below by
        # `Mem0Embedder`, so every vector comes from the study's single Embedder (memoised,
        # L2-normalised, same as ExpRAG / A-MEM / the stratification). The LLM entry is a
        # placeholder for the same whitelist reason and is replaced by `Mem0LLM`.
        emb_cfg = {"model": embedder.model_name, "embedding_dims": 1024, "api_key": "x"}
        if embedder.backend == "remote":
            emb_cfg["openai_base_url"] = embedder.base_url          # Qwen3-Embedding via vLLM /v1
        cfg = {
            "llm": {"provider": "openai", "config": {"model": "placeholder", "api_key": "x"}},
            "embedder": {"provider": "openai", "config": emb_cfg},
            "vector_store": {"provider": "faiss", "config": {
                "collection_name": "stream", "path": str(self.state_dir / "faiss"),
                "embedding_model_dims": 1024, "distance_strategy": "cosine"}},
            "history_db_path": str(self.state_dir / "history.db"),
        }
        self.m = _Mem0.from_config(cfg)
        self.m.llm = Mem0LLM(client)                      # every LLM call → our client, tag=memory
        self.m.embedding_model = Mem0Embedder(embedder)   # every embedding → the study's Embedder
        assert type(self.m.embedding_model).__name__ == "Mem0Embedder"
        self.user_id = "stream"
        self.n_added = 0

    # ------------------------------------------------------------ interface
    def retrieve(self, question: str) -> str:
        if self.n_added == 0:
            return ""
        res = self.m.search(question, user_id=self.user_id, limit=TOP_K)
        hits = res.get("results", res) if isinstance(res, dict) else res
        entries = [f"- {h['memory']}" for h in hits if h.get("memory")]
        body = fit_entries(entries, cap=self.cap - 24, sep="\n")
        return "## Memory (retrieved notes, most relevant first)\n" + body if body else ""

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        msgs = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Answer: {answer or '(none)'}. "
                                             f"This answer was {'correct' if correct else 'wrong'}."},
        ]
        self.m.add(msgs, user_id=self.user_id)
        self.n_added += 1
        (self.state_dir / "n_added").write_text(str(self.n_added))

    def size_tokens(self) -> int:
        try:
            allm = self.m.get_all(user_id=self.user_id)
            items = allm.get("results", allm) if isinstance(allm, dict) else allm
            return sum(count_tokens(h["memory"]) for h in items if h.get("memory"))
        except Exception:
            return -1

    # ------------------------------------------------------------ resume
    def load(self) -> bool:
        p = self.state_dir / "n_added"
        if p.exists() and (self.state_dir / "faiss").exists():
            self.n_added = int(p.read_text())
            return True
        return False

    def dump(self) -> None:      # FAISS + history.db persist on every add
        pass
