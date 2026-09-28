"""One embedder for the whole study: Qwen/Qwen3-Embedding-0.6B via sentence-transformers.

Used by ExpRAG retrieval, Mem0 / A-MEM adapters (later), and the similarity
stratification at analysis time. Every stream item is embedded ONCE into
`cache/embeddings/<stream>.npy` (+ `<stream>.ids.json`); the same vectors serve
retrieval and the stratification, so both see the same geometry.

Symmetric setting: items are embedded as plain text (no query instruction) and compared
by cosine — the retrieval question is "which earlier item resembles this one", a
document-document comparison, not query-document.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CACHE_DIR = Path(__file__).parent / "cache" / "embeddings"


class Embedder:
    """Backend chosen by env: EMBED_BASE_URL set → remote OpenAI-compatible /v1/embeddings
    (e.g. vLLM serving Qwen3-Embedding-0.6B; EMBED_MODEL = served name); otherwise the
    model runs in-process via sentence-transformers. Same vectors either way: L2-normalised
    1024-d. Provenance records which backend produced them."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.base_url = os.environ.get("EMBED_BASE_URL")
        self.model_name = model_name or os.environ.get("EMBED_MODEL", EMBED_MODEL)
        self._memo: dict[str, np.ndarray] = {}
        if self.base_url:
            import openai
            self.backend = "remote"
            self._client = openai.OpenAI(base_url=self.base_url, api_key=os.environ.get("EMBED_API_KEY", "x"))
            self.model = None
        else:
            from sentence_transformers import SentenceTransformer
            self.backend = "local"
            self.model = SentenceTransformer(self.model_name, device=device)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _encode_raw(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self.backend == "local":
            return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                show_progress_bar=len(texts) > 64), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            resp = self._client.embeddings.create(model=self.model_name, input=texts[i:i + batch_size])
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        vecs = np.asarray(out, dtype=np.float32)
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """L2-normalised vectors, memoised by text so a question is never embedded twice."""
        todo = [t for t in dict.fromkeys(texts) if self._key(t) not in self._memo]
        if todo:
            for t, v in zip(todo, self._encode_raw(todo, batch_size)):
                self._memo[self._key(t)] = v
        return np.stack([self._memo[self._key(t)] for t in texts])

    # ------------------------------------------------------------ stream cache
    def embed_stream(self, stream: str, items: dict[str, dict]) -> dict[str, np.ndarray]:
        """Embed every item of a stream once; cached on disk; returns {item_id: vec}."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        npy, ids_path = CACHE_DIR / f"{stream}.npy", CACHE_DIR / f"{stream}.ids.json"
        ids = sorted(items)
        if npy.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
            vecs = np.load(npy)
        else:
            vecs = self.encode([items[i]["input_text"] for i in ids])
            np.save(npy, vecs); ids_path.write_text(json.dumps(ids))
            (CACHE_DIR / f"{stream}.meta.json").write_text(json.dumps(
                {"model": self.model_name, "backend": self.backend, "dim": int(vecs.shape[1])}))
        for i, v in zip(ids, vecs):                       # prefill the memo
            self._memo[self._key(items[i]["input_text"])] = v
        return dict(zip(ids, vecs))
