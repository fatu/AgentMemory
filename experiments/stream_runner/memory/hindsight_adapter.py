"""Adapter for Hindsight (https://github.com/vectorize-io/hindsight), a Postgres-backed
agent memory system with retain()/recall()/reflect() operations.

Unlike mem0/A-Mem/ACE, Hindsight is not a pure in-process Python object - even its
"embedded" mode runs a real FastAPI+uvicorn server (in a background thread of THIS
process, not a subprocess) backed by an embedded Postgres ("pg0"). We monkeypatch its
LLM and embedding provider factories so that server routes every LLM/embedding call
through our own instrumented client/embedder instead of a real external provider.

Scope: only retain() (-> evolve()) and recall() (-> retrieve()) are wired up.
reflect() (Hindsight's separate "deep analysis" operation, an agentic tool-use loop)
is intentionally NOT supported - it needs LLMInterface.call_with_tools(), which none
of our other adapters need either, and doesn't map onto this project's Memory ABC.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import tiktoken

from .base import CAP, Memory, count_tokens, fit_entries
from .llm_shim import extract_json_text

_QUERY_ENC = tiktoken.get_encoding("o200k_base")


def _truncate_query(text: str, max_tokens: int) -> str:
    toks = _QUERY_ENC.encode(text, disallowed_special=())
    return text if len(toks) <= max_tokens else _QUERY_ENC.decode(toks[:max_tokens])


EMBED_DIM = 1024  # must match embed.py's EMBED_DIM (qwen3-embedding-0.6b)
DEFAULT_MAX_TOKENS = 2000
# Server-side recall() query cap is raised to 1000 via HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS
# (see _set_hindsight_env) from Hindsight's 500-token default. Measured by Hindsight's own
# tokenizer, which can count a bit differently from ours - keep a safety margin below 1000.
MAX_QUERY_TOKENS = 900

_shared_server = None  # one embedded Hindsight server (Postgres + API) per process


def _install_llm_seam(client):
    """Point hindsight_api's LLM provider factory at OUR instrumented client.

    create_llm_provider() is only ever called as a bare name from inside its own
    module (llm_wrapper.py), so patching that one attribute is enough - unlike the
    embeddings factory below, no other module imports it by name.
    """
    import hindsight_api.engine.llm_wrapper as llm_wrapper
    from hindsight_api.engine.llm_interface import LLMInterface
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    def _messages_to_system_user(messages: list[dict]) -> tuple[str, str]:
        def text_of(content):
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal content parts; we have no vision
                return " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text")
            return str(content)

        system = "\n\n".join(text_of(m["content"]) for m in messages if m.get("role") == "system")
        rest = [m for m in messages if m.get("role") != "system"]
        if len(rest) == 1:
            user = text_of(rest[0]["content"])
        else:
            user = "\n\n".join(f"[{m['role']}]\n{text_of(m['content'])}" for m in rest)
        # The Anthropic API attaches cache_control to the system block and 400s if
        # that block is empty text - always pass something.
        return system or "You are a careful, precise assistant.", user

    class HindsightLLMBridge(LLMInterface):
        async def verify_connection(self) -> None:
            return None

        async def cleanup(self) -> None:
            return None

        async def call(
            self, messages, response_format=None, max_completion_tokens=None,
            temperature=None, scope="memory", max_retries=10, initial_backoff=1.0,
            max_backoff=60.0, skip_validation=False, strict_schema=False,
            cached_prefix=None, attempt_context=None,
        ) -> LLMCallResult:
            system, user = _messages_to_system_user(messages)
            if response_format is not None:
                schema = json.dumps(response_format.model_json_schema())
                user += (f"\n\nRespond with a single JSON object matching this schema "
                         f"and nothing else:\n{schema}")
            row_id = getattr(client, "current_row_id", None) or f"hindsight:{scope}"
            max_tokens = max_completion_tokens or DEFAULT_MAX_TOKENS

            text, content, last_err = "", None, None
            for _ in range(2):  # one retry, only for structured-output parse failures
                # client.complete() is a blocking network call - run it off-thread so it
                # doesn't stall hindsight's single-threaded event loop (observed as
                # "EVENT LOOP BLOCKED" watchdog warnings / stalled /health checks).
                text = await asyncio.to_thread(
                    client.complete, system, user, tag="memory", row_id=row_id,
                    max_tokens=max_tokens, thinking=False,
                ) or ""
                if response_format is None:
                    content = text
                    break
                try:
                    raw = json.loads(extract_json_text(text))
                    content = raw if skip_validation else response_format.model_validate(raw)
                    break
                except Exception as e:
                    last_err = e
                    user += f"\n\n(Your previous reply could not be parsed: {e}. Try again.)"
            else:
                raise ValueError(f"Hindsight LLM bridge: could not parse structured output: {last_err}")

            usage = TokenUsage(
                input_tokens=count_tokens(system) + count_tokens(user),
                output_tokens=count_tokens(text),
                total_tokens=count_tokens(system) + count_tokens(user) + count_tokens(text),
            )
            return LLMCallResult(content=content, usage=usage)

        async def call_with_tools(self, *args, **kwargs):
            raise NotImplementedError(
                "HindsightMemory does not support reflect()/tool-calling - only retain()/recall()"
            )

    bridge = HindsightLLMBridge(provider="bridge", api_key="x", base_url="",
                                model=getattr(client, "model", "bridge"))
    llm_wrapper.create_llm_provider = lambda *a, **kw: bridge


def _install_embedding_seam(embedder):
    """Point hindsight_api's embeddings factory at OUR shared Embedder.

    memory_engine.py does `from .embeddings import create_embeddings_from_env`, which
    binds its own copy of the name at import time - same gotcha as ACE's
    `from llm import timed_llm_call`. Patch both modules' copies.
    """
    import hindsight_api.engine.embeddings as embeddings_mod
    import hindsight_api.engine.memory_engine as memory_engine_mod
    from hindsight_api.engine.embeddings import Embeddings

    class HindsightEmbeddingsBridge(Embeddings):
        @property
        def provider_name(self) -> str:
            return "bridge"

        @property
        def dimension(self) -> int:
            return EMBED_DIM

        async def initialize(self) -> None:
            return None

        async def encode(self, texts: list[str]) -> list[list[float]]:
            vecs = await asyncio.to_thread(embedder.encode, texts)
            return [v.tolist() for v in vecs]

    bridge = HindsightEmbeddingsBridge()
    embeddings_mod.create_embeddings_from_env = lambda: bridge
    memory_engine_mod.create_embeddings_from_env = lambda: bridge


def _set_hindsight_env():
    """Must run before ANY hindsight_api submodule gets imported - not just before
    `import hindsight` itself. The real trigger turned out to be _install_llm_seam's
    `import hindsight_api.engine.llm_wrapper`, whose module-level code calls
    configure_http_logging() -> get_config(), caching HindsightConfig.from_env() for the
    rest of the process. That happens BEFORE _get_shared_hindsight_client() ever runs
    (it's called later in HindsightMemory.__init__), which is why setting these env vars
    there was too late - confirmed by tracing HindsightConfig.from_env()'s call site back
    to _install_llm_seam. So this must be the very first thing HindsightMemory.__init__
    does, full stop.
    """
    # The cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) is already fully
    # cached locally, but huggingface_hub still does a network HEAD request to check for
    # updates before falling back to cache - flaky on this network (observed repeated SSL
    # handshake timeouts). Nothing new needs downloading, so skip that check entirely.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Server-side recall() query cap (default 500 tokens, see hindsight_api/config.py's
    # DEFAULT_RECALL_MAX_QUERY_TOKENS) - raise it so long MMLU-Pro questions don't 400.
    os.environ.setdefault("HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS", "1000")


def _get_shared_hindsight_client():
    """Start the one embedded server this process needs, lazily, on first use.

    Starting embedded Postgres per HindsightMemory instance would be far too slow
    (this project creates a fresh memory instance per conversation in some experiments,
    e.g. experiments/part_a/locomo/run.py) - banks (see HindsightMemory.bank_id) are
    Hindsight's own isolation unit, so one shared server is the right granularity.
    """
    global _shared_server
    if _shared_server is None:
        from hindsight import start_server
        # llm_provider/llm_api_key/llm_model are required args but never actually used -
        # create_llm_provider is fully monkeypatched to ignore them (see _install_llm_seam).
        # Default timeout=30s isn't always enough for a cold start (loading the local
        # cross-encoder reranker + local embedding scaffolding the first time).
        _shared_server = start_server(llm_provider="mock", llm_api_key="x", llm_model="bridge", timeout=180.0)
    from hindsight import HindsightClient
    return HindsightClient(base_url=_shared_server.url)


class HindsightMemory(Memory):
    name = "hindsight"

    def __init__(self, client, embedder, state_path: str | Path, cap: int = CAP):
        _set_hindsight_env()  # must run before _install_llm_seam's hindsight_api import
        self.client = client
        self.embedder = embedder
        self.cap = cap
        self.state_path = Path(state_path)
        self.n_retained = 0

        _install_llm_seam(client)
        _install_embedding_seam(embedder)
        self.hclient = _get_shared_hindsight_client()

        # Deterministic from state_path (not random) so a rerun against the same
        # state_path lands on the same bank - NOTE: whether that bank's data actually
        # survives a process restart depends on where embedded Postgres ("pg0")
        # persists to, which this adapter does not control or verify. Treat resume
        # as untested until checked against a real run.
        self.bank_id = "b" + hashlib.sha256(str(self.state_path).encode()).hexdigest()[:16]
        try:
            self.hclient.create_bank(bank_id=self.bank_id)
        except Exception:
            pass  # bank already exists

    def retrieve(self, question: str) -> str:
        if self.n_retained == 0:
            return ""
        resp = self.hclient.recall(bank_id=self.bank_id, query=_truncate_query(question, MAX_QUERY_TOKENS),
                                   max_tokens=max(self.cap - 32, 256))
        entries = [r.text for r in resp.results if getattr(r, "text", None)]
        body = fit_entries(entries, cap=self.cap - 32)
        return "## Memory (Hindsight recall, most relevant first)\n" + body if body else ""

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        content = (f"Question: {question}\n"
                   f"Answer given: {answer or '(none)'}\n"
                   f"Result: {'correct' if correct else 'wrong'}")
        self.hclient.retain(bank_id=self.bank_id, content=content)
        self.n_retained += 1

    def size_tokens(self) -> int:
        try:
            resp = self.hclient.list_memories(bank_id=self.bank_id, limit=1000)
            return sum(count_tokens(it.text or "") for it in resp.items)
        except Exception:
            return -1
