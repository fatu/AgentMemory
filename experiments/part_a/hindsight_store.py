"""Hindsight (vectorize-io/hindsight, `hindsight-all` 0.10.x) as a Part A memory system — step 9b.

Hindsight runs as its own server (`hindsight-api`, embedded Postgres via pg0://) and makes its
LLM calls server-side (fact extraction on retain, reflect); they cannot pass through our
instrumented client. Two things keep the accounting honest without a proxy:
  - the server is started with the ROW'S backbone as its LLM (same-backbone rule — see
    hindsight_server.sh) and the study's embedder as its embeddings provider;
  - every retain / reflect response carries `usage` (input / output / cached / thoughts
    tokens), which this wrapper logs into calls.jsonl in the live schema (tag=memory for
    retain, tag=agent for reflect). recall makes no LLM call (embeddings + reranker only).

Interface used by the three Part A systems:
    store = HindsightStore(bank_id, log_client)          # HINDSIGHT_URL env (default localhost:8888)
    store.fresh()                                        # delete + create the bank
    store.retain(text, timestamp=dt, context="session 3")
    block = store.recall(question)                       # ≤ CAP o200k tokens, our formatting
    text  = store.reflect(question)                      # Hindsight's own reader — separate row
Provenance: store.info() → server version, bank id, recall budget.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

STREAM_RUNNER = Path(__file__).resolve().parents[1] / "stream_runner"
sys.path.insert(0, str(STREAM_RUNNER))
from memory.base import CAP, fit_entries                                    # noqa: E402

HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "http://localhost:8888")
RECALL_BUDGET = os.environ.get("HINDSIGHT_RECALL_BUDGET", "mid")            # low | mid | high
RECALL_MAX_TOKENS = int(os.environ.get("HINDSIGHT_RECALL_MAX_TOKENS", "4096"))


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        from dateutil import parser as dp
        return dp.parse(s, fuzzy=True)
    except Exception:                                    # noqa: BLE001
        return None


class HindsightStore:
    def __init__(self, bank_id: str, log_client, url: str = HINDSIGHT_URL):
        from hindsight_client import Hindsight
        self.h = Hindsight(base_url=url, timeout=600.0)
        self.bank_id = bank_id
        self.log = log_client                            # our LLMClient: only its _log() is used
        self.n_retained = 0
        self.retain_usage = {"in": 0, "out": 0, "cached": 0, "calls": 0, "seconds": 0.0}

    # ------------------------------------------------------------ bank
    def fresh(self):
        try:
            self.h.delete_bank(self.bank_id)
        except Exception:                                # noqa: BLE001  (no such bank)
            pass
        self.h.create_bank(self.bank_id, name=self.bank_id)
        self.n_retained = 0

    def exists(self) -> bool:
        try:
            self.h.get_bank_config(self.bank_id)
            return True
        except Exception:                                # noqa: BLE001
            return False

    # ------------------------------------------------------------ write
    def retain(self, text: str, *, timestamp: datetime | None = None, context: str | None = None,
               document_id: str | None = None, row_id: str = "ingest"):
        t0 = time.perf_counter()
        r = self.h.retain(self.bank_id, text, timestamp=timestamp, context=context, document_id=document_id)
        dt = time.perf_counter() - t0
        u = getattr(r, "usage", None)
        usage = {"in": int(getattr(u, "input_tokens", 0) or 0), "out": int(getattr(u, "output_tokens", 0) or 0),
                 "cache_read": int(getattr(u, "cached_tokens", 0) or 0), "cache_write": 0}
        self.n_retained += 1
        self.retain_usage["in"] += usage["in"]; self.retain_usage["out"] += usage["out"]
        self.retain_usage["cached"] += usage["cache_read"]; self.retain_usage["calls"] += 1
        self.retain_usage["seconds"] += dt
        self.log._log(row_id, "memory", usage, dt, retries=0)
        return r

    # ------------------------------------------------------------ read
    def recall(self, question: str, *, query_timestamp: str | None = None, cap: int = CAP) -> str:
        resp = self.h.recall(self.bank_id, question, max_tokens=RECALL_MAX_TOKENS, budget=RECALL_BUDGET,
                             query_timestamp=query_timestamp)
        entries = []
        for r in resp.results or []:
            when = r.occurred_start or r.mentioned_at or ""
            kind = f"[{r.type}] " if r.type else ""
            entries.append(f"- {kind}{r.text}" + (f" ({when[:10]})" if when else ""))
        body = fit_entries(entries, cap=cap - 24, sep="\n")
        return "## Memory (recalled facts, most relevant first)\n" + body if body else ""

    def reflect(self, question: str, *, row_id: str, budget: str = "low") -> str:
        t0 = time.perf_counter()
        r = self.h.reflect(self.bank_id, question, budget=budget)
        dt = time.perf_counter() - t0
        u = getattr(r, "usage", None)
        usage = {"in": int(getattr(u, "input_tokens", 0) or 0), "out": int(getattr(u, "output_tokens", 0) or 0),
                 "cache_read": int(getattr(u, "cached_tokens", 0) or 0), "cache_write": 0}
        self.log._log(row_id, "agent", usage, dt, retries=0)
        return r.text or ""

    # ------------------------------------------------------------ provenance
    def info(self) -> dict:
        try:
            v = self.h.get_version()
            version = getattr(v, "version", None) or str(v)
        except Exception as e:                           # noqa: BLE001
            version = f"unknown ({e!r})"
        return {"hindsight_url": HINDSIGHT_URL, "hindsight_server_version": version, "bank_id": self.bank_id,
                "recall_budget": RECALL_BUDGET, "recall_max_tokens": RECALL_MAX_TOKENS,
                "server_llm": {k: os.environ.get(k) for k in ("HINDSIGHT_API_LLM_PROVIDER", "HINDSIGHT_API_LLM_MODEL",
                                                             "HINDSIGHT_API_LLM_BASE_URL", "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
                                                             "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL", "HINDSIGHT_API_RERANKER_PROVIDER")},
                "note": "server-side LLM usage taken from retain/reflect responses; recall makes no LLM call"}
