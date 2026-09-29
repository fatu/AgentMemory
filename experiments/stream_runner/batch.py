"""Anthropic Message Batches for the Sonnet Part A slices (step 9.3 proof → step 12 runs).

Same request assembly and the same calls.jsonl schema as LLMClient — a batched row is
indistinguishable from a live one downstream, except for `batch_id` and a per-batch (not
per-request) latency. Batching applies to independent calls only: answering and judging.
Memory ingestion (Mem0 / A-MEM) is a chain of dependent calls and stays on the live client.

    bc = BatchClient(run_id, log_path)
    reqs = [bc.build(custom_id="12", tag="agent", system=history, user=prompt, max_tokens=64,
                     cache_system=True), ...]
    bid = bc.submit(reqs)            # → batch id; manifest under <log dir>/batches/<id>.json
    bc.wait(bid)                     # polls processing_status until "ended"
    out = bc.collect(bid)            # {custom_id: {"text", "usage", "error"}}, logged with tag/row_id

Batch price = 50 % of list on input and output; prompt-cache hits inside a batch are not
guaranteed (requests run in parallel) — `usage.cache_read` says what actually happened.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from client import MODEL_IDS, THINK_BUDGET, ANSWER_HEADROOM

POLL_S = int(os.environ.get("BATCH_POLL_S", "30"))


class BatchClient:
    backbone = "sonnet5"

    def __init__(self, run_id: str, log_path: str | Path, model: str | None = None):
        import anthropic
        self._client = anthropic.Anthropic()
        self.model = model or MODEL_IDS["sonnet5"]
        self.run_id = run_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_dir = self.log_path.parent / "batches"
        self.batch_dir.mkdir(exist_ok=True)
        self.calls: list[dict] = []

    # ------------------------------------------------------------ requests
    def build(self, *, custom_id: str, tag: str, system: str, user: str, max_tokens: int = 1024,
              temperature: float = 0.0, cache_system: bool = False, thinking: bool = False,
              history: list[dict] | None = None) -> dict:
        """One batch request; the message assembly mirrors LLMClient._call_anthropic."""
        if tag not in ("agent", "memory", "judge"):
            raise ValueError(tag)
        msgs = [dict(m) for m in (history or [])] + [{"role": "user", "content": user}]
        if cache_system and history:
            last = msgs[-2]
            last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
        params: dict = {"model": self.model, "messages": msgs}
        if system:
            params["system"] = ([{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                                if cache_system else system)
        if thinking:
            params.update(max_tokens=THINK_BUDGET + ANSWER_HEADROOM,
                          thinking={"type": "enabled", "budget_tokens": THINK_BUDGET})
        else:
            params.update(max_tokens=max_tokens, temperature=temperature)
        return {"custom_id": custom_id, "params": params, "_tag": tag, "_thinking": thinking}

    # ------------------------------------------------------------ lifecycle
    def submit(self, requests: list[dict]) -> str:
        assert requests, "empty batch"
        ids = [r["custom_id"] for r in requests]
        assert len(ids) == len(set(ids)), "custom_id must be unique within a batch"
        batch = self._client.messages.batches.create(
            requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in requests])
        manifest = {"batch_id": batch.id, "run_id": self.run_id, "model": self.model, "submitted": time.time(),
                    "n": len(requests), "tags": {r["custom_id"]: r["_tag"] for r in requests},
                    "thinking": {r["custom_id"]: r["_thinking"] for r in requests}}
        (self.batch_dir / f"{batch.id}.json").write_text(json.dumps(manifest, indent=1))
        print(f"[batch] submitted {batch.id}: {len(requests)} requests")
        return batch.id

    def status(self, batch_id: str):
        b = self._client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        return b.processing_status, {"processing": c.processing, "succeeded": c.succeeded, "errored": c.errored,
                                     "canceled": c.canceled, "expired": c.expired}

    def wait(self, batch_id: str, poll_s: int = POLL_S, timeout_s: int = 24 * 3600):
        t0 = time.time()
        while True:
            st, counts = self.status(batch_id)
            if st == "ended":
                m = self._manifest(batch_id); m["ended"] = time.time(); m["counts"] = counts
                (self.batch_dir / f"{batch_id}.json").write_text(json.dumps(m, indent=1))
                print(f"[batch] {batch_id} ended after {time.time() - t0:.0f}s: {counts}")
                return counts
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"batch {batch_id} still {st} after {timeout_s}s")
            print(f"[batch] {batch_id} {st} {counts} — {time.time() - t0:.0f}s", flush=True)
            time.sleep(poll_s)

    def collect(self, batch_id: str) -> dict[str, dict]:
        """Results keyed by custom_id; every result is logged to calls.jsonl (row_id = custom_id)."""
        m = self._manifest(batch_id)
        wall = (m.get("ended") or time.time()) - m["submitted"]
        out: dict[str, dict] = {}
        for r in self._client.messages.batches.results(batch_id):
            cid, tag = r.custom_id, m["tags"].get(r.custom_id, "agent")
            if r.result.type == "succeeded":
                msg = r.result.message
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                u = msg.usage
                usage = {"in": u.input_tokens, "out": u.output_tokens,
                         "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                         "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                         "truncated": msg.stop_reason == "max_tokens", "thinking": m["thinking"].get(cid, False)}
                out[cid] = {"text": text, "usage": usage, "error": None}
                self._log(cid, tag, usage, wall, batch_id)
            else:
                err = getattr(r.result, "error", None)
                out[cid] = {"text": "", "usage": {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}, "error": f"{r.result.type}: {err}"}
                self._log(cid, tag, out[cid]["usage"], wall, batch_id, error=out[cid]["error"])
        missing = set(m["tags"]) - set(out)
        if missing:
            print(f"[batch] WARNING {len(missing)} requests have no result yet: {sorted(missing)[:5]}…")
        return out

    # ------------------------------------------------------------ internals
    def _manifest(self, batch_id: str) -> dict:
        return json.load(open(self.batch_dir / f"{batch_id}.json"))

    def _log(self, row_id, tag, usage, batch_wall_s, batch_id, error=None):
        line = {"ts": time.time(), "run_id": self.run_id, "row_id": row_id, "tag": tag, "backbone": self.backbone,
                "model": self.model, "in": usage.get("in", 0), "out": usage.get("out", 0),
                "cache_read": usage.get("cache_read", 0), "cache_write": usage.get("cache_write", 0),
                "latency_s": round(batch_wall_s, 1), "retries": 0, "thinking": usage.get("thinking", False),
                "truncated": usage.get("truncated", False), "batch_id": batch_id, "batch": True}
        if error is not None:
            line["error"] = error
        self.calls.append(line)
        with self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")


def batch_cost_usd(calls: list[dict], in_per_m: float = 3.0, out_per_m: float = 15.0) -> float:
    """List-price cost × 0.5 (batch); cache reads at 0.1×, cache writes at 1.25× of the batch input price."""
    inp = sum(c["in"] for c in calls); cr = sum(c.get("cache_read", 0) for c in calls)
    cw = sum(c.get("cache_write", 0) for c in calls); out = sum(c["out"] for c in calls)
    return 0.5 * ((inp * 1.0 + cr * 0.1 + cw * 1.25) * in_per_m + out * out_per_m) / 1e6
