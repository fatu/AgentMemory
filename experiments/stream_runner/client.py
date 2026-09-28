"""Instrumented LLM client — the one call site for every model request in the study.

Two backbones behind one interface:
  sonnet5  → Anthropic SDK, model "claude-sonnet-5"
  qwen27b  → OpenAI-compatible SDK against a local vLLM server (QWEN_BASE_URL, QWEN_MODEL)

Every call appends exactly one JSON line to `log_path` — including failed calls after the
retry budget is spent (with an "error" field, then re-raised). `tag` is mandatory so each
row's tokens are attributed to agent / memory / judge; harness-reported token fields are
never trusted, this log is the source of `tokens_agent` / `tokens_mem` / `tokens_judge`.

Env: ANTHROPIC_API_KEY · QWEN_BASE_URL (e.g. http://box:8000/v1) · QWEN_MODEL (served
name; if unset, the first entry of /v1/models is used and recorded).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Literal

Backbone = Literal["sonnet5", "qwen27b"]
Tag = Literal["agent", "memory", "judge"]

MODEL_IDS = {"sonnet5": "claude-sonnet-5"}
MAX_ATTEMPTS = 5

# Thinking (user decision 2026-09-18): ON for `agent` calls only, capped; OFF for memory /
# judge calls. Sonnet: extended thinking with budget_tokens=THINK_BUDGET (the API forces
# temperature=1 in this mode). Qwen: chat_template enable_thinking + the same budget as
# max_tokens headroom, temperature 0.6 / top_p 0.95 (Qwen's thinking-mode recommendation;
# greedy decoding loops), seed=0. Serve vLLM with `--reasoning-parser qwen3` so the
# reasoning lands in `reasoning_content`; <think> blocks in `content` are stripped anyway.
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "8192"))   # pilot: 2048 truncated 35-40% of Qwen answers, 4096 still 20%
ANSWER_HEADROOM = 512
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


class LLMClient:
    def __init__(self, backbone: Backbone, run_id: str, log_path: str | Path):
        self.backbone = backbone
        self.run_id = run_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict] = []          # in-memory mirror of the log (per-row sums)

        if backbone == "sonnet5":
            import anthropic
            self._sdk = anthropic
            self._client = anthropic.Anthropic()
            self.model = MODEL_IDS["sonnet5"]
        elif backbone == "qwen27b":
            import openai
            self._sdk = openai
            base_url = os.environ["QWEN_BASE_URL"]
            self._client = openai.OpenAI(base_url=base_url, api_key=os.environ.get("QWEN_API_KEY", "x"))
            self.model = os.environ.get("QWEN_MODEL") or self._client.models.list().data[0].id
        else:
            raise ValueError(f"unknown backbone {backbone!r}")

    # ------------------------------------------------------------------ public
    def complete(
        self,
        system: str,
        user: str,
        *,
        tag: Tag,
        row_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cache_system: bool = False,
        thinking: bool = False,
        history: list[dict] | None = None,
    ) -> str:
        """One chat completion. `cache_system=True` marks the system block as a prompt-cache
        prefix (Anthropic only; no-op on vLLM, which prefix-caches automatically) — used by
        the Part A full-context rows where many questions share one history.
        `history` = prior {role, content} turns placed between the system block and `user`
        (BEAM's official long-context path feeds the chat as real turns); with
        `cache_system` the last history turn carries the cache breakpoint too. An empty
        `system` sends no system block at all (official prompts that have none)."""
        if tag not in ("agent", "memory", "judge"):
            raise ValueError(f"tag must be agent|memory|judge, got {tag!r}")

        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                if self.backbone == "sonnet5":
                    text, usage = self._call_anthropic(system, user, max_tokens, temperature, cache_system, thinking, history)
                else:
                    text, usage = self._call_openai(system, user, max_tokens, temperature, thinking, history)
                usage["thinking"] = thinking
                self._log(row_id, tag, usage, time.perf_counter() - t0, retries=attempt)
                return text
            except self._retryable() as e:  # rate limit, 5xx, connection, timeout
                last_err = e
                time.sleep(2 ** attempt)
        # retry budget spent: log the failure, then raise — never a silent gap in the log
        self._log(row_id, tag, {"in": 0, "out": 0, "cache_read": 0},
                  time.perf_counter() - t0, retries=MAX_ATTEMPTS, error=repr(last_err))
        assert last_err is not None
        raise last_err

    # ---------------------------------------------------------------- backends
    def _call_anthropic(self, system, user, max_tokens, temperature, cache_system, thinking, history=None):
        msgs = [dict(m) for m in (history or [])] + [{"role": "user", "content": user}]
        if cache_system and history:            # breakpoint after the shared history turns
            last = msgs[-2]
            last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
        kwargs = dict(model=self.model, messages=msgs)
        if system:                              # empty system → no system block at all
            kwargs["system"] = (
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if cache_system else system
            )
        if thinking:
            kwargs.update(max_tokens=THINK_BUDGET + ANSWER_HEADROOM,
                          thinking={"type": "enabled", "budget_tokens": THINK_BUDGET})
            # temperature must be left at the API default (1) in thinking mode
        else:
            kwargs.update(max_tokens=max_tokens, temperature=temperature)
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        usage = {
            "in": u.input_tokens,
            "out": u.output_tokens,          # includes thinking tokens
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "truncated": resp.stop_reason == "max_tokens",
        }
        return text, usage

    def _call_openai(self, system, user, max_tokens, temperature, thinking, history=None):
        msgs = ([{"role": "system", "content": system}] if system else []) \
            + [dict(m) for m in (history or [])] + [{"role": "user", "content": user}]
        kwargs = dict(
            model=self.model, seed=0, messages=msgs,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}},
        )
        if thinking:
            kwargs.update(max_tokens=THINK_BUDGET + ANSWER_HEADROOM, temperature=0.6, top_p=0.95)
        else:
            kwargs.update(max_tokens=max_tokens, temperature=temperature)
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = _THINK_RE.sub("", choice.message.content or "")   # no-op when --reasoning-parser is on
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        usage = {
            "in": u.prompt_tokens,
            "out": u.completion_tokens,      # includes reasoning tokens
            "cache_read": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
            "cache_write": 0,
            "truncated": choice.finish_reason == "length",
        }
        return text, usage

    def _retryable(self):
        s = self._sdk
        return (s.RateLimitError, s.APIConnectionError, s.APITimeoutError, s.InternalServerError)

    # --------------------------------------------------------------------- log
    def _log(self, row_id, tag, usage, latency_s, *, retries, error=None):
        line = {
            "ts": time.time(),
            "run_id": self.run_id,
            "row_id": row_id,
            "tag": tag,
            "backbone": self.backbone,
            "model": self.model,
            "in": usage.get("in", 0),
            "out": usage.get("out", 0),
            "cache_read": usage.get("cache_read", 0),
            "cache_write": usage.get("cache_write", 0),
            "latency_s": round(latency_s, 3),
            "retries": retries,
            "thinking": usage.get("thinking", False),
            "truncated": usage.get("truncated", False),
        }
        if error is not None:
            line["error"] = error
        self.calls.append(line)
        with self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")


# ------------------------------------------------------------------ smoke test
if __name__ == "__main__":
    import sys
    backbones = sys.argv[1:] or ["sonnet5", "qwen27b"]
    for b in backbones:
        c = LLMClient(b, run_id="smoke", log_path="runs/smoke/calls.jsonl")
        out = c.complete("You are terse.", "Say OK.", tag="memory", row_id="smoke-plain")
        print(b, "plain →", out.strip()[:40])
        out = c.complete("Answer with a final line 'Answer: <integer>'.",
                         "What is 17 * 23?", tag="agent", row_id="smoke-think", thinking=True)
        print(b, "think →", out.strip().splitlines()[-1][:40])
    print(Path("runs/smoke/calls.jsonl").read_text())
