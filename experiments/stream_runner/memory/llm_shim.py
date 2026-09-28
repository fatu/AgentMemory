"""Shims that make third-party memory systems call OUR instrumented client.

Every LLM call a memory system makes must go through `LLMClient.complete()` with
tag="memory" — that is where `tokens_mem_*` comes from. Mem0 and A-MEM each expect a
different object; both shims below route to the same place, never think, and attribute
the call to the row the runner is currently on (`client.current_row_id`).
"""
from __future__ import annotations

import json
import re

MEM_MAX_TOKENS = 2000
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json_text(text: str) -> str:
    """Return the JSON object inside a reply: strip code fences, then take the outermost {...}."""
    text = text or ""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    text = text[a:b + 1] if a != -1 and b > a else text.strip()
    # A-MEM's prompt shows `True or False`; models sometimes echo Python booleans
    return re.sub(r":\s*True\b", ": true", re.sub(r":\s*False\b", ": false", text))


def _split(messages: list[dict]) -> tuple[str, str]:
    """(system, user) from an OpenAI-style message list; extra turns are folded into `user`."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    if len(rest) == 1:
        return system, rest[0]["content"]
    return system, "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in rest)


class _Base:
    def __init__(self, client):
        self.client = client

    def _call(self, system: str, user: str, *, want_json: bool, max_tokens: int = MEM_MAX_TOKENS) -> str:
        row_id = getattr(self.client, "current_row_id", None) or "memory"
        if want_json:
            system = (system + "\n\n" if system else "") + "Respond with a single JSON object and nothing else."
        reply = self.client.complete(system, user, tag="memory", row_id=row_id,
                                     max_tokens=max_tokens, thinking=False) or ""
        return extract_json_text(reply) if want_json else reply


class Mem0LLM(_Base):
    """Drop-in for `Memory.llm` (mem0's LLMBase contract)."""

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        system, user = _split(messages)
        return self._call(system, user, want_json=True)          # mem0's add/update paths all parse JSON


class AmemLLM(_Base):
    """Drop-in for `AgenticMemorySystem.llm_controller.llm` (get_completion contract)."""

    def get_completion(self, prompt: str, response_format=None, temperature: float = 0.7) -> str:
        return self._call("You must respond with a JSON object.", prompt, want_json=True)


class Mem0Embedder:
    """Drop-in for `Memory.embedding_model`, sharing the study's single Embedder."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.config = type("cfg", (), {"embedding_dims": 1024})()

    def embed(self, text, memory_action=None):
        return self.embedder.encode([text])[0].tolist()

    def embed_batch(self, texts, memory_action=None):
        return [v.tolist() for v in self.embedder.encode(list(texts))]
