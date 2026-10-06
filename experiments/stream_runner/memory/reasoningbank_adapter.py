"""ReasoningBank (Google DeepMind, arXiv:2509.25140) - "Scaling Agent Self-Evolving with
Reasoning Memory". No official code release exists for this paper (unlike ACE/A-mem, which
we clone from the authors' own repos - see ace_adapter.py / amem_adapter.py); this is a
from-scratch implementation of the paper's core memory mechanism, written from the published
methodology rather than adapted from someone else's code.

Core idea (distinct from ACE's single append-only playbook and A-mem's per-note linking):
distill BOTH successful and failed trajectories into small, titled "memory items" - a
title, a one-line description, and the actual reusable strategy/pitfall content - then
retrieve the most relevant items by embedding similarity and splice them into the next
prompt. Two deliberate adaptations to this project's harness:

1. Self-judgment: the paper has no ground truth at inference time, so it runs an
   LLM-as-a-judge (same backbone, temperature 0.0) over the trajectory to decide
   success/failure before extraction, and reports that performance stays stable even
   when that judge is only 70-90% accurate. Every benchmark in this project already
   knows the real `correct` flag (it's how score() computes accuracy), so there is no
   reason to spend an extra LLM call re-deriving a noisier version of a signal we
   already have exactly - extraction is conditioned directly on `correct`.
2. Retrieval top-k: the paper's agentic web-task setting uses a large context budget
   and defaults to k=1. This project retrieves top `top_k` candidates by cosine
   similarity and then token-budget-fits them to `cap` (same pattern as exprag.py),
   since our CAP is much smaller and a single item may not fill it.

Extraction output is requested as JSON (title/description/content, max 3 items) rather
than the paper's structured Markdown, purely so it can be parsed the same way every
other adapter in this project parses LLM output (see llm_shim.extract_json_text) -
the two success/failure system prompts below otherwise follow the paper's described
instructions (Figure 9: analyze why a trajectory succeeded vs. reflect on why it failed
and state the lesson). The paper's "MaTTS" test-time-scaling companion technique (running
several sampled trajectories per task and contrasting them before extraction) is a
separate inference-time compute strategy, not a memory-storage mechanism, and is out of
scope here - this adapter implements only the memory bank itself.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .base import CAP, Memory, count_tokens, fit_entries
from .llm_shim import extract_json_text

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json_array(text: str) -> list:
    """The extraction prompt asks for a JSON *array*. llm_shim.extract_json_text takes the
    outermost {...}, which would keep only the first item of an array and then fail the
    `isinstance(parsed, list)` check - so take the outermost [...] here, falling back to a
    single object wrapped in a list."""
    text = text or ""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b > a:
        parsed = json.loads(text[a:b + 1])
    else:
        parsed = json.loads(extract_json_text(text))
    return parsed if isinstance(parsed, list) else [parsed]


TOP_K = 8
EXTRACT_MAX_TOKENS = 1024
MAX_ITEMS_PER_TRAJECTORY = 3

# Paraphrased from the paper's Figure 9 (exact wording isn't published): one instruction
# for trajectories the agent succeeded at, one for trajectories it failed.
_SUCCESS_SYSTEM = f"""You are distilling a reusable reasoning strategy from an agent's SUCCESSFUL \
attempt at a task, so it can be reused on future, different tasks.

Analyze WHY this trajectory led to success: what approach, check, or decision was responsible. \
Extract at most {MAX_ITEMS_PER_TRAJECTORY} memory items. Each item must be general enough to \
transfer to a different question - never reference this specific question's content, numbers, \
or answer choices.

Respond with a JSON array (and nothing else) of up to {MAX_ITEMS_PER_TRAJECTORY} objects, each:
{{"title": "short identifier for the strategy",
  "description": "one-sentence summary of when this applies",
  "content": "the reusable reasoning step, decision rule, or operational insight"}}
If nothing general enough is worth keeping, respond with an empty array []."""

_FAILURE_SYSTEM = f"""You are distilling a reusable lesson from an agent's FAILED attempt at a \
task, so the same mistake is avoided on future, different tasks.

Reflect on WHY this trajectory failed: what assumption, check, or step was missing or wrong. \
State the lesson as something to do differently next time, not the correct answer to this \
specific question. Extract at most {MAX_ITEMS_PER_TRAJECTORY} memory items, each general enough \
to transfer to a different question - never reference this specific question's content, \
numbers, or answer choices.

Respond with a JSON array (and nothing else) of up to {MAX_ITEMS_PER_TRAJECTORY} objects, each:
{{"title": "short identifier for the pitfall/lesson",
  "description": "one-sentence summary of when this applies",
  "content": "what went wrong and what to check or do instead"}}
If nothing general enough is worth keeping, respond with an empty array []."""

# Paper's verbatim framing for how retrieved items are presented to the agent.
_RETRIEVAL_PREFACE = ("Below are some memory items that I accumulated from past interaction "
                      "from the environment that may be helpful to solve the task.")


class ReasoningBankMemory(Memory):
    name = "reasoningbank"

    def __init__(self, client, embedder, state_path: str | Path, cap: int = CAP, top_k: int = TOP_K):
        self.client = client
        self.embedder = embedder
        self.cap = cap
        self.top_k = top_k
        self.state_path = Path(state_path)
        self.items: list[dict] = []    # each: {title, description, content}
        self._vecs: list[np.ndarray] = []
        self.position = 0
        self.n_extracted = 0
        self.parse_fails = 0

    def retrieve(self, question: str) -> str:
        if not self.items:
            return ""
        qv = self.embedder.encode([question])[0]
        sims = np.stack(self._vecs) @ qv
        order = np.argsort(-sims)[: self.top_k]
        entries = [self._format(self.items[j]) for j in order]
        body = fit_entries(entries, cap=self.cap - count_tokens(_RETRIEVAL_PREFACE) - 32)
        return f"## Memory (ReasoningBank, {len(self.items)} items)\n{_RETRIEVAL_PREFACE}\n\n{body}" if body else ""

    def evolve(self, question: str, answer: str, correct: bool,
               trajectory: str | None = None, feedback: str | None = None) -> None:
        self.position += 1
        trace = trajectory or answer or "(no trajectory)"
        system = _SUCCESS_SYSTEM if correct else _FAILURE_SYSTEM
        user = (f"## Task\n{question}\n\n## Agent's trajectory / final answer\n{trace}\n\n"
                f"## Outcome\n{'Success' if correct else 'Failure'}"
                + (f"\n\n## Environment feedback\n{feedback}" if feedback else ""))
        row_id = getattr(self.client, "current_row_id", None) or f"reasoningbank:{self.position}"
        reply = self.client.complete(system, user, tag="memory", row_id=row_id,
                                     max_tokens=EXTRACT_MAX_TOKENS, thinking=False) or ""
        try:
            parsed = _extract_json_array(reply)
        except Exception:
            self.parse_fails += 1
            self.dump()
            return
        for it in parsed[:MAX_ITEMS_PER_TRAJECTORY]:
            if not isinstance(it, dict) or not it.get("content"):
                continue
            item = {"title": str(it.get("title", "")).strip(),
                    "description": str(it.get("description", "")).strip(),
                    "content": str(it["content"]).strip()}
            self.items.append(item)
            self._vecs.append(self.embedder.encode([self._format(item)])[0])
            self.n_extracted += 1
        self.dump()

    def size_tokens(self) -> int:
        return sum(count_tokens(self._format(it)) for it in self.items)

    @staticmethod
    def _format(item: dict) -> str:
        return f"### {item['title']}\n{item['description']}\n{item['content']}"

    def dump(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "items": self.items, "vecs": [v.tolist() for v in self._vecs],
            "position": self.position, "n_extracted": self.n_extracted,
            "parse_fails": self.parse_fails,
        }))

    def load(self) -> bool:
        if not self.state_path.exists():
            return False
        st = json.loads(self.state_path.read_text())
        self.items = st["items"]
        self._vecs = [np.array(v) for v in st["vecs"]]
        self.position, self.n_extracted = st["position"], st["n_extracted"]
        self.parse_fails = st.get("parse_fails", 0)
        return True
