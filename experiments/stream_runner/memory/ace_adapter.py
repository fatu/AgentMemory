"""ACE — Agentic Context Engineering (Zhang et al. 2025, arXiv 2510.04618; code
`ace-agent/ace`, checkout at projects/AgentMemory/code-repo/ace) behind the Memory interface.

As shipped: a structured playbook of bullets `[id] helpful=N harmful=M :: content` in fixed
sections; after each episode the REFLECTOR reads the trajectory + environment feedback and
tags which bullets helped/hurt; counts are updated; the CURATOR proposes incremental
operations (ADD / UPDATE / REMOVE) that are applied programmatically — no whole-playbook
rewrite (the "delta vs. rewrite" contrast with Dynamic Cheatsheet, blog §2.1). Optional
BulletpointAnalyzer de-duplicates by embedding similarity.

Wired to the study: the ONE seam is `llm.timed_llm_call(client, api_provider, model,
prompt, role, call_id, ...)` used by both reflector and curator — replaced at import with
a function that routes to our instrumented client (tag=memory, no thinking). The playbook
is injected whole (<= CAP), like DC. `use_ground_truth=False` everywhere: ACE sees the
task, the trajectory summary, the model's output, and binary feedback — never a gold
answer/patch. Curator runs every episode (ACE's default `curator_frequency` is
per-step in the online loop).

Intended stream: the lite SWE-bench execution stream (10c), where a trajectory exists.
On QA streams the reasoning_trace is the answer text — a degenerate mode, run only as a
comparison row if ever.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .base import CAP, Memory, count_tokens

_DEFAULT_ACE = Path(__file__).resolve().parents[3] / "code-repo" / "ace"
MEM_MAX_TOKENS = 4096          # ACE's own default max_tokens for reflector/curator


def _install_llm_seam(client):
    """Make `llm.timed_llm_call` (imported by ace.core.reflector / curator) call our client."""
    import llm as ace_llm

    def timed_llm_call(_api_client, _api_provider, model, prompt, role, call_id,
                       max_tokens=MEM_MAX_TOKENS, log_dir=None, use_json_mode=False, **_):
        row_id = getattr(client, "current_row_id", None) or f"ace:{call_id}"
        system = "Respond with a single JSON object and nothing else." if use_json_mode else ""
        text = client.complete(system, prompt, tag="memory", row_id=row_id,
                               max_tokens=max_tokens, thinking=False) or ""
        return text, {"role": role, "call_id": call_id, "model": getattr(client, "model", "?"),
                      "prompt": prompt, "response": text}

    ace_llm.timed_llm_call = timed_llm_call
    # reflector/curator did `from llm import timed_llm_call` → patch their module globals too
    import ace.core.reflector as r, ace.core.curator as c
    r.timed_llm_call = timed_llm_call
    c.timed_llm_call = timed_llm_call


class ACEMemory(Memory):
    name = "ace"

    def __init__(self, client, state_path: str | Path, cap: int = CAP, dedup: bool = False):
        ace_dir = Path(os.environ.get("ACE_PATH", _DEFAULT_ACE))
        if str(ace_dir) not in sys.path:
            sys.path.insert(0, str(ace_dir))
        from ace.core.reflector import Reflector
        from ace.core.curator import Curator
        from ace.ace import ACE as _ACE
        import playbook_utils as pu

        _install_llm_seam(client)
        self.pu = pu
        self.client = client
        self.cap = cap
        self.state_path = Path(state_path)
        # api_client / api_provider / model are unused once the seam is installed
        self.reflector = Reflector(None, "seam", "seam", max_tokens=MEM_MAX_TOKENS)
        self.curator = Curator(None, "seam", "seam", max_tokens=MEM_MAX_TOKENS)
        self.playbook: str = _ACE._initialize_empty_playbook(self)     # the shipped section skeleton
        self.next_global_id = 1
        self.position = 0
        self.n_ops = 0
        self.reflect_fails = 0
        self.curate_fails = 0
        self.curate_noops = 0        # curator returned no operations (incl. parse failures inside ACE)
        self.dedup = dedup           # BulletpointAnalyzer needs its own embedder; off by default

    # ------------------------------------------------------------ interface
    def retrieve(self, question: str) -> str:
        if not self._has_bullets():
            return ""
        return f"## Memory (playbook, {self.n_ops} updates)\n{self.playbook}"

    def evolve(self, question: str, answer: str, correct: bool,
               trajectory: str | None = None, feedback: str | None = None) -> None:
        """`trajectory` = summary of the episode (tool calls / reasoning); `feedback` = the
        environment's verdict text (test output). On QA streams both default to the answer."""
        self.position += 1
        trace = trajectory or answer or "(no trajectory)"
        env = feedback or ("Result: correct" if correct else "Result: wrong")
        bullets_used = self._bullets_text()

        # 1) reflector: what helped / hurt, which bullets to tag
        try:
            reflection, bullet_tags, _ = self.reflector.reflect(
                question=question, reasoning_trace=trace, predicted_answer=answer or "(none)",
                ground_truth=None, environment_feedback=env, bullets_used=bullets_used,
                use_ground_truth=False, use_json_mode=False,
                call_id=f"q{self.position}_reflect", log_dir=None)
            if bullet_tags:
                self.playbook = self.pu.update_bullet_counts(self.playbook, bullet_tags)
        except Exception as e:                       # never let a memory failure kill the stream
            self.reflect_fails += 1
            reflection = f"(reflection failed: {type(e).__name__})"

        # 2) curator: incremental operations on the playbook
        try:
            stats = self.pu.get_playbook_stats(self.playbook)
            new_pb, self.next_global_id, ops, _ = self.curator.curate(
                current_playbook=self.playbook, recent_reflection=reflection,
                question_context=question, current_step=self.position, total_samples=0,
                token_budget=self.cap, playbook_stats=stats, use_ground_truth=False,
                use_json_mode=False, call_id=f"q{self.position}_curate", log_dir=None,
                next_global_id=self.next_global_id)
            self.playbook = self._fit(new_pb)
            if ops:
                self.n_ops += len(ops)
            else:
                self.curate_noops += 1       # ACE swallows JSON/validation errors and returns the
                                             # playbook unchanged — count those steps explicitly
        except Exception as e:
            self.curate_fails += 1
        self.dump()

    def size_tokens(self) -> int:
        return count_tokens(self.playbook)

    # ------------------------------------------------------------ internals
    def _has_bullets(self) -> bool:
        return any(l.strip().startswith("[") for l in self.playbook.splitlines())

    def _bullets_text(self) -> str:
        return "\n".join(l for l in self.playbook.splitlines() if l.strip().startswith("[")) or "(none)"

    def _fit(self, playbook: str) -> str:
        """Hard cap: drop the LOWEST-value bullets (harmful − helpful, then newest) until
        header + playbook <= cap. Section headers are always kept."""
        budget = self.cap - count_tokens("## Memory (playbook, 000 updates)\n")
        if count_tokens(playbook) <= budget:
            return playbook
        lines = playbook.splitlines()
        bullets = [(i, self.pu.parse_playbook_line(l)) for i, l in enumerate(lines) if l.strip().startswith("[")]
        # parse_playbook_line → {'id','helpful','harmful','content','raw_line'} or None
        ranked = sorted([(i, p) for i, p in bullets if p],
                        key=lambda ip: (ip[1]["harmful"] - ip[1]["helpful"], -ip[0]), reverse=True)  # worst first
        drop = set()
        for i, _ in ranked:
            drop.add(i)
            if count_tokens("\n".join(l for j, l in enumerate(lines) if j not in drop)) <= budget:
                break
        return "\n".join(l for j, l in enumerate(lines) if j not in drop)

    # ------------------------------------------------------------ persistence
    def dump(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "playbook": self.playbook, "next_global_id": self.next_global_id,
            "position": self.position, "n_ops": self.n_ops,
            "reflect_fails": self.reflect_fails, "curate_fails": self.curate_fails,
            "curate_noops": self.curate_noops}, indent=1))

    def load(self) -> bool:
        if not self.state_path.exists():
            return False
        st = json.loads(self.state_path.read_text())
        self.playbook, self.next_global_id = st["playbook"], st["next_global_id"]
        self.position, self.n_ops = st["position"], st["n_ops"]
        self.reflect_fails, self.curate_fails = st.get("reflect_fails", 0), st.get("curate_fails", 0)
        self.curate_noops = st.get("curate_noops", 0)
        return True
