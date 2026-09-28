"""ACE adapter smoke, in three layers.

    python smoke_ace.py offline                      # fake client, no API — contract + parsing + cap
    python smoke_ace.py live --backbone sonnet5      # 3 real episodes through the instrumented client
    python smoke_ace.py live --backbone qwen27b

Layer 3 (a real trajectory) is the SWE stream itself — step 10c.3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("ACE_PATH", str(Path(__file__).resolve().parents[2] / "code-repo" / "ace"))

from memory.ace_adapter import ACEMemory
from memory.base import count_tokens

# Three fake SWE-style episodes: two failures with a shared lesson, one success.
EPISODES = [
    dict(question="django/forms: BoundField.label_tag drops the 'for' attribute when a custom id is set.",
         answer="patch: edited forms/boundfield.py", correct=False,
         trajectory="grep for label_tag; edited boundfield.py; wrote no test; submitted",
         feedback="FAIL: 3 tests failed (test_label_tag_custom_id, test_label_tag_no_id, test_forms)"),
    dict(question="django/forms: ModelChoiceField raises on empty queryset with to_field_name.",
         answer="patch: edited forms/models.py", correct=False,
         trajectory="read models.py; edited ModelChoiceField.__init__; ran full test suite once, 2 failures, submitted anyway",
         feedback="FAIL: 2 tests failed (test_empty_queryset, test_to_field_name)"),
    dict(question="django/forms: DateField accepts ISO strings with trailing whitespace.",
         answer="patch: edited forms/fields.py", correct=True,
         trajectory="ran the failing test first; saw the exact assertion; edited DateField.to_python; re-ran tests: pass",
         feedback="PASS: all tests passed"),
]


class FakeClient:
    """Deterministic stand-in: answers ACE's reflector and curator prompts in their JSON shapes."""
    model = "fake"

    def __init__(self):
        self.n = 0

    def complete(self, system, user, **kw):
        self.n += 1
        assert kw["tag"] == "memory" and kw["thinking"] is False, kw
        if "operations" in user.lower():
            return json.dumps({"reasoning": "distil the lesson from this episode", "operations": [
                {"type": "ADD", "section": "strategies_and_insights",
                 "content": "Run the failing test first and read the exact assertion before editing."},
                {"type": "ADD", "section": "common_mistakes_to_avoid",
                 "content": "Do not submit while the suite still reports failures."}]})
        return json.dumps({"reasoning": "agent skipped verification", "error_identification": "no test run",
                           "correct_approach": "run tests first", "key_insight": "assertion text guides the fix",
                           "bullet_tags": []})


def run(client, state_path: Path, label: str):
    ace = ACEMemory(client, state_path=state_path)
    assert ace.retrieve("x") == "", "playbook must be empty before any episode"
    for i, ep in enumerate(EPISODES, 1):
        block = ace.retrieve(ep["question"])
        ace.evolve(ep["question"], ep["answer"], ep["correct"],
                   trajectory=ep["trajectory"], feedback=ep["feedback"])
        st = json.loads(state_path.read_text())
        print(f"[{label}] episode {i}: injected {count_tokens(block):4d} tok | ops so far {st['n_ops']} "
              f"| noops {st['curate_noops']} | fails r{st['reflect_fails']} c{st['curate_fails']} "
              f"| playbook {ace.size_tokens()} tok")
    print(f"\n[{label}] final playbook:\n{ace.playbook}\n")
    return ace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "live"])
    ap.add_argument("--backbone", choices=["sonnet5", "qwen27b"], default="sonnet5")
    a = ap.parse_args()

    if a.mode == "offline":
        ace = run(FakeClient(), Path("runs/smoke-ace/offline.ace.json"), "offline")
        assert ace.n_ops == 6 and ace.curate_fails == 0 and ace.reflect_fails == 0
        assert ace.client.n == 6, "2 memory calls per episode expected"
        # resume: a fresh instance loads the same playbook
        again = ACEMemory(FakeClient(), state_path=Path("runs/smoke-ace/offline.ace.json"))
        assert again.load() and again.playbook == ace.playbook and again.position == 3
        print("offline OK: contract, parsing, counters, resume")
    else:
        from client import LLMClient
        client = LLMClient(a.backbone, run_id=f"smoke-ace-{a.backbone}",
                           log_path=f"runs/smoke-ace/{a.backbone}.calls.jsonl")
        ace = run(client, Path(f"runs/smoke-ace/{a.backbone}.ace.json"), a.backbone)
        calls = [json.loads(l) for l in open(f"runs/smoke-ace/{a.backbone}.calls.jsonl")]
        mem = [c for c in calls if c["tag"] == "memory"]
        print(f"[{a.backbone}] memory calls: {len(mem)} | in/out per call: "
              f"{sum(c['in'] for c in mem)//max(len(mem),1)} / {sum(c['out'] for c in mem)//max(len(mem),1)} "
              f"| truncated: {sum(bool(c.get('truncated')) for c in mem)} | thinking flags: "
              f"{sorted({c.get('thinking') for c in mem})}")
        st = json.loads(Path(f"runs/smoke-ace/{a.backbone}.ace.json").read_text())
        print(f"[{a.backbone}] n_ops {st['n_ops']} · curate_noops {st['curate_noops']} · "
              f"reflect_fails {st['reflect_fails']} · curate_fails {st['curate_fails']}")


if __name__ == "__main__":
    main()
