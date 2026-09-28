"""Sonnet-5 judge running Mem0's LoCoMo rubric VERBATIM (memory-benchmarks
benchmarks/locomo/prompts.py, no-evidence variant, categories 1-4 only).

The template is imported from the vendored harness so it cannot drift from the published
rubric; `preprocess_answer` (cat-3 gold = text before the ';') is theirs too. The judge
call goes through our instrumented client with tag="judge".
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

MB = Path(os.environ.get("MEMORY_BENCHMARKS_PATH",
                         Path(__file__).resolve().parents[3] / "code-repo" / "memory-benchmarks"))
sys.path.insert(0, str(MB))
from benchmarks.locomo.prompts import JUDGE_SYSTEM_PROMPT, get_judge_prompt, preprocess_answer  # noqa: E402

_LABEL_RE = re.compile(r'"label"\s*:\s*"?(CORRECT|WRONG)', re.I)


def judge(client, *, category: int, question: str, gold: str, response: str, row_id: str) -> tuple[bool | None, str]:
    """Returns (correct, raw). correct=None when the judge output has no label."""
    prompt = get_judge_prompt(category, question, preprocess_answer(category, gold), response)
    raw = client.complete(JUDGE_SYSTEM_PROMPT, prompt, tag="judge", row_id=row_id,
                          max_tokens=200, thinking=False) or ""
    m = _LABEL_RE.search(raw)
    if not m:
        try:
            m2 = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]).get("label", "")
            return (m2.upper() == "CORRECT") if m2 else None, raw
        except Exception:
            return None, raw
    return m.group(1).upper() == "CORRECT", raw
