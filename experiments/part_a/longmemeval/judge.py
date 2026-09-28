"""Sonnet-5 judge running LongMemEval's official answer-check templates VERBATIM
(`get_anscheck_prompt` imported from the vendored src/evaluation/evaluate_qa.py @ 9e0b455:
five task templates + the abstention template; the official judge is gpt-4o-2024-08-06,
temperature 0, max_tokens 10, label = 'yes' in response.lower()).

Two-number abstention rule (ours): `_abs` items are judged with the abstention template
(abstention accuracy); answerable items the judge marks wrong get ONE more call with the
abstention template to tell a refusal from a wrong answer (false-abstention rate).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LME = Path(os.environ.get("LONGMEMEVAL_PATH", Path(__file__).resolve().parents[3] / "code-repo" / "LongMemEval"))
sys.path.insert(0, str(LME / "src" / "evaluation"))
try:
    import backoff  # noqa: F401  evaluate_qa.py decorates ITS OWN gpt-4o call with it; we never call that
except ImportError:                                  # stub the decorator so the templates import anywhere
    import types
    _b = types.ModuleType("backoff")
    _b.on_exception = lambda *a, **k: (lambda fn: fn)
    _b.expo = None
    sys.modules["backoff"] = _b
from evaluate_qa import get_anscheck_prompt   # noqa: E402  (also imports openai, tqdm, numpy at module level)

# The official call is ONE user message with no system prompt (evaluate_qa.py:103-110). Our
# client always sends a system block, so it repeats the template's own last line and adds no rule.
JUDGE_SYSTEM = "Answer yes or no only."
JUDGE_MAX_TOKENS = 10                                                 # evaluate_qa.py


def _yes(client, prompt: str, row_id: str) -> tuple[bool, str]:
    raw = client.complete(JUDGE_SYSTEM, prompt, tag="judge", row_id=row_id,
                          max_tokens=JUDGE_MAX_TOKENS, temperature=0.0, thinking=False) or ""
    return ("yes" in raw.lower()), raw


def judge(client, *, question_type: str, is_abs: bool, question: str, gold: str, response: str,
          row_id: str) -> dict:
    """Returns {correct, raw, false_abstention, raw_abs}. For `_abs` items `correct` means
    the model abstained; `false_abstention` is only computed for wrong answerable items."""
    prompt = get_anscheck_prompt(question_type, question, gold, response, abstention=is_abs)
    ok, raw = _yes(client, prompt, row_id)
    out = {"correct": ok, "raw": raw, "false_abstention": None, "raw_abs": ""}
    if not is_abs and not ok:
        p2 = get_anscheck_prompt(question_type, question, gold, response, abstention=True)
        fa, raw2 = _yes(client, p2, row_id)
        out.update(false_abstention=fa, raw_abs=raw2)
    return out
