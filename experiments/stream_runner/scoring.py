"""Prompt construction + answer parsing + exact-match scoring for the streams."""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a careful expert solving one question at a time. "
    "If a '## Memory' section is present, it contains notes from your earlier work on "
    "similar questions — use it when relevant, ignore it when not. "
    "Work through the question, then end your reply with exactly one final line of the form\n"
    "Answer: X\n"
    "where X is the option letter for multiple-choice questions or the integer for math "
    "problems. Nothing may follow that line."
)

_ANS_RE = re.compile(r"Answer:\s*\**\(?\s*([A-J]|-?\d+)\s*\)?\**\s*$", re.I | re.M)


def build_user(memory_block: str, input_text: str) -> str:
    q = f"## Question\n{input_text}"
    return f"## Memory\n{memory_block}\n\n{q}" if memory_block else q


def parse_answer(text: str, kind: str) -> str | None:
    """kind: 'letter' (MMLU-Pro, GPQA) or 'integer' (AIME). Uses the LAST 'Answer:' line."""
    hits = _ANS_RE.findall(text or "")
    if not hits:
        return None
    a = hits[-1].strip()
    if kind == "letter":
        return a.upper() if a.isalpha() else None
    try:
        return str(int(a))
    except ValueError:
        return None


def answer_kind(stream: str) -> str:
    return "integer" if stream.startswith("aime") else "letter"


def score(pred: str | None, target: str) -> bool:
    return pred is not None and pred == target
