"""LoCoMo data access + the OFFICIAL context serialization and answer prompts.

Everything here is ported verbatim from snap-research/locomo @ 3eb6f2c (task_eval/gpt_utils.py):
  - history text: per session  "DATE: <date_time>\\nCONVERSATION:\\n" + one line per turn
    '<speaker> said, "<text>"' (+ ' and shared <blip_caption>' when an image is present)
  - QA_PROMPT (cat 1-4), QA_PROMPT_CAT_5, the cat-2 temporal add-on, and the cat-5
    two-option construction (randomised order; we seed it per question so it is fixed).
Data: code-repo/locomo/data/locomo10.json (CC BY-NC 4.0 — personal study only).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve()
LOCOMO_DIR = Path(os.environ.get("LOCOMO_PATH", HERE.parents[3] / "code-repo" / "locomo"))
DATA = LOCOMO_DIR / "data" / "locomo10.json"

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""
TEMPORAL_SUFFIX = " Use DATE of CONVERSATION to answer with an approximate date."
NOT_MENTIONED = "Not mentioned in the conversation"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load_conversation(sample_id: str) -> dict:
    for conv in json.load(open(DATA)):
        if conv["sample_id"] == sample_id:
            return conv
    raise KeyError(sample_id)


def session_ids(conv: dict) -> list[int]:
    c = conv["conversation"]
    return sorted(int(k.split("_")[-1]) for k in c if k.startswith("session_") and "date_time" not in k)


def turn_line(d: dict) -> str:
    line = f'{d["speaker"]} said, "{d["text"]}"'
    if d.get("blip_caption"):
        line += f' and shared {d["blip_caption"]}'
    return line


def session_text(conv: dict, i: int) -> str:
    c = conv["conversation"]
    turns = "\n".join(turn_line(d) for d in c[f"session_{i}"])
    return f"DATE: {c[f'session_{i}_date_time']}\nCONVERSATION:\n{turns}"


def history_text(conv: dict) -> str:
    return "\n\n".join(session_text(conv, i) for i in session_ids(conv))


def turns_as_documents(conv: dict) -> list[dict]:
    """Dialog-level documents for retrieval: one turn each, with the session date and dia_id."""
    docs = []
    for i in session_ids(conv):
        date = conv["conversation"][f"session_{i}_date_time"]
        for d in conv["conversation"][f"session_{i}"]:
            docs.append({"dia_id": d["dia_id"], "session": i, "date": date, "text": f"{date}: {turn_line(d)}"})
    return docs


def build_question(qa: dict, qa_key: str) -> tuple[str, str, dict]:
    """Return (question_text, prompt_template, cat5_info). Cat 5 follows the official
    two-option construction with a per-question fixed random order."""
    cat = qa["category"]
    if cat == 2:
        return qa["question"] + TEMPORAL_SUFFIX, QA_PROMPT, {}
    if cat == 5:
        adv = qa.get("adversarial_answer") or qa.get("answer") or ""
        rng = random.Random(int(hashlib.sha256(qa_key.encode()).hexdigest(), 16))
        q = qa["question"] + " Select the correct answer: (a) {} (b) {}. "
        if rng.random() < 0.5:
            q = q.format(NOT_MENTIONED, adv); info = {"not_mentioned_letter": "a"}
        else:
            q = q.format(adv, NOT_MENTIONED); info = {"not_mentioned_letter": "b"}
        return q, QA_PROMPT_CAT_5, info
    return qa["question"], QA_PROMPT, {}


def gold(qa: dict) -> str:
    """Gold for scoring: cat 5 has no gold answer (the official scorer checks the abstention phrase)."""
    return qa.get("answer") if qa["category"] != 5 else NOT_MENTIONED
