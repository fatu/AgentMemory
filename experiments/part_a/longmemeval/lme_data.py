"""LongMemEval-S data access + the OFFICIAL history serialization and answer prompt.

Ported verbatim from xiaowu0162/LongMemEval @ 9e0b455, src/generation/run_generation.py
(`prepare_prompt`, the `run_generation.sh` defaults: history_format=json, useronly=false,
cot=false, no key expansion):
  - every session becomes  "\\n### Session {i}:\\nSession Date: {date}\\nSession Content:\\n{json}\\n"
    where {json} = json.dumps(list of {role, content}) — `has_answer` stripped first
    (evaluation-only field; the official code pops it before dumping, lines 176-191);
  - sessions are sorted by date before numbering (line 227);
  - answer prompt = the non-CoT template (line 57), `question_date` as "Current Date";
  - generation: temperature 0, max_tokens 500 (gen_length, line 342).
Data: code-repo/LongMemEval/data/longmemeval_s_cleaned.json (277 MB, one element per
question with its OWN haystack; sha256 d6f21ea9…). `answer`, `answer_session_ids` and
`has_answer` never reach the system under test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve()
LME_DIR = Path(os.environ.get("LONGMEMEVAL_PATH", HERE.parents[3] / "code-repo" / "LongMemEval"))
DATA = Path(os.environ.get("LONGMEMEVAL_DATA", LME_DIR / "data" / "longmemeval_s_cleaned.json"))
SMOKE_IDS = HERE.parent / "smoke_ids.json"

# run_generation.py:57 — non-CoT, no key expansion. The official code sends this as ONE user
# message; we split it at the first blank line into system (the instruction sentence) and
# user (history / date / question) so no backend sees an empty system block. Same text.
ANSWER_SYSTEM = ("I will give you several history chats between you and a user. "
                 "Please answer the question based on the relevant chat history.")
ANSWER_USER = "History Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
SESSION_WRAPPER = "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n"   # line 253
GEN_MAX_TOKENS = 500                                                              # line 342

TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "temporal-reasoning", "knowledge-update"]


def is_abs(entry_or_id) -> bool:
    qid = entry_or_id if isinstance(entry_or_id, str) else entry_or_id["question_id"]
    return qid.endswith("_abs")


def load_all() -> list[dict]:
    return json.load(open(DATA))


def load_entries(ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Entries in file order, filtered to `ids` (keeping the file order) or the first `limit`."""
    data = load_all()
    if ids is not None:
        want = set(ids)
        data = [e for e in data if e["question_id"] in want]
        missing = want - {e["question_id"] for e in data}
        if missing:
            raise KeyError(f"question ids not in {DATA.name}: {sorted(missing)}")
    if limit:
        data = data[:limit]
    return data


def smoke_ids(data: list[dict] | None = None, refresh: bool = False) -> list[str]:
    """The pinned 7-question smoke set: first of each question_type in file order + the
    first `_abs` question. Cached in smoke_ids.json so every system/backbone sees the same 7."""
    if SMOKE_IDS.exists() and not refresh:
        return json.load(open(SMOKE_IDS))["ids"]
    data = data or load_all()
    first: dict[str, str] = {}
    for e in data:
        key = "_abs" if is_abs(e) else e["question_type"]
        first.setdefault(key, e["question_id"])
    ids = [first[t] for t in TYPES] + [first["_abs"]]
    SMOKE_IDS.write_text(json.dumps({"ids": ids, "by_type": first, "source": DATA.name}, indent=2))
    return ids


# ---------------------------------------------------------------- serialization (official)
def clean_session(session: list[dict]) -> list[dict]:
    """The session as the model may see it: role + content only (has_answer popped, as the
    official clean-up does)."""
    return [{"role": t["role"], "content": t["content"]} for t in session]


def sessions(entry: dict) -> list[tuple[str, str, list[dict]]]:
    """(date, session_id, cleaned turns) in FILE order (index-aligned fields)."""
    return [(d, sid, clean_session(s)) for d, sid, s in
            zip(entry["haystack_dates"], entry["haystack_session_ids"], entry["haystack_sessions"])]


def wrap(i: int, date: str, chunk) -> str:
    """One `### Session` block; `chunk` is a list of turns (a session or a round)."""
    return SESSION_WRAPPER.format(i, date, json.dumps(chunk))


def history_string(chunks: list[tuple[str, list[dict]]]) -> str:
    """Official assembly: sort chunks by date (stable), then number and wrap them."""
    ordered = sorted(chunks, key=lambda x: x[0])
    return "".join(wrap(i + 1, date, chunk) for i, (date, chunk) in enumerate(ordered))


def full_history(entry: dict) -> str:
    """`orig-session` retriever: every session, whole."""
    return history_string([(d, turns) for d, _, turns in sessions(entry)])


def rounds(entry: dict) -> list[dict]:
    """Round-level documents for retrieval (the official flat-turn retriever expands each hit
    to [turn, next turn]; we index those rounds directly): one dict per user turn + reply."""
    docs = []
    for si, (date, sid, turns) in enumerate(sessions(entry)):
        i = 0
        while i < len(turns):
            chunk = turns[i:i + 2] if turns[i]["role"] == "user" else turns[i:i + 1]
            docs.append({"session": si, "session_id": sid, "date": date, "turn": i, "chunk": chunk,
                         "text": " ".join(t["content"] for t in chunk)})
            i += len(chunk)
    return docs


def session_plain(date: str, turns: list[dict]) -> str:
    """Plain-text session for LLM-ingesting systems (one line per turn)."""
    return f"Session date: {date}\n" + "\n".join(f"{t['role']}: {t['content'].strip()}" for t in turns)


def gold(entry: dict) -> str:
    return entry["answer"]
