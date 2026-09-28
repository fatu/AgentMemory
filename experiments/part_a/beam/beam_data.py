"""BEAM data access + the OFFICIAL context serialization and answer prompts.

Ported from Mohammadta/BEAM @ b2da22e (src/answer_probing_questions/long_term_memory_methods.py,
src/prompts.py):
  - long-context ("Vanilla"): the whole chat as REAL turns [{role, content}, ...] in file
    order (batches → turns → messages; `time_anchor` etc. dropped), then one user message
    NOTE_TEMPLATE with the question (lines 555-592). When the chat does not fit,
    `prune_from_tail` keeps the MOST RECENT messages that fit (lines 245-265).
  - RAG documents: `pair_chunk` — one user message + its reply as one document, text
    "USER: … \\n\\n ASSISTANT: …" (lines 291-325; the official loop appends each pair once per
    message of its turn — a duplication we do not reproduce); RAG answer prompt =
    `answer_generation_for_rag` with <context>/<question> (line 11682), context = hits
    concatenated in rank order.
  - probing questions: chats/<group>/<conv>/probing_questions/probing_questions.json — a dict
    of 10 ability keys → 2 items with `question`, `rubric` (nugget list), `difficulty`, and a
    reference answer under an ability-specific key. Rubric and reference are evaluation-only.
Data: chats/100K (20 conversations) and chats/1M (35); CC BY-SA 4.0.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BEAM_DIR = Path(os.environ.get("BEAM_PATH", HERE.parents[3] / "code-repo" / "BEAM"))
if str(BEAM_DIR) not in sys.path:
    sys.path.insert(0, str(BEAM_DIR))
from src.prompts import answer_generation_for_rag, unified_llm_judge_base_prompt   # noqa: E402  (pure string constants)

# long_term_memory_methods.py:587-591 — the f-string's indentation and the trailing space
# after "explanations." are part of the prompt
NOTE_TEMPLATE = "\n            NOTE: Only provide the answer without any explanations. \n            Question: {query}"
ABILITIES = ["abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning",
             "preference_following", "summarization", "temporal_reasoning"]
GOLD_KEYS = ("answer", "ideal_response", "ideal_answer", "ideal_summary", "expected_compliance")
PAIR_TEMPLATE = """
                            USER: {u} \n\n
                            ASSISTANT: {a}
                            """                                              # create_chunking, pair_chunk branch


def conv_dir(group: str, conv: str | int) -> Path:
    return BEAM_DIR / "chats" / group / str(conv)


def load_chat(group: str, conv: str | int) -> list[dict]:
    return json.load(open(conv_dir(group, conv) / "chat.json"))


def load_questions(group: str, conv: str | int) -> list[dict]:
    """Flattened: [{ability, idx, question, rubric, gold, difficulty}] in ability order."""
    q = json.load(open(conv_dir(group, conv) / "probing_questions" / "probing_questions.json"))
    out = []
    for ab in ABILITIES:
        for i, item in enumerate(q.get(ab, [])):
            out.append({"ability": ab, "idx": i, "question": item["question"], "rubric": list(item["rubric"]),
                        "gold": next((str(item[k]) for k in GOLD_KEYS if k in item), ""),
                        "difficulty": item.get("difficulty", "")})
    return out


# ------------------------------------------------------------------ serializations
def messages(chat: list[dict]) -> list[dict]:
    """Official long-context turns: role + content only, file order."""
    return [{"role": m["role"], "content": m["content"]}
            for batch in chat for turn in batch["turns"] for m in turn]


def turns(chat: list[dict]) -> list[dict]:
    """One entry per turn (a user message with its reply and any follow-ups): the unit the
    LLM-ingesting systems add at a time. `text` = the official turn_chunk format."""
    out = []
    for bi, batch in enumerate(chat):
        for ti, turn in enumerate(batch["turns"]):
            msgs = [{"role": m["role"], "content": m["content"]} for m in turn]
            text = "".join(f"{m['role'].upper()}: {m['content']} \n\n" for m in msgs)
            out.append({"batch": bi + 1, "turn": ti + 1, "messages": msgs, "text": text,
                        "time_anchor": batch.get("time_anchor") or next((m.get("time_anchor") for m in turn if m.get("time_anchor")), None)})
    return out


def pairs(chat: list[dict]) -> list[dict]:
    """Official pair_chunk documents (each pair once)."""
    docs = []
    for bi, batch in enumerate(chat):
        for ti, turn in enumerate(batch["turns"]):
            for pi in range(0, len(turn), 2):
                u = turn[pi]["content"]
                a = turn[pi + 1]["content"] if pi + 1 < len(turn) else "N/A"
                docs.append({"batch": bi + 1, "turn": ti + 1, "pair": pi // 2 + 1,
                             "text": PAIR_TEMPLATE.format(u=u, a=a), "search": f"{u} {a}"})
    return docs


def history_plain(chat: list[dict]) -> str:
    """`role: content\\n` per message — the audit's token-count serialization."""
    return "".join(f"{m['role']}: {m['content']}\n" for m in messages(chat))


def note_prompt(question: str) -> str:
    return NOTE_TEMPLATE.format(query=question)


def rag_prompt(context: str, question: str) -> str:
    return answer_generation_for_rag.replace("<context>", context).replace("<question>", question)
