"""Count LoCoMo full-history tokens without calling an inference API.

Usage: python3 count_history_tokens.py [path/to/locomo10.json]
Dependency: tiktoken==0.13.0
The zero-based qa_index_0based indexes the source sample's qa array.
These are explicitly defined counts, not official reported benchmark counts
or measured API prompt-token usage. No truncation or retrieval is applied.
"""

import hashlib
import json
from pathlib import Path
import statistics
import sys
import urllib.request

import tiktoken

COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
SOURCE = f"https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}/data/locomo10.json"


def render_history(conversation):
    sessions = sorted(
        (key for key in conversation
         if key.startswith("session_") and key.removeprefix("session_").isdigit()),
        key=lambda key: int(key.split("_")[1]),
    )
    history_parts = []
    text_parts = []
    turns = 0
    for session in sessions:
        history_parts.append(
            "DATE: " + conversation.get(session + "_date_time", "")
            + "\nCONVERSATION:\n"
        )
        for turn in conversation[session]:
            turns += 1
            text_parts.append(turn["text"])
            history_parts.append(turn["speaker"] + ": " + turn["text"] + "\n")
            if turn.get("blip_caption"):
                history_parts.append("[Image caption: " + turn["blip_caption"] + "]\n")
        history_parts.append("\n")
    return "".join(history_parts), "\n".join(text_parts), len(sessions), turns


def main():
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_bytes()
    else:
        with urllib.request.urlopen(SOURCE) as response:
            raw = response.read()
    samples = json.loads(raw)
    encoder = tiktoken.get_encoding("cl100k_base")
    output_dir = Path(__file__).resolve().parent
    rows = []
    conversations = []
    for sample in samples:
        history, text_only, sessions, turns = render_history(sample["conversation"])
        history_tokens = len(encoder.encode(history))
        text_tokens = len(encoder.encode(text_only))
        conversations.append({
            "sample_id": sample["sample_id"],
            "sessions": sessions,
            "turns": turns,
            "questions": len(sample["qa"]),
            "dialogue_text_tokens": text_tokens,
            "history_tokens": history_tokens,
        })
        for index, question in enumerate(sample["qa"]):
            rows.append({
                "sample_id": sample["sample_id"],
                "qa_index_0based": index,
                "category": question["category"],
                "dialogue_text_tokens": text_tokens,
                "history_tokens": history_tokens,
            })

    assert len({row["sample_id"] for row in conversations}) == len(samples)
    assert len({(row["sample_id"], row["qa_index_0based"]) for row in rows}) == len(rows)
    if len(samples) == 10:
        assert len(rows) == 1986, "Source question count differs from audited release"

    manifest = {
        "source_url": SOURCE,
        "source_commit": COMMIT,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "tiktoken_version": tiktoken.__version__,
        "encoding": "cl100k_base",
        "count_type": "custom full-history text serialization; not API usage",
        "history_tokens_includes": ["chronological session dates", "speaker names", "dialogue text", "nonempty BLIP image captions", "format labels and whitespace"],
        "history_tokens_excludes": ["QA questions and answers", "system and task instructions", "chat protocol overhead", "evidence IDs", "JSON structure", "image URLs and image token charges", "observations", "session summaries", "event summaries"],
        "dialogue_text_tokens_format": "all turn.text values in chronological order, joined with one newline",
        "history_tokens_format": "For each chronological session append 'DATE: {date}\\nCONVERSATION:\\n'; for each turn append '{speaker}: {text}\\n' and, if a nonempty blip_caption exists, '[Image caption: {caption}]\\n'; after each session append '\\n'.",
        "question_locator": "sample_id + qa_index_0based (zero-based position in that sample's qa array)",
        "category_ids": {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop", "5": "adversarial"},
        "conversation_count": len(conversations),
        "question_count": len(rows),
        "conversation_mean_history_tokens": statistics.mean(c["history_tokens"] for c in conversations),
        "question_weighted_mean_history_tokens": statistics.mean(r["history_tokens"] for r in rows),
        "conversation_mean_dialogue_text_tokens": statistics.mean(c["dialogue_text_tokens"] for c in conversations),
        "question_weighted_mean_dialogue_text_tokens": statistics.mean(r["dialogue_text_tokens"] for r in rows),
        "min_history_tokens": min(c["history_tokens"] for c in conversations),
        "max_history_tokens": max(c["history_tokens"] for c in conversations),
        "conversations": conversations,
    }
    (output_dir / "question_history_tokens.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "token_count_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
