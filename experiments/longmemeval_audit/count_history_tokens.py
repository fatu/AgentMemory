#!/usr/bin/env python3
"""Per-question history token counts for LongMemEval (cleaned release).

Usage: python3 count_history_tokens.py longmemeval_s_cleaned.json [out.jsonl]

Counts two serializations with tiktoken cl100k_base (a proxy — Claude 4.7+ models
tokenize ~1.3x heavier; Llama-3 tokenization is what the README's "~115k" refers to):
  content_tokens : all message `content` joined by newlines
  history_tokens : "[date]\\nrole: content" per session, sessions joined by newlines
Excludes the question, prompts, and API overhead. `disallowed_special=()` because one
message in the S release contains the literal string "<|endoftext|>".
"""
import json, statistics, sys
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
E = lambda s: len(enc.encode(s, disallowed_special=()))

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
data = json.load(open(src))
rows = []
for q in data:
    content = "\n".join(m["content"] for s in q["haystack_sessions"] for m in s)
    full = "\n".join(
        f"[{dt}]\n" + "\n".join(f"{m['role']}: {m['content']}" for m in s)
        for dt, s in zip(q["haystack_dates"], q["haystack_sessions"])
    )
    rows.append({
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "abstention": q["question_id"].endswith("_abs"),
        "n_sessions": len(q["haystack_sessions"]),
        "content_tokens": E(content),
        "history_tokens": E(full),
    })

def stats(xs):
    return dict(min=min(xs), median=int(statistics.median(xs)), mean=int(statistics.mean(xs)), max=max(xs), sum=sum(xs))

print("questions:", len(rows))
for k in ("n_sessions", "content_tokens", "history_tokens"):
    print(k, stats([r[k] for r in rows]))
if out:
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", out)
