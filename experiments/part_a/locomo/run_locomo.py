"""LoCoMo pilot (step 8): official data + official prompts + official scorer, our execution layer.

    python run_locomo.py --conv conv-26 --system full --backbone sonnet5
    python run_locomo.py --conv conv-26 --system bm25 --backbone qwen27b --limit 20
    python run_locomo.py --conv conv-26 --system mem0 --backbone sonnet5

Per question: context = system.context_for(question) → official prompt → backbone
(no thinking, temperature 0, max_tokens 64) → prediction → (1) official scorer
(task_eval/evaluation.py: F1 / multi-hop F1 / cat-5 binary) and (2) Sonnet judge on
Mem0's rubric (cat 1-4). Rows carry tokens per phase; ingestion calls are in calls.jsonl
under row_id "ingest:*". Resumable per question.

Outputs runs/<conv>_<system>_<backbone>/{results.csv, calls.jsonl, provenance.json,
predictions.json}; predictions.json is the conv record with `<key>_prediction` filled,
i.e. exactly what the official evaluate_qa.py would write.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM_RUNNER = HERE.parents[1] / "stream_runner"
LOCOMO_DIR = Path(os.environ.get("LOCOMO_PATH", HERE.parents[2] / "code-repo" / "locomo"))
sys.path.insert(0, str(STREAM_RUNNER))
sys.path.insert(0, str(LOCOMO_DIR / "task_eval"))

from client import LLMClient                                   # noqa: E402
from embed import Embedder, EMBED_MODEL                        # noqa: E402
from memory.base import CAP, count_tokens                      # noqa: E402
from locomo_data import (DATA, CATEGORY_NAMES, NOT_MENTIONED, load_conversation,  # noqa: E402
                         build_question, gold)
from systems import make_system                                # noqa: E402
from judge import judge as judge_fn                            # noqa: E402
# task_eval/evaluation.py imports bert_score (a ~1 GB model) and nltk at module load for
# metrics we never call; stub them so the official F1 code runs unmodified.
import types                                                   # noqa: E402
sys.modules.setdefault("bert_score", types.ModuleType("bert_score"))
sys.modules["bert_score"].score = lambda *a, **k: None         # never called (we don't use bert_score)
try:
    import nltk.stem                                           # the official F1 STEMS tokens — keep it real
except ImportError:                                            # pip install nltk on the server
    raise SystemExit("pip install nltk  (the official LoCoMo F1 uses nltk's PorterStemmer)")
import evaluation as official                                  # noqa: E402  (task_eval/evaluation.py, unmodified)

JUDGE_BACKBONE = "sonnet5"     # one fixed judge for every row (decision 2026-09-14)
FIELDS = ["part", "benchmark", "conv", "qa_idx", "category", "category_name", "system", "backbone",
          "pred", "gold", "score_official", "score_judge", "judge_label",
          "tokens_agent_in", "tokens_agent_cache_read", "tokens_agent_cache_write", "tokens_agent_out",
          "tokens_judge_in", "tokens_judge_out", "block_tokens", "latency_s", "cat5_not_mentioned_letter"]


def official_score(category: int, pred: str, gold_text: str) -> float:
    """The official per-question rule (evaluation.py eval_question_answering), applied to one row."""
    if category in (2, 3, 4):
        g = gold_text.split(";")[0].strip() if category == 3 else gold_text
        return float(official.f1_score(pred, g))
    if category == 1:
        return float(official.f1(pred, gold_text))
    if category == 5:
        return 1.0 if ("no information available" in pred.lower() or "not mentioned" in pred.lower()) else 0.0
    raise ValueError(category)


def _sum(calls, tag, key):
    return sum(c.get(key, 0) for c in calls if c["tag"] == tag)


def _ingest_calls_from_log(path: Path) -> list[dict]:
    """All `ingest:*` memory calls ever logged for this run (calls.jsonl is append-only)."""
    if not path.exists():
        return []
    out = []
    for line in path.open():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if str(c.get("row_id", "")).startswith("ingest:") and c.get("tag") == "memory":
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--system", choices=["full", "bm25", "mem0", "amem", "hindsight", "hindsight-reflect"], required=True)
    ap.add_argument("--backbone", choices=["sonnet5", "qwen27b"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    run_id = f"{a.conv}_{a.system}_{a.backbone}"
    out = HERE / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    conv = load_conversation(a.conv)
    qas = conv["qa"][: a.limit] if a.limit else conv["qa"]

    client = LLMClient(a.backbone, run_id=run_id, log_path=out / "calls.jsonl")
    judge_client = client if a.backbone == JUDGE_BACKBONE else LLMClient(JUDGE_BACKBONE, run_id=run_id, log_path=out / "calls.jsonl")
    embedder = Embedder() if a.system in ("mem0", "amem") else None
    system = make_system(a.system, client=client, embedder=embedder, state_dir=out / "mem0",
                         bank_id=f"locomo-{a.conv}-{a.backbone}")

    t_ing = time.perf_counter()
    system.ingest(conv)
    ingest_s = round(time.perf_counter() - t_ing, 1)
    ingest_calls = [c for c in client.calls if str(c["row_id"]).startswith("ingest:")]
    if not ingest_calls:                       # resumed run: the store already existed — take the
        ingest_calls = _ingest_calls_from_log(out / "calls.jsonl")   # original ingestion from the log
        ingest_s = round(sum(c.get("latency_s", 0) for c in ingest_calls), 1) or ingest_s

    (out / "provenance.json").write_text(json.dumps({
        "benchmark": "locomo", "data": str(DATA), "data_sha256_prefix": "79fa87e9", "scorer_commit": "3eb6f2c",
        "conv": a.conv, "n_sessions": len([k for k in conv["conversation"] if k.startswith("session_") and "date" not in k]),
        "system": a.system, "backbone": a.backbone, "model": client.model, "judge": JUDGE_BACKBONE,
        "thinking": False, "temperature": 0.0, "max_tokens": 64, "cap_tokens": CAP,
        "embedder": (embedder.model_name if embedder else None),
        "history_tokens_o200k": count_tokens(system.context_for("") if a.system == "full" else ""),
        "ingest_seconds": ingest_s, "ingest_tokens_in": _sum(ingest_calls, "memory", "in") + _sum(ingest_calls, "memory", "cache_read") + _sum(ingest_calls, "memory", "cache_write"),
        "ingest_tokens_out": _sum(ingest_calls, "memory", "out"), "ingest_calls": len(ingest_calls),
        "hindsight": (system.store.info() if hasattr(system, "store") else None),
        "ts": time.time(),
    }, indent=2))
    print(f"ingested {a.conv} with {a.system}: {ingest_s}s, {len(ingest_calls)} memory calls")

    results_path = out / "results.csv"
    done = set()
    if results_path.exists():
        with results_path.open() as f:
            done = {int(r["qa_idx"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} rows done")

    pred_key = f"{a.system}_{a.backbone}_prediction"
    with results_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for idx, qa in enumerate(qas):
            if idx in done:
                continue
            row_id = f"{idx}"
            client.current_row_id = row_id
            t0 = time.perf_counter()
            qtext, template, c5 = build_question(qa, qa_key=f"{a.conv}:{idx}")
            if getattr(system, "kind", "context") == "reader":          # hindsight-reflect: its own reader answers
                block_tokens = 0
                pred = (system.answer(qtext, row_id) or "").strip()
            else:
                context = system.context_for(qa["question"])
                block_tokens = count_tokens(context)
                if a.system != "full":
                    assert block_tokens <= CAP, (idx, block_tokens)
                user = template.format(qtext).strip()
                pred = (client.complete(context, user, tag="agent", row_id=row_id, max_tokens=64,
                                        temperature=0.0, thinking=False,
                                        cache_system=getattr(system, "uses_cache", False)) or "").strip()
            g = gold(qa)
            s_off = official_score(qa["category"], pred, g)
            s_judge, label = None, ""
            if not a.no_judge and qa["category"] in (1, 2, 3, 4):
                ok, raw = judge_fn(judge_client, category=qa["category"], question=qa["question"],
                                   gold=g, response=pred, row_id=row_id)
                s_judge, label = (None if ok is None else float(ok)), ("CORRECT" if ok else "WRONG" if ok is not None else "NONE")
            calls = [c for c in client.calls + (judge_client.calls if judge_client is not client else []) if c["row_id"] == row_id]
            row = {
                "part": "A", "benchmark": "locomo", "conv": a.conv, "qa_idx": idx, "category": qa["category"],
                "category_name": CATEGORY_NAMES[qa["category"]], "system": a.system, "backbone": a.backbone,
                "pred": pred, "gold": g, "score_official": round(s_off, 3), "score_judge": s_judge, "judge_label": label,
                "tokens_agent_in": _sum(calls, "agent", "in") + _sum(calls, "agent", "cache_read") + _sum(calls, "agent", "cache_write"),
                "tokens_agent_cache_read": _sum(calls, "agent", "cache_read"), "tokens_agent_cache_write": _sum(calls, "agent", "cache_write"),
                "tokens_agent_out": _sum(calls, "agent", "out"),
                "tokens_judge_in": _sum(calls, "judge", "in") + _sum(calls, "judge", "cache_read") + _sum(calls, "judge", "cache_write"),
                "tokens_judge_out": _sum(calls, "judge", "out"),
                "block_tokens": block_tokens, "latency_s": round(time.perf_counter() - t0, 3),
                "cat5_not_mentioned_letter": c5.get("not_mentioned_letter", ""),
            }
            w.writerow(row); f.flush()
            print(f"[{idx+1}/{len(qas)}] cat{qa['category']} off={s_off:.2f} judge={label:7s} "
                  f"pred={pred[:40]!r} gold={str(g)[:30]!r} in={row['tokens_agent_in']} blk={block_tokens}")

    # predictions.json in the official layout (what evaluate_qa.py would write)
    rows = list(csv.DictReader(results_path.open()))
    conv_out = json.loads(json.dumps(conv))
    for r in rows:
        conv_out["qa"][int(r["qa_idx"])][pred_key] = r["pred"]
    (out / "predictions.json").write_text(json.dumps([conv_out], indent=1))
    summarize(rows, run_id)


def summarize(rows, run_id):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["category"])].append(r)
    print(f"\n== {run_id} ==")
    print(f"{'cat':>4} {'name':>12} {'n':>4} {'official':>9} {'judge':>7} {'gap':>6}")
    tot_off, tot_n = 0.0, 0
    for cat in (4, 1, 2, 3, 5):
        rs = by.get(cat, [])
        if not rs:
            continue
        off = sum(float(r["score_official"]) for r in rs) / len(rs)
        jv = [float(r["score_judge"]) for r in rs if r["score_judge"] not in ("", "None")]
        j = sum(jv) / len(jv) if jv else float("nan")
        tot_off += sum(float(r["score_official"]) for r in rs); tot_n += len(rs)
        print(f"{cat:>4} {CATEGORY_NAMES[cat]:>12} {len(rs):>4} {off:>9.3f} {j:>7.3f} {j-off:>+6.3f}")
    print(f"{'all':>4} {'micro':>12} {tot_n:>4} {tot_off/max(tot_n,1):>9.3f}")
    ag_in = sum(int(r["tokens_agent_in"]) for r in rows) / max(len(rows), 1)
    cr = sum(int(r["tokens_agent_cache_read"]) for r in rows) / max(len(rows), 1)
    jd = sum(int(r["tokens_judge_in"]) for r in rows) / max(len(rows), 1)
    print(f"tokens/question: agent_in {ag_in:.0f} (cache_read {cr:.0f}) · judge_in {jd:.0f} · "
          f"latency {sum(float(r['latency_s']) for r in rows)/max(len(rows),1):.1f}s")


if __name__ == "__main__":
    main()
