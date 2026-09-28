"""LongMemEval-S pilot (step 9.1): official data + official prompt + official judge templates,
our execution layer — a fresh memory store per question.

    python run_lme.py --system full --backbone sonnet5 --smoke          # the pinned 7 questions
    python run_lme.py --system mem0 --backbone qwen27b --smoke
    python run_lme.py --system bm25 --backbone qwen27b --limit 50       # first 50 in file order
    python run_lme.py --system amem --backbone sonnet5 --ids my_ids.json

Per question: system.ingest(entry) (timed; LLM calls tagged memory under row_id
ingest:<qid>:<k>) → context_for(question) → official prompt → backbone (no thinking,
temperature 0, max_tokens 500) → Sonnet judge on the official template for the question
type (+ the abstention template on wrong answerable items → false_abstention).
Resumable per question. Outputs runs/<system>_<backbone>[_smoke]/{results.csv, calls.jsonl,
provenance.json, hypotheses.jsonl}; hypotheses.jsonl is {question_id, hypothesis} per line —
exactly what the official evaluate_qa.py consumes, should anyone want to re-judge with gpt-4o.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM_RUNNER = HERE.parents[1] / "stream_runner"
sys.path.insert(0, str(STREAM_RUNNER))

from client import LLMClient                                              # noqa: E402
from embed import Embedder                                                # noqa: E402
from memory.base import CAP, count_tokens                                 # noqa: E402
from lme_data import (DATA, ANSWER_SYSTEM, ANSWER_USER, GEN_MAX_TOKENS, TYPES, full_history, gold,  # noqa: E402
                      is_abs, load_entries, smoke_ids)
from systems import make_system                                           # noqa: E402
from judge import judge as judge_fn                                       # noqa: E402

JUDGE_BACKBONE = "sonnet5"     # one fixed judge for every row (decision 2026-09-14)
FIELDS = ["part", "benchmark", "question_id", "question_type", "is_abs", "system", "backbone",
          "pred", "gold", "score_judge", "judge_label", "false_abstention",
          "n_sessions", "history_tokens", "block_tokens", "truncated",
          "tokens_ingest_in", "tokens_ingest_out", "ingest_calls", "ingest_s",
          "tokens_agent_in", "tokens_agent_cache_read", "tokens_agent_out",
          "tokens_judge_in", "tokens_judge_out", "latency_s"]


def _sum(calls, tag, key):
    return sum(c.get(key, 0) for c in calls if c["tag"] == tag)


def _tot_in(calls, tag):
    return _sum(calls, tag, "in") + _sum(calls, tag, "cache_read") + _sum(calls, tag, "cache_write")


def max_context(client, backbone: str) -> int | None:
    """Context window the full-context system must fit: Sonnet from env (default 200K);
    Qwen from vLLM's /v1/models `max_model_len` (env QWEN_MAX_MODEL_LEN overrides)."""
    if backbone == "sonnet5":
        return int(os.environ.get("SONNET_MAX_CONTEXT", "200000"))
    if os.environ.get("QWEN_MAX_MODEL_LEN"):
        return int(os.environ["QWEN_MAX_MODEL_LEN"])
    # raw GET: vLLM puts max_model_len in the /v1/models JSON, which the OpenAI SDK's typed
    # Model object does not surface reliably
    try:
        import urllib.request
        base = os.environ["QWEN_BASE_URL"].rstrip("/")
        with urllib.request.urlopen(f"{base}/models", timeout=10) as r:
            models = json.load(r)["data"]
        for m in models:
            if m.get("id") == client.model and m.get("max_model_len"):
                return int(m["max_model_len"])
        for m in models:
            if m.get("max_model_len"):
                return int(m["max_model_len"])
    except Exception as e:
        print(f"WARNING: /v1/models read failed ({e!r})")
    print("WARNING: could not read max_model_len from vLLM; assuming 131072 (set QWEN_MAX_MODEL_LEN)")
    return 131072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["full", "bm25", "mem0", "amem"], required=True)
    ap.add_argument("--backbone", choices=["sonnet5", "qwen27b"], required=True)
    ap.add_argument("--smoke", action="store_true", help="the pinned 7-question set (smoke_ids.json)")
    ap.add_argument("--ids", default=None, help="JSON file with {'ids': [...]} or a list")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--rejudge", choices=["missing", "all"], default=None,
                    help="judge existing rows in place (no new answers): rows without a judge score, or every row")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    if a.rejudge:
        return rejudge(a)

    ids = None
    if a.smoke:
        ids = smoke_ids()
    elif a.ids:
        j = json.load(open(a.ids)); ids = j["ids"] if isinstance(j, dict) else j
    entries = load_entries(ids=ids, limit=a.limit)

    run_id = a.run_id or f"{a.system}_{a.backbone}" + ("_smoke" if a.smoke else "")
    out = HERE / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)

    client = LLMClient(a.backbone, run_id=run_id, log_path=out / "calls.jsonl")
    judge_client = client if a.backbone == JUDGE_BACKBONE else LLMClient(JUDGE_BACKBONE, run_id=run_id, log_path=out / "calls.jsonl")
    embedder = Embedder() if a.system in ("mem0", "amem") else None
    ctx_limit = max_context(client, a.backbone) if a.system == "full" else None
    system = make_system(a.system, client=client, embedder=embedder, state_root=out / a.system,
                         max_context=ctx_limit, backbone=a.backbone)

    (out / "provenance.json").write_text(json.dumps({
        "benchmark": "longmemeval_s", "data": str(DATA), "data_sha256_prefix": "d6f21ea9", "repo_commit": "9e0b455",
        "history_format": "json", "useronly": False, "cot": False, "gen_max_tokens": GEN_MAX_TOKENS,
        "system": a.system, "ingest_unit": system.ingest_unit, "backbone": a.backbone, "model": client.model,
        "judge": JUDGE_BACKBONE, "judge_max_tokens": 10, "judge_system_prompt": "Answer yes or no only.",
        "thinking": False, "temperature": 0.0, "cap_tokens": CAP,
        "max_context": ctx_limit, "truncation_budget_o200k": getattr(system, "budget", None),
        "embedder": (embedder.model_name if embedder else None),
        "embedder_backend": (embedder.backend if embedder else None),
        "question_ids": [e["question_id"] for e in entries], "n_questions": len(entries),
        "ts": time.time(),
    }, indent=2))

    results_path = out / "results.csv"
    done = set()
    if results_path.exists():
        with results_path.open() as f:
            done = {r["question_id"] for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} rows done")

    with results_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for n, e in enumerate(entries):
            qid = e["question_id"]
            if qid in done:
                continue
            t0 = time.perf_counter()
            # ---- ingest (fresh store per question)
            t_ing = time.perf_counter()
            system.ingest(e)
            ingest_s = round(time.perf_counter() - t_ing, 1)
            ing = [c for c in client.calls if str(c["row_id"]).startswith(f"ingest:{qid}:")]
            # ---- answer
            client.current_row_id = qid
            context = system.context_for(e["question"])
            block_tokens = count_tokens(context)
            if a.system != "full":
                assert block_tokens <= CAP, (qid, block_tokens)
            user = ANSWER_USER.format(context, e["question_date"], e["question"])
            pred = (client.complete(ANSWER_SYSTEM, user, tag="agent", row_id=qid, max_tokens=GEN_MAX_TOKENS,
                                    temperature=0.0, thinking=False) or "").strip()
            # ---- judge
            g = gold(e)
            s_judge, label, fa = None, "", ""
            if not a.no_judge:
                j = judge_fn(judge_client, question_type=e["question_type"], is_abs=is_abs(e),
                             question=e["question"], gold=g, response=pred, row_id=qid)
                s_judge, label = float(j["correct"]), ("CORRECT" if j["correct"] else "WRONG")
                fa = "" if j["false_abstention"] is None else int(j["false_abstention"])
            calls = [c for c in client.calls + (judge_client.calls if judge_client is not client else []) if c["row_id"] == qid]
            row = {
                "part": "A", "benchmark": "longmemeval_s", "question_id": qid, "question_type": e["question_type"],
                "is_abs": int(is_abs(e)), "system": a.system, "backbone": a.backbone,
                "pred": pred, "gold": g, "score_judge": s_judge, "judge_label": label, "false_abstention": fa,
                "n_sessions": len(e["haystack_sessions"]),
                "history_tokens": count_tokens(full_history(e)),        # o200k, official serialization, before any truncation
                "block_tokens": block_tokens, "truncated": int(getattr(system, "truncated", False)),
                "tokens_ingest_in": _tot_in(ing, "memory"), "tokens_ingest_out": _sum(ing, "memory", "out"),
                "ingest_calls": len([c for c in ing if c["tag"] == "memory"]), "ingest_s": ingest_s,
                "tokens_agent_in": _tot_in(calls, "agent"), "tokens_agent_cache_read": _sum(calls, "agent", "cache_read"),
                "tokens_agent_out": _sum(calls, "agent", "out"),
                "tokens_judge_in": _tot_in(calls, "judge"), "tokens_judge_out": _sum(calls, "judge", "out"),
                "latency_s": round(time.perf_counter() - t0, 1),
            }
            w.writerow(row); f.flush()
            print(f"[{n+1}/{len(entries)}] {e['question_type'][:22]:22s}{' abs' if is_abs(e) else '    '} "
                  f"judge={label:7s} fa={fa!s:1s} ingest={ingest_s}s/{row['ingest_calls']}c/{row['tokens_ingest_in']}t "
                  f"in={row['tokens_agent_in']} blk={block_tokens}{' TRUNC' if row['truncated'] else ''} "
                  f"pred={pred[:50]!r}")

    rows = list(csv.DictReader(results_path.open()))
    with (out / "hypotheses.jsonl").open("w") as hf:
        for r in rows:
            hf.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["pred"]}) + "\n")
    summarize(rows, run_id)


def rejudge(a):
    """Judge rows already in results.csv without re-answering (a run made with --no-judge, or
    a later re-judge). Rewrites the CSV in place; judge calls append to the same calls.jsonl."""
    run_id = a.run_id or f"{a.system}_{a.backbone}" + ("_smoke" if a.smoke else "")
    out = HERE / "runs" / run_id
    results_path = out / "results.csv"
    rows = list(csv.DictReader(results_path.open()))
    todo = [r for r in rows if a.rejudge == "all" or r["score_judge"] in ("", "None")]
    print(f"{run_id}: {len(rows)} rows, judging {len(todo)}")
    if not todo:
        summarize(rows, run_id); return
    by_id = {e["question_id"]: e for e in load_entries(ids=[r["question_id"] for r in todo])}
    jc = LLMClient(JUDGE_BACKBONE, run_id=run_id, log_path=out / "calls.jsonl")
    for r in todo:
        e = by_id[r["question_id"]]
        j = judge_fn(jc, question_type=e["question_type"], is_abs=is_abs(e), question=e["question"],
                     gold=gold(e), response=r["pred"], row_id=r["question_id"])
        calls = [c for c in jc.calls if c["row_id"] == r["question_id"]]
        r.update(score_judge=float(j["correct"]), judge_label=("CORRECT" if j["correct"] else "WRONG"),
                 false_abstention=("" if j["false_abstention"] is None else int(j["false_abstention"])),
                 tokens_judge_in=_tot_in(calls, "judge"), tokens_judge_out=_sum(calls, "judge", "out"))
        print(f"  {r['question_id']:>16} {r['question_type'][:22]:22s} judge={r['judge_label']:7s} fa={r['false_abstention']!s:1s} pred={r['pred'][:50]!r}")
    with results_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    summarize(rows, run_id)


def summarize(rows, run_id):
    """Our aggregator (print_qa_metrics.py asserts a gpt-4o judge): per-type accuracy with abs
    items inside their type (official), task-averaged over the 6 types, overall, abstention
    accuracy on `_abs`, false-abstention rate on answerable items."""
    def acc(rs):
        v = [float(r["score_judge"]) for r in rs if r["score_judge"] not in ("", "None")]
        return (sum(v) / len(v)) if v else float("nan")
    print(f"\n== {run_id} ==")
    print(f"{'type':>26} {'n':>4} {'acc':>6}   ingest(t/calls/s)      agent_in   blk   trunc")
    type_accs = []
    for t in TYPES:
        rs = [r for r in rows if r["question_type"] == t]
        if not rs:
            continue
        type_accs.append(acc(rs))
        ing = f"{sum(int(r['tokens_ingest_in']) for r in rs)/len(rs):8.0f}/{sum(int(r['ingest_calls']) for r in rs)/len(rs):4.0f}/{sum(float(r['ingest_s']) for r in rs)/len(rs):5.0f}"
        print(f"{t:>26} {len(rs):>4} {acc(rs):>6.3f}   {ing}   {sum(int(r['tokens_agent_in']) for r in rs)/len(rs):8.0f} {sum(int(r['block_tokens']) for r in rs)/len(rs):5.0f}   {sum(int(r['truncated']) for r in rs):>3}")
    abs_rows = [r for r in rows if r["is_abs"] == "1"]
    ans_rows = [r for r in rows if r["is_abs"] != "1"]
    fa = [r for r in ans_rows if r["false_abstention"] == "1"]
    print(f"{'task-averaged':>26} {len(rows):>4} {sum(type_accs)/max(len(type_accs),1):>6.3f}")
    print(f"{'overall':>26} {len(rows):>4} {acc(rows):>6.3f}")
    print(f"{'abstention acc (_abs)':>26} {len(abs_rows):>4} {acc(abs_rows):>6.3f}")
    print(f"{'false-abstention rate':>26} {len(ans_rows):>4} {len(fa)/max(len(ans_rows),1):>6.3f}")
    n = max(len(rows), 1)
    print(f"tokens/question: ingest_in {sum(int(r['tokens_ingest_in']) for r in rows)/n:.0f} · agent_in "
          f"{sum(int(r['tokens_agent_in']) for r in rows)/n:.0f} · judge_in {sum(int(r['tokens_judge_in']) for r in rows)/n:.0f} · "
          f"latency {sum(float(r['latency_s']) for r in rows)/n:.1f}s (ingest {sum(float(r['ingest_s']) for r in rows)/n:.1f}s)")


if __name__ == "__main__":
    main()
