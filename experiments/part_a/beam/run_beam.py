"""BEAM pilot (step 9.2): official data + official prompts + official judge, our execution layer.

    python run_beam.py --group 100K --conv 1 --system full --backbone sonnet5     # the smoke: 20 Qs
    python run_beam.py --group 100K --conv 1 --system mem0 --backbone qwen27b
    python run_beam.py --group 100K --conv 1 --system bm25 --backbone sonnet5 --rejudge missing

Per conversation: system.ingest(chat) once (LLM calls tagged memory, row_id ingest:<conv>:<k>);
per question: full → the chat as real turns + the official NOTE message (cache breakpoint on
the last turn: 20 questions share it); others → the official RAG prompt with the retrieved
block (≤ 4,096 tokens). Backbone: no thinking, temperature 0, max_tokens 1024. Judge: Sonnet
on the official nugget prompt per rubric item (+ LLM event alignment → tau_norm for
event_ordering). Resumable per question; outputs runs/<group>-<conv>_<system>_<backbone>/.
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

from client import LLMClient                                                  # noqa: E402
from embed import Embedder                                                    # noqa: E402
from memory.base import CAP, count_tokens                                     # noqa: E402
from beam_data import (ABILITIES, BEAM_DIR, history_plain, load_chat, load_questions,   # noqa: E402
                       note_prompt, rag_prompt)
from systems import make_system                                               # noqa: E402
from judge import Judge                                                       # noqa: E402

JUDGE_BACKBONE = "sonnet5"
GEN_MAX_TOKENS = 1024
FIELDS = ["part", "benchmark", "group", "conv", "ability", "q_idx", "difficulty", "system", "backbone",
          "pred", "gold", "score", "llm_judge_score", "tau_norm", "f1", "n_nuggets", "judge_calls", "judge_parse_fails",
          "history_tokens", "block_tokens", "truncated",
          "tokens_agent_in", "tokens_agent_cache_read", "tokens_agent_cache_write", "tokens_agent_out",
          "tokens_judge_in", "tokens_judge_out", "latency_s"]


def _sum(calls, tag, key):
    return sum(c.get(key, 0) for c in calls if c["tag"] == tag)


def _tot_in(calls, tag):
    return _sum(calls, tag, "in") + _sum(calls, tag, "cache_read") + _sum(calls, tag, "cache_write")


def _ingest_calls_from_log(path: Path, conv_id: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if str(c.get("row_id", "")).startswith(f"ingest:{conv_id}:") and c.get("tag") == "memory":
            out.append(c)
    return out


def max_context(client, backbone: str) -> int | None:
    if backbone == "sonnet5":
        return int(os.environ.get("SONNET_MAX_CONTEXT", "200000"))
    if os.environ.get("QWEN_MAX_MODEL_LEN"):
        return int(os.environ["QWEN_MAX_MODEL_LEN"])
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
    ap.add_argument("--group", default="100K", choices=["100K", "1M"])
    ap.add_argument("--conv", default="1")
    ap.add_argument("--system", choices=["full", "bm25", "mem0", "amem", "hindsight", "hindsight-reflect"], required=True)
    ap.add_argument("--backbone", choices=["sonnet5", "qwen27b"], required=True)
    ap.add_argument("--abilities", nargs="+", default=None, help="subset of the 10 ability keys")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--rejudge", choices=["missing", "all"], default=None)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    conv_id = f"{a.group}-{a.conv}"
    run_id = a.run_id or f"{conv_id}_{a.system}_{a.backbone}"
    out = HERE / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(a.group, a.conv)
    if a.abilities:
        questions = [q for q in questions if q["ability"] in a.abilities]

    if a.rejudge:
        return rejudge(a, out, run_id, questions)

    chat = load_chat(a.group, a.conv)
    client = LLMClient(a.backbone, run_id=run_id, log_path=out / "calls.jsonl")
    judge_client = client if a.backbone == JUDGE_BACKBONE else LLMClient(JUDGE_BACKBONE, run_id=run_id, log_path=out / "calls.jsonl")
    judge = Judge(judge_client)
    embedder = Embedder() if a.system in ("mem0", "amem") else None
    ctx_limit = max_context(client, a.backbone) if a.system == "full" else None
    system = make_system(a.system, client=client, embedder=embedder, state_dir=out / a.system,
                         max_context=ctx_limit, backbone=a.backbone)

    t_ing = time.perf_counter()
    if a.system in ("mem0", "amem", "hindsight", "hindsight-reflect"):
        system.ingest(chat, conv_id=conv_id, resume=True)
    else:
        system.ingest(chat)
    ingest_s = round(time.perf_counter() - t_ing, 1)
    ing = _ingest_calls_from_log(out / "calls.jsonl", conv_id) or \
        [c for c in client.calls if str(c["row_id"]).startswith(f"ingest:{conv_id}:")]
    history_tokens = count_tokens(history_plain(chat))
    ingest = {"ingest_calls": len([c for c in ing if c["tag"] == "memory"]),
              "ingest_tokens_in": _tot_in(ing, "memory"), "ingest_tokens_out": _sum(ing, "memory", "out"),
              "ingest_seconds": (round(sum(c.get("latency_s", 0) for c in ing), 1) or ingest_s)}
    print(f"ingested {conv_id} with {a.system}: {ingest} · history {history_tokens} o200k tokens"
          f"{' · TRUNCATED' if getattr(system, 'truncated', False) else ''}")

    (out / "provenance.json").write_text(json.dumps({
        "benchmark": "beam", "group": a.group, "conv": a.conv, "data": str(BEAM_DIR / "chats" / a.group / a.conv),
        "repo_commit": "b2da22e", "system": a.system, "ingest_unit": system.ingest_unit, "backbone": a.backbone,
        "model": client.model, "judge": JUDGE_BACKBONE, "judge_protocol": "unified_llm_judge_base_prompt per nugget; "
        "event_ordering = llm_equivalence alignment + kendall tau_b → tau_norm", "thinking": False, "temperature": 0.0,
        "gen_max_tokens": GEN_MAX_TOKENS, "cap_tokens": CAP, "max_context": ctx_limit,
        "history_tokens_o200k": history_tokens, "truncated": bool(getattr(system, "truncated", False)),
        "embedder": (embedder.model_name if embedder else None), "embedder_backend": (embedder.backend if embedder else None),
        **ingest, "n_questions": len(questions),
        "hindsight": (system.store.info() if hasattr(system, "store") else None), "ts": time.time(),
    }, indent=2))

    results_path = out / "results.csv"
    done = set()
    if results_path.exists():
        with results_path.open() as f:
            done = {(r["ability"], int(r["q_idx"])) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} rows done")

    with results_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for n, q in enumerate(questions):
            key = (q["ability"], q["idx"])
            if key in done:
                continue
            row_id = f"{q['ability']}:{q['idx']}"
            client.current_row_id = row_id
            t0 = time.perf_counter()
            if system.kind == "history":
                block_tokens = 0
                for attempt in range(3):
                    history = system.context_for(q["question"])
                    try:
                        pred = client.complete("", note_prompt(q["question"]), tag="agent", row_id=row_id,
                                               max_tokens=GEN_MAX_TOKENS, temperature=0.0, thinking=False,
                                               cache_system=True, history=history)
                        break
                    except Exception as e:                         # the estimate undershot the real tokenizer:
                        msg = str(e).lower()                       # prune harder (official rule) and flag the row
                        if attempt == 2 or not any(s in msg for s in ("too long", "context length", "maximum context", "too many tokens")):
                            raise
                        system.factor += 0.15
                        system.ingest(chat)
                        print(f"  prompt too long for {a.backbone}; re-pruned with token factor {system.factor:.2f} (truncated={system.truncated})")
            elif system.kind == "reader":                          # hindsight-reflect: its own reader
                block_tokens = 0
                pred = system.answer(q["question"], row_id)
            else:
                context = system.context_for(q["question"])
                block_tokens = count_tokens(context)
                assert block_tokens <= CAP, (row_id, block_tokens)
                pred = client.complete("", rag_prompt(context, q["question"]), tag="agent", row_id=row_id,
                                       max_tokens=GEN_MAX_TOKENS, temperature=0.0, thinking=False)
            pred = (pred or "").strip()
            j = {"score": None, "llm_judge_score": None, "tau_norm": None, "f1": None,
                 "n_nuggets": len(q["rubric"]), "judge_calls": 0, "parse_fails": 0}
            if not a.no_judge:
                j = judge.score(ability=q["ability"], rubric=q["rubric"], response=pred, row_id=row_id)
            with (out / "judge_log.jsonl").open("a") as jl:       # raw judge outputs per row (E4 audit)
                jl.write(json.dumps({"row_id": row_id, "nuggets": q["rubric"], "nugget_raw": j.get("raw", []),
                                     "equiv": j.get("equiv", [])}) + "\n")
            calls = [c for c in client.calls + (judge_client.calls if judge_client is not client else []) if c["row_id"] == row_id]
            row = {
                "part": "A", "benchmark": "beam", "group": a.group, "conv": a.conv, "ability": q["ability"], "q_idx": q["idx"],
                "difficulty": q["difficulty"], "system": a.system, "backbone": a.backbone, "pred": pred, "gold": q["gold"],
                "score": _fmt(j["score"]), "llm_judge_score": _fmt(j["llm_judge_score"]), "tau_norm": _fmt(j["tau_norm"]),
                "f1": _fmt(j["f1"]), "n_nuggets": j["n_nuggets"], "judge_calls": j["judge_calls"], "judge_parse_fails": j["parse_fails"],
                "history_tokens": history_tokens, "block_tokens": block_tokens, "truncated": int(getattr(system, "truncated", False)),
                "tokens_agent_in": _tot_in(calls, "agent"), "tokens_agent_cache_read": _sum(calls, "agent", "cache_read"),
                "tokens_agent_cache_write": _sum(calls, "agent", "cache_write"), "tokens_agent_out": _sum(calls, "agent", "out"),
                "tokens_judge_in": _tot_in(calls, "judge"), "tokens_judge_out": _sum(calls, "judge", "out"),
                "latency_s": round(time.perf_counter() - t0, 1),
            }
            w.writerow(row); f.flush()
            print(f"[{n+1}/{len(questions)}] {q['ability']:>24}/{q['idx']} score={row['score']:>5} "
                  f"(nuggets {j['n_nuggets']}, judge calls {j['judge_calls']}) in={row['tokens_agent_in']} "
                  f"(cr {row['tokens_agent_cache_read']}) blk={block_tokens} pred={pred[:60]!r}")

    rows = list(csv.DictReader(results_path.open()))
    write_official_layout(rows, questions, out)
    summarize(rows, run_id)


def _fmt(v):
    return "" if v is None else round(float(v), 4)


def write_official_layout(rows, questions, out: Path):
    """answers.json in the shape answer_generation.py writes (question objects + llm_response),
    so the official run_evaluation.py could re-judge it with gpt-4.1-mini."""
    by = {(r["ability"], int(r["q_idx"])): r["pred"] for r in rows}
    data = {}
    for q in questions:
        data.setdefault(q["ability"], []).append({"question": q["question"], "rubric": q["rubric"],
                                                 "llm_response": by.get((q["ability"], q["idx"]), "")})
    (out / "answers.json").write_text(json.dumps(data, indent=2))


def rejudge(a, out: Path, run_id: str, questions):
    results_path = out / "results.csv"
    rows = list(csv.DictReader(results_path.open()))
    todo = [r for r in rows if a.rejudge == "all" or r["score"] in ("", "None")]
    print(f"{run_id}: {len(rows)} rows, judging {len(todo)}")
    if todo:
        jc = LLMClient(JUDGE_BACKBONE, run_id=run_id, log_path=out / "calls.jsonl")
        judge = Judge(jc)
        qmap = {(q["ability"], q["idx"]): q for q in questions}
        for r in todo:
            q = qmap[(r["ability"], int(r["q_idx"]))]
            row_id = f"{r['ability']}:{r['q_idx']}"
            j = judge.score(ability=q["ability"], rubric=q["rubric"], response=r["pred"], row_id=row_id)
            calls = [c for c in jc.calls if c["row_id"] == row_id]
            r.update(score=_fmt(j["score"]), llm_judge_score=_fmt(j["llm_judge_score"]), tau_norm=_fmt(j["tau_norm"]),
                     f1=_fmt(j["f1"]), n_nuggets=j["n_nuggets"], judge_calls=j["judge_calls"], judge_parse_fails=j["parse_fails"],
                     tokens_judge_in=_tot_in(calls, "judge"), tokens_judge_out=_sum(calls, "judge", "out"))
            print(f"  {r['ability']:>24}/{r['q_idx']} score={r['score']:>5} judge calls {j['judge_calls']}")
        with results_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    summarize(rows, run_id)


def summarize(rows, run_id):
    """Per ability: mean question score (nugget mean; tau_norm for event_ordering) — the
    official per-(group, ability) aggregate for one conversation; 'average' = equal-weight
    mean of the ability means (Table 1's composite)."""
    def mean(xs):
        xs = [float(x) for x in xs if x not in ("", "None")]
        return sum(xs) / len(xs) if xs else float("nan")
    print(f"\n== {run_id} ==")
    print(f"{'ability':>26} {'n':>3} {'score':>6} {'nuggets':>7} {'jcalls':>6} {'agent_in':>9} {'cache_rd':>8} {'blk':>5}")
    ability_means = []
    for ab in ABILITIES:
        rs = [r for r in rows if r["ability"] == ab]
        if not rs:
            continue
        m = mean(r["score"] for r in rs); ability_means.append(m)
        print(f"{ab:>26} {len(rs):>3} {m:>6.3f} {sum(int(r['n_nuggets']) for r in rs):>7} {sum(int(r['judge_calls']) for r in rs):>6} "
              f"{sum(int(r['tokens_agent_in']) for r in rs)/len(rs):>9.0f} {sum(int(r['tokens_agent_cache_read']) for r in rs)/len(rs):>8.0f} "
              f"{sum(int(r['block_tokens']) for r in rs)/len(rs):>5.0f}")
    valid = [m for m in ability_means if m == m]
    print(f"{'average (10 abilities)':>26} {len(rows):>3} {sum(valid)/max(len(valid),1):>6.3f}")
    n = max(len(rows), 1)
    print(f"tokens/question: agent_in {sum(int(r['tokens_agent_in']) for r in rows)/n:.0f} "
          f"(cache_read {sum(int(r['tokens_agent_cache_read']) for r in rows)/n:.0f}, cache_write {sum(int(r['tokens_agent_cache_write']) for r in rows)/n:.0f}) · "
          f"judge_in {sum(int(r['tokens_judge_in']) for r in rows)/n:.0f} over {sum(int(r['judge_calls']) for r in rows)/n:.1f} calls · "
          f"parse fails {sum(int(r['judge_parse_fails']) for r in rows)} · latency {sum(float(r['latency_s']) for r in rows)/n:.1f}s · "
          f"truncated {sum(int(r['truncated']) for r in rows)}")


if __name__ == "__main__":
    main()
