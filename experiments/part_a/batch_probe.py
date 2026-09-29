"""Step 9.3 — prove the Part A pipeline runs through the Anthropic Message Batches API.

    python batch_probe.py                      # 5 LoCoMo conv-26 questions, full-context, Sonnet
    python batch_probe.py --n 5 --conv conv-26 --batch-id msgbatch_… --judge-batch-id msgbatch_…   # re-collect

Two batches: (1) the five answer calls with exactly step 8's prompt (official serialization +
official template, history as the cached system block); (2) the five Mem0-rubric judge calls on
the returned answers. Then the batch answers are printed next to the live answers from
runs/conv-26_full_sonnet5/results.csv (step 8), with per-request usage, turnaround and cost.
Outputs runs/batch_probe/{calls.jsonl, batches/*.json, probe.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stream_runner"))
sys.path.insert(0, str(HERE / "locomo"))

from batch import BatchClient, batch_cost_usd                                   # noqa: E402
from locomo_data import load_conversation, build_question, gold, CATEGORY_NAMES   # noqa: E402
from systems import make_system                                                 # noqa: E402
from run_locomo import official_score                                           # noqa: E402
from judge import JUDGE_SYSTEM_PROMPT, get_judge_prompt, preprocess_answer, _LABEL_RE   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--n", type=int, default=5, help="first N cat 1-4 questions")
    ap.add_argument("--batch-id", default=None, help="skip submission, collect this answer batch")
    ap.add_argument("--judge-batch-id", default=None)
    a = ap.parse_args()

    out = HERE / "runs" / "batch_probe"
    out.mkdir(parents=True, exist_ok=True)
    bc = BatchClient(run_id="batch_probe", log_path=out / "calls.jsonl")
    conv = load_conversation(a.conv)
    picks = [(i, qa) for i, qa in enumerate(conv["qa"]) if qa["category"] in (1, 2, 3, 4)][: a.n]
    system = make_system("full"); system.ingest(conv)
    history = system.context_for("")

    # ---- batch 1: answers (step 8's exact prompt)
    t0 = time.time()
    if a.batch_id:
        bid = a.batch_id
    else:
        reqs = []
        for i, qa in picks:
            qtext, template, _ = build_question(qa, qa_key=f"{a.conv}:{i}")
            reqs.append(bc.build(custom_id=str(i), tag="agent", system=history, user=template.format(qtext).strip(),
                                 max_tokens=64, temperature=0.0, cache_system=True))
        bid = bc.submit(reqs)
    bc.wait(bid)
    answers = bc.collect(bid)
    t_answers = time.time() - t0

    # ---- batch 2: judge (Mem0 rubric, cat 1-4)
    t1 = time.time()
    if a.judge_batch_id:
        jid = a.judge_batch_id
    else:
        jreqs = []
        for i, qa in picks:
            pred = (answers[str(i)]["text"] or "").strip()
            prompt = get_judge_prompt(qa["category"], qa["question"], preprocess_answer(qa["category"], gold(qa)), pred)
            jreqs.append(bc.build(custom_id=f"judge:{i}", tag="judge", system=JUDGE_SYSTEM_PROMPT, user=prompt, max_tokens=200))
        jid = bc.submit(jreqs)
    bc.wait(jid)
    judged = bc.collect(jid)
    t_judge = time.time() - t1

    # ---- live rows from step 8, if present
    live = {}
    p = HERE / "locomo" / "runs" / f"{a.conv}_full_sonnet5" / "results.csv"
    if p.exists():
        live = {int(r["qa_idx"]): r for r in csv.DictReader(p.open())}

    print(f"\n== batch probe · {a.conv} · full-context · Sonnet · {len(picks)} questions ==")
    print(f"{'idx':>4} {'cat':>12} {'batch pred':<34} {'live pred':<34} {'same':>4} {'off_b':>5} {'off_l':>5} {'judge':>7} {'in':>6} {'cr':>6} {'cw':>6} {'out':>4}")
    rows = []
    for i, qa in picks:
        r = answers[str(i)]; pred = (r["text"] or "").strip(); u = r["usage"]
        lp = (live.get(i) or {}).get("pred", "")
        off_b = official_score(qa["category"], pred, gold(qa))
        off_l = float(live[i]["score_official"]) if i in live else float("nan")
        jraw = judged.get(f"judge:{i}", {}).get("text", "") or ""
        m = _LABEL_RE.search(jraw); jlabel = m.group(1).upper() if m else "NONE"
        same = "yes" if lp and lp.strip() == pred else ("n/a" if not lp else "no")
        print(f"{i:>4} {CATEGORY_NAMES[qa['category']]:>12} {pred[:32]!r:<34} {lp[:32]!r:<34} {same:>4} {off_b:>5.2f} {off_l:>5.2f} {jlabel:>7} "
              f"{u['in']:>6} {u['cache_read']:>6} {u['cache_write']:>6} {u['out']:>4}")
        rows.append({"qa_idx": i, "category": qa["category"], "batch_pred": pred, "live_pred": lp, "same": same,
                     "official_batch": off_b, "official_live": off_l, "judge": jlabel, "usage": u, "error": r["error"]})
    agent_calls = [c for c in bc.calls if c["tag"] == "agent"]; judge_calls = [c for c in bc.calls if c["tag"] == "judge"]
    live_cost = sum((int(r["tokens_agent_in"]) - int(r["tokens_agent_cache_read"])) * 3.0 + int(r["tokens_agent_cache_read"]) * 0.3
                    + int(r["tokens_agent_out"]) * 15.0 for r in (live.get(i) for i, _ in picks) if r) / 1e6
    summary = {
        "n": len(picks), "answer_batch": bid, "judge_batch": jid, "turnaround_s": {"answers": round(t_answers), "judge": round(t_judge)},
        "identical_to_live": sum(1 for r in rows if r["same"] == "yes"), "compared": sum(1 for r in rows if r["same"] != "n/a"),
        "errors": sum(1 for r in rows if r["error"]), "cache_read_seen": any(c["cache_read"] > 0 for c in agent_calls),
        "cost_usd": {"batch_answers": round(batch_cost_usd(agent_calls), 4), "batch_judge": round(batch_cost_usd(judge_calls), 4),
                     "live_answers_with_cache_est": round(live_cost, 4)},
    }
    print(f"\nturnaround: answers {t_answers:.0f}s · judge {t_judge:.0f}s | identical to live: {summary['identical_to_live']}/{summary['compared']} | "
          f"errors {summary['errors']} | cache_read in batch: {summary['cache_read_seen']}")
    print(f"cost: batch answers ${summary['cost_usd']['batch_answers']} + judge ${summary['cost_usd']['batch_judge']} "
          f"vs live answers (with cache reads) ≈ ${summary['cost_usd']['live_answers_with_cache_est']}")
    (out / "probe.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
