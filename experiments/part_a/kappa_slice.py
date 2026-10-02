"""Judge audit on LoCoMo (step 15): a blind human-labelled slice → Cohen's κ.

    python kappa_slice.py sample              # draw the slice → kappa/slice.csv (seed 0)
    python kappa_slice.py label               # label blind, one item at a time (resumable)
    python kappa_slice.py report              # κ: human vs judge, human vs official F1; per backbone
    python kappa_slice.py rejudge             # re-run the Sonnet judge on the slice → judge–judge κ (~$0.3)

Slice: cat 1-4 rows from runs/conv-*_{full,bm25,mem0}_{sonnet5,qwen27b}, N/2 per backbone,
and within each backbone half judge-CORRECT / half judge-WRONG (balanced so κ is not computed on
a 90 %-correct sample; raw agreement on a balanced slice is NOT the population agreement — the
report says so). While labelling you see only question / gold / answer — never the backbone,
system, judge label or F1. Label by the benchmark's intent: is the answer right given the gold?
(c = correct, w = wrong, s = skip, u = undo last, q = quit).
"""
from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "kappa"
SLICE, LABELS, REJUDGE = OUT / "slice.csv", OUT / "labels.csv", OUT / "rejudge.csv"
SYSTEMS, BACKBONES = ("full", "bm25", "mem0"), ("sonnet5", "qwen27b")
F1_THRESHOLD = 0.5


def _rows():
    out = []
    for d in sorted(glob.glob(str(HERE / "locomo/runs/conv-*"))):
        p = next((Path(d) / n for n in ("results.csv", "result.csv") if (Path(d) / n).exists()), None)
        if not p:
            continue
        for r in csv.DictReader(p.open()):
            if r["system"] in SYSTEMS and r["backbone"] in BACKBONES and r["category"] in "1234" \
                    and r["score_judge"] not in ("", "None"):
                r["judge"] = int(float(r["score_judge"]))
                out.append(r)
    return out


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_sample(a):
    rows = _rows()
    rng = random.Random(a.seed)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["backbone"], r["judge"])].append(r)
    per = a.n // 4
    picked = []
    for bb in BACKBONES:
        for j in (1, 0):
            pool = cells[(bb, j)]
            rng.shuffle(pool)
            picked += pool[:per]
            print(f"{bb} judge={'CORRECT' if j else 'WRONG'}: {min(per, len(pool))} of {len(pool)}")
    rng.shuffle(picked)                                   # blind order
    OUT.mkdir(exist_ok=True)
    with SLICE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "conv", "qa_idx", "category", "system", "backbone", "pred", "gold", "score_official", "judge"])
        for i, r in enumerate(picked):
            w.writerow([i, r["conv"], r["qa_idx"], r["category"], r["system"], r["backbone"], r["pred"], r["gold"],
                        r["score_official"], r["judge"]])
    print(f"wrote {len(picked)} items → {SLICE}  (don't open it before labelling — it contains the judge column)")


def _questions():
    sys.path.insert(0, str(HERE / "locomo"))
    from locomo_data import load_conversation
    cache = {}

    def q(conv, idx):
        if conv not in cache:
            cache[conv] = load_conversation(conv)["qa"]
        return cache[conv][int(idx)]["question"]
    return q


def cmd_label(a):
    items = list(csv.DictReader(SLICE.open()))
    done = {r["item"]: r["human"] for r in csv.DictReader(LABELS.open())} if LABELS.exists() else {}
    question = _questions()
    order = [r for r in items if r["item"] not in done]
    print(f"{len(done)}/{len(items)} labelled. c = correct · w = wrong · s = skip · u = undo · q = quit\n")
    history = []
    i = 0
    while i < len(order):
        r = order[i]
        print(f"── {len(done) + 1}/{len(items)} ─────────────────────────────")
        print(f"Q:      {question(r['conv'], r['qa_idx'])}")
        print(f"GOLD:   {r['gold']}")
        print(f"ANSWER: {r['pred']}")
        try:
            k = input("label [c/w/s/u/q] > ").strip().lower()
        except EOFError:
            break
        if k == "q":
            break
        if k == "u" and history:
            done.pop(history.pop(), None); i -= 1
        elif k in ("c", "w"):
            done[r["item"]] = "1" if k == "c" else "0"; history.append(r["item"]); i += 1
        elif k == "s":
            i += 1
        with LABELS.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["item", "human"]); w.writerows(done.items())
    print(f"\n{len(done)}/{len(items)} labelled → {LABELS}")


def _stats(name, h, x):
    n = len(h)
    agree = sum(p == q for p, q in zip(h, x)) / n if n else float("nan")
    fp = sum(1 for p, q in zip(h, x) if q == 1 and p == 0)   # scorer says correct, human says wrong
    fn = sum(1 for p, q in zip(h, x) if q == 0 and p == 1)
    print(f"  {name:<22} n={n:>3}  κ={kappa(h, x):+.3f}  agreement={agree:.3f}  "
          f"scorer-lenient={fp}  scorer-strict={fn}")


def cmd_report(a):
    items = {r["item"]: r for r in csv.DictReader(SLICE.open())}
    lab = {r["item"]: int(r["human"]) for r in csv.DictReader(LABELS.open())}
    rows = [dict(items[i], human=h) for i, h in lab.items()]
    print(f"{len(rows)} labelled items (slice balanced on backbone × judge label — agreement here is "
          f"not the population agreement)\n")
    for scope, sub in [("all", rows)] + [(bb, [r for r in rows if r["backbone"] == bb]) for bb in BACKBONES]:
        h = [r["human"] for r in sub]
        print(f"{scope}:")
        _stats("human vs judge", h, [int(r["judge"]) for r in sub])
        _stats(f"human vs F1≥{F1_THRESHOLD}", h, [int(float(r["score_official"]) >= F1_THRESHOLD) for r in sub])
    print("\nby category (human vs judge / human vs F1):")
    for c in "4123":
        sub = [r for r in rows if r["category"] == c]
        if sub:
            h = [r["human"] for r in sub]
            print(f"  cat {c}  n={len(sub):>3}  κ_judge={kappa(h, [int(r['judge']) for r in sub]):+.3f}  "
                  f"κ_F1={kappa(h, [int(float(r['score_official']) >= F1_THRESHOLD) for r in sub]):+.3f}")
    print("\nwhere the judge is lenient (judge CORRECT, human WRONG), by backbone — the self-preference test:")
    for bb in BACKBONES:
        sub = [r for r in rows if r["backbone"] == bb and int(r["judge"]) == 1]
        if sub:
            bad = sum(1 for r in sub if r["human"] == 0)
            print(f"  {bb:<8} {bad}/{len(sub)} judge-CORRECT answers you marked wrong ({bad / len(sub):.0%})")
    if REJUDGE.exists():
        rj = {r["item"]: int(r["judge2"]) for r in csv.DictReader(REJUDGE.open()) if r["judge2"] != ""}
        both = [i for i in items if i in rj]
        j1, j2 = [int(items[i]["judge"]) for i in both], [rj[i] for i in both]
        print(f"\njudge vs re-judge (same answers, temperature 0): n={len(both)}  κ={kappa(j1, j2):+.3f}  "
              f"flips={sum(x != y for x, y in zip(j1, j2))}")


def cmd_rejudge(a):
    sys.path.insert(0, str(HERE.parent / "stream_runner")); sys.path.insert(0, str(HERE / "locomo"))
    from client import LLMClient
    from judge import judge as judge_fn
    items = list(csv.DictReader(SLICE.open()))
    question = _questions()
    client = LLMClient("sonnet5", run_id="kappa_rejudge", log_path=OUT / "calls.jsonl")
    with REJUDGE.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "judge2"])
        for r in items:
            ok, _ = judge_fn(client, category=int(r["category"]), question=question(r["conv"], r["qa_idx"]),
                             gold=r["gold"], response=r["pred"], row_id=f"rejudge:{r['item']}")
            w.writerow([r["item"], "" if ok is None else int(ok)]); f.flush()
    print(f"re-judged {len(items)} items → {REJUDGE}; run `report`")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "label", "report", "rejudge"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"sample": cmd_sample, "label": cmd_label, "report": cmd_report, "rejudge": cmd_rejudge}[a.cmd](a)
