"""Paired comparison of execution-stream runs on the same seeded task order (step 10c.3).

    python compare.py runs/batch8 runs/swe_s0_dc_qwen27b runs/swe_s0_ace_qwen27b
    python compare.py --baseline runs/batch8 runs/swe_s0_*_qwen27b --strata tier

The first run (or --baseline) is the no-memory reference. Every other run is joined to it on
task_id, so each comparison is paired: the same task, same order, same harness, only the memory
differs. Reports per run:
  resolve rate overall and per stratum; paired conversions (baseline failed → resolved) and
  regressions (baseline resolved → failed), McNemar exact p; resolve rate in the first vs
  second half of the stream (a learning curve needs the second half to improve *relative to the
  baseline's own second half*); and the efficiency side — turns, tool calls, wall time and agent
  input tokens per episode, first vs second half, paired against the baseline on the tasks both
  resolved (memory can pay off as "same success, fewer turns" when resolve rate is saturated).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load(run: str) -> dict[str, dict]:
    p = Path(run)
    p = p / "results.csv" if p.is_dir() else p
    rows = {r["task_id"]: r for r in csv.DictReader(p.open())}
    for r in rows.values():
        r["_pos"] = int(r.get("position", 0) or 0)
        r["_res"] = int(r["resolved"])
    return rows


def f(r, k):
    try:
        return float(r.get(k) or 0)
    except ValueError:
        return 0.0


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--strata", nargs="*", default=["tier"])
    a = ap.parse_args()
    base_path = a.baseline or a.runs[0]
    others = [r for r in a.runs if r != base_path]
    base = load(base_path)
    order = sorted(base, key=lambda t: base[t]["_pos"])
    half = len(order) // 2
    first, second = set(order[:half]), set(order[half:])
    print(f"baseline {Path(base_path).name}: {len(base)} tasks, resolved {sum(r['_res'] for r in base.values())}")

    for run in others:
        m = load(run)
        shared = [t for t in order if t in m]
        if not shared:
            print(f"\n{Path(run).name}: no tasks in common with the baseline"); continue
        mismatch = sum(1 for t in shared if m[t]["_pos"] != base[t]["_pos"])
        b_res = sum(base[t]["_res"] for t in shared); m_res = sum(m[t]["_res"] for t in shared)
        conv = [t for t in shared if not base[t]["_res"] and m[t]["_res"]]
        regr = [t for t in shared if base[t]["_res"] and not m[t]["_res"]]
        print(f"\n== {Path(run).name} vs baseline — {len(shared)} paired tasks"
              + (f"  (⚠ {mismatch} at a different position: orders differ)" if mismatch else ""))
        print(f"  resolved            baseline {b_res}  memory {m_res}  (Δ {m_res - b_res:+d})")
        print(f"  conversions         {len(conv)} baseline-failed → resolved   regressions {len(regr)}   McNemar p = {mcnemar_p(len(conv), len(regr)):.3f}")
        for s in a.strata:
            vals = sorted({base[t].get(s, "") for t in shared})
            parts = []
            for v in vals:
                ts = [t for t in shared if base[t].get(s, "") == v]
                parts.append(f"{v or '?'} {sum(base[t]['_res'] for t in ts)}→{sum(m[t]['_res'] for t in ts)}/{len(ts)}")
            print(f"  by {s:<17} " + " · ".join(parts))
        for name, ts in (("first half", [t for t in shared if t in first]), ("second half", [t for t in shared if t in second])):
            print(f"  {name:<19} baseline {mean(base[t]['_res'] for t in ts):.3f}  memory {mean(m[t]['_res'] for t in ts):.3f}"
                  f"  Δ {mean(m[t]['_res'] for t in ts) - mean(base[t]['_res'] for t in ts):+.3f}")
        both = [t for t in shared if base[t]["_res"] and m[t]["_res"]]
        print(f"  efficiency on the {len(both)} tasks both resolved (memory − baseline, mean per episode):")
        for k, label in (("num_turns", "turns"), ("n_tool_calls", "tool calls"), ("wall_s", "wall s"), ("cc_in", "agent input tok")):
            for name, ts in (("1st half", [t for t in both if t in first]), ("2nd half", [t for t in both if t in second])):
                if ts:
                    d = mean(f(m[t], k) - f(base[t], k) for t in ts); rel = d / max(mean(f(base[t], k) for t in ts), 1e-9)
                    print(f"    {label:<16} {name}  {d:+10.1f}  ({rel:+.0%})")
        mem_in = mean(f(m[t], "tokens_mem_in") for t in shared)
        print(f"  memory-side cost    {mem_in:.0f} input tokens per episode; block {mean(f(m[t], 'block_tokens') for t in shared):.0f} tokens")


if __name__ == "__main__":
    main()
