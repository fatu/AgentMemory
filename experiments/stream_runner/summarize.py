"""Aggregate runs/*/results.csv into one table (accuracy, parse failures, truncation,
tokens per step, list-price cost). Used for pilots, the step-10 cost table, and as the
input to the plots later.

    python summarize.py            # all runs
    python summarize.py --glob 'runs/mmlu_pro_philosophy_s0_*'
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

PRICE = {"sonnet5": (2.0, 10.0), "qwen27b": (0.0, 0.0)}   # $/M tokens (in, out), standard rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*")
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(a.glob)):
        p = next((Path(d) / n for n in ("results.csv", "result.csv") if (Path(d) / n).exists()), None)
        if p is None:
            continue
        df = pd.read_csv(p)
        b = df["backbone"].iloc[0]
        pin, pout = PRICE.get(b, (0, 0))
        col = lambda n: df[n].sum() if n in df else 0
        cr = col("tokens_agent_cache_read") + col("tokens_mem_cache_read")
        cw = col("tokens_agent_cache_write") + col("tokens_mem_cache_write")
        tin = col("tokens_agent_in") + col("tokens_mem_in")
        # Anthropic: uncached 1x, cache read 0.1x, cache write 1.25x (5-min); Qwen all 0
        cost = (tin - cr - cw) / 1e6 * pin + cr / 1e6 * pin * 0.1 + cw / 1e6 * pin * 1.25 \
             + (col("tokens_agent_out") + col("tokens_mem_out")) / 1e6 * pout
        rows.append({
            "run": Path(d).name, "n": len(df),
            "acc": round(df["correct"].astype(str).eq("True").mean(), 3),
            "null_pred": int(df["pred"].isna().sum()),
            "truncated": int(df["truncated"].astype(str).eq("True").sum()),
            "mem_trunc": int(df["mem_truncated"].astype(str).eq("True").sum()) if "mem_truncated" in df else -1,
            "in/step": int(df["tokens_agent_in"].mean()), "out/step": int(df["tokens_agent_out"].mean()),
            "cached/step": int(df["tokens_agent_cache_read"].mean()) if "tokens_agent_cache_read" in df else -1,
            "cwrite/step": int(df["tokens_agent_cache_write"].mean()) if "tokens_agent_cache_write" in df else -1,
            "mem_in/step": int(df["tokens_mem_in"].mean()), "mem_out/step": int(df["tokens_mem_out"].mean()),
            "block_mean": int(df["block_tokens"].mean()), "block_max": int(df["block_tokens"].max()),
            "mem_size_end": int(df["memory_size_tokens"].iloc[-1]),
            "s/step": round(df["latency_s"].mean(), 1),
            "$/step": round(cost / len(df), 4), "$_total": round(cost, 3),
        })
    if not rows:
        print("no results under", a.glob); return
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
