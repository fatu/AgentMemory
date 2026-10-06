"""Print the f4 learning curves as a compact text table (for machines that can't ship PNGs).

    python dump_curves.py --backbone qwen27b            # every 25 positions from 50, 3 digits each
    python dump_curves.py --backbone qwen27b --every 50

One line per stream × system: cumulative accuracy (orders averaged) sampled at the listed
positions, ×1000 with no decimal point ("887" = 0.887). Cross-domain control runs (*-x*)
are skipped, as in figures.py. Photograph the output; figures.py's f4 can be rebuilt from it.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

STREAMS = ["mmlu_pro_economics", "mmlu_pro_engineering", "mmlu_pro_philosophy", "gpqa_diamond"]
SHORT = {"mmlu_pro_economics": "econ", "mmlu_pro_engineering": "eng", "mmlu_pro_philosophy": "phil", "gpqa_diamond": "gpqa"}
SYSTEMS = ["none", "exprag", "dc", "dc-frozen", "mem0", "amem", "ace", "hindsight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*")
    ap.add_argument("--backbone", default="qwen27b")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--min-rows", type=int, default=100)
    a = ap.parse_args()

    frames = []
    for d in sorted(glob.glob(a.runs)):
        if "-x" in Path(d).name:
            continue
        p = next((Path(d) / n for n in ("results.csv", "result.csv") if (Path(d) / n).exists()), None)
        if p is None:
            continue
        df = pd.read_csv(p)
        if len(df) < a.min_rows:
            continue
        df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(int)
        frames.append(df)
    df = pd.concat(frames)
    df = df[df.backbone == a.backbone]

    for st in STREAMS:
        d = df[df.stream == st]
        if d.empty:
            continue
        n = int(d.groupby(["system", "seed"]).size().min())
        pos = list(range(a.start, n + 1, a.every))
        if pos[-1] != n:
            pos.append(n)
        print(f"# {SHORT[st]} n={n} positions: {a.start}..{n} every {a.every} (+{n} last)")
        for s in SYSTEMS:
            g = d[d.system == s]
            if g.empty:
                continue
            curves = []
            for _, gs in g.groupby("seed"):
                gs = gs.sort_values("position")
                curves.append(np.cumsum(gs["correct"].to_numpy()) / (np.arange(len(gs)) + 1))
            m = np.vstack([c[:n] for c in curves]).mean(0)
            vals = " ".join(f"{int(round(m[p - 1] * 1000)):03d}" for p in pos)
            print(f"{SHORT[st]} {s:<9} {vals}")
        print()


if __name__ == "__main__":
    main()
