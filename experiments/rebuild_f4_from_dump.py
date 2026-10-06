"""Rebuild f4 (Part B learning curves) from a `dump_curves.py` text table.

    python rebuild_f4_from_dump.py figures/f4_dump_qwen27b.txt --backbone qwen27b

For a run box that cannot ship PNGs: `stream_runner/dump_curves.py` prints the sampled
cumulative-accuracy curves as text, the text is photographed/transcribed into a file with
the same layout (comment lines give n per stream), and this draws the same panels as
figures.f4 — minus the min–max band across orders, which the dump does not carry.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures import BB, COL, STREAM, _order, _save

SHORT = {"econ": "mmlu_pro_economics", "eng": "mmlu_pro_engineering", "phil": "mmlu_pro_philosophy", "gpqa": "gpqa_diamond"}


def load(path: Path, start: int = 50, every: int = 25):
    data, n_of = {}, {}
    for line in path.read_text().splitlines():
        m = re.match(r"#\s*(\w+)\s+n=(\d+)", line)
        if m:
            n_of[m.group(1)] = int(m.group(2)); continue
        if not line.strip() or line.startswith("#"):
            continue
        short, system, *vals = line.split()
        n = n_of[short]
        pos = list(range(start, n + 1, every)) + ([n] if (n - start) % every else [])
        ys = np.array([int(v) / 1000 for v in vals])
        assert len(ys) == len(pos), f"{short} {system}: {len(ys)} values for {len(pos)} positions"
        data.setdefault(SHORT[short], {})[system] = (np.array(pos), ys)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--backbone", default="qwen27b")
    ap.add_argument("--orders", type=int, default=3, help="orders averaged on the MMLU-Pro streams (label only)")
    a = ap.parse_args()
    data = load(Path(a.dump))
    streams = [s for s in STREAM if s in data]
    fig, axes = plt.subplots(1, len(streams), figsize=(4.2 * len(streams), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, streams):
        finals = []
        for s in _order(data[st]):
            if s not in COL:
                continue                                     # systems not in the post (e.g. reasoningbank)
            pos, ys = data[st][s]
            style = dict(color="black", lw=2.2, ls="--", zorder=5) if s == "none" else dict(color=COL[s], lw=1.9)
            multi = st.startswith("mmlu") and a.orders > 1
            ax.plot(pos, ys, label=f"{s}" + (f" ({a.orders} orders)" if multi and s != "none" else ""), **style)
            finals += list(ys)
        ax.set_title(STREAM[st], fontsize=10); ax.set_xlabel(f"position in stream (from {pos[0]})"); ax.grid(alpha=0.3)
        lo, hi = min(finals), max(finals); pad = (hi - lo) * 0.25 + 0.01
        ax.set_xlim(pos[0], None); ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("cumulative accuracy"); axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Learning curves on {BB[a.backbone]} — flat for every memory system (dashed black = no memory)", y=0.99)
    fig.subplots_adjust(top=0.82)
    _save(fig, f"f4_curves_{a.backbone}")


if __name__ == "__main__":
    main()
