"""The post's figures (step 16), from the rolled-up tables — nothing is typed in by hand.

    python figures.py                 # part_a/tables/*.csv + stream_runner/plots/curves.csv → figures/*.png
    python figures.py --only f3 f5

Inputs (run `part_a/rollup.py` and `stream_runner/plot_curves.py` first):
    part_a/tables/locomo.csv · longmemeval.csv · beam.csv
    stream_runner/plots/curves.csv
    part_a/kappa/{slice,labels,rejudge}.csv          (f6; optional)

Figures
    f1  LoCoMo: judge vs official F1 per system × backbone — the scorer disagreement
    f2  LongMemEval per-type bars — the ability signatures (Qwen; all systems)
    f3  BEAM 100K → 1M — the crossover
    f4  Part B learning curves, cumulative, Qwen seeds-averaged, one panel per stream (+ Sonnet)
    f5  Part B "static vs learned": none / dc-frozen / dc / exprag final accuracy per stream
    f6  Judge audit: κ by scorer and backbone (from kappa/)
    f7  Cost: accuracy vs input tokens per question, log-x, LoCoMo + LongMemEval
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = HERE / "part_a" / "tables"
OUT = HERE / "figures"
SYS_ORDER = ["full", "bm25", "mem0", "amem", "hindsight", "hindsight-reflect", "none", "exprag", "dc", "dc-frozen", "ace"]
COL = {"full": "#4c4c4c", "bm25": "#1f77b4", "mem0": "#2ca02c", "amem": "#9467bd", "hindsight": "#ff7f0e",
       "hindsight-reflect": "#ffbb78", "none": "#7f7f7f", "exprag": "#17becf", "dc": "#d62728", "dc-frozen": "#ff9896", "ace": "#8c564b"}
BB = {"qwen27b": "Qwen3.8-27B", "sonnet5": "Claude Sonnet 5"}
STREAM = {"mmlu_pro_economics": "MMLU-Pro Economics", "mmlu_pro_engineering": "MMLU-Pro Engineering",
          "mmlu_pro_philosophy": "MMLU-Pro Philosophy", "gpqa_diamond": "GPQA-Diamond", "aime24": "AIME-24", "aime25": "AIME-25"}


def _order(systems):
    return [s for s in SYS_ORDER if s in set(systems)]


def _save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=150); plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- f1 LoCoMo: judge vs F1
def f1():
    df = pd.read_csv(T / "locomo.csv")
    systems = _order(df["system"]); x = np.arange(len(systems)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, (bb, col) in enumerate([("sonnet5", "cat1-4_judge"), ("qwen27b", "cat1-4_judge"), ("sonnet5", "cat1-4_off"), ("qwen27b", "cat1-4_off")]):
        color = "#d62728" if bb == "sonnet5" else "#1f77b4"
        for j, s in enumerate(systems):
            r = df[(df.system == s) & (df.backbone == bb)]
            if r.empty:
                continue
            partial = int(r["convs"].iloc[0]) < 10
            ax.bar(x[j] + (i - 1.5) * w, r[col].iloc[0], w, color=color, alpha=(0.9 if "judge" in col else 0.45) * (0.5 if partial else 1),
                   hatch=("" if "judge" in col else "//"), edgecolor="white",
                   label=(f"{BB[bb]} · {'judge' if 'judge' in col else 'official F1'}" if j == 0 else None))
            if partial:
                ax.text(x[j] + (i - 1.5) * w, r[col].iloc[0] + 0.01, f"{int(r['convs'].iloc[0])} conv", ha="center", fontsize=6, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(systems); ax.set_ylim(0, 1.08); ax.set_ylabel("LoCoMo cat 1–4 (10 conversations unless marked)")
    ax.set_title("Two scorers, two backbone rankings: the judge puts Sonnet first, the F1 puts Qwen first")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    _save(fig, "f1_locomo_scorers")


# ---------------------------------------------------------------- f2 LongMemEval per-type
def f2():
    df = pd.read_csv(T / "longmemeval.csv")
    df = df[(df.backbone == "qwen27b") & (df.n >= 150)]
    types = ["ss-user", "ss-assistant", "ss-preference", "multi-session", "temporal", "knowledge-upd"]
    labels = ["single-session\nuser", "single-session\nassistant", "single-session\npreference", "multi-\nsession", "temporal", "knowledge\nupdate"]
    systems = _order(df["system"]); x = np.arange(len(types)); w = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, s in enumerate(systems):
        r = df[df.system == s].iloc[0]
        ax.bar(x + (i - (len(systems) - 1) / 2) * w, [r[t] for t in types], w, color=COL[s], label=f"{s} (n={int(r['n'])})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 1.05); ax.set_ylabel("judge accuracy")
    ax.set_title("LongMemEval-S on Qwen: the ability signature follows the storage unit (extracted facts vs raw turns)")
    ax.legend(fontsize=8, ncol=3, loc="lower left"); ax.grid(axis="y", alpha=0.3)
    _save(fig, "f2_longmemeval_types")


# ---------------------------------------------------------------- f3 BEAM crossover
def f3():
    df = pd.read_csv(T / "beam.csv")
    df = df[df.backbone == "qwen27b"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s in _order(df["system"]):
        a = df[(df.system == s) & (df.benchmark == "beam-100K")]["average"]
        b = df[(df.system == s) & (df.benchmark == "beam-1M")]["average"]
        if len(a) and len(b):
            ax.plot([0, 1], [a.iloc[0], b.iloc[0]], "o-", color=COL[s], lw=2.2, label=s)
        elif len(a):
            ax.plot([0], [a.iloc[0]], "o", color=COL[s], label=f"{s} (100K only)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["100K tokens\n(20 conversations)", "1M tokens\n(5 conversations)"])
    ax.set_ylabel("BEAM 10-ability average (Qwen)"); ax.set_ylim(0.25, 0.55); ax.grid(alpha=0.3)
    ax.set_title("At 1M the history no longer fits: full context falls to BM25's level,\nthe LLM-written memories hold their score")
    ax.legend(fontsize=8)
    _save(fig, "f3_beam_crossover")


# ---------------------------------------------------------------- f4 learning curves
def f4():
    runs = HERE / "stream_runner" / "runs"
    frames = []
    for d in sorted(runs.glob("*")):
        p = next((d / n for n in ("results.csv", "result.csv") if (d / n).exists()), None)
        if p:
            df = pd.read_csv(p)
            if len(df) >= 100:
                df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(int); frames.append(df)
    df = pd.concat(frames)
    for bb in ("qwen27b", "sonnet5"):
        d = df[df.backbone == bb]
        streams = [s for s in STREAM if s in set(d.stream)]
        if not streams:
            continue
        fig, axes = plt.subplots(1, len(streams), figsize=(4.2 * len(streams), 3.8), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, st in zip(axes, streams):
            start = 50 if st.startswith("mmlu") or st == "gpqa_diamond" else 10
            finals = []
            for s in _order(d[d.stream == st].system):
                g = d[(d.stream == st) & (d.system == s)]
                curves = []
                for seed, gs in g.groupby("seed"):
                    gs = gs.sort_values("position"); curves.append(np.cumsum(gs["correct"].to_numpy()) / (np.arange(len(gs)) + 1))
                n = min(map(len, curves)); stack = np.vstack([c[:n] for c in curves]); m = stack.mean(0)
                xs = np.arange(1, n + 1); keep = xs >= start
                style = dict(color="black", lw=2.2, ls="--", zorder=5) if s == "none" else dict(color=COL[s], lw=1.9)
                ax.plot(xs[keep], m[keep], label=f"{s}" + (f" ({len(curves)} orders)" if len(curves) > 1 else ""), **style)
                if len(curves) > 1 and s != "none":
                    ax.fill_between(xs[keep], stack.min(0)[keep], stack.max(0)[keep], color=COL[s], alpha=0.08, lw=0)
                finals += list(m[keep])
            ax.set_title(STREAM[st], fontsize=10); ax.set_xlabel(f"position in stream (from {start})"); ax.grid(alpha=0.3)
            lo, hi = min(finals), max(finals); pad = (hi - lo) * 0.25 + 0.01
            ax.set_xlim(start, None); ax.set_ylim(lo - pad, hi + pad)
        axes[0].set_ylabel("cumulative accuracy"); axes[-1].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Learning curves on {BB[bb]} — flat for every memory system (dashed black = no memory)", y=0.99)
        fig.subplots_adjust(top=0.82)
        _save(fig, f"f4_curves_{bb}")


# ---------------------------------------------------------------- f5 static vs learned
def f5():
    cv = pd.read_csv(HERE / "stream_runner" / "plots" / "curves.csv")
    cv = cv[cv.n >= 100]
    for bb in ("qwen27b", "sonnet5"):
        d = cv[cv.backbone == bb].groupby(["stream", "system"])["acc"].mean().unstack()
        streams = [s for s in STREAM if s in d.index]
        systems = [s for s in ["none", "exprag", "dc-frozen", "dc", "ace", "mem0", "amem", "hindsight"] if s in d.columns]
        if not streams or not systems:
            continue
        x = np.arange(len(streams)); w = 0.8 / len(systems)
        fig, ax = plt.subplots(figsize=(8.5, 4))
        for i, s in enumerate(systems):
            ax.bar(x + (i - (len(systems) - 1) / 2) * w, [d.loc[st, s] for st in streams], w, color=COL[s], label=s)
        ax.set_xticks(x); ax.set_xticklabels([STREAM[s] for s in streams], fontsize=9)
        ax.set_ylabel("final accuracy (orders averaged)"); ax.set_ylim(0.5, 0.95); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{BB[bb]}: dc ≈ dc-frozen (its own final sheet, never updated) — the gain is advice, not learning")
        ax.legend(fontsize=8, ncol=4, loc="lower left")
        _save(fig, f"f5_static_vs_learned_{bb}")


# ---------------------------------------------------------------- f6 judge audit
def f6():
    k = HERE / "part_a" / "kappa"
    if not (k / "labels.csv").exists():
        print("f6: no kappa/labels.csv — skipped"); return
    items = {r["item"]: r for r in csv.DictReader((k / "slice.csv").open())}
    lab = {r["item"]: int(r["human"]) for r in csv.DictReader((k / "labels.csv").open())}

    def kappa(a, b):
        n = len(a); po = sum(x == y for x, y in zip(a, b)) / n; pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb); return (po - pe) / (1 - pe)
    rows = []
    for scope in ("all", "sonnet5", "qwen27b"):
        sel = [i for i in lab if scope == "all" or items[i]["backbone"] == scope]
        h = [lab[i] for i in sel]
        rows.append((scope, kappa(h, [int(items[i]["judge"]) for i in sel]), kappa(h, [int(float(items[i]["score_official"]) >= 0.5) for i in sel])))
    fig, ax = plt.subplots(figsize=(6, 3.6)); x = np.arange(3)
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color="#d62728", label="judge (Sonnet 5, Mem0 rubric)")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color="#1f77b4", label="official F1 ≥ 0.5")
    ax.set_xticks(x); ax.set_xticklabels(["all 98", "Sonnet answers", "Qwen answers"]); ax.set_ylim(0, 1)
    ax.set_ylabel("Cohen's κ vs blind human labels"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Which scorer to trust: the judge agrees with a human more than the F1 does,\nand no more on Sonnet's answers than on Qwen's")
    _save(fig, "f6_judge_kappa")


# ---------------------------------------------------------------- f7 cost
def f7():
    lo = pd.read_csv(T / "locomo.csv"); lo = lo[lo.system != "hindsight-reflect"]   # its own reader: not a block-under-the-cap point
    lm = pd.read_csv(T / "longmemeval.csv"); lm = lm[lm.n >= 150]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, df, ycol, title in ((axes[0], lo, "cat1-4_judge", "LoCoMo (judge, cat 1–4)"), (axes[1], lm, "task_avg", "LongMemEval-S (judge, task-avg)")):
        for _, r in df.iterrows():
            m = "o" if r.backbone == "sonnet5" else "s"
            partial = "convs" in r and int(r["convs"]) < 10
            ax.scatter(r["agent_in"], r[ycol], facecolors=("none" if partial else COL.get(r.system, "k")), edgecolors=COL.get(r.system, "k"),
                       marker=m, s=60, zorder=3, linewidths=1.5)
            ax.annotate(r.system + (f" ({int(r['convs'])} conv)" if partial else ""), (r["agent_in"], r[ycol]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("input tokens per question (log)"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].scatter([], [], marker="o", color="k", label="Sonnet"); axes[0].scatter([], [], marker="s", color="k", label="Qwen"); axes[0].legend(fontsize=8)
    fig.suptitle("Accuracy vs. what the answer costs to read (ingestion not included; hollow = partial coverage)", y=1.02)
    _save(fig, "f7_cost")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None); a = ap.parse_args()
    for name, fn in [("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4), ("f5", f5), ("f6", f6), ("f7", f7)]:
        if a.only and name not in a.only:
            continue
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            print(f"{name}: {type(e).__name__}: {e}")
