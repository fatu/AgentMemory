"""Learning curves from the stream runs (step 14) — works on partial rows.

    python plot_curves.py                              # all runs/ → plots/*.png + plots/curves.csv
    python plot_curves.py --backbone qwen27b --streams mmlu_pro_philosophy gpqa_diamond
    python plot_curves.py --bucket 50 --min-rows 40

Per (stream, backbone, system): accuracy vs position — cumulative (the Evo-Memory protocol's
curve) and per-bucket (a moving window, less flattering to early luck) — averaged over seeds,
with a shaded min–max band across seeds when ≥ 2 seeds exist. Two derived curves when the
rows exist: dc − dc-frozen (what the sheet LEARNED during the stream) and dc-frozen − none
(what advice-in-context is worth without learning). Also a one-line-per-run table
(curves.csv): n, final cumulative accuracy, first-half vs second-half accuracy, the
half-to-half delta (the single number a learning curve reduces to), and tokens per step.
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM_ORDER = ["none", "exprag", "dc", "dc-frozen", "mem0", "amem", "ace"]
COLORS = {"none": "#7f7f7f", "exprag": "#1f77b4", "dc": "#d62728", "dc-frozen": "#ff9896",
          "mem0": "#2ca02c", "amem": "#9467bd", "ace": "#ff7f0e"}


def load_runs(runs_glob: str, min_rows: int) -> pd.DataFrame:
    frames = []
    for d in sorted(glob.glob(runs_glob)):
        p = next((Path(d) / n for n in ("results.csv", "result.csv") if (Path(d) / n).exists()), None)
        if p is None:
            continue
        df = pd.read_csv(p)
        if len(df) < min_rows or not {"stream", "system", "backbone", "seed", "position", "correct"} <= set(df.columns):
            continue
        df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(int)
        df["run"] = Path(d).name
        frames.append(df.sort_values("position"))
    if not frames:
        raise SystemExit(f"no runs with ≥ {min_rows} rows under {runs_glob}")
    return pd.concat(frames, ignore_index=True)


def curves(df: pd.DataFrame, bucket: int):
    """{(stream, backbone, system): {seed: (positions, cumulative, bucketed)}}."""
    out = defaultdict(dict)
    for (stream, backbone, system, seed), g in df.groupby(["stream", "backbone", "system", "seed"]):
        g = g.sort_values("position")
        pos = g["position"].to_numpy() + 1
        cum = np.cumsum(g["correct"].to_numpy()) / pos
        buck = pd.Series(g["correct"].to_numpy()).rolling(bucket, min_periods=max(10, bucket // 2)).mean().to_numpy()
        out[(stream, backbone, system)][seed] = (pos, cum, buck)
    return out


def mean_over_seeds(series: dict, kind: int):
    """Align seeds on the shortest common length; return positions, mean, min, max."""
    n = min(len(v[0]) for v in series.values())
    stack = np.vstack([v[kind][:n] for v in series.values()])
    return series[next(iter(series))][0][:n], np.nanmean(stack, 0), np.nanmin(stack, 0), np.nanmax(stack, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--streams", nargs="+", default=None)
    ap.add_argument("--bucket", type=int, default=50)
    ap.add_argument("--min-rows", type=int, default=20)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_runs(a.runs, a.min_rows)
    if a.backbone:
        df = df[df["backbone"] == a.backbone]
    if a.streams:
        df = df[df["stream"].isin(a.streams)]
    out = Path(a.out); out.mkdir(exist_ok=True)
    cv = curves(df, a.bucket)

    # ---- per-run table
    rows = []
    for run, g in df.groupby("run"):
        g = g.sort_values("position"); c = g["correct"].to_numpy(); n = len(c); h = n // 2
        rows.append({"run": run, "stream": g["stream"].iloc[0], "backbone": g["backbone"].iloc[0], "system": g["system"].iloc[0],
                     "seed": g["seed"].iloc[0], "n": n, "acc": round(c.mean(), 4),
                     "first_half": round(c[:h].mean(), 4) if h else np.nan, "second_half": round(c[h:].mean(), 4) if h else np.nan,
                     "delta_halves": round(c[h:].mean() - c[:h].mean(), 4) if h else np.nan,
                     "in_per_step": int(g["tokens_agent_in"].mean()) if "tokens_agent_in" in g else -1,
                     "mem_in_per_step": int(g["tokens_mem_in"].mean()) if "tokens_mem_in" in g else -1,
                     "block_tokens": int(g["block_tokens"].mean()) if "block_tokens" in g else -1,
                     "mem_size_end": int(g["memory_size_tokens"].iloc[-1]) if "memory_size_tokens" in g else -1})
    table = pd.DataFrame(rows).sort_values(["stream", "backbone", "system", "seed"])
    table.to_csv(out / "curves.csv", index=False)
    print(table.to_string(index=False))

    # ---- one figure per (stream, backbone): cumulative + bucketed, systems overlaid
    keys = sorted({(s, b) for (s, b, _) in cv})
    for stream, backbone in keys:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for kind, ax, title in ((1, axes[0], "cumulative accuracy"), (2, axes[1], f"accuracy, window {a.bucket}")):
            for system in SYSTEM_ORDER:
                series = cv.get((stream, backbone, system))
                if not series:
                    continue
                pos, m, lo, hi = mean_over_seeds(series, kind)
                ax.plot(pos, m, label=f"{system} ({len(series)} seed{'s' if len(series) > 1 else ''})", color=COLORS.get(system), lw=1.8)
                if len(series) > 1:
                    ax.fill_between(pos, lo, hi, color=COLORS.get(system), alpha=0.12, lw=0)
            ax.set_title(f"{stream} · {backbone} — {title}"); ax.set_xlabel("position in stream"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(out / f"{stream}_{backbone}.png", dpi=130); plt.close(fig)

        # ---- differences: learning vs advice-in-context
        have = {s for (st, b, s) in cv if st == stream and b == backbone}
        if {"dc", "dc-frozen", "none"} <= have:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            p_dc, m_dc, _, _ = mean_over_seeds(cv[(stream, backbone, "dc")], 1)
            p_fr, m_fr, _, _ = mean_over_seeds(cv[(stream, backbone, "dc-frozen")], 1)
            p_no, m_no, _, _ = mean_over_seeds(cv[(stream, backbone, "none")], 1)
            n = min(len(p_dc), len(p_fr), len(p_no))
            ax.plot(p_dc[:n], m_dc[:n] - m_fr[:n], label="dc − dc-frozen  (learned during the stream)", color="#d62728")
            ax.plot(p_no[:n], m_fr[:n] - m_no[:n], label="dc-frozen − none  (advice-in-context, no learning)", color="#ff9896")
            ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("position in stream"); ax.set_ylabel("Δ cumulative accuracy")
            ax.set_title(f"{stream} · {backbone}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(out / f"{stream}_{backbone}_delta.png", dpi=130); plt.close(fig)
    print(f"\nwrote {len(keys)} stream figures (+ delta figures where dc/dc-frozen/none exist) and curves.csv → {out}/")


if __name__ == "__main__":
    main()
