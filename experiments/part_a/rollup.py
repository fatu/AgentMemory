"""Roll the Part A runs up into the E1 tables (step 14): one row per (benchmark, system, backbone).

    python rollup.py                 # all three benchmarks → tables/*.csv + markdown on stdout
    python rollup.py --only beam

LoCoMo   runs/<conv>_<system>_<backbone>/          → per category: official F1 (micro over rows) and judge;
                                                     cat 5 official only; all-5 micro; n convs / n rows
LongMemEval runs/<system>_<backbone>[_sN][_shNN]/  → shards merged; per type; task-avg; overall; abs acc;
                                                     false-abstention; ingest tokens / question
BEAM     runs/<group>-<conv>_<system>_<backbone>/  → per ability mean over questions (= the official
                                                     per-group aggregation), 10-ability average; n convs
Partial runs roll up as they are — n tells you how much is in.
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"


def _read(d: Path) -> pd.DataFrame | None:
    p = next((d / n for n in ("results.csv", "result.csv") if (d / n).exists()), None)
    return pd.read_csv(p) if p else None


def locomo() -> pd.DataFrame:
    rows = []
    groups = defaultdict(list)
    for d in sorted(glob.glob(str(HERE / "locomo/runs/conv-*"))):
        df = _read(Path(d))
        if df is None or df.empty:
            continue
        groups[(df["system"].iloc[0], df["backbone"].iloc[0])].append(df)
    for (system, backbone), dfs in sorted(groups.items()):
        df = pd.concat(dfs)
        r = {"benchmark": "locomo", "system": system, "backbone": backbone, "convs": len(dfs), "n": len(df)}
        for cat, name in ((4, "single"), (1, "multi"), (2, "temporal"), (3, "open"), (5, "cat5")):
            g = df[df["category"] == cat]
            r[f"{name}_off"] = round(g["score_official"].mean(), 3) if len(g) else None
            if cat != 5:
                j = pd.to_numeric(g["score_judge"], errors="coerce")
                r[f"{name}_judge"] = round(j.mean(), 3) if j.notna().any() else None
        c14 = df[df["category"] != 5]
        r["cat1-4_off"] = round(c14["score_official"].mean(), 3)
        j = pd.to_numeric(c14["score_judge"], errors="coerce"); r["cat1-4_judge"] = round(j.mean(), 3) if j.notna().any() else None
        r["all_off"] = round(df["score_official"].mean(), 3)
        r["agent_in"] = int(df["tokens_agent_in"].mean()); r["s_per_q"] = round(df["latency_s"].mean(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def longmemeval() -> pd.DataFrame:
    rows = []
    groups = defaultdict(list)
    for d in sorted(glob.glob(str(HERE / "longmemeval/runs/*"))):
        name = Path(d).name
        if name.startswith("_") or name.endswith("_smoke"):
            continue
        df = _read(Path(d))
        if df is None or df.empty:
            continue
        key = re.sub(r"_sh\d+$", "", name)                    # merge shards
        groups[key].append(df)
    types = ["single-session-user", "single-session-assistant", "single-session-preference", "multi-session", "temporal-reasoning", "knowledge-update"]
    for key, dfs in sorted(groups.items()):
        df = pd.concat(dfs).drop_duplicates("question_id", keep="last")
        j = pd.to_numeric(df["score_judge"], errors="coerce")
        r = {"benchmark": "longmemeval", "run": key, "system": df["system"].iloc[0], "backbone": df["backbone"].iloc[0], "n": len(df)}
        accs = []
        for t in types:
            g = j[df["question_type"] == t]
            r[t.replace("single-session-", "ss-").replace("-reasoning", "").replace("-update", "-upd")] = round(g.mean(), 3) if len(g) else None
            if len(g):
                accs.append(g.mean())
        r["task_avg"] = round(sum(accs) / len(accs), 3) if accs else None
        r["overall"] = round(j.mean(), 3) if j.notna().any() else None
        ab = df["is_abs"].astype(str) == "1"
        r["abs_acc"] = round(j[ab].mean(), 3) if ab.any() else None
        fa = pd.to_numeric(df.loc[~ab, "false_abstention"], errors="coerce")
        r["false_abst"] = round(fa.fillna(0).mean(), 3) if (~ab).any() else None
        r["ingest_in/q"] = int(df["tokens_ingest_in"].mean()); r["ingest_s/q"] = round(df["ingest_s"].mean(), 1)
        r["agent_in"] = int(df["tokens_agent_in"].mean()); r["truncated"] = int(df["truncated"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def beam() -> pd.DataFrame:
    rows = []
    groups = defaultdict(list)
    for d in sorted(glob.glob(str(HERE / "beam/runs/*-*_*_*"))):
        df = _read(Path(d))
        if df is None or df.empty:
            continue
        groups[(df["group"].iloc[0], df["system"].iloc[0], df["backbone"].iloc[0])].append(df)
    import sys
    sys.path.insert(0, str(HERE / "beam"))
    from beam_data import ABILITIES  # noqa: E402
    for (group, system, backbone), dfs in sorted(groups.items()):
        df = pd.concat(dfs)
        sc = pd.to_numeric(df["score"], errors="coerce")
        r = {"benchmark": f"beam-{group}", "system": system, "backbone": backbone, "convs": len(dfs), "n": len(df),
             "judged": int(sc.notna().sum())}
        means = []
        for ab in ABILITIES:
            g = sc[df["ability"] == ab].dropna()
            r[ab[:12]] = round(g.mean(), 3) if len(g) else None
            if len(g):
                means.append(g.mean())
        r["average"] = round(sum(means) / len(means), 3) if means else None
        r["agent_in"] = int(df["tokens_agent_in"].mean()); r["truncated"] = int(df["truncated"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["locomo", "longmemeval", "beam"], default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for name, fn in (("locomo", locomo), ("longmemeval", longmemeval), ("beam", beam)):
        if a.only and a.only != name:
            continue
        try:
            t = fn()
        except Exception as e:                           # noqa: BLE001
            print(f"== {name}: {e!r}"); continue
        print(f"\n== {name} ==")
        if t.empty:
            print("(no runs)"); continue
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(t.to_string(index=False))
    print(f"\ntables → {OUT}/")


if __name__ == "__main__":
    main()
