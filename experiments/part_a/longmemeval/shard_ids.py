"""Split a LongMemEval id set into N disjoint shards for parallel `run_lme.py --ids` runs.

    python shard_ids.py --all --n 12 --tag mem0_qwen27b        # 500 ids → shards/mem0_qwen27b_{00..11}.json
    python shard_ids.py --slice 150 --n 6 --tag amem_qwen27b   # the frozen 150 → 6 shards

Each shard runs as its own run id (`--run-id <tag>_sh<k>`) so no two processes append to one
results.csv; rollup.py merges `runs/<tag>_sh*` back into one table. Interleaved assignment
(id i → shard i mod N) keeps the per-type mix roughly equal across shards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lme_data import load_all

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--slice", type=int, choices=[20, 50, 100, 150])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ids = [e["question_id"] for e in load_all()] if a.all else json.load(open(HERE / "slices.json"))["slices"][str(a.slice)]
    out = HERE / "shards"; out.mkdir(exist_ok=True)
    for k in range(a.n):
        shard = ids[k::a.n]
        (out / f"{a.tag}_{k:02d}.json").write_text(json.dumps({"ids": shard, "tag": a.tag, "shard": k, "of": a.n}))
    print(f"{len(ids)} ids → {a.n} shards of ~{len(ids) // a.n} under {out}/{a.tag}_NN.json")


if __name__ == "__main__":
    main()
