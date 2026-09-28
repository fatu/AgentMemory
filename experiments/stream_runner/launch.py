"""Concurrency launcher for the streams (step 13.0 / 13).

    python launch.py --jobs jobs.txt --concurrency 16            # run a job list
    python launch.py --jobs jobs.txt --concurrency 48 \
        --base-urls http://localhost:8000/v1 http://localhost:8002/v1 http://localhost:8003/v1
                                                                  # spread jobs round-robin over vLLM replicas
    python launch.py --plan --backbone qwen27b --seeds 0 1 2      # write jobs.txt for the E2 grid
    python launch.py --plan --backbone sonnet5 --seeds 0 \
        --streams mmlu_pro_philosophy gpqa_diamond aime24 aime25

Each job line is one `run.py` invocation (args after `run.py`). Jobs run as separate
processes, N at a time; vLLM batches their requests. Every job is resumable, so killing
the launcher and restarting it continues where it stopped. Progress lines report
aggregate throughput (rows/min) so the step-13.0 measurement is read straight off.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

STREAMS = ["mmlu_pro_philosophy", "mmlu_pro_economics", "mmlu_pro_engineering", "gpqa_diamond", "aime24", "aime25"]
SYSTEMS = ["none", "exprag", "dc", "mem0", "amem"]      # dc-frozen is appended by --frozen-from


def plan(args) -> list[str]:
    jobs = []
    for stream, seed, system in itertools.product(args.streams, args.seeds, args.systems):
        if seed > 0 and not stream.startswith("mmlu_pro"):
            continue                                  # only the MMLU-Pro streams have seeds 1/2
        jobs.append(f"--stream {stream} --seed {seed} --system {system} --backbone {args.backbone}"
                    + (f" --limit {args.limit}" if args.limit else ""))
    return jobs


def count_rows(job: str) -> int:
    kv = dict(zip(job.split()[0::2], job.split()[1::2]))
    run_id = f"{kv['--stream']}_s{kv['--seed']}_{kv['--system']}_{kv['--backbone']}"
    for name in ("results.csv", "result.csv"):
        p = Path("runs") / run_id / name
        if p.exists():
            return max(0, sum(1 for _ in p.open()) - 1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.txt")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--backbone", default="qwen27b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--streams", nargs="+", default=STREAMS)
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-urls", nargs="+", default=None,
                    help="vLLM replica base URLs; jobs are assigned round-robin (sets QWEN_BASE_URL per child)")
    a = ap.parse_args()

    if a.plan:
        jobs = plan(a)
        Path(a.jobs).write_text("\n".join(jobs) + "\n")
        print(f"wrote {len(jobs)} jobs → {a.jobs}")
        return

    jobs = [l.strip() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    Path("logs").mkdir(exist_ok=True)
    pending = list(jobs)
    running: dict[subprocess.Popen, str] = {}
    urls = a.base_urls or [os.environ.get("QWEN_BASE_URL", "")]
    launched = 0
    done_rows0 = sum(count_rows(j) for j in jobs)
    t0 = time.time()
    last_report = t0

    while pending or running:
        while pending and len(running) < a.concurrency:
            job = pending.pop(0)
            log = Path("logs") / (job.replace("--", "").replace(" ", "_") + ".log")
            env = dict(os.environ)
            if urls[0]:
                env["QWEN_BASE_URL"] = urls[launched % len(urls)]      # round-robin over replicas
            launched += 1
            p = subprocess.Popen([sys.executable, "run.py", *job.split()],
                                 stdout=log.open("a"), stderr=subprocess.STDOUT, env=env)
            running[p] = job
        for p in [p for p in running if p.poll() is not None]:
            job = running.pop(p)
            print(f"[{'ok' if p.returncode == 0 else 'FAIL rc=' + str(p.returncode)}] {job}", flush=True)
        if time.time() - last_report >= 60:
            rows = sum(count_rows(j) for j in jobs) - done_rows0
            mins = (time.time() - t0) / 60
            print(f"  {rows} rows in {mins:.1f} min → {rows / max(mins, 1e-9):.1f} rows/min "
                  f"| running {len(running)} | pending {len(pending)}", flush=True)
            last_report = time.time()
        time.sleep(5)

    rows = sum(count_rows(j) for j in jobs) - done_rows0
    mins = (time.time() - t0) / 60
    print(f"done: {rows} new rows in {mins:.1f} min → {rows / max(mins, 1e-9):.1f} rows/min at concurrency {a.concurrency}")


if __name__ == "__main__":
    main()
