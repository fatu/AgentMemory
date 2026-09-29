"""The execution stream: the Evo-Memory protocol over a coding benchmark, with Claude Code as
the agent and our memory systems as the only carrier of state across episodes.

    python run_swe.py --tasks-dir tasks/ --harness harness_api --system none --backbone qwen27b \
        --claude-bin claude --cc-model <served-name> --seed 0 --limit 5
    python run_swe.py --tasks-dir … --harness my_harness --system ace --backbone qwen27b …

Per episode (seeded order over the task ids):
    block  = memory.retrieve(brief)                       (≤ 4,096 tokens) → memory.md
    agent  = claude -p <initial message> --bare --append-system-prompt-file memory.md
             --disallowedTools … --max-turns N            (in a fresh copy of the task workspace)
    verdict= harness.grade(): resolved ∈ {0,1} + the failing stage and a log tail
    memory.evolve(brief, trajectory_summary, resolved, trajectory=…, feedback=<stage + log tail>)
    one row.  Resume by position; memory state via load()/dump() or replay.
The harness (task loading, prompt composition, workspace, grading) is a module implementing
the contract in harness_api.py; `--opts k=v` are passed to its initial_message(). Limits:
--max-turns and --timeout are enforced; per-turn / per-episode token caps are reported only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM_RUNNER = HERE.parent / "stream_runner"
sys.path.insert(0, str(STREAM_RUNNER))
sys.path.insert(0, str(HERE))

from client import LLMClient                                                # noqa: E402
from embed import Embedder                                                  # noqa: E402
from memory import make_memory, count_tokens, CAP                           # noqa: E402
from harness_api import load as load_harness                                # noqa: E402
from claude_cc import claude_version, run_episode, trajectory_summary, ALLOWED_TOOLS, DISALLOWED_TOOLS   # noqa: E402

BASE_FIELDS = ["run_id", "system", "backbone", "seed", "position", "task_id"]
TAIL_FIELDS = ["opts", "resolved", "gate_failed", "tests_passed", "expected_tests",
               "cum_resolved", "num_turns", "n_tool_calls", "n_tool_errors", "stop_reason", "timed_out", "wall_s", "grade_s",
               "cc_model", "cc_in", "cc_out", "cc_cache_read", "cc_cache_write",
               "tokens_mem_in", "tokens_mem_out", "tokens_mem_cache_read", "block_tokens", "memory_size_tokens",
               "files_edited", "pred", "traj_summary", "feedback", "latency_s"]
TRUNC = 6000


def _tot(c):
    return c["in"] + c.get("cache_read", 0) + c.get("cache_write", 0)


def git_rev(path: Path | None = None) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path or HERE), "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def feedback_text(v: dict) -> str:
    gates = v.get("gates") or {}
    stages = " ".join(f"{g}={'pass' if ok else 'FAIL' if ok is False else '-'}" for g, ok in gates.items())
    tests = f"; tests {v['tests_passed']}/{v['expected_tests']}" if v.get("tests_passed") is not None else ""
    head = f"Result: {'RESOLVED' if v['resolved'] else 'NOT RESOLVED'}" + (f" ({stages}{tests})" if stages or tests else "")
    if v.get("gate_failed"):
        head += f"; failed at {v['gate_failed']}"
    return head + ("\nGrader log tail:\n" + v["log_tail"] if v.get("log_tail") else "")


def parse_opts(items: list[str]) -> dict:
    out = {}
    for kv in items or []:
        k, _, v = kv.partition("=")
        out[k] = {"1": True, "true": True, "0": False, "false": False}.get(v.lower(), v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--harness", default="harness_api", help="module implementing the harness contract (see harness_api.py)")
    ap.add_argument("--opts", nargs="*", default=[], help="k=v options passed to the harness's initial_message()")
    ap.add_argument("--system", required=True, help="none | exprag | dc | dc-frozen | ace | mem0 | amem")
    ap.add_argument("--backbone", choices=["sonnet5", "qwen27b"], required=True, help="memory-side client + row label")
    ap.add_argument("--claude-bin", default="claude", help="the Claude Code binary/wrapper (e.g. a vLLM shim wrapper)")
    ap.add_argument("--cc-model", default=None, help="--model for Claude Code (served name on the open-model path; default claude-sonnet-5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=2700)
    ap.add_argument("--keep-work", action="store_true", help="keep each episode's workspace after grading")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    harness = load_harness(a.harness)
    opts = parse_opts(a.opts)
    cc_model = a.cc_model or ("claude-sonnet-5" if a.backbone == "sonnet5" else os.environ.get("QWEN_MODEL", ""))
    assert cc_model, "--cc-model (served model name) is required on the open-model path"
    fields = BASE_FIELDS + list(harness.STRATA) + TAIL_FIELDS

    run_id = a.run_id or f"swe_s{a.seed}_{a.system}_{a.backbone}"
    out = HERE / "runs" / run_id
    (out / "events").mkdir(parents=True, exist_ok=True)

    tasks = {t["task_id"]: t for t in harness.load_tasks(a.tasks_dir)}
    ids = sorted(tasks)
    random.Random(a.seed).shuffle(ids)
    if a.limit:
        ids = ids[: a.limit]
    (out / "order.json").write_text(json.dumps({"seed": a.seed, "task_ids": ids}, indent=1))

    client = LLMClient(a.backbone, run_id=run_id, log_path=out / "calls.jsonl")   # memory-side calls only
    embedder = Embedder() if a.system in ("exprag", "mem0", "amem") else None
    memory = make_memory(a.system, embedder=embedder, client=client, state_path=out / "memory_state.json")

    (out / "provenance.json").write_text(json.dumps({
        "benchmark": harness.NAME, "harness_module": a.harness, "harness_opts": opts,
        "our_rev": git_rev(), "agent": "claude-code", "claude_version": claude_version(a.claude_bin), "claude_bin": a.claude_bin,
        "cc_model_alias": cc_model, "served_model": (os.environ.get("QWEN_MODEL") or client.model) if a.backbone == "qwen27b" else client.model,
        "flags": ["--bare", "--output-format stream-json", f"--max-turns {a.max_turns}", "--disallowedTools …", "--dangerously-skip-permissions"],
        "allowed_tools": list(ALLOWED_TOOLS), "disallowed_tools": DISALLOWED_TOOLS.split(","),
        "limits": {"max_turns": a.max_turns, "episode_timeout_s": a.timeout, "note": "token caps per turn/episode are reported, not enforced"},
        "system": a.system, "backbone": a.backbone, "memory_model": client.model, "memory_thinking": False, "cap_tokens": CAP,
        "embedder": (embedder.model_name if embedder else None), "seed": a.seed, "n_episodes": len(ids), "ts": time.time(),
    }, indent=2))

    # ---- resume
    results_path = out / "results.csv"
    done: dict[int, dict] = {}
    if results_path.exists():
        with results_path.open() as f:
            for r in csv.DictReader(f):
                done[int(r["position"])] = r
        if hasattr(memory, "load") and memory.load():
            print(f"resumed {len(done)} rows (memory state loaded)")
        else:
            for pos in sorted(done):
                r = done[pos]
                memory.evolve(tasks[r["task_id"]]["prompt"], r["traj_summary"], r["resolved"] == "1")
            print(f"resumed {len(done)} rows (memory replayed)")
    n_resolved = sum(1 for r in done.values() if r["resolved"] == "1")

    env = {"DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    with results_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        for pos, task_id in enumerate(ids):
            if pos in done:
                continue
            t = tasks[task_id]
            row_id = f"{pos}:{task_id}"
            client.current_row_id = row_id
            t0 = time.perf_counter()

            block = memory.retrieve(t["prompt"])
            block_tokens = count_tokens(block)
            assert block_tokens <= CAP, (pos, block_tokens)
            memory_md = None
            if block.strip():
                memory_md = out / "events" / f"{pos:04d}_memory.md"
                memory_md.write_text(block)

            work = harness.prepare_workspace(t, out / "work" / f"{pos:04d}_{task_id}")
            ep = run_episode(work, harness.initial_message(t, opts), model=cc_model, memory_md=memory_md,
                             max_turns=a.max_turns, timeout_s=a.timeout, env=env, claude_bin=a.claude_bin,
                             events_path=out / "events" / f"{pos:04d}_{task_id}.jsonl")
            t_g = time.perf_counter()
            v = harness.grade(t, work)
            grade_s = round(time.perf_counter() - t_g, 1)
            (out / "events" / f"{pos:04d}_{task_id}.grade.json").write_text(json.dumps(v.get("raw", {}), indent=1, default=str))

            traj = trajectory_summary(ep)
            fb = feedback_text(v)
            try:
                memory.evolve(t["prompt"], traj, v["resolved"], trajectory=traj, feedback=fb)
            except TypeError:
                memory.evolve(t["prompt"], traj, v["resolved"])

            mem = [c for c in client.calls if c["row_id"] == row_id and c["tag"] == "memory"]
            n_resolved += int(v["resolved"])
            row = {
                "run_id": run_id, "system": a.system, "backbone": a.backbone, "seed": a.seed, "position": pos, "task_id": task_id,
                **{k: t.get("meta", {}).get(k, "") for k in harness.STRATA},
                "opts": json.dumps(opts, sort_keys=True), "resolved": int(v["resolved"]), "gate_failed": v.get("gate_failed", ""),
                "tests_passed": v.get("tests_passed"), "expected_tests": v.get("expected_tests"),
                "cum_resolved": round(n_resolved / (pos + 1), 4),
                "num_turns": ep["num_turns"], "n_tool_calls": ep["n_tool_calls"], "n_tool_errors": ep["n_tool_errors"],
                "stop_reason": ep["stop_reason"], "timed_out": int(ep["timed_out"]), "wall_s": ep["wall_s"], "grade_s": grade_s,
                "cc_model": cc_model, "cc_in": ep["usage"]["in"] + ep["usage"]["cache_read"] + ep["usage"]["cache_write"],
                "cc_out": ep["usage"]["out"], "cc_cache_read": ep["usage"]["cache_read"], "cc_cache_write": ep["usage"]["cache_write"],
                "tokens_mem_in": sum(_tot(c) for c in mem), "tokens_mem_out": sum(c["out"] for c in mem),
                "tokens_mem_cache_read": sum(c.get("cache_read", 0) for c in mem),
                "block_tokens": block_tokens, "memory_size_tokens": memory.size_tokens(),
                "files_edited": ";".join(ep["files_edited"]), "pred": (ep["result_text"] or "")[:300],
                "traj_summary": traj[:TRUNC], "feedback": fb[:TRUNC], "latency_s": round(time.perf_counter() - t0, 1),
            }
            w.writerow(row); f.flush()
            if not a.keep_work:
                import shutil; shutil.rmtree(work, ignore_errors=True)
            strata = "/".join(str(row.get(k, "")) for k in harness.STRATA[:2])
            print(f"[{pos+1}/{len(ids)}] {task_id[:32]:32s} {strata:14s} "
                  f"{'RESOLVED' if v['resolved'] else 'fail@' + (v.get('gate_failed') or '?'):12s} cum={row['cum_resolved']:.3f} "
                  f"turns={ep['num_turns']} calls={ep['n_tool_calls']} wall={ep['wall_s']}s cc_in={row['cc_in']} "
                  f"mem_in={row['tokens_mem_in']} blk={block_tokens} memsize={row['memory_size_tokens']}", flush=True)

    summarize(list(csv.DictReader(results_path.open())), run_id, harness.STRATA)


def summarize(rows, run_id, strata):
    import collections
    n = len(rows)
    print(f"\n== {run_id} == {n} episodes")
    if not n:
        return
    res = [int(r["resolved"]) for r in rows]
    print(f"resolved {sum(res)}/{n} = {sum(res)/n:.3f}  (first half {sum(res[:n//2])/max(n//2,1):.3f} · second half {sum(res[n//2:])/max(n-n//2,1):.3f})")
    for key in strata:
        by = collections.defaultdict(list)
        for r in rows:
            by[r.get(key, "")].append(int(r["resolved"]))
        print(f"  by {key}: " + " · ".join(f"{k or '?'} {sum(v)}/{len(v)}" for k, v in sorted(by.items())))
    gates = collections.Counter(r["gate_failed"] or "none" for r in rows if r["resolved"] != "1")
    print(f"  failures by stage: {dict(gates)}")
    f = lambda k: sum(float(r[k] or 0) for r in rows) / n   # noqa: E731
    print(f"  per episode: turns {f('num_turns'):.1f} · tool calls {f('n_tool_calls'):.1f} · wall {f('wall_s'):.0f}s · grade {f('grade_s'):.0f}s · "
          f"cc_in {f('cc_in'):.0f} (cache_read {f('cc_cache_read'):.0f}) · cc_out {f('cc_out'):.0f} · mem_in {f('tokens_mem_in'):.0f} · "
          f"block {f('block_tokens'):.0f} · memsize end {rows[-1]['memory_size_tokens']} · timeouts {sum(int(r['timed_out']) for r in rows)}")


if __name__ == "__main__":
    main()
