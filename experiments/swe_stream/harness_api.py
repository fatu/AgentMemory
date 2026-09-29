"""Harness contract for the execution stream + a minimal example harness.

`run_swe.py --harness <module>` imports a module that provides:

    NAME: str                                   # benchmark label for provenance / rows
    STRATA: list[str]                           # task["meta"] keys reported as CSV columns and summary groups
    load_tasks(tasks_dir) -> list[dict]         # {task_id, prompt, dir, meta: {...}, raw: ...}
    initial_message(task, opts: dict) -> str    # the agent's first message (the benchmark's own composition)
    prepare_workspace(task, dest) -> Path       # fresh copy of the agent's starting state; NEVER the tests/solution
    grade(task, work) -> dict                   # {resolved: bool, gate_failed: str, tests_passed, expected_tests,
                                                #  log_tail: str, raw: dict}

Only `resolved` is fed back to the memory as the binary outcome; `gate_failed` and `log_tail` are
what an engineer would see (build error, failing tests) and go to the reflector as environment
feedback. Reference answers never reach the memory.

The example below implements the contract for a simple layout — the smallest thing that lets
someone run the stream on their own tasks:

    tasks/<task_id>/task.json     {"task_id": ..., "prompt": ..., "meta": {"difficulty": ..., ...}}
    tasks/<task_id>/workspace/    the agent's starting directory
    tasks/<task_id>/grade.sh      run with the finished workspace as $1 (cwd = task dir);
                                  exit 0 = resolved; stdout/stderr tail = log

A benchmark with its own grader replaces this module (same four functions); keep such adapters
out of public repos when they describe a private benchmark.
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path

NAME = "example"
STRATA = ["difficulty"]


def load(name: str):
    """`run_swe.py --harness <name>` → the module implementing the contract (this one by default)."""
    mod = importlib.import_module(name)
    for fn in ("load_tasks", "initial_message", "prepare_workspace", "grade"):
        if not hasattr(mod, fn):
            raise AttributeError(f"harness {name!r} lacks {fn}()")
    mod.NAME = getattr(mod, "NAME", name)
    mod.STRATA = list(getattr(mod, "STRATA", []))
    return mod


# ------------------------------------------------------------------ example implementation
def load_tasks(tasks_dir: str | Path) -> list[dict]:
    out = []
    for d in sorted(Path(tasks_dir).iterdir()):
        tj = d / "task.json"
        if not tj.exists():
            continue
        t = json.load(open(tj))
        out.append({"task_id": t.get("task_id", d.name), "prompt": t["prompt"], "dir": d,
                    "meta": dict(t.get("meta", {})), "raw": t})
    return out


def initial_message(task: dict, opts: dict) -> str:
    return task["prompt"].strip()


def prepare_workspace(task: dict, dest: str | Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(task["dir"] / "workspace", dest)
    return dest


def grade(task: dict, work: str | Path) -> dict:
    script = task["dir"] / "grade.sh"
    if not script.exists():
        raise RuntimeError(f"{script} missing — the example harness grades with tasks/<id>/grade.sh")
    p = subprocess.run(["bash", str(script), str(work)], cwd=task["dir"], capture_output=True, text=True, timeout=1800)
    log = (p.stdout + p.stderr)[-1200:]
    return {"resolved": p.returncode == 0, "gate_failed": "" if p.returncode == 0 else f"exit {p.returncode}",
            "tests_passed": None, "expected_tests": None, "log_tail": log, "raw": {"returncode": p.returncode}}
