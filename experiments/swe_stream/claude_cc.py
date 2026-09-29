"""Claude Code (headless) as the coding agent — one episode = one `claude -p` run in the task's
workspace, memory injected with --append-system-prompt-file, trajectory read from stream-json.

Facts this relies on (10c.0, 2026-09-23/28): the injection seam works on the Qwen path; the
result event carries usage / modelUsage / num_turns / stop_reason / duration_ms; on the Qwen
path modelUsage is keyed by the alias, so the served model name is recorded by the caller;
`--bare` skips hooks, CLAUDE.md and Claude Code's own auto-memory — required, otherwise the
harness itself would carry state across episodes and confound the memory system under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# The four tools a minimal coding scaffold allows; everything else Claude Code ships is disallowed
# by name (--allowedTools alone does not remove the built-ins). Confirm the list against the
# installed version; override with CC_DISALLOWED_TOOLS.
ALLOWED_TOOLS = ("Bash", "Read", "Edit", "Write")
DISALLOWED_TOOLS = os.environ.get("CC_DISALLOWED_TOOLS", ",".join([
    "Agent", "AskUserQuestion", "BashOutput", "KillShell", "Glob", "Grep", "NotebookEdit", "WebFetch", "WebSearch",
    "TodoWrite", "Skill", "SlashCommand", "EnterPlanMode", "ExitPlanMode", "ListMcpResources", "ReadMcpResource",
    "TaskCreate", "TaskGet", "TaskList", "TaskOutput", "TaskStop", "TaskUpdate", "ToolSearch", "LSP", "Monitor",
]))


def claude_version(claude_bin: str = "claude") -> str:
    try:
        return subprocess.check_output([claude_bin, "--version"], text=True, timeout=30).strip()
    except Exception as e:                                   # noqa: BLE001
        return f"unknown ({e!r})"


def run_episode(workdir: Path, prompt: str, *, model: str, memory_md: Path | None, max_turns: int,
                timeout_s: int, events_path: Path, env: dict | None = None, claude_bin: str = "claude") -> dict:
    cmd = [claude_bin, "-p", prompt, "--bare", "--model", model, "--output-format", "stream-json", "--verbose",
           "--max-turns", str(max_turns), "--disallowedTools", DISALLOWED_TOOLS, "--dangerously-skip-permissions"]
    if memory_md is not None:
        cmd += ["--append-system-prompt-file", str(memory_md)]
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s,
                              env={**os.environ, **(env or {})})
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, rc, timed_out = (e.stdout or ""), (e.stderr or ""), -9, True
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
    wall = time.perf_counter() - t0
    events_path.write_text(stdout)
    if stderr.strip():
        events_path.with_suffix(".stderr").write_text(stderr)
    return parse_events(stdout, wall_s=wall, rc=rc, timed_out=timed_out)


def parse_events(stdout: str, *, wall_s: float, rc: int, timed_out: bool) -> dict:
    """Trajectory + usage from the stream-json lines."""
    tool_calls, results_by_id, final_text, result = [], {}, "", None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        t = ev.get("type")
        if t == "assistant":
            for b in (ev.get("message") or {}).get("content", []) or []:
                if b.get("type") == "tool_use":
                    tool_calls.append({"id": b.get("id"), "name": b.get("name"), "input": b.get("input", {})})
                elif b.get("type") == "text" and b.get("text"):
                    final_text = b["text"]
        elif t == "user":
            for b in (ev.get("message") or {}).get("content", []) or []:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
                    results_by_id[b.get("tool_use_id")] = {"is_error": bool(b.get("is_error")), "text": text or ""}
        elif t == "result":
            result = ev
    usage = (result or {}).get("usage") or {}
    edited = sorted({str(c["input"].get("file_path")) for c in tool_calls
                     if c["name"] in ("Edit", "Write") and c["input"].get("file_path")})
    steps = []
    for i, c in enumerate(tool_calls, 1):
        r = results_by_id.get(c["id"], {})
        if c["name"] == "Bash":
            what = str(c["input"].get("command", ""))[:160]
        else:
            what = str(c["input"].get("file_path", ""))
        tail = (r.get("text") or "").strip().replace("\n", " ")[-160:]
        steps.append(f"{i}. {c['name']}: {what}" + (f" → ERROR: {tail}" if r.get("is_error") else (f" → {tail}" if c['name'] == 'Bash' and tail else "")))
    return {
        "result_text": (result or {}).get("result") or final_text or "",
        "num_turns": (result or {}).get("num_turns"), "stop_reason": (result or {}).get("stop_reason") or (result or {}).get("subtype"),
        "duration_ms": (result or {}).get("duration_ms"), "wall_s": round(wall_s, 1), "rc": rc, "timed_out": timed_out,
        "is_error": bool((result or {}).get("is_error")) or rc != 0,
        "usage": {"in": usage.get("input_tokens", 0) or 0, "out": usage.get("output_tokens", 0) or 0,
                  "cache_read": usage.get("cache_read_input_tokens", 0) or 0,
                  "cache_write": usage.get("cache_creation_input_tokens", 0) or 0},
        "model_usage": (result or {}).get("modelUsage") or {},
        "n_tool_calls": len(tool_calls), "n_tool_errors": sum(1 for r in results_by_id.values() if r["is_error"]),
        "files_edited": edited, "steps": steps,
    }


def trajectory_summary(ep: dict, *, max_chars: int = 6000) -> str:
    """Compact text for evolve() / the reflector: what the agent did, in order, with errors."""
    head = [f"Tool calls: {ep['n_tool_calls']} ({ep['n_tool_errors']} errors); turns: {ep['num_turns']}; "
            f"stop: {ep['stop_reason']}; files edited: {', '.join(ep['files_edited']) or 'none'}"]
    body = "\n".join(ep["steps"])
    text = "\n".join(head) + "\n" + body
    if len(text) > max_chars:                       # keep the beginning and the end of the episode
        text = text[: max_chars // 2] + "\n…\n" + text[-max_chars // 2:]
    final = (ep.get("result_text") or "").strip()
    if final:
        text += "\nFinal message: " + final[:600]
    return text
