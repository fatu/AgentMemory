"""The stream loop (Evo-Memory protocol, our implementation).

    python run.py --stream mmlu_pro_philosophy --seed 0 --system exprag --backbone sonnet5 --limit 20

For each item in the manifest's seeded order:
    block = memory.retrieve(question)         (<= CAP tokens)
    reply = backbone(system_prompt, memory block + question)   [thinking: Qwen only]
    correct = exact_match(parse(reply), target)
    memory.evolve(question, reply_answer, correct)             (binary feedback only)
    write one row

Outputs runs/<run_id>/{config.json, provenance.json, calls.jsonl, results.csv}.
Token columns are summed from the instrumented client's log by row_id × tag.
Resume: if results.csv exists, finished positions are skipped and the memory is rebuilt
by replaying evolve() from the CSV (systems whose evolve() is not replayable must
implement dump()/load(); ExpRAG and no-memory replay for free).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from client import LLMClient, THINK_BUDGET
from embed import Embedder, EMBED_MODEL
from manifests import load_manifest, PINS
from memory import make_memory, count_tokens, CAP
from scoring import SYSTEM_PROMPT, build_user, parse_answer, answer_kind, score

THINKING_BACKBONES = {"qwen27b"}           # user decision 2026-09-18: Qwen thinks, Sonnet does not
FIELDS = ["run_id", "stream", "backbone", "system", "seed", "position", "item_id", "pred",
          "target", "correct", "cum_acc", "tokens_agent_in", "tokens_agent_out",
          "tokens_agent_cache_read", "tokens_agent_cache_write",
          "tokens_mem_in", "tokens_mem_out", "tokens_mem_cache_read", "tokens_mem_cache_write",
          "truncated", "mem_truncated", "latency_s", "memory_size_tokens", "block_tokens"]
# tokens_*_in = ALL prompt tokens processed (uncached + cache_read + cache_write);
# tokens_*_cache_read = the part served from cache (priced at 0.1x). Anthropic's raw
# `input_tokens` excludes cached tokens, which made the pilot's ExpRAG/Sonnet rows read
# "84 tokens in" with a 1.4K memory block — fixed 2026-09-20.


def _tot(c):
    return c["in"] + c.get("cache_read", 0) + c.get("cache_write", 0)


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def provenance(args, client, embedder) -> dict:
    prov = {
        "backbone": args.backbone, "model": client.model,
        "thinking": args.backbone in THINKING_BACKBONES, "think_budget": THINK_BUDGET,
        "temperature": "0.6/top_p 0.95 (thinking)" if args.backbone in THINKING_BACKBONES else 0.0,
        "embedder": embedder.model_name, "embedder_backend": embedder.backend,
        "embedder_base_url": embedder.base_url, "cap_tokens": CAP, "dataset_pins": json.loads(PINS.read_text()),
        "runner_git": git_rev(), "python": sys.version.split()[0], "ts": time.time(),
    }
    try:
        from huggingface_hub import HfApi
        prov["embedder_revision"] = HfApi().model_info(EMBED_MODEL).sha   # HF revision of the weights
    except Exception:
        prov["embedder_revision"] = "unknown"
    if args.backbone == "qwen27b":
        import os
        prov["vllm_base_url"] = os.environ.get("QWEN_BASE_URL")
        prov["gpu"] = os.environ.get("GPU_NAME", "RTX 6000 PRO (set GPU_NAME to override)")
        try:
            import requests
            prov["vllm_version"] = requests.get(os.environ["QWEN_BASE_URL"].rsplit("/v1", 1)[0] + "/version", timeout=5).json()
        except Exception:
            prov["vllm_version"] = "unknown"
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", required=True)
    ap.add_argument("--backbone", choices=["sonnet5", "qwen27b"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"{args.stream}_s{args.seed}_{args.system}_{args.backbone}"
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    manifest, items = load_manifest(args.stream, args.seed)
    order = manifest["order"][: args.limit] if args.limit else manifest["order"]
    kind = answer_kind(args.stream)

    client = LLMClient(args.backbone, run_id=run_id, log_path=out / "calls.jsonl")
    embedder = Embedder()
    embedder.embed_stream(args.stream, items)            # every item once, regardless of system
    memory = make_memory(args.system, embedder=embedder, client=client,
                         state_path=out / "memory_state.json")
    thinking = args.backbone in THINKING_BACKBONES

    (out / "config.json").write_text(json.dumps(vars(args) | {"run_id": run_id, "n": len(order)}, indent=2))
    (out / "provenance.json").write_text(json.dumps(provenance(args, client, embedder), indent=2))

    # ---- resume: replay finished rows into memory
    results_path = out / "results.csv"
    done: dict[int, dict] = {}
    if results_path.exists():
        with results_path.open() as f:
            for r in csv.DictReader(f):
                done[int(r["position"])] = r
        if hasattr(memory, "load") and memory.load():          # LLM-updated memories: exact state
            print(f"resumed {len(done)} rows (memory state loaded)")
        else:                                                    # replayable memories: rebuild
            for pos in sorted(done):
                r = done[pos]
                memory.evolve(items[r["item_id"]]["input_text"], r["pred"] or "", r["correct"] == "True")
            print(f"resumed {len(done)} rows")
    n_correct = sum(1 for r in done.values() if r["correct"] == "True")

    with results_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for pos, item_id in enumerate(order):
            if pos in done:
                continue
            item = items[item_id]
            row_id = f"{pos}:{item_id}"
            client.current_row_id = row_id          # memory-side calls (curator etc.) attribute to this row
            t0 = time.perf_counter()

            block = memory.retrieve(item["input_text"])
            block_tokens = count_tokens(block)
            assert block_tokens <= CAP, (pos, block_tokens)

            reply = client.complete(SYSTEM_PROMPT, build_user(block, item["input_text"]),
                                    tag="agent", row_id=row_id, thinking=thinking,
                                    max_tokens=2048)          # non-thinking cap; pilot hit 1024 once
            pred = parse_answer(reply, kind)
            correct = score(pred, item["target"])
            memory.evolve(item["input_text"], pred or "", correct)   # binary feedback, no reference

            calls = [c for c in client.calls if c["row_id"] == row_id]
            agent = [c for c in calls if c["tag"] == "agent"]
            mem = [c for c in calls if c["tag"] == "memory"]
            n_correct += int(correct)
            row = {
                "run_id": run_id, "stream": args.stream, "backbone": args.backbone,
                "system": args.system, "seed": args.seed, "position": pos, "item_id": item_id,
                "pred": pred, "target": item["target"], "correct": correct,
                "cum_acc": round(n_correct / (pos + 1), 4),
                "tokens_agent_in": sum(_tot(c) for c in agent), "tokens_agent_out": sum(c["out"] for c in agent),
                "tokens_agent_cache_read": sum(c.get("cache_read", 0) for c in agent),
                "tokens_agent_cache_write": sum(c.get("cache_write", 0) for c in agent),
                "tokens_mem_in": sum(_tot(c) for c in mem), "tokens_mem_out": sum(c["out"] for c in mem),
                "tokens_mem_cache_read": sum(c.get("cache_read", 0) for c in mem),
                "tokens_mem_cache_write": sum(c.get("cache_write", 0) for c in mem),
                "truncated": any(c.get("truncated") for c in agent),
                "mem_truncated": any(c.get("truncated") for c in mem),
                "latency_s": round(time.perf_counter() - t0, 3),
                "memory_size_tokens": memory.size_tokens(), "block_tokens": block_tokens,
            }
            w.writerow(row); f.flush()
            print(f"[{pos+1}/{len(order)}] {item_id} pred={pred} target={item['target']} "
                  f"{'✓' if correct else '✗'} cum={row['cum_acc']:.3f} "
                  f"in={row['tokens_agent_in']} out={row['tokens_agent_out']} block={block_tokens}")

    print(f"\ndone: {run_id}  acc={n_correct/len(order):.3f}  rows={len(order)}  → {results_path}")


if __name__ == "__main__":
    main()
