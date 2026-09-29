#!/usr/bin/env bash
# One stream run inside the derived image. Usage:
#   ./run_in_container.sh swe-stream <tasks-dir> <repo>/experiments -- --harness harness_api --system none --backbone qwen27b --claude-bin claude --cc-model <served> --seed 0 --limit 5
# Host network so Claude Code reaches the model endpoint / vLLM / the embedder on this box; grading
# itself is whatever the harness does offline. Pass the model/embedder envs through.
set -euo pipefail
IMAGE=$1; TASKS=$2; EXP=$3; shift 3; [ "${1:-}" = "--" ] && shift
docker run --rm --network host \
  -e QWEN_BASE_URL -e QWEN_MODEL -e EMBED_BASE_URL -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL \
  -e CC_DISALLOWED_TOOLS -e FROZEN_SHEET_PATH -e THINK_BUDGET -e BENCH_PATH \
  -v "$TASKS":/tasks:ro -v "$EXP":/wp/experiments \
  "$IMAGE" python3 run_swe.py --tasks-dir /tasks "$@"
