#!/usr/bin/env bash
# Start a Hindsight server for one backbone (step 9b). Same-backbone rule: the server's LLM is
# the row's backbone; embeddings = the study's Qwen3-Embedding endpoint; reranker = Hindsight's
# local default (recorded in provenance). Embedded Postgres via pg0:// — no Docker.
#
#   python -m venv ~/hs && ~/hs/bin/pip install "hindsight-all==0.10.1"
#   ./hindsight_server.sh qwen27b    # → http://localhost:8888  (QWEN_BASE_URL / QWEN_MODEL / EMBED_BASE_URL from env)
#   ./hindsight_server.sh sonnet5    # → http://localhost:8889  (ANTHROPIC_API_KEY)
# Then in the runner shell:  export HINDSIGHT_URL=http://localhost:8888  (or :8889)
set -euo pipefail
BACKBONE=${1:?usage: hindsight_server.sh qwen27b|sonnet5}
HS_BIN=${HS_BIN:-$HOME/hs/bin}

export HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
export HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL="${EMBED_BASE_URL:?EMBED_BASE_URL}"
export HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS=1024
export HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY="${EMBED_API_KEY:-x}"
export HINDSIGHT_API_RERANKER_PROVIDER="${HINDSIGHT_API_RERANKER_PROVIDER:-local}"
export HINDSIGHT_API_LOG_LEVEL="${HINDSIGHT_API_LOG_LEVEL:-info}"
export HINDSIGHT_API_LLM_TIMEOUT="${HINDSIGHT_API_LLM_TIMEOUT:-300}"

case "$BACKBONE" in
  qwen27b)
    export HINDSIGHT_API_LLM_PROVIDER=openai
    export HINDSIGHT_API_LLM_BASE_URL="${QWEN_BASE_URL:?QWEN_BASE_URL}"
    export HINDSIGHT_API_LLM_MODEL="${QWEN_MODEL:?QWEN_MODEL}"
    export HINDSIGHT_API_LLM_API_KEY="${QWEN_API_KEY:-x}"
    export HINDSIGHT_API_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'   # memory calls never think
    export HINDSIGHT_API_PORT="${HINDSIGHT_API_PORT:-8888}"
    export HINDSIGHT_API_DATABASE_URL="${HINDSIGHT_API_DATABASE_URL:-pg0://hindsight-agentmemory-qwen}"
    ;;
  sonnet5)
    export HINDSIGHT_API_LLM_PROVIDER=anthropic
    export HINDSIGHT_API_LLM_MODEL="${SONNET_MODEL:-claude-sonnet-5}"
    export HINDSIGHT_API_LLM_API_KEY="${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY}"
    export HINDSIGHT_API_PORT="${HINDSIGHT_API_PORT:-8889}"
    export HINDSIGHT_API_DATABASE_URL="${HINDSIGHT_API_DATABASE_URL:-pg0://hindsight-agentmemory-sonnet}"
    ;;
  *) echo "unknown backbone $BACKBONE"; exit 1 ;;
esac

env | grep '^HINDSIGHT_API_' | sed 's/\(API_KEY=\).*/\1***/' | sort
exec "$HS_BIN/hindsight-api"
