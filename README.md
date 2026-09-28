# AgentMemory — a hands-on study of self-evolving memory for LLM agents

Code for a self-study that measures what memory systems actually deliver for LLM agents,
in two parts:

- **Part A — memory itself.** Standard long-term-memory benchmarks (LoCoMo, LongMemEval-S,
  BEAM) run with their *official* data, prompts and scoring, on our own execution layer, so
  that full-context, BM25 retrieval and LLM-written memories (Mem0, A-MEM) are compared under
  one retrieved-context cap, one judge, and per-phase token accounting (ingest / answer / judge).
- **Part B — self-evolving memory.** The Evo-Memory protocol on task streams (MMLU-Pro
  subjects, GPQA-Diamond, AIME): the agent answers, gets binary correct/wrong feedback, and
  the memory updates — no reference answer ever reaches it. Learning curves compare
  no-memory, experience retrieval (ExpRAG), a Dynamic-Cheatsheet-style rewritten sheet, its
  frozen counterpart (advice-in-context without learning), Mem0, A-MEM and ACE.

Backbones: an open-weight model served with vLLM (Qwen3-family) and Claude Sonnet 5 through
the Anthropic API; one embedder everywhere (Qwen3-Embedding-0.6B via an OpenAI-compatible
`/v1/embeddings` endpoint). Every LLM call goes through one instrumented client that tags it
`agent | memory | judge` and logs tokens, cache hits and latency per row.

## Layout

```
experiments/
  stream_runner/        Part B: manifests + dataset pins, instrumented client, embedder,
                        memory systems (memory/), the stream loop (run.py), summarize.py,
                        launch.py (parallel jobs over vLLM replicas)
  part_a/
    locomo/             official serialization + prompts + task_eval scorer; Mem0-rubric judge
    longmemeval/        official JSON-session prompt + the five answer-check templates
    beam/               official long-context turns, pair-chunk BM25, nugget judge + tau_norm
  locomo_audit/         token audits of the three datasets (question ids + counts only)
  longmemeval_audit/
  beam_audit/
code-repo/              (not tracked) vendored benchmark repos, see below
```

Every benchmark script imports the *official* prompt / scorer code from a vendored checkout
rather than re-typing it, so the reproduction cannot drift from the published protocol.

## Setup

```bash
pip install tiktoken openai anthropic numpy rank_bm25 nltk scipy mem0ai faiss-cpu litellm
mkdir -p code-repo && cd code-repo
git clone https://github.com/mem0ai/memory-benchmarks
git clone https://github.com/snap-research/locomo        && git -C locomo checkout 3eb6f2c
git clone https://github.com/xiaowu0162/LongMemEval      && git -C LongMemEval checkout 9e0b455
git clone https://github.com/mohammadtavakoli78/BEAM && git -C BEAM checkout b2da22e
git clone https://github.com/WujiangXu/AgenticMemory A-mem && git -C A-mem checkout 0c8039f
git clone https://github.com/ace-agent/ace               && git -C ace checkout 82709de
# LongMemEval data (not in its repo): huggingface-cli download xiaowu0162/longmemeval-cleaned \
#   --repo-type dataset --local-dir LongMemEval/data/
```

Environment:

| variable | purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Sonnet rows and the judge |
| `QWEN_BASE_URL`, `QWEN_MODEL` | vLLM OpenAI-compatible endpoint + served model name |
| `EMBED_BASE_URL` | OpenAI-compatible `/v1/embeddings` serving Qwen3-Embedding-0.6B (else a local sentence-transformers fallback) |
| `THINK_BUDGET` | thinking budget for the open model's *agent* calls (memory and judge calls never think) |
| `MEM0_TELEMETRY=False` | Mem0 phones home by default |
| `LOCOMO_PATH`, `LONGMEMEVAL_PATH`, `BEAM_PATH`, `AMEM_PATH`, `MEMORY_BENCHMARKS_PATH` | override the `code-repo/` locations |

## Running

Part B streams:

```bash
cd experiments/stream_runner
python manifests.py build-all                      # seeded orders, pinned dataset revisions
python run.py --stream mmlu_pro_philosophy --seed 0 --system dc --backbone qwen27b --limit 20
python summarize.py runs/<run_id>
python launch.py --plan --backbone qwen27b --seeds 0 1 2 --jobs jobs.txt && python launch.py --jobs jobs.txt --concurrency 48 --base-urls http://localhost:8000/v1 http://localhost:8001/v1
```

Part A pilots (one conversation / the pinned smoke set; the same scripts scale to the full sets):

```bash
cd experiments/part_a/locomo      && python run_locomo.py --conv conv-26 --system full --backbone sonnet5
cd experiments/part_a/longmemeval && python run_lme.py --system mem0 --backbone qwen27b --smoke
cd experiments/part_a/beam        && python run_beam.py --group 100K --conv 1 --system bm25 --backbone sonnet5
```

Each run writes `runs/<run_id>/{results.csv, calls.jsonl, provenance.json}` plus the
benchmark's own output layout (`predictions.json`, `hypotheses.jsonl`, `answers.json`) so
the official evaluators can re-score the same answers. Runs are resumable; `--rejudge` re-scores
existing rows without re-answering.

## Protocol notes

- Retrieved context is capped at 4,096 o200k tokens for every non-full-context system.
- One judge (Claude Sonnet 5) scores every row, running each benchmark's published judge
  template verbatim; LoCoMo is additionally scored with the official F1/substring scorer so
  the two can be compared on the same answers.
- Token columns are per backbone tokenizer; `block_tokens` / `history_tokens` (o200k) are the
  cross-backbone comparable ones.
- Part B memories receive only the binary outcome, never the reference answer.

## Data licenses

LoCoMo is CC BY-NC 4.0 (non-commercial use only); BEAM is CC BY-SA 4.0; LongMemEval follows its
repository license. The datasets are not redistributed here — only question ids and token
counts appear in `experiments/*_audit/`.
