#!/usr/bin/env bash
# E1 Sonnet slices (step 12, frozen 2026-09-29) — LongMemEval + BEAM-100K; LoCoMo runs separately.
#   LongMemEval: full 50 · bm25 100 · mem0 20 (nested slices, longmemeval/slices.json)
#   BEAM-100K:   full/bm25 convs 1-5 · mem0 convs 1-3
# Every run resumes; rerunning the script continues where it stopped.
#   nohup ./sonnet_slices.sh > sonnet_slices.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export MEM0_TELEMETRY=False

cd longmemeval
python run_lme.py --system full --backbone sonnet5 --slice 50
python run_lme.py --system bm25 --backbone sonnet5 --slice 100
python run_lme.py --system mem0 --backbone sonnet5 --slice 20

cd ../beam
for c in 1 2 3 4 5; do for s in full bm25; do python run_beam.py --group 100K --conv $c --system $s --backbone sonnet5; done; done
for c in 1 2 3; do python run_beam.py --group 100K --conv $c --system mem0 --backbone sonnet5; done
echo "sonnet slices done — python rollup.py"
