#!/usr/bin/env bash
# E1 on Qwen (step 11) — the frozen queue in three nights, every job resumable.
#
#   ./queue_e1.sh night1     # LoCoMo all convs × all systems · LongMemEval full/bm25 500 · BEAM-100K full/bm25 20 convs
#   ./queue_e1.sh night2     # LongMemEval mem0 500 (12 shards) · BEAM-100K mem0 20 convs
#   ./queue_e1.sh night3     # LongMemEval amem/hindsight 150 (6 shards each) · BEAM-100K amem/hindsight · BEAM-1M convs 1-5
#   ./queue_e1.sh amem       # LoCoMo A-MEM on Qwen, 10 convs (moved out of night1 2026-09-30; run after night3)
#   ./queue_e1.sh judge      # Sonnet judge on judged sets with missing scores (BEAM convs 11-20 stay unjudged by design)
#   DRY=1 ./queue_e1.sh night1   # print the commands instead of running them
#
# Env: QWEN_BASE_URL QWEN_MODEL EMBED_BASE_URL ANTHROPIC_API_KEY (judge) MEM0_TELEMETRY=False
#      HINDSIGHT_URL=http://localhost:8888 with `./hindsight_server.sh qwen27b` running (hindsight jobs are
#      skipped with a warning when the server is down).  P = parallel processes per group (default 6).
set -uo pipefail
cd "$(dirname "$0")"
P=${P:-6}
LOG=logs; mkdir -p $LOG longmemeval/shards
hs_up() { curl -sf "${HINDSIGHT_URL:-http://localhost:8888}/health" >/dev/null 2>&1 || curl -sf "${HINDSIGHT_URL:-http://localhost:8888}/version" >/dev/null 2>&1; }
par() {   # run the command lines on stdin, P at a time (DRY=1 → print)
  local n=${1:-$P}
  # NUL-separated, one command per bash: no `xargs -I` (BSD/macOS caps -I lines at 255 bytes)
  if [ "${DRY:-0}" = "1" ]; then cat; else tr '\n' '\0' | xargs -0 -n 1 -P "$n" bash -c; fi
}
LOCOMO_CONVS=$(python3 -c "import json,os;p=os.environ.get('LOCOMO_PATH','../../code-repo/locomo')+'/data/locomo10.json';print(' '.join(x['sample_id'] for x in json.load(open(p))))")

locomo_cmds() {           # $1 = system
  for c in $LOCOMO_CONVS; do echo "cd locomo && python3 run_locomo.py --conv $c --system $1 --backbone qwen27b > ../$LOG/locomo_${c}_$1.log 2>&1"; done
}
lme_shard_cmds() {        # $1 = system, $2 = shard tag (files longmemeval/shards/<tag>_NN.json), $3 = run-id prefix
  for f in longmemeval/shards/$2_*.json; do k=${f##*_}; k=${k%.json}
    echo "cd longmemeval && python3 run_lme.py --system $1 --backbone qwen27b --ids ../$f --run-id $3_sh$k > ../$LOG/lme_$3_sh$k.log 2>&1"; done
}
beam_cmds() {             # $1 = group, $2 = system, $3 = first conv, $4 = last conv, $5 = extra flags
  for c in $(seq "$3" "$4"); do echo "cd beam && python3 run_beam.py --group $1 --conv $c --system $2 --backbone qwen27b $5 > ../$LOG/beam${1}_${c}_$2.log 2>&1"; done
}
beam100k() { beam_cmds 100K "$1" 1 10 "" ; beam_cmds 100K "$1" 11 20 "--no-judge"; }   # frozen: judge Qwen BEAM on convs 1-10

case "${1:-}" in
night1)
  for s in full bm25 mem0; do echo "[$(date +%H:%M)] LoCoMo $s"; locomo_cmds $s | par; done   # amem → its own step
  if hs_up; then for s in hindsight hindsight-reflect; do echo "[$(date +%H:%M)] LoCoMo $s"; locomo_cmds $s | par; done
  else echo "  !! Hindsight server down — LoCoMo hindsight rows skipped (rerun night1 later; done rows resume)"; fi
  (cd longmemeval && python3 shard_ids.py --all --n 8 --tag full_qwen27b >/dev/null && python3 shard_ids.py --all --n 8 --tag bm25_qwen27b >/dev/null)
  for s in full bm25; do echo "[$(date +%H:%M)] LongMemEval $s (8 shards)"; lme_shard_cmds $s ${s}_qwen27b ${s}_qwen27b | par 8; done
  for s in full bm25; do echo "[$(date +%H:%M)] BEAM-100K $s"; beam100k $s | par; done ;;
night2)
  (cd longmemeval && python3 shard_ids.py --all --n 12 --tag mem0_qwen27b >/dev/null)
  echo "[$(date +%H:%M)] LongMemEval mem0 (12 shards)"; lme_shard_cmds mem0 mem0_qwen27b mem0_qwen27b | par 12
  echo "[$(date +%H:%M)] BEAM-100K mem0"; beam100k mem0 | par 4 ;;
night3)
  (cd longmemeval && python3 shard_ids.py --slice 150 --n 6 --tag amem_qwen27b >/dev/null && python3 shard_ids.py --slice 150 --n 6 --tag hindsight_qwen27b >/dev/null)
  echo "[$(date +%H:%M)] LongMemEval amem (150, 6 shards)"; lme_shard_cmds amem amem_qwen27b amem_qwen27b_s150 | par 6
  echo "[$(date +%H:%M)] BEAM-100K amem"; beam100k amem | par 4
  if hs_up; then
    echo "[$(date +%H:%M)] LongMemEval hindsight (150, 6 shards)"; lme_shard_cmds hindsight hindsight_qwen27b hindsight_qwen27b_s150 | par 6
    echo "[$(date +%H:%M)] BEAM-100K hindsight"; beam100k hindsight | par 4
  else echo "  !! Hindsight server down — hindsight rows skipped"; fi
  for s in bm25 full mem0; do echo "[$(date +%H:%M)] BEAM-1M $s (convs 1-5)"; beam_cmds 1M $s 1 5 "" | par 3; done   # frozen D3: no A-MEM on 1M
  if hs_up; then echo "[$(date +%H:%M)] BEAM-1M hindsight"; beam_cmds 1M hindsight 1 5 "" | par 3; fi ;;
amem)
  echo "[$(date +%H:%M)] LoCoMo amem"; locomo_cmds amem | par ;;
judge)
  for d in longmemeval/runs/*_qwen27b*; do [ -f "$d/results.csv" ] || continue; r=$(basename "$d"); s=${r%%_*}
    echo "cd longmemeval && python3 run_lme.py --system $s --backbone qwen27b --run-id $r --rejudge missing > ../$LOG/judge_$r.log 2>&1"; done | par 4
  for c in $(seq 1 10); do for s in full bm25 mem0 amem hindsight; do [ -f "beam/runs/100K-${c}_${s}_qwen27b/results.csv" ] || continue
    echo "cd beam && python3 run_beam.py --group 100K --conv $c --system $s --backbone qwen27b --rejudge missing > ../$LOG/judge_beam100k_${c}_$s.log 2>&1"; done; done | par 4 ;;
*) echo "usage: $0 night1|night2|night3|amem|judge   (DRY=1 to print)"; exit 1 ;;
esac
echo "[$(date +%H:%M)] ${1} done — python3 rollup.py for the tables"
