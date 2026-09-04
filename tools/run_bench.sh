#!/usr/bin/env bash
# Run the real-prompt category benchmark against a lane at C1, C4 and C16 and write results/categories_<lane>_off_c<N>.json
# Usage: tools/run_bench.sh <base_url> <served_model> <lane>      e.g. tools/run_bench.sh http://100.92.77.51:30000 deepseek-ai/DeepSeek-V4-Flash-Vision-Exp sglang_tp2
set -euo pipefail
BASE="${1:?base url}"; MODEL="${2:?served model id}"; LANE="${3:?lane name}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; mkdir -p results
PY="${PY:-/opt/homebrew/bin/python3.11}"
for C in 1 4 16; do
  echo "== $LANE C$C  $(date '+%H:%M:%S')"
  "$PY" tools/bench_categories.py "$BASE" "$MODEL" "$LANE" --thinking off --concurrency "$C" 2>&1 | tail -3
done
echo "done: $(ls results/categories_${LANE}_off_c*.json)"
