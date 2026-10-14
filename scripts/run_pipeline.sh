#!/usr/bin/env bash
# Reproduce every artifact: data check -> EDA -> full training -> held-out evaluation -> forecast
# -> fast-mode run (preliminary subset) -> reports. Usage: bash scripts/run_pipeline.sh [csv]
set -euo pipefail
CSV_ARG=()
if [ "${1:-}" != "" ]; then CSV_ARG=(--csv "$1"); fi
uv run motor-thermal data "${CSV_ARG[@]}"
uv run motor-thermal eda "${CSV_ARG[@]}"
for MODE in full fast; do
  uv run motor-thermal train --mode "$MODE" "${CSV_ARG[@]}"
  uv run motor-thermal evaluate --mode "$MODE" "${CSV_ARG[@]}"
  uv run motor-thermal forecast --mode "$MODE" "${CSV_ARG[@]}"
done
uv run motor-thermal report --mode full
