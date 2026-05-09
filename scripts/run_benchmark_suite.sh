#!/usr/bin/env bash
set -euo pipefail

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  CONFIGS=(
    "config/cstamoerec_amazon_video_games_50k.yaml"
    "config/cstamoerec_amazon_sports_50k.yaml"
    "config/cstamoerec_amazon_toys_50k.yaml"
  )
fi

for config in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running benchmark pipeline for $config"
  echo "============================================================"
  bash scripts/run_benchmark_pipeline.sh "$config"
done
