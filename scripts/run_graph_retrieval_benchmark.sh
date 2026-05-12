#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/cstamoerec_amazon_video_games_50k.yaml}"
DEVICE="${DEVICE:-cuda}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}"
LIGHTGCN_EPOCHS="${LIGHTGCN_EPOCHS:-20}"
LIGHTGCN_BATCH_SIZE="${LIGHTGCN_BATCH_SIZE:-4096}"
SIMILARITY_TOPK="${SIMILARITY_TOPK:-50}"
SIMILARITY_BATCH_SIZE="${SIMILARITY_BATCH_SIZE:-384}"
PER_SOURCE_K="${PER_SOURCE_K:-500}"
MAX_CANDIDATES="${MAX_CANDIDATES:-1000}"
LIMIT_USERS="${LIMIT_USERS:-0}"

LIMIT_ARGS=()
if [ "$LIMIT_USERS" -gt 0 ]; then
  LIMIT_ARGS=(--limit-users "$LIMIT_USERS")
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "== Graph retrieval benchmark: $CONFIG =="

python scripts/prepare_amazon2023.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --text-batch-size "$TEXT_BATCH_SIZE"

python scripts/train_lightgcn.py \
  --config "$CONFIG" \
  --epochs "$LIGHTGCN_EPOCHS" \
  --batch-size "$LIGHTGCN_BATCH_SIZE" \
  --similarity-topk "$SIMILARITY_TOPK" \
  --similarity-batch-size "$SIMILARITY_BATCH_SIZE" \
  --device "$DEVICE"

python scripts/evaluate_candidates.py \
  --config "$CONFIG" \
  --split test \
  "${LIMIT_ARGS[@]}" \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --topk 10 20 50 100 200 500 1000 \
  --device "$DEVICE"

python scripts/rerank_candidates.py \
  --config "$CONFIG" \
  --mode candidate \
  --split test \
  "${LIMIT_ARGS[@]}" \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --device "$DEVICE"
