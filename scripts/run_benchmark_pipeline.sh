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
RERANKER_EPOCHS="${RERANKER_EPOCHS:-8}"
SOURCE_RANKER_EPOCHS="${SOURCE_RANKER_EPOCHS:-250}"
DEMO_USERS="${DEMO_USERS:-50}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SAVE_DIR="$(python - "$CONFIG" <<'PY'
import sys
from cstamoerec.config import load_config
cfg = load_config(sys.argv[1])
print(cfg.train.save_dir)
PY
)"

CHECKPOINT="$SAVE_DIR/best_cstamoerec.pt"
CANDIDATE_CHECKPOINT="$SAVE_DIR/best_candidate_reranker.pt"
SOURCE_RANKER="$SAVE_DIR/learned_source_ranker.json"

echo "== Config: $CONFIG =="
echo "== Save dir: $SAVE_DIR =="

echo "== Prepare Amazon Reviews 2023 benchmark data =="
python scripts/prepare_amazon2023.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --text-batch-size "$TEXT_BATCH_SIZE"

echo "== Train modal-aware LightGCN graphs =="
python scripts/train_lightgcn.py \
  --config "$CONFIG" \
  --epochs "$LIGHTGCN_EPOCHS" \
  --batch-size "$LIGHTGCN_BATCH_SIZE" \
  --similarity-topk "$SIMILARITY_TOPK" \
  --similarity-batch-size "$SIMILARITY_BATCH_SIZE" \
  --device "$DEVICE"

echo "== Train CS-TAMoERec =="
python scripts/train_cstamoerec.py \
  --config "$CONFIG" \
  --device "$DEVICE"

echo "== Train candidate reranker =="
python scripts/train_candidate_reranker.py \
  --config "$CONFIG" \
  --checkpoint-in "$CHECKPOINT" \
  --checkpoint-out "$CANDIDATE_CHECKPOINT" \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --epochs "$RERANKER_EPOCHS" \
  --lr 0.0001 \
  --device "$DEVICE"

echo "== Train learned source ranker =="
python scripts/tune_source_ranker.py \
  --config "$CONFIG" \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --epochs "$SOURCE_RANKER_EPOCHS" \
  --output "$SOURCE_RANKER" \
  --device "$DEVICE"

echo "== Evaluate comparable 99-negative ranking baselines =="
python scripts/evaluate_traditional_baselines.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --num-negatives 99 \
  --device "$DEVICE"

echo "== Evaluate full-pool candidate generation =="
python scripts/evaluate_candidates.py \
  --config "$CONFIG" \
  --split test \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --topk 50 100 200 500 1000 \
  --checkpoint "$CHECKPOINT" \
  --include-model-candidates \
  --device "$DEVICE"

echo "== Evaluate true two-stage retrieval + reranking =="
python scripts/rerank_candidates.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --mode hybrid \
  --split test \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --include-model-candidates \
  --device "$DEVICE"

python scripts/rerank_candidates.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --mode learned_source \
  --source-ranker "$SOURCE_RANKER" \
  --split test \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --include-model-candidates \
  --device "$DEVICE"

python scripts/rerank_candidates.py \
  --config "$CONFIG" \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --mode adaptive \
  --split test \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --include-model-candidates \
  --device "$DEVICE"

echo "== Explainability and report artifacts =="
python scripts/evaluate_perturbation.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --device "$DEVICE"

python scripts/evaluate_counterfactual.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --limit-users 200 \
  --device "$DEVICE"

python scripts/summarize_experiments.py \
  --config "$CONFIG"

python scripts/export_demo_cache.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --mode learned_source \
  --source-ranker "$SOURCE_RANKER" \
  --only-hits \
  --num-users "$DEMO_USERS" \
  --per-source-k "$PER_SOURCE_K" \
  --max-candidates "$MAX_CANDIDATES" \
  --topk 10 \
  --include-model-candidates \
  --device "$DEVICE"

echo "Done. Report: $SAVE_DIR/experiment_report.md"
