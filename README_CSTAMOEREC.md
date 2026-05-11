# Quickstart

The current main method is **graph-enhanced multimodal candidate retrieval**. MoE is experimental and should not be treated as the main result unless it improves metrics.

## Install

```bash
pip install -r requirements_cstamoerec.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Run Video Games

```bash
CONFIG=config/cstamoerec_amazon_video_games_50k.yaml
```

Prepare data:

```bash
python scripts/prepare_amazon2023.py --config $CONFIG --device cuda --text-batch-size 256
```

Train graph edges/embeddings:

```bash
python scripts/train_lightgcn.py \
  --config $CONFIG \
  --epochs 20 \
  --batch-size 4096 \
  --similarity-topk 50 \
  --similarity-batch-size 384 \
  --device cuda
```

Evaluate candidate recall:

```bash
python scripts/evaluate_candidates.py \
  --config $CONFIG \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --topk 10 20 50 100 200 500 1000 \
  --device cuda
```

Evaluate graph candidate-order ranking:

```bash
python scripts/rerank_candidates.py \
  --config $CONFIG \
  --mode candidate \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

For debugging:

```bash
--limit-users 2000
```

## Current Full Video Games Result

```text
R@5   = 0.0241
R@10  = 0.0383
R@20  = 0.0579
N@5   = 0.0177
N@10  = 0.0221
N@20  = 0.0270
PoolHit@1000 = 0.3966
```

This is `mode=candidate`, full test, no target oracle append.

## MuSICRec-Style Datasets

Use these configs for Baby/Sports/Electronics:

```text
config/cstamoerec_amazon_baby_50k.yaml
config/cstamoerec_amazon_sports_50k.yaml
config/cstamoerec_amazon_electronics_50k.yaml
```

Report:

```text
R@10, R@20, N@10, N@20
```

## Optional MoE

Train only for ablation/analysis:

```bash
python scripts/train_cstamoerec.py --config $CONFIG --device cuda
```

Adaptive MoE reranking:

```bash
python scripts/rerank_candidates.py \
  --config $CONFIG \
  --checkpoint checkpoints/cstamoerec_video_games_5core_50k/best_cstamoerec.pt \
  --mode adaptive \
  --split test \
  --limit-users 2000 \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

Do not use MoE as the main claim unless it beats candidate-order.
