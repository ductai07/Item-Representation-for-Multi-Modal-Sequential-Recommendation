# Graph-Enhanced Multimodal Sequential Recommendation

This repository implements a two-stage recommendation pipeline for product next-item recommendation on Amazon Reviews data.

The current experimental direction is **not** to use the MoE model as the main full-catalog ranker. After debugging and benchmark runs, the strongest and most stable component is:

```text
multimodal graph candidate retrieval -> candidate-order / lightweight reranking
```

The MoE model is kept as an experimental module for representation learning, analysis, and future reranking work, but it is not the main reported method unless it improves ranking metrics.

## Current Claim

The project focuses on:

```text
Graph-enhanced multimodal candidate retrieval for sequential recommendation.
```

Signals used:

- item transition graph from user sequences;
- itemCF co-occurrence graph;
- text similarity graph from product metadata embeddings;
- image similarity graph from product image embeddings;
- popularity prior;
- optional model/MoE scores for analysis or reranking experiments.

## Why This Direction

On Amazon Reviews 2023 Video Games, full test evaluation with `max_candidates=1000` gives:

```text
R@5   = 0.0241
R@10  = 0.0383
R@20  = 0.0579
N@5   = 0.0177
N@10  = 0.0221
N@20  = 0.0270
PoolHit@1000 = 0.3966
```

This was measured with `mode=candidate`, meaning no target oracle append and no MoE scoring. The candidate pool is retrieved from the full catalog.

The direct MoE/full-catalog model was weaker than the graph candidate method in current experiments, so the final experimental story should be graph retrieval first.

## Relation to MuSICRec / Multimodal SR Papers

The project can be compared at a protocol level with multimodal sequential recommendation papers such as MuSICRec, MISSRec, MMSR, HM4SR, SMORE, MGCN, FREEDOM, BM3, SASRec, BERT4Rec, and LightGCN.

For a fair benchmark-style comparison, use:

```text
datasets: Baby, Sports and Outdoors, Electronics
filtering: 5-core users/items
split: leave-two-out
metrics: R@10, R@20, N@10, N@20
```

This repo uses MiniLM text embeddings and CLIP image embeddings by default. These features are part of this method. If comparing to papers that use published 384-d text and 4096-d image features, report this difference clearly.

## Repository Layout

```text
cstamoerec/
  candidate.py       # popularity, transition, itemCF, text/image candidate retrieval
  config.py          # YAML config loader
  data.py            # sequence split and dataset artifacts
  features.py        # text/image embedding extraction
  graph.py           # LightGCN and graph utilities
  metrics.py         # HR/Recall, MRR, NDCG, Coverage
  model.py           # experimental CS-TAMoERec model
  reranker.py        # source prior and optional reranking utilities
  source_ranker.py   # lightweight learned source ranker
  train.py           # MoE/full-catalog training logic

scripts/
  prepare_amazon2023.py
  train_lightgcn.py
  evaluate_candidates.py
  rerank_candidates.py
  evaluate_traditional_baselines.py
  train_cstamoerec.py
  train_candidate_reranker.py
  tune_source_ranker.py
  summarize_experiments.py
```

## Install

```bash
pip install -r requirements_cstamoerec.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Main Configs

Debug/current main:

```text
config/cstamoerec_amazon_video_games_50k.yaml
```

MuSICRec-style benchmark domains:

```text
config/cstamoerec_amazon_baby_50k.yaml
config/cstamoerec_amazon_sports_50k.yaml
config/cstamoerec_amazon_electronics_50k.yaml
```

Older quick/demo configs are still available for All Beauty, but they should not be used as the main benchmark evidence.

## Recommended Pipeline

Set a config:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
CONFIG=config/cstamoerec_amazon_video_games_50k.yaml
```

Prepare data and features:

```bash
python scripts/prepare_amazon2023.py \
  --config $CONFIG \
  --device cuda \
  --text-batch-size 256
```

Train graph embeddings and graph edges:

```bash
python scripts/train_lightgcn.py \
  --config $CONFIG \
  --epochs 20 \
  --batch-size 4096 \
  --similarity-topk 50 \
  --similarity-batch-size 384 \
  --device cuda
```

Evaluate candidate retrieval recall:

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

For quick debugging, add:

```bash
--limit-users 2000
```

## Optional Experimental MoE

The MoE model is optional and currently not the main method.

Train it only if you want to test model-based reranking, expert analysis, or future work:

```bash
python scripts/train_cstamoerec.py \
  --config $CONFIG \
  --device cuda
```

Evaluate adaptive MoE reranking:

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

Do not report MoE as the main result unless it beats candidate-order and simple baselines.

## Metrics

Report these metrics in paper-style tables:

```text
R@10, R@20, N@10, N@20
```

In this codebase:

```text
HR@K == Recall@K == R@K
NDCG@K == N@K
```

Also report retrieval quality:

```text
CandidatePoolHitRate
Recall@1000
AvgCandidatePoolSize
```

## Baselines to Report

At minimum, report:

```text
Popularity
Transition graph
ItemCF graph
Text kNN
Image kNN
Combined graph candidate-order
```

Optional:

```text
LightGCN standalone
SASRec / BERT4Rec from external frameworks
MoE direct / adaptive as an ablation
learned source ranker
candidate reranker
```

## Important Notes

- `mode=candidate` in `rerank_candidates.py` is graph candidate-order evaluation and does not require a checkpoint.
- `mode=adaptive` and `mode=hybrid` require a trained MoE checkpoint.
- 99-negative evaluation is useful only as a secondary diagnostic. Do not mix it with full-catalog/two-stage metrics.
- Full test should be used for final tables. `--limit-users` is only for debugging.
- The candidate pool does not append the target by default. `--append-target-for-oracle` is only for diagnostics.

## Suggested Experimental Story

The safest report framing is:

```text
We build a two-stage multimodal sequential recommendation system.
Stage 1 retrieves candidates from transition, itemCF, text, image, and modal graph sources.
Stage 2 evaluates candidate-order and optional learned reranking.
Experiments show that graph-enhanced multimodal candidate retrieval provides a strong and stable signal, while the current MoE scorer is kept as an experimental ablation.
```
