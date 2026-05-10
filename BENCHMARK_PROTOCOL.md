# Benchmark Protocol for CS-TAMoERec

This project should be reported as an experiment-first recommender, not only as a demo. Use the same split, candidate protocol, and metrics for every method in a table.

## Main Dataset

Use **Amazon Reviews 2023 - Video_Games** as the main benchmark:

- It is large enough for sequential recommendation experiments: 2.8M users, 137.2K items, and 4.6M ratings in the raw category.
- The official Amazon Reviews 2023 page also provides a 5-core version with 94.8K users, 25.6K items, and 814.6K interactions before splitting.
- It has item metadata and images, so it is aligned with the multimodal design of CS-TAMoERec.
- It is closer to the HM4SR direction than the tiny `All_Beauty` subset currently used for quick local runs.

Main config:

```bash
config/cstamoerec_amazon_video_games_50k.yaml
```

## Secondary Datasets

Run at least one secondary dataset if time/GPU budget allows:

```bash
config/cstamoerec_amazon_sports_50k.yaml
config/cstamoerec_amazon_toys_50k.yaml
```

Sports and Toys are common sequential-recommendation domains in older papers, and both have text/image metadata in Amazon Reviews 2023.

## Filtering and Split

For paper-style comparison, use the 5-core-inspired setting in the benchmark configs:

- `min_user_interactions: 5`
- `min_item_interactions: 5`
- leave-last-two-out split:
  - train: first `N-2` interactions
  - validation: interaction `N-1`
  - test: interaction `N`

Do not compare a min4 demo run directly against 5-core paper results. Min4 is useful for coverage and demo density, but 5-core is easier to defend in a report.

## Metrics to Report

Use two separate tables.

**Table A: Comparable ranking protocol**

Use sampled-negative evaluation with 99 negatives:

```bash
python scripts/evaluate_traditional_baselines.py \
  --config config/cstamoerec_amazon_video_games_50k.yaml \
  --checkpoint checkpoints/cstamoerec_video_games_5core_50k/best_cstamoerec.pt \
  --num-negatives 99 \
  --device cuda
```

Report `HR@5`, `HR@10`, `HR@20`, `NDCG@5`, `NDCG@10`, `NDCG@20`. This is the table used to compare against classic baselines such as Popularity, ItemCF, SASRec-like scoring, and multimodal nearest-neighbor sources.

**Table B: Real two-stage recommendation protocol**

Use full-pool candidate generation and reranking:

```bash
python scripts/evaluate_candidates.py \
  --config config/cstamoerec_amazon_video_games_50k.yaml \
  --checkpoint checkpoints/cstamoerec_video_games_5core_50k/best_cstamoerec.pt \
  --include-model-candidates \
  --per-source-k 500 \
  --max-candidates 1000 \
  --topk 50 100 200 500 1000 \
  --device cuda

python scripts/rerank_candidates.py \
  --config config/cstamoerec_amazon_video_games_50k.yaml \
  --checkpoint checkpoints/cstamoerec_video_games_5core_50k/best_cstamoerec.pt \
  --mode learned_source \
  --source-ranker checkpoints/cstamoerec_video_games_5core_50k/learned_source_ranker.json \
  --include-model-candidates \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

Report `CandidatePoolHitRate`, `Recall@200`, `Recall@500`, `Recall@1000`, `HR@10`, `NDCG@10`, and `Coverage@10`. This table is stricter and should be presented as real retrieval + reranking, not as the direct paper comparison table.

The benchmark configs use `train.num_eval_negatives: 0`, so validation/test inside `train_cstamoerec.py` is full-catalog evaluation. Use `evaluate_traditional_baselines.py --num-negatives 99` only when creating the sampled-negative comparison table.

## Baselines

Run these local baselines first because they share exactly the same processed data and split:

- Popularity
- Transition graph
- ItemCF graph
- Text kNN
- Image kNN
- Combined reciprocal-rank baseline
- CS-TAMoERec adaptive scoring
- Candidate reranker
- Learned source ranker

Then compare the final table qualitatively with papers that use similar Amazon sequential-recommendation settings:

- HM4SR, WWW 2025: hierarchical multimodal sequential recommendation.
- MISSRec, MM 2023: multimodal interest-aware sequence representation.
- RecFormer, KDD 2023: text-based sequential recommendation.
- BERT4Rec and SASRec-style sequential baselines.
- Recent Amazon multimodal SR papers such as MDSRec/MuSTRec can be discussed as related work, but only compare numbers when preprocessing and evaluation protocol match.

## Full Linux Run

Run the main benchmark:

```bash
chmod +x scripts/run_benchmark_pipeline.sh
bash scripts/run_benchmark_pipeline.sh config/cstamoerec_amazon_video_games_50k.yaml
```

Run the full three-dataset suite:

```bash
chmod +x scripts/run_benchmark_pipeline.sh scripts/run_benchmark_suite.sh
bash scripts/run_benchmark_suite.sh
```

For a 24GB+ GPU, keep 50K items. For 8-12GB GPU, lower `max_image_items` and `max_items` to 30K first, then scale back up once the pipeline is stable.

## Recommended Report Claim

Use this framing:

> We evaluate CS-TAMoERec on Amazon Reviews 2023 using a 5-core-inspired leave-last-two-out protocol. We report both sampled-negative ranking metrics for comparability with classic sequential-recommendation baselines and full-pool two-stage metrics for real retrieval quality. This separates model ranking ability from candidate-generation recall, avoiding inflated conclusions from sampled-negative-only evaluation.

## Sources

- Amazon Reviews 2023 dataset card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Amazon Reviews 2023 processing docs: https://amazon-reviews-2023.github.io/data_processing/5core.html
- RecFormer: https://www.amazon.science/publications/text-is-all-you-need-learning-language-representations-for-sequential-recommendation
- MISSRec: https://arxiv.org/abs/2308.11175
