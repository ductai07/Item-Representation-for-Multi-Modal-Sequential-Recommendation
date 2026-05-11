# Graph-Enhanced Multimodal Sequential Recommendation

Repository này xây dựng hệ gợi ý sản phẩm tiếp theo trong chuỗi hành vi người dùng, sử dụng dữ liệu Amazon Reviews 2023 có tương tác, metadata văn bản và ảnh sản phẩm.

Sau quá trình thử nghiệm, hướng chính của dự án hiện tại là:

```text
Graph-enhanced multimodal candidate retrieval
```

Nói ngắn gọn:

```text
Graph tìm candidate tốt trước, sau đó mới xét reranking nhẹ.
```

MoE/CS-TAMoERec vẫn còn trong code, nhưng hiện tại **không còn là claim chính** vì kết quả direct full-catalog và adaptive reranking chưa tốt bằng candidate-order từ graph.

## Kết Luận Thực Nghiệm Hiện Tại

Trên Amazon Reviews 2023 Video Games, chạy full test với `max_candidates=1000`, `mode=candidate`, không append target oracle:

```text
R@5   = 0.0241
R@10  = 0.0383
R@20  = 0.0579
N@5   = 0.0177
N@10  = 0.0221
N@20  = 0.0270
PoolHit@1000 = 0.3966
```

Ý nghĩa:

- Stage 1 graph retrieval có tín hiệu tốt.
- Candidate pool lấy được target khoảng 39.66% ở top 1000.
- Candidate-order ranking đang tốt hơn direct MoE ranker trong thử nghiệm hiện tại.
- Vì vậy báo cáo nên lấy graph candidate retrieval làm phương pháp chính, còn MoE là ablation/future work.

## Ý Tưởng Chính

Pipeline hiện tại:

```text
Amazon Reviews 2023
        |
        v
5-core-style filtering + leave-two-out split
        |
        v
Text embedding + Image embedding
        |
        v
Build transition / itemCF / text / image graphs
        |
        v
Stage 1: multimodal graph candidate generation
        |
        v
Stage 2: candidate-order hoặc learned reranking
        |
        v
R@10, R@20, N@10, N@20
```

Các nguồn candidate:

```text
Popularity
Transition graph
ItemCF graph
Text similarity graph
Image similarity graph
Text kNN
Image kNN
Combined candidate order
```

## Quan Hệ Với MuSICRec Và Các Paper Multimodal SR

Dự án có thể đối chiếu theo protocol với các hướng như MuSICRec, MISSRec, MMSR, HM4SR, SMORE, MGCN, FREEDOM, BM3, SASRec, BERT4Rec và LightGCN.

Để gần với MuSICRec, nên chạy trên:

```text
Baby
Sports and Outdoors
Electronics
```

Protocol nên dùng:

```text
5-core users/items
leave-two-out split
R@10, R@20, N@10, N@20
```

Lưu ý: repo này dùng MiniLM cho text embedding và CLIP cho image embedding. Đây là một phần của phương pháp hiện tại. Nếu paper khác dùng published 384-d text feature và 4096-d visual feature, cần ghi rõ khác biệt này khi báo cáo.

## Cấu Trúc Code

```text
cstamoerec/
  candidate.py       # popularity, transition, itemCF, text/image candidate retrieval
  config.py          # load YAML config
  data.py            # xử lý sequence, split, artifact
  features.py        # encode text/image feature
  graph.py           # LightGCN và graph utilities
  metrics.py         # HR/Recall, MRR, NDCG, Coverage
  model.py           # CS-TAMoERec/MoE experimental model
  reranker.py        # source prior và optional reranking
  source_ranker.py   # learned source ranker
  train.py           # train/evaluate MoE experimental model

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

## Cài Đặt

```bash
pip install -r requirements_cstamoerec.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Config Chính

Debug và kiểm tra protocol:

```text
config/cstamoerec_amazon_video_games_50k.yaml
```

Các dataset gần với MuSICRec:

```text
config/cstamoerec_amazon_baby_50k.yaml
config/cstamoerec_amazon_sports_50k.yaml
config/cstamoerec_amazon_electronics_50k.yaml
```

## Chạy Pipeline Graph Retrieval

Chọn config:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
CONFIG=config/cstamoerec_amazon_video_games_50k.yaml
```

Chuẩn bị dữ liệu:

```bash
python scripts/prepare_amazon2023.py \
  --config $CONFIG \
  --device cuda \
  --text-batch-size 256
```

Train graph / build graph embeddings:

```bash
python scripts/train_lightgcn.py \
  --config $CONFIG \
  --epochs 20 \
  --batch-size 4096 \
  --similarity-topk 50 \
  --similarity-batch-size 384 \
  --device cuda
```

Đánh giá candidate recall:

```bash
python scripts/evaluate_candidates.py \
  --config $CONFIG \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --topk 10 20 50 100 200 500 1000 \
  --device cuda
```

Đánh giá graph candidate-order ranking:

```bash
python scripts/rerank_candidates.py \
  --config $CONFIG \
  --mode candidate \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

Khi debug, có thể thêm:

```bash
--limit-users 2000
```

## MoE / CS-TAMoERec Hiện Tại

MoE hiện vẫn có trong project, nhưng chỉ nên dùng cho:

```text
ablation
expert analysis
future reranking
demo giải thích
```

Train MoE:

```bash
python scripts/train_cstamoerec.py \
  --config $CONFIG \
  --device cuda
```

Đánh giá adaptive MoE reranking:

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

Không dùng MoE làm kết quả chính nếu nó không vượt candidate-order và baseline đơn giản.

## Metric Báo Cáo

Bảng chính nên dùng:

```text
R@10
R@20
N@10
N@20
```

Trong code:

```text
HR@K == Recall@K == R@K
NDCG@K == N@K
```

Nên báo cáo thêm:

```text
CandidatePoolHitRate
Recall@1000
AvgCandidatePoolSize
```

## Baseline Nên Có

Tối thiểu:

```text
Popularity
Transition graph
ItemCF graph
Text kNN
Image kNN
Combined candidate-order
```

Nếu có thêm thời gian:

```text
LightGCN standalone
SASRec / BERT4Rec từ framework ngoài
learned source ranker
candidate reranker
MoE direct/adaptive như ablation
```

## Lưu Ý Quan Trọng

- `mode=candidate` trong `rerank_candidates.py` là graph candidate-order, không cần checkpoint.
- `mode=adaptive` và `mode=hybrid` cần checkpoint MoE.
- 99-negative chỉ nên dùng làm diagnostic hoặc bảng phụ, không trộn với full/two-stage metric.
- `--limit-users` chỉ để debug. Bảng chính nên chạy full test.
- Mặc định không append target vào candidate pool. `--append-target-for-oracle` chỉ dùng để chẩn đoán.

## Câu Chuyện Báo Cáo Đề Xuất

Nên trình bày dự án như sau:

```text
Dự án xây dựng pipeline two-stage cho multimodal sequential recommendation.
Stage 1 sử dụng transition, itemCF, text, image và modal graph để retrieve candidate từ full catalog.
Stage 2 đánh giá candidate-order và các hướng reranking nhẹ.
Kết quả cho thấy graph-enhanced multimodal candidate retrieval là tín hiệu mạnh và ổn định, trong khi MoE hiện được giữ như hướng ablation/future work.
```
