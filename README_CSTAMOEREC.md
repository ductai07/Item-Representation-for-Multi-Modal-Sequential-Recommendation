# Quickstart Tiếng Việt

Phương pháp chính hiện tại:

```text
Graph-enhanced multimodal candidate retrieval
```

MoE/CS-TAMoERec không còn là claim chính. MoE chỉ là module thử nghiệm/ablation nếu chưa vượt được candidate-order.

Đã thêm `sequence_graph` lấy cảm hứng từ MuSICRec: mỗi train sequence được xem như một sequence node, tìm sequence gần với history hiện tại rồi lấy item/target từ sequence đó làm candidate.

## Cài Đặt

```bash
pip install -r requirements_cstamoerec.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Chạy Video Games

```bash
CONFIG=config/cstamoerec_amazon_video_games_50k.yaml
```

Chuẩn bị dữ liệu:

```bash
python scripts/prepare_amazon2023.py --config $CONFIG --device cuda --text-batch-size 256
```

Train graph:

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

Đánh giá candidate-order:

```bash
python scripts/rerank_candidates.py \
  --config $CONFIG \
  --mode candidate \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

Chạy gọn graph-only pipeline:

```bash
chmod +x scripts/run_graph_retrieval_benchmark.sh
bash scripts/run_graph_retrieval_benchmark.sh $CONFIG
```

Debug nhanh:

```bash
--limit-users 2000
```

## Kết Quả Full Video Games Hiện Tại

```text
R@5   = 0.0241
R@10  = 0.0383
R@20  = 0.0579
N@5   = 0.0177
N@10  = 0.0221
N@20  = 0.0270
PoolHit@1000 = 0.3966
```

Đây là `mode=candidate`, full test, không append target oracle.

## Dataset Gần MuSICRec

```text
config/cstamoerec_amazon_baby_50k.yaml
config/cstamoerec_amazon_sports_50k.yaml
config/cstamoerec_amazon_electronics_50k.yaml
```

Ví dụ chạy Baby:

```bash
CONFIG=config/cstamoerec_amazon_baby_50k.yaml
bash scripts/run_graph_retrieval_benchmark.sh $CONFIG
```

Metric cần báo cáo:

```text
R@10, R@20, N@10, N@20
```

## MoE Tùy Chọn

Train MoE:

```bash
python scripts/train_cstamoerec.py --config $CONFIG --device cuda
```

Đánh giá adaptive MoE:

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

Không dùng MoE làm kết quả chính nếu nó không vượt candidate-order.
