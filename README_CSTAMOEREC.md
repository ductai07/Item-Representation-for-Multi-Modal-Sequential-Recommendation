# Quickstart CS-TAMoERec++

File này là bản chạy nhanh. README chính mô tả đầy đủ kiến trúc, pipeline và phần báo cáo.

## 1. Cài Thư Viện

```bash
pip install -r requirements_cstamoerec.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

PowerShell:

```powershell
$env:PYTHONPATH="$PWD;$env:PYTHONPATH"
```

## 2. Chuẩn Bị Dữ Liệu

Bản đầy đủ có text + image:

```bash
python scripts/prepare_amazon2023.py --config config/cstamoerec_all_beauty.yaml
```

Bản debug nhanh, bỏ ảnh:

```bash
python scripts/prepare_amazon2023.py --config config/cstamoerec_all_beauty.yaml --skip-images
```

## 3. Train Modal-aware LightGCN Graph Embeddings

```bash
python scripts/train_lightgcn.py \
  --config config/cstamoerec_all_beauty.yaml \
  --epochs 20 \
  --similarity-topk 50 \
  --similarity-batch-size 512 \
  --device cuda
```

Script này build/train 3 graph và ghi embedding vào:

```text
data/processed/all_beauty/features.pt
```

Các key chính:

```text
id_graph_embeddings
text_graph_embeddings
image_graph_embeddings
text_graph_edges
image_graph_edges
```

## 4. Train CS-TAMoERec++

```bash
python scripts/train_cstamoerec.py --config config/cstamoerec_all_beauty.yaml --device cuda
```

Checkpoint:

```text
checkpoints/cstamoerec/best_cstamoerec.pt
```

## 5. Đánh Giá Stage 1 Candidate Generation

```bash
python scripts/evaluate_candidates.py \
  --config config/cstamoerec_all_beauty.yaml \
  --split test \
  --per-source-k 200 \
  --max-candidates 500
```

## 6. Đánh Giá Two-stage Reranking

```bash
python scripts/rerank_candidates.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test \
  --per-source-k 100 \
  --max-candidates 300
```

## 7. Ablation

```bash
python scripts/run_ablation.py \
  --config config/cstamoerec_all_beauty.yaml \
  --epochs 10 \
  --variants full id_only no_text no_image no_time no_cold_router no_cross no_graph id_graph_only no_text_graph no_image_graph no_aux_loss no_router_balance
```

## 8. Expert Weight Analysis

```bash
python scripts/analyze_experts.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test
```

## 9. Perturbation Test

```bash
python scripts/evaluate_perturbation.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test
```

## 10. Counterfactual Rank Test

```bash
python scripts/evaluate_counterfactual.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test \
  --limit-users 100
```

## 11. Export Demo Cache

```bash
python scripts/export_demo_cache.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --num-users 50 \
  --per-source-k 100 \
  --max-candidates 300 \
  --topk 10
```

## 12. Demo UI

```bash
streamlit run demo_streamlit.py
```

## Expert Hiện Tại

```text
ID, Text, Image, Time, Cross, Graph
```

Modal-aware Graph Expert dùng:

- LightGCN embedding từ ID transition graph;
- LightGCN embedding từ text similarity graph;
- LightGCN embedding từ image similarity graph;
- `modal_graph_scores(candidate | user_history)` gồm ID/Text/Image scores trong two-stage reranker.

Checkpoint cũ 5 expert cần train lại để dùng bản 6 expert hiện tại.
