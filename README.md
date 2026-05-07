# CS-TAMoERec++

**Cold-Start, Time-Aware and Graph-Enhanced Mixture-of-Experts Recommendation**

Repo gốc dựa trên HM4SR/RecBole và vẫn giữ phần HM4SR làm paper nền/reference. Nhánh phát triển chính của project hiện tại nằm trong package `cstamoerec/`.

Mục tiêu của project là xây dựng hệ gợi ý sản phẩm tiếp theo trong chuỗi hành vi người dùng, kết hợp:

- lịch sử tương tác theo thời gian;
- ID sản phẩm;
- metadata dạng text;
- ảnh sản phẩm;
- tín hiệu cold-start;
- transition graph và LightGCN;
- Mixture-of-Experts để vừa rerank vừa giải thích model đang dựa vào nguồn tín hiệu nào.

## Ý Tưởng Chính

CS-TAMoERec++ dùng pipeline hai giai đoạn:

```text
Amazon Reviews 2023 All_Beauty
        ↓
Data processing + sequence split
        ↓
Text embedding + Image embedding + Time feature
        ↓
Train transition graph + LightGCN item embedding
        ↓
Stage 1: Candidate Generation
        ↓
Stage 2: MoE Reranking
        ↓
Recommendation + Expert-weight Explanation + Demo UI
```

Model lấy cảm hứng từ các hướng nghiên cứu:

- **HM4SR**: time-aware multimodal sequential recommendation và MoE.
- **MAMEX**: router thích ứng với cold-start item.
- **Molar**: alignment giữa collaborative ID signal và multimodal content.
- **RecFormer-style item sentence**: biểu diễn item bằng câu metadata giàu thông tin.
- **LightGCN**: học item graph embedding từ transition graph.

Không fine-tune LLM làm core recommender. LLM/fine-tune chỉ phù hợp cho explanation layer hoặc future work.

## Kiến Trúc Hiện Tại

Với mỗi item, model tạo các representation:

```text
ID embedding       = trainable item embedding
Text embedding     = MiniLM/BGE embedding từ item sentence
Image embedding    = CLIP image embedding
Time embedding     = month, weekday, interval
Graph embedding    = LightGCN item embedding
Graph score        = transition score từ recent history tới candidate
Cold feature       = log(popularity + 1), cold item flag
```

MoE hiện có 6 expert:

```text
E_id
E_text
E_image
E_time
E_cross
E_graph
```

Router học trọng số:

```text
w_id, w_text, w_image, w_time, w_cross, w_graph
```

Item representation cuối:

```text
item_repr =
    w_id    * E_id
  + w_text  * E_text
  + w_image * E_image
  + w_time  * E_time
  + w_cross * E_cross
  + w_graph * E_graph
```

Sequence backbone là SASRec-style Transformer. Stage 2 dùng representation này để rerank candidate pool.

## Graph Expert

Graph được build từ **train split only** để tránh leakage.

Transition edge:

```text
item_i → item_j
```

nếu trong lịch sử train, nhiều user tương tác `item_i` rồi sau đó tới `item_j`.

Graph Expert dùng hai tín hiệu:

1. **LightGCN item embedding**

```text
train sequences
      ↓
transition graph
      ↓
LightGCN
      ↓
graph_embeddings[item]
```

2. **Graph transition score**

Với user history `[A, B, C]` và candidate `X`:

```text
graph_score(X) =
  w(A → X) * decay_A
+ w(B → X) * decay_B
+ w(C → X) * decay_C
```

Recent item có trọng số cao hơn.

## Dataset

Dataset chính:

```text
McAuley-Lab/Amazon-Reviews-2023
raw_review_All_Beauty
raw_meta_All_Beauty
```

Các trường quan trọng:

- `user_id`
- `parent_asin`
- `timestamp`
- `rating`
- `title`
- `store` / brand
- `categories`
- `features`
- `description`
- image URL

Prepare script đọc trực tiếp Parquet từ HuggingFace, không phụ thuộc `trust_remote_code`.

## Cấu Trúc Code

```text
cstamoerec/
  config.py          # load YAML config
  data.py            # dataset, sequence, artifact helpers
  features.py        # text/image feature extraction
  graph.py           # LightGCN + graph utilities
  model.py           # CS-TAMoERec++ model
  metrics.py         # HR, MRR, NDCG, Recall, Coverage
  train.py           # train/evaluate logic
  candidate.py       # popularity, transition, itemcf, text/image candidates
  reranker.py        # MoE candidate reranking

scripts/
  prepare_amazon2023.py      # chuẩn bị dữ liệu Amazon Reviews 2023
  train_lightgcn.py          # train LightGCN graph embeddings
  train_cstamoerec.py        # train CS-TAMoERec++
  evaluate_candidates.py     # đánh giá Stage 1 Recall@K
  rerank_candidates.py       # đánh giá Stage 1 + Stage 2
  run_ablation.py            # ablation
  analyze_experts.py         # phân tích expert weights
  evaluate_perturbation.py   # mask/shuffle modality
  evaluate_counterfactual.py # counterfactual rank test
  export_demo_cache.py       # export cache cho Streamlit demo

config/
  cstamoerec_all_beauty.yaml

demo_streamlit.py
requirements_cstamoerec.txt
```

## Cài Đặt

Nên chạy từ root repo:

```bash
cd HM4SR-main
pip install -r requirements_cstamoerec.txt
```

Nếu chạy trên Kaggle/Colab/SSH, nên set:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Trên PowerShell:

```powershell
$env:PYTHONPATH="$PWD;$env:PYTHONPATH"
```

## Config Chính

File:

```text
config/cstamoerec_all_beauty.yaml
```

Các cấu hình quan trọng hiện tại:

```yaml
data:
  max_items: 30000
  min_user_interactions: 5
  min_item_interactions: 2
  max_seq_len: 50
  use_images: true
  max_image_items: 8000

model:
  hidden_size: 128
  n_layers: 2
  n_heads: 4
  num_experts: 6
  use_text: true
  use_image: true
  use_time: true
  use_cold: true
  use_cross: true
  use_graph: true
  graph_dim: 64
  graph_layers: 2

train:
  batch_size: 256
  epochs: 20
  lr: 0.001
  num_eval_negatives: 0
  device: cuda
```

## Pipeline Chạy Chuẩn

### 1. Chuẩn Bị Dữ Liệu Đầy Đủ Có Ảnh

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml
```

Output:

```text
data/processed/all_beauty/
  examples.pt
  features.pt
  meta.json
  item_cards.json
```

Nếu chỉ muốn smoke test nhanh, có thể bỏ ảnh:

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml \
  --skip-images
```

Lưu ý: nếu bỏ ảnh thì `image` candidate gần như không có ý nghĩa và Image Expert chỉ là kiểm tra pipeline, không phải kết quả cuối.

### 2. Train LightGCN Graph Embeddings

Chạy sau khi prepare xong:

```bash
python scripts/train_lightgcn.py \
  --config config/cstamoerec_all_beauty.yaml \
  --epochs 20 \
  --device cuda
```

Script này sẽ ghi thêm:

```text
features["graph_embeddings"]
```

vào:

```text
data/processed/all_beauty/features.pt
```

### 3. Train CS-TAMoERec++

```bash
python scripts/train_cstamoerec.py \
  --config config/cstamoerec_all_beauty.yaml \
  --device cuda
```

Checkpoint tốt nhất:

```text
checkpoints/cstamoerec/best_cstamoerec.pt
```

Training history:

```text
checkpoints/cstamoerec/history.json
```

Khuyến nghị epoch:

- smoke test: `3` epoch;
- chạy có thể báo cáo: `20` epoch;
- nếu GPU ổn và metric còn tăng: `30-50` epoch với early stopping thủ công theo `NDCG@10`.

Với bản hiện tại, checkpoint cũ 5 expert không tương thích trực tiếp với model mới 6 expert. Sau khi bật Graph Expert nên train lại từ đầu.

## Stage 1: Candidate Generation

Stage 1 tạo candidate pool bằng nhiều nguồn:

```text
popularity
transition graph
itemcf / co-occurrence
text similarity
image similarity
combined
```

Đánh giá Recall@50/100/200:

```bash
python scripts/evaluate_candidates.py \
  --config config/cstamoerec_all_beauty.yaml \
  --split test \
  --per-source-k 200 \
  --max-candidates 500
```

Output:

```text
checkpoints/cstamoerec/candidate_recall_test.json
```

Bảng nên đưa vào báo cáo:

```text
Candidate Source     Recall@50   Recall@100   Recall@200
Popularity           ...
Transition graph     ...
ItemCF               ...
Text similarity      ...
Image similarity     ...
Combined             ...
```

## Stage 2: MoE Reranking

Stage 2 rerank candidate pool bằng CS-TAMoERec++.

```bash
python scripts/rerank_candidates.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test \
  --per-source-k 100 \
  --max-candidates 300
```

Output:

```text
checkpoints/cstamoerec/two_stage_rerank_test.json
```

Trong bước này, reranker nhận thêm:

```text
graph_score(candidate | user_history)
```

để Graph Expert có tín hiệu transition trực tiếp.

## Ablation

Chạy ablation:

```bash
python scripts/run_ablation.py \
  --config config/cstamoerec_all_beauty.yaml \
  --epochs 10 \
  --variants full id_only no_text no_image no_time no_cold_router no_cross no_graph no_aux_loss no_router_balance
```

Output:

```text
checkpoints/cstamoerec/ablation/ablation_summary.json
```

Các variant quan trọng cho báo cáo:

```text
full
id_only
no_text
no_image
no_time
no_cross
no_graph
no_aux_loss
```

Bảng nên có:

```text
Variant                  NDCG@10   MRR@10   Recall@10
ID-only                  ...
CS-TAMoERec              ...
CS-TAMoERec++ w/o graph  ...
Full CS-TAMoERec++       ...
```

## Expert Weight Analysis

Phân tích router đang dựa vào expert nào:

```bash
python scripts/analyze_experts.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test
```

Output:

```text
checkpoints/cstamoerec/expert_weights_test.json
```

Các nhóm:

- `cold_items`
- `warm_items`
- `short_time_gap`
- `long_time_gap`
- `all`

Expert hiện tại:

```text
ID, Text, Image, Time, Cross, Graph
```

## Perturbation Test

Dùng để kiểm tra modality có đóng góp thật không:

```bash
python scripts/evaluate_perturbation.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test
```

Các mode:

```text
full
mask_text
mask_image
shuffle_text
shuffle_image
mask_text_image
```

Nếu metric giảm rõ khi mask/shuffle một modality, có thể lập luận modality đó có đóng góp.

## Counterfactual Rank Test

Dùng để xem ranking thay đổi thế nào khi bỏ text/image/time:

```bash
python scripts/evaluate_counterfactual.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test \
  --limit-users 100
```

Output:

```text
checkpoints/cstamoerec/counterfactual_test.json
```

Phù hợp để chọn vài case study đưa vào báo cáo/demo.

## Export Demo Cache

Nên export cache trước để UI chạy mượt:

```bash
python scripts/export_demo_cache.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --num-users 50 \
  --per-source-k 100 \
  --max-candidates 300 \
  --topk 10
```

Output:

```text
demo_cache/recommendations.json
```

Cache chứa:

- user history;
- target item;
- top recommendations;
- candidate sources;
- graph transition score;
- expert weights;
- main expert;
- cold/warm flag;
- image URL nếu có.

## Demo UI

Chạy Streamlit:

```bash
streamlit run demo_streamlit.py
```

Demo có:

- lịch sử tương tác của user;
- top-k recommendation;
- candidate source;
- image/title/category;
- cold/warm label;
- graph transition score;
- expert weight chart;
- explanation đơn giản dựa trên main expert và source.

Nếu đã export cache, bật `Use demo cache` để chạy nhanh và ổn định.

## Metrics

Metric chính:

```text
HR@5 / HR@10 / HR@20
MRR@5 / MRR@10 / MRR@20
NDCG@5 / NDCG@10 / NDCG@20
Recall@5 / Recall@10 / Recall@20
Coverage@10
```

Ngoài overall metric, project còn báo cáo:

- cold item performance;
- warm item performance;
- Stage 1 candidate recall;
- Stage 2 reranking metric;
- ablation;
- perturbation;
- expert-weight analysis;
- counterfactual case study.

## Gợi Ý Chạy Trên GPU

Với GPU ổn:

```yaml
data:
  max_items: 30000
  use_images: true
  max_image_items: 8000

model:
  hidden_size: 128
  n_layers: 2
  n_heads: 4

train:
  batch_size: 256
  epochs: 20
```

Nếu thiếu VRAM:

```yaml
data:
  max_items: 10000
  max_image_items: 2000

model:
  hidden_size: 64
  n_layers: 1
  n_heads: 2

train:
  batch_size: 128
  epochs: 10
```

Nếu chỉ debug pipeline:

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml \
  --skip-images \
  --limit-reviews 20000
```

## Checklist Báo Cáo

Nên có các bảng/hình sau:

1. Dataset statistics:
   - số user;
   - số item;
   - số train/valid/test examples;
   - số cold/warm item.

2. Stage 1 candidate recall:
   - popularity;
   - transition;
   - itemcf;
   - text;
   - image;
   - combined.

3. Main metric:
   - HR@10;
   - MRR@10;
   - NDCG@10;
   - Recall@10.

4. Ablation:
   - full;
   - id_only;
   - no_text;
   - no_image;
   - no_time;
   - no_graph;
   - no_aux_loss.

5. Expert weight analysis:
   - cold vs warm;
   - short time gap vs long time gap.

6. Demo/case study:
   - user history;
   - recommended item;
   - candidate source;
   - graph score;
   - expert weights;
   - explanation.

## Đóng Góp Chính

1. Xây dựng pipeline Amazon Reviews 2023 All_Beauty cho multimodal sequential recommendation.
2. Đề xuất CS-TAMoERec: Cold-start and Time-aware MoE item representation.
3. Nâng cấp thành CS-TAMoERec++ với Graph Expert.
4. Tích hợp ID, text, image, time, cross-modal và graph signal.
5. Dùng LightGCN item transition graph embedding.
6. Dùng graph transition score trong MoE reranker.
7. Có two-stage candidate generation + reranking.
8. Có category loss, ID-MM alignment loss và router balance loss.
9. Có ablation, perturbation, counterfactual và expert-weight analysis.
10. Có Streamlit UI demo giải thích recommendation.

## Tài Liệu Tham Khảo

HM4SR gốc:

```bibtex
@inproceedings{zhang2025hierarchical,
  title={Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation},
  author={Zhang, Shengzhe and Chen, Liyi and Shen, Dazhong and Wang, Chao and Xiong, Hui},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={3672--3682},
  year={2025}
}
```
