# Item Representation for Multi-Modal Sequential Recommendation

## CS-TAMoERec

**Cold-Start and Time-Aware Mixture-of-Experts for Multi-Modal Sequential Recommendation**

 Phần HM4SR cũ vẫn được giữ lại như paper nền/reference, còn phần phát triển chính nằm trong package `cstamoerec/`.

Mục tiêu của project là xây dựng mô hình gợi ý sản phẩm tiếp theo trong chuỗi hành vi người dùng, sử dụng đồng thời:

- lịch sử tương tác theo thời gian,
- ID sản phẩm,
- text metadata của sản phẩm,
- ảnh sản phẩm,
- độ phổ biến/cold-start của item,
- Mixture-of-Experts để giải thích mô hình đang dựa vào modality nào.

## Ý Tưởng Chính

CS-TAMoERec lấy cảm hứng từ các hướng nghiên cứu:

- **MM-GPT2Rec**: coi chuỗi item như chuỗi token trong NLP.
- **HM4SR**: dùng multimodal representation, MoE và temporal signal.
- **MAMEX**: router thích ứng với cold-start item.
- **Molar**: giữ tín hiệu collaborative ID và multimodal content rồi align chúng.

Khác với việc fine-tune GPT-2 nặng và khó kiểm soát, project này dùng **SASRec-style Transformer** làm sequential backbone.

## Dataset

Dataset chính:

```text
Amazon Reviews 2023 - All_Beauty
```

Nguồn HuggingFace:

```text
McAuley-Lab/Amazon-Reviews-2023
```

Các config sử dụng:

```text
raw_review_All_Beauty
raw_meta_All_Beauty
```

Dataset này phù hợp vì có:

- `user_id`
- `parent_asin`
- `timestamp`
- review/rating
- title
- description
- features
- categories
- store/brand
- image URL

## Pipeline Tổng Thể

```text
Amazon Reviews 2023
        |
        |-- raw_review_All_Beauty
        |       |-- user_id
        |       |-- parent_asin
        |       |-- timestamp
        |
        |-- raw_meta_All_Beauty
                |-- title
                |-- store / brand
                |-- categories
                |-- description
                |-- features
                |-- image URL

        ↓

Data Processing
        |-- lọc user có ít nhất 5 interaction
        |-- lọc item có ít nhất 2 interaction
        |-- nếu giới hạn max_items:
        |       |-- giữ item phổ biến
        |       |-- luôn giữ valid/test target items
        |       |-- giữ thêm một phần cold items
        |-- sort interaction theo timestamp
        |-- tạo chuỗi hành vi user
        |-- train / valid / test theo sequence
        |-- tính item_popularity chỉ trên train split
        |-- gán cold_item_flag theo train popularity

        ↓

Feature Extraction
        |-- Text sentence:
        |       Title + Brand + Category + Features + Description
        |
        |-- Text Encoder:
        |       MiniLM / BGE-small
        |
        |-- Image Encoder:
        |       CLIP ViT-B/32
        |
        |-- Time Feature:
        |       interval, month, weekday

        ↓

CS-TAMoERec
        |-- ID Expert
        |-- Text Expert
        |-- Image Expert
        |-- Time Expert
        |-- Cross Expert
        |
        |-- Cold-start & Time-aware Router
        |
        |-- Adaptive item representation

        ↓

SASRec-style Transformer
        ↓
Next-item Prediction
        ↓
Top-K Recommendation + Expert Weight Explanation
```

## Pipeline 2-Stage Bản Nâng Cấp

Bản mới của project không chỉ train một model rồi full-sort toàn bộ item. Project có thêm pipeline giống hệ thống recommendation thực tế:

```text
Stage 1: Candidate Generation
        |-- Popularity candidates
        |-- ItemCF / co-occurrence candidates
        |-- Transition graph candidates
        |-- Text similarity candidates
        |-- Image similarity candidates
        |
        ↓
        Candidate pool 200-500 items

Stage 2: CS-TAMoERec MoE Reranker
        |-- score(user_history, candidate)
        |-- expert weights
        |-- candidate source explanation
        ↓
        Top-K recommendation
```

Stage 1 giúp tăng Recall và mô phỏng hệ recommender thật. Stage 2 dùng MoE để rerank và giải thích vì sao item được chọn.

Lưu ý hiện tại:

```text
Transition graph đang được dùng như candidate source ở Stage 1.
Graph chưa phải neural expert trong MoE.
```

Đây là lựa chọn an toàn để có graph signal mà không phải train GNN phức tạp. Nếu còn thời gian có thể nâng cấp thành `Graph Expert`.

## Kiến Trúc Mô Hình

Với mỗi item trong sequence, mô hình tạo các representation:

```text
ID embedding       = trainable item embedding
Text embedding     = encoded product metadata
Image embedding    = encoded product image
Time embedding     = timestamp / interval feature
Cold feature       = log(popularity + 1), cold_item_flag
```

Sau đó đưa vào MoE:

```text
E_id      : ID Expert
E_text    : Text Expert
E_image   : Image Expert
E_time    : Time Expert
E_cross   : Cross-modal Expert
```

Router học trọng số:

```text
w_id, w_text, w_image, w_time, w_cross = softmax(MLP(router_input))
```

Item representation cuối:

```text
item_repr =
    w_id    * E_id
  + w_text  * E_text
  + w_image * E_image
  + w_time  * E_time
  + w_cross * E_cross
```

Chuỗi `item_repr` được đưa vào Transformer để dự đoán item tiếp theo.

## Loss Function

Mô hình dùng:

```text
L = L_next + lambda1 * L_category + lambda2 * L_id_mm_alignment
```

Trong đó:

- `L_next`: cross entropy cho next-item prediction.
- `L_category`: dự đoán category của item, lấy cảm hứng từ CP task của HM4SR.
- `L_id_mm_alignment`: kéo ID-based representation và multimodal representation lại gần nhau.
- `L_router_balance`: tránh router collapse vào một expert duy nhất.

Loss thực tế trong code:

```text
L = L_next
  + lambda1 * L_category
  + lambda2 * L_id_mm_alignment
  + lambda3 * L_router_balance
```

## Cấu Trúc Code

```text
cstamoerec/
  config.py          # load config yaml
  data.py            # dataset, artifact, sequence processing helpers
  features.py        # text/image feature extraction
  model.py           # CS-TAMoERec model
  metrics.py         # HR, NDCG, MRR, Recall, Coverage
  train.py           # train/evaluate logic
  candidate.py       # popularity, ItemCF, transition, text/image candidates
  reranker.py        # MoE reranking logic

scripts/
  prepare_amazon2023.py      # chuẩn bị Amazon Reviews 2023
  train_cstamoerec.py        # train model
  run_ablation.py            # chạy ablation
  evaluate_perturbation.py   # mask/shuffle text/image
  evaluate_counterfactual.py # rank thay đổi khi bỏ text/image/time
  analyze_experts.py         # phân tích expert weights
  evaluate_candidates.py     # đánh giá Recall@K của candidate generation
  rerank_candidates.py       # đánh giá two-stage reranking
  export_demo_cache.py       # export cache cho demo Streamlit

config/
  cstamoerec_all_beauty.yaml # config chính

demo_streamlit.py            # demo giao diện Streamlit
requirements_cstamoerec.txt  # thư viện cần cài
```

HM4SR gốc vẫn nằm ở:

```text
recbole_model/HM4SR.py
run_hm4sr.py
recbole/
```

## Cài Đặt

Trên Kaggle/Colab:

```bash
pip install -r requirements_cstamoerec.txt
```

## Chuẩn Bị Dữ Liệu

Chạy bản nhanh, chưa tải ảnh:

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml \
  --skip-images
```

Chạy bản đầy đủ có image embedding:

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml
```

Debug nhanh với ít review:

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml \
  --skip-images \
  --limit-reviews 20000
```

Sau khi prepare, dữ liệu cache nằm ở:

```text
data/processed/all_beauty/
  examples.pt
  features.pt
  meta.json
  item_cards.json
```

## Train Model

```bash
python scripts/train_cstamoerec.py \
  --config config/cstamoerec_all_beauty.yaml
```

Checkpoint tốt nhất:

```text
checkpoints/cstamoerec/best_cstamoerec.pt
```

Kết quả training lưu ở:

```text
checkpoints/cstamoerec/history.json
```

## Baseline Và Ablation

Baseline hiện có:

- Popularity baseline
- ID-only, gần tương đương SASRec-ID
- No-text
- No-image
- No-time
- No-cold-router
- No-cross
- No-aux-loss
- No-router-balance

Chạy ablation:

```bash
python scripts/run_ablation.py \
  --config config/cstamoerec_all_beauty.yaml \
  --variants full id_only no_text no_image no_time no_cold_router \
  --epochs 5
```

Kết quả:

```text
checkpoints/cstamoerec/ablation/ablation_summary.json
```

## Candidate Generation Evaluation

Đánh giá chất lượng Stage 1 bằng Recall@50/100/200:

```bash
python scripts/evaluate_candidates.py \
  --config config/cstamoerec_all_beauty.yaml \
  --split test \
  --per-source-k 200 \
  --max-candidates 500
```

Kết quả:

```text
checkpoints/cstamoerec/candidate_recall_test.json
```

Bảng nên đưa vào báo cáo:

```text
Candidate Source     Recall@50   Recall@100   Recall@200
Popularity           ...
ItemCF               ...
Transition graph     ...
Text similarity      ...
Image similarity     ...
Combined             ...
```

## Two-Stage Reranking Evaluation

Đánh giá Stage 1 + Stage 2:

```bash
python scripts/rerank_candidates.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --split test \
  --per-source-k 100 \
  --max-candidates 300
```

Kết quả:

```text
checkpoints/cstamoerec/two_stage_rerank_test.json
```

## Perturbation Test

Dùng để kiểm tra text/image có thật sự đóng góp không.

```bash
python scripts/evaluate_perturbation.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt
```

Các mode:

- `full`
- `mask_text`
- `mask_image`
- `shuffle_text`
- `shuffle_image`
- `mask_text_image`

Nếu metric giảm khi mask/shuffle, có thể kết luận modality đó có đóng góp thực sự.

## Counterfactual Rank Test

Dùng để xem thứ hạng thay đổi thế nào khi bỏ một modality:

```bash
python scripts/evaluate_counterfactual.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --limit-users 100
```

Kết quả:

```text
checkpoints/cstamoerec/counterfactual_test.json
```

Ví dụ case study:

```text
Original recommendation: rank #1
After mask text: rank #5
After mask image: rank #3
After mask time: rank #7
```

## Expert Weight Analysis

Script này xuất trọng số trung bình của từng expert theo nhóm item:

- all items
- cold items
- warm items
- short time gap
- long time gap

Chạy:

```bash
python scripts/analyze_experts.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt
```

Kết quả:

```text
checkpoints/cstamoerec/expert_weights_test.json
```

Ví dụ bảng mong muốn trong báo cáo:

```text
Group           ID     Text   Image  Time   Cross
Cold items      0.14   0.36   0.28   0.10   0.12
Warm items      0.42   0.20   0.14   0.09   0.15
Long time gap   0.22   0.21   0.17   0.28   0.12
```

## Demo Streamlit

Sau khi train xong, nên export demo cache trước để demo chạy mượt:

```bash
python scripts/export_demo_cache.py \
  --config config/cstamoerec_all_beauty.yaml \
  --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt \
  --num-users 50 \
  --max-candidates 300
```

Cache:

```text
demo_cache/recommendations.json
```

Sau đó chạy demo:

```bash
streamlit run demo_streamlit.py
```

Demo có các tab:

- lịch sử mua hàng của user,
- top-10 sản phẩm được gợi ý,
- ảnh sản phẩm,
- category,
- cold/warm label,
- score,
- expert weights.
- candidate source,
- explainability,
- comparison/counterfactual analysis.

Ý nghĩa demo:

```text
Nếu item là cold-start, router có thể tăng trọng số Text/Image.
Nếu hành vi gần đây có khoảng cách thời gian lớn, Time Expert có thể tăng.
Nếu item phổ biến/warm, ID Expert có thể đóng vai trò lớn hơn.
```

## Metrics

Các metric chính:

- `HR@5`, `HR@10`, `HR@20`
- `NDCG@5`, `NDCG@10`, `NDCG@20`
- `MRR@5`, `MRR@10`, `MRR@20`
- `Recall@5`, `Recall@10`, `Recall@20`
- `Coverage@10`

Ngoài overall metric, project còn báo cáo:

- cold item performance,
- warm item performance,
- perturbation result,
- expert-weight analysis.

## Gợi Ý Chạy Trên Colab/Kaggle

Nếu GPU yếu hoặc muốn chạy thử nhanh:

Trong `config/cstamoerec_all_beauty.yaml`, giảm:

```yaml
data:
  max_items: 10000
  max_image_items: 0

model:
  hidden_size: 64
  n_layers: 1
  n_heads: 2

train:
  batch_size: 128
  epochs: 3
```

Và prepare bằng:

```bash
python scripts/prepare_amazon2023.py \
  --config config/cstamoerec_all_beauty.yaml \
  --skip-images
```

Khi mọi thứ chạy ổn, tăng dần:

- `max_items`
- `hidden_size`
- `epochs`
- bật image embedding.

## Đóng Góp Chính Của Project

1. Xây dựng pipeline Amazon Reviews 2023 `All_Beauty` cho multi-modal sequential recommendation.
2. Đề xuất CS-TAMoERec: cold-start and time-aware MoE item representation.
3. Kết hợp ID, text, image, time và cold-start feature trong adaptive item representation.
4. Thêm auxiliary supervision: category loss và ID-multimodal alignment loss.
5. Cung cấp explainability bằng expert weights.
6. Có đầy đủ ablation, perturbation test và cold/warm evaluation.
7. Bổ sung two-stage recommendation pipeline gồm candidate generation và MoE reranking.
8. Thêm router balance loss để giảm nguy cơ MoE router collapse.
9. Thêm counterfactual rank test để kiểm tra recommendation thay đổi khi bỏ text/image/time.

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

Repo này ban đầu dựa trên HM4SR/RecBole, sau đó được mở rộng thêm nhánh CS-TAMoERec cho project môn học.
