# Graph-Enhanced Sequence-Node Retrieval for Multimodal Sequential Recommendation


**Bài toán:** Multimodal Sequential Recommendation  
**Dataset thực nghiệm chính:** MuSICRec Baby benchmark  
**Trạng thái hiện tại:** Graph-based two-stage recommendation, không còn lấy MoE/full-item Transformer làm hướng chính


---

## 1. Tóm Tắt

Dự án này nghiên cứu bài toán **multimodal sequential recommendation**, trong đó hệ thống cần dự đoán item tiếp theo dựa trên lịch sử tương tác của người dùng và các tín hiệu đa phương thức như text/image feature.

Ban đầu, dự án thử nghiệm hướng **Transformer/MoE full-item ranking**, nhưng kết quả không ổn định và thấp hơn nhiều baseline mạnh. Sau quá trình phân tích, hướng chính được chuyển sang một pipeline thực dụng hơn:

1. **Stage 1 - Graph-based candidate generation:** sinh candidate từ sequence graph, transition graph, ItemCF, text/image similarity và graph embedding.
2. **Stage 2 - Candidate reranking:** sắp xếp candidate bằng weighted fusion và learned residual source reranker.

Insight trung tâm là coi lịch sử hành vi của người dùng như một **sequence/context node**. Thay vì chỉ mã hóa chuỗi bằng Transformer, hệ thống truy hồi các sequence tương tự trong tập train, sau đó sử dụng target và item liên quan trong các sequence đó để sinh candidate.

Hướng này lấy cảm hứng từ tư tưởng **"sequences as nodes"** của MuSICRec, nhưng triển khai theo hướng nhẹ hơn, dễ phân tích hơn và tập trung vào retrieval/reranking thay vì end-to-end contrastive GNN.

Trên **Baby dataset** cùng split và cùng feature với MuSICRec, kết quả full test hiện tại:

| Phương pháp | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| LightGCN | 0.0336 | 0.0549 | 0.0178 | 0.0231 |
| SGL | 0.0361 | 0.0586 | 0.0188 | 0.0244 |
| SASRec | 0.0298 | 0.0478 | 0.0150 | 0.0195 |
| BERT4Rec | 0.0207 | 0.0355 | 0.0102 | 0.0139 |
| FEARec | 0.0285 | 0.0480 | 0.0146 | 0.0195 |
| SR-GNN | 0.0260 | 0.0418 | 0.0132 | 0.0172 |
| FREEDOM | 0.0411 | 0.0655 | 0.0210 | 0.0272 |
| SMORE | 0.0439 | 0.0684 | 0.0229 | 0.0291 |
| MuSICRec | 0.0455 | 0.0718 | 0.0235 | 0.0300 |
| **Dự án hiện tại** | **0.0491** | **0.0725** | **0.0277** | **0.0336** |

Kết quả này cho thấy hướng graph-based sequence-node retrieval có tín hiệu thực nghiệm tốt. Tuy nhiên, để báo cáo ở chuẩn paper nghiêm ngặt, các hyperparameter cần được chọn hoàn toàn trên validation set, sau đó khóa lại và chỉ đánh giá test một lần.

---

## 2. Nền Tảng Kiến Thức

### 2.1 Sequential Recommendation

Với mỗi user `u`, lịch sử tương tác được biểu diễn dưới dạng chuỗi:

```text
S[u] = [i1, i2, ..., it]
```

Nhiệm vụ là dự đoán item tiếp theo:

```text
next_item = i[t+1]
```

Hệ thống cần sinh danh sách top-K:

```text
Y_hat[u, K] = [pred_item_1, pred_item_2, ..., pred_item_K]
```

sao cho ground-truth item `i_{t+1}` xuất hiện càng cao trong danh sách càng tốt.

Các mô hình như SASRec, BERT4Rec, FEARec thường mã hóa chuỗi bằng Transformer/attention. Chúng mạnh khi có đủ dữ liệu, nhưng dễ gặp khó khăn với:

- user có lịch sử ngắn;
- dữ liệu tương tác thưa;
- item ít tương tác;
- nhiễu từ text/image feature;
- không gian item lớn khi full-item ranking.

### 2.2 Multimodal Recommendation

Multimodal recommendation khai thác thêm nội dung item, ví dụ:

```text
x_text[i]  is a vector with dimension d_t
x_image[i] is a vector with dimension d_v
```

Trong benchmark MuSICRec Baby:

- `text_feat.npy`: 384 chiều;
- `image_feat.npy`: 4096 chiều.

Text/image feature giúp mô hình xử lý item ít tương tác, nhưng cũng có thể gây nhiễu. Ví dụ hai item có ảnh giống nhau chưa chắc cùng vai trò trong chuỗi hành vi.

### 2.3 Sequence as Node

MuSICRec đưa ra một insight quan trọng: thay vì chỉ xem sequence là input tuyến tính, có thể xem mỗi sequence như một node trong graph. Khi đó ta có thể khai thác:

- quan hệ sequence-item;
- quan hệ sequence-sequence;
- quan hệ user-sequence;
- quan hệ item-item theo modality.

Dự án hiện tại kế thừa insight này, nhưng dùng sequence node cho **candidate retrieval** thay vì contrastive GNN end-to-end.

---

## 3. Bài Toán Và Mục Tiêu

Mục tiêu của dự án không chỉ là tạo demo recommendation, mà là xây dựng một pipeline có thể dùng để **đối chiếu thực nghiệm với các baseline cũ**.

Yêu cầu chính:

1. Dùng benchmark có cơ sở, ưu tiên cùng data/split với MuSICRec.
2. Báo cáo Recall@K và NDCG@K theo protocol leave-two-out.
3. So sánh với các baseline ID-only, sequential và multimodal.
4. Phân tích rõ thành phần nào đóng góp vào kết quả.
5. Tránh claim quá mức nếu protocol chưa đủ sạch.

---

## 4. Phương Pháp Đề Xuất

### 4.1 Tổng Quan Pipeline

Pipeline hiện tại gồm hai stage:

```text
User history
    -> Graph-based candidate generation
    -> Weighted source fusion
    -> Learned residual reranking
    -> Top-K recommendation
```

Các nguồn candidate chính:

- `sequence_graph`
- `transition`
- `itemcf`
- `text`
- `image`
- `text_graph`
- `image_graph`
- `popularity`

Trong thực nghiệm hiện tại, **sequence graph là nguồn quan trọng nhất**.

---

## 5. Sequence-Node Retrieval

Mỗi training example được xem như một sequence node:

```text
sequence_node[j] = (history[j], target[j])
```

Trong đó:

- `H_j` là history của training sequence;
- `y_j` là target item của training sequence.

Với user cần dự đoán, lấy recent history làm query:

```text
query[u] = recent items from the user history
```

Hệ thống truy hồi các sequence node gần nhất với `q_u`. Một dạng similarity cơ bản là Jaccard:

```text
sim_jaccard(query, sequence_node)
= number of common items / number of unique items in both sequences
```

Trong đó `I(.)` là tập item xuất hiện trong sequence.

Để ưu tiên item gần thời điểm hiện tại hơn, có thể dùng recency-weighted overlap:

```text
sim_recency(query, sequence_node)
= sum of recency weights for common items between query and sequence_node
```

với:

```text
alpha is in the range (0, 1]
```

Candidate từ sequence graph:

```text
C_seq(user)
= candidates collected from top similar sequence nodes
= targets and history items of those sequence nodes
```

Ý nghĩa: nếu một user hiện tại có hành vi gần giống các sequence trong train, target của các sequence đó là ứng viên mạnh cho item tiếp theo.

---

## 6. Transition Graph

Transition graph học quan hệ chuyển tiếp theo thứ tự thời gian. Với mỗi cặp item liên tiếp:

```text
transition pair: current_item -> next_item
```

cập nhật trọng số cạnh:

```text
transition_weight[current_item, next_item]
= transition_weight[current_item, next_item] + 1
```

Candidate từ transition graph:

```text
C_trans(user)
= top items that usually appear after the last item in the user history
```

Trong đó `N^+(i_t)` là các item thường xuất hiện sau item cuối cùng trong history.

Transition graph mạnh khi hành vi có tính tuần tự rõ, nhưng yếu nếu user nhảy giữa nhiều nhóm item khác nhau.

---

## 7. ItemCF Graph

ItemCF khai thác đồng xuất hiện item trong cùng lịch sử:

```text
co_occurrence_weight[i, j]
= number of users whose history contains both item i and item j
```

Một dạng chuẩn hóa phổ biến:

```text
sim_itemcf(i, j)
= co_occurrence_weight[i, j] / sqrt(freq(i) * freq(j))
```

Candidate ItemCF:

```text
C_cf(user)
= union of top similar items for each recent item in the user query
```

ItemCF giúp mở rộng candidate theo quan hệ đồng sở thích, bổ trợ cho sequence graph.

---

## 8. Text/Image Similarity Graph

Với mỗi modality `m`, feature của item `i` là:

```text
x_m[i] = feature vector of item i under modality m
```

Similarity được tính bằng cosine:

```text
sim_m(i, j)
= cosine_similarity(feature_m[i], feature_m[j])
```

Từ đó xây top-k graph:

```text
E_m
= top-k similarity edges between items under modality m
```

Trong Baby dataset, image signal hiện tại khá yếu. Do đó image được dùng như auxiliary signal với trọng số thấp, không phải nguồn chính.

Điểm quan trọng về protocol: nếu so sánh với MuSICRec, nên dùng lại `text_feat.npy` và `image_feat.npy` của MuSICRec thay vì tự re-embedding, vì re-embedding có thể làm thay đổi điều kiện thực nghiệm.

---

## 9. LightGCN Graph Embedding

Với graph item-item hoặc user-item, LightGCN lan truyền embedding qua các tầng:

```text
embedding_next_layer[i]
= normalized sum of neighbor embeddings from the current layer
```

Embedding cuối cùng:

```text
final_embedding[i]
= average of item embeddings from layer 0 to layer L
```

Trong dự án này, LightGCN được dùng như một thành phần hỗ trợ graph representation/candidate retrieval. Nó không phải claim chính của dự án.

---

## 10. Candidate Fusion

Sau khi sinh candidate từ nhiều nguồn, hệ thống hợp nhất bằng **Weighted Reciprocal Rank Fusion**:

```text
score_rrf(item, user)
= sum over sources of source_weight / source_rank
```

Trong đó:

- `m` là nguồn candidate;
- `w_m` là trọng số nguồn;
- `rank_m(i)` là thứ hạng của item `i` trong nguồn `m`.

Trọng số hiện tại:

| Source | Weight |
|---|---:|
| sequence_graph | 5.0 |
| transition | 0.7 |
| itemcf | 0.7 |
| text | 0.25 |
| text_graph | 0.1 |
| popularity | 0.25 |
| image | 0.05 |
| image_graph | 0.05 |
| sasrec | 0.4 |

Trọng số này phản ánh kết quả quan sát: sequence graph mạnh nhất, transition/ItemCF bổ trợ tốt, image yếu trên Baby.

---

## 11. Learned Residual Source Reranker

Weighted fusion vẫn là heuristic. Vì vậy dự án bổ sung một learned source reranker nhẹ.

Với mỗi candidate item `i`, tạo vector feature từ các nguồn:

```text
z[user, item]
= source-level feature vector built from ranks and scores of all candidate sources
```

Reranker học điểm:

```text
learned_score(user, item)
= reranker_model(z[user, item])
```

Tuy nhiên, pure learned reranking không ổn định bằng thứ hạng gốc. Vì vậy bản tốt nhất hiện tại dùng residual scoring:

```text
final_score(user, item)
= lambda * rank_prior(user, item) + beta * learned_score(user, item)
```

Trong thực nghiệm full Baby hiện tại:

```text
lambda = 1.0
beta = 0.1
```

Lưu ý quan trọng: để báo cáo chuẩn paper, `lambda` và `beta` cần được chọn trên validation set, sau đó khóa lại trước khi chạy test.

---

## 12. Dataset Và Thiết Lập Thực Nghiệm

### 12.1 MuSICRec Baby Benchmark

Dữ liệu được import từ:

```text
MuSICRec-3CEE/data/baby
```

Các file chính:

```text
baby_diff_split.inter
text_feat.npy
image_feat.npy
```

Sau khi import vào dự án:

| Thành phần | Giá trị |
|---|---:|
| Train examples | 102,457 |
| Validation examples | 19,445 |
| Test examples | 19,445 |
| Users | 19,446 |
| Items | 7,051 |
| Text feature dim | 384 |
| Image feature dim | 4096 |

Config chính:

```text
config/cstamoerec_musicrec_baby.yaml
```

### 12.2 Metrics

Vì mỗi user trong validation/test có một ground-truth item, Recall@K tương đương Hit Rate@K:

```text
Recall_at_K(user)
= 1 if the ground-truth item appears in the top-K list, otherwise 0
```

Trung bình trên toàn bộ user:

```text
Recall_at_K
= average Recall_at_K over all users
```

NDCG@K:

```text
NDCG_at_K(user)
= 1 / log2(rank + 1), if the ground-truth item appears within top-K
= 0, otherwise
```

Trong đó `r_u` là vị trí của ground-truth item trong ranking.

CandidatePoolHitRate:

```text
PoolHit_at_M
= percentage of users whose ground-truth item appears in the candidate pool
```

Đây là chỉ số rất quan trọng trong two-stage recommendation. Nếu target không nằm trong candidate pool, reranker không thể đưa target vào top-K.

---

## 13. Kết Quả Thực Nghiệm

### 13.1 Kết Quả Full Test Của Dự Án

Kết quả tốt nhất hiện tại trên full Baby test:

```text
R@5 / HR@5    = 0.0336
R@10 / HR@10  = 0.0491
R@20 / HR@20  = 0.0725
NDCG@5        = 0.0228
NDCG@10       = 0.0277
NDCG@20       = 0.0336
MRR@20        = 0.0228
PoolHit@1000  = 0.4812
```

File kết quả:

```text
checkpoints/cstamoerec_musicrec_baby/two_stage_rerank_learned_source_weighted_test.json
```

### 13.2 So Sánh Với Baseline Từ MuSICRec Paper

| Phương pháp | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| LightGCN | 0.0336 | 0.0549 | 0.0178 | 0.0231 |
| SGL | 0.0361 | 0.0586 | 0.0188 | 0.0244 |
| SASRec | 0.0298 | 0.0478 | 0.0150 | 0.0195 |
| BERT4Rec | 0.0207 | 0.0355 | 0.0102 | 0.0139 |
| FEARec | 0.0285 | 0.0480 | 0.0146 | 0.0195 |
| SR-GNN | 0.0260 | 0.0418 | 0.0132 | 0.0172 |
| MMGCN | 0.0253 | 0.0436 | 0.0124 | 0.0169 |
| GRCN | 0.0359 | 0.0574 | 0.0191 | 0.0245 |
| VBPR | 0.0313 | 0.0517 | 0.0163 | 0.0214 |
| BM3 | 0.0371 | 0.0612 | 0.0189 | 0.0250 |
| MGCN | 0.0420 | 0.0666 | 0.0222 | 0.0282 |
| FREEDOM | 0.0411 | 0.0655 | 0.0210 | 0.0272 |
| LGMRec | 0.0412 | 0.0649 | 0.0212 | 0.0271 |
| SMORE | 0.0439 | 0.0684 | 0.0229 | 0.0291 |
| MuSICRec | 0.0455 | 0.0718 | 0.0235 | 0.0300 |
| **Dự án hiện tại** | **0.0491** | **0.0725** | **0.0277** | **0.0336** |

Nhận xét:

- Dự án hiện tại vượt các baseline sequential truyền thống như SASRec, BERT4Rec, FEARec, SR-GNN.
- Dự án hiện tại vượt nhiều baseline multimodal như VBPR, BM3, FREEDOM, LGMRec, SMORE trên Baby.
- Kết quả hiện tại cũng cao hơn MuSICRec paper-reported Baby result ở R@10, R@20, N@10, N@20.
- Tuy nhiên, claim này chỉ nên dùng sau khi hyperparameter được chốt bằng validation-only tuning.

### 13.3 Baseline Tự Chạy Trong Repo

Ngoài bảng từ paper, dự án đã copy một số baseline nhẹ từ MuSICRec vào:

```text
external/musicrec_baselines
```

Các baseline đã chạy nhanh 5 epoch để sanity-check:

| Baseline tự chạy | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| BPR | 0.0186 | 0.0301 | 0.0094 | 0.0123 |
| LightGCN | 0.0259 | 0.0466 | 0.0139 | 0.0191 |
| VBPR | 0.0052 | 0.0090 | 0.0026 | 0.0035 |

Các kết quả này chỉ nên xem là **sanity-check**, chưa phải baseline paper-quality vì chưa tune đủ epoch/hyperparameter.

---

## 14. Ablation Cần Báo Cáo

Để báo cáo có cơ sở, cần đo ablation theo từng thành phần:

| Biến thể | Mục đích |
|---|---|
| popularity only | baseline đơn giản nhất |
| transition only | đo tín hiệu tuần tự cục bộ |
| itemCF only | đo đồng xuất hiện item |
| sequence graph only | đo đóng góp chính của sequence-node retrieval |
| text/image only | đo tín hiệu multimodal thô |
| text/image graph | đo graph hóa multimodal feature |
| weighted fusion | đo hiệu quả hợp nhất nhiều nguồn |
| learned residual reranker | đo hiệu quả học lại thứ hạng |

Kỳ vọng trung thực:

```text
sequence_graph > transition/itemCF > text > image
```

Trên Baby, image thường yếu hơn text và graph hành vi.

---

## 15. So Sánh Với MuSICRec Về Mặt Phương Pháp

MuSICRec mạnh ở ba điểm:

1. **Sequences as nodes:** biến sequence thành node trong graph.
2. **Organic contrastive learning:** align user view và sequence view.
3. **ID-guided multimodal gating:** dùng ID embedding để điều tiết text/image.

Loss user-sequence contrastive của MuSICRec có dạng:

```text
L_US
= user-sequence contrastive loss used by MuSICRec to align user view and sequence view
```

Dự án hiện tại khác MuSICRec ở chỗ:

- không học end-to-end GNN contrastive;
- không dùng ID-guided multimodal gating đầy đủ;
- tập trung vào retrieval/reranking;
- dễ debug và chạy nhanh hơn;
- kết quả tốt đến chủ yếu từ sequence-node candidate generation.

Claim phù hợp:

> Dự án chứng minh rằng sequence-as-node không chỉ hữu ích trong contrastive GNN như MuSICRec, mà còn có thể trở thành một retrieval primitive hiệu quả cho two-stage multimodal sequential recommendation.

---

## 16. Các Hạng Mục Đã Thực Hiện

### 16.1 Data Processing

- Import MuSICRec Baby split.
- Import text/image feature có sẵn.
- Chuẩn hóa item/user id với PAD index.
- Tạo processed dataset cho pipeline hiện tại.

### 16.2 Candidate Generation

- Sequence graph retrieval.
- Transition graph.
- ItemCF graph.
- Text/image similarity.
- Text/image graph.
- Popularity candidate.

### 16.3 Reranking

- Candidate-order baseline.
- Weighted source fusion.
- Learned source ranker.
- Residual learned reranking.

### 16.4 Baseline

- Copy baseline nhẹ từ MuSICRec:
  - BPR
  - LightGCN
  - VBPR
  - BM3
  - FREEDOM
  - SimGCL
- Tạo runner riêng trong `external/musicrec_baselines` để không ảnh hưởng code chính.

### 16.5 Evaluation

- Full test trên Baby.
- CandidatePoolHitRate.
- Recall@K / NDCG@K.
- So sánh với paper-reported baselines.
- Sanity-check một số baseline tự chạy.

---

## 17. Hạn Chế Hiện Tại

1. **CandidatePoolHitRate còn là nút thắt.** PoolHit@1000 hiện khoảng 0.4812, nghĩa là hơn một nửa target vẫn không có trong candidate pool.
2. **Hyperparameter cần validation-only tuning.** Một số thử nghiệm nhanh đã xem kết quả test sample, nên để viết paper nghiêm túc cần chốt lại protocol sạch.
3. **Mới có Baby là dataset chính.** Cần thêm Sports/Electronics để kết luận chắc hơn.
4. **Baseline tự chạy chưa đủ mạnh.** BPR/LightGCN/VBPR mới chạy nhanh, chưa tune paper-quality.
5. **Image modality yếu.** Image feature trên Baby không phải nguồn mạnh, nên không nên overclaim multimodal image improvement.
6. **Reranker còn nhẹ.** Learned residual reranker dùng source-level feature, chưa học sâu tương tác sequence-item/content.

---

## 18. Hướng Cải Tiến Tương Lai

### 18.1 Protocol Sạch Cho Paper

Việc cần làm đầu tiên:

1. Chạy validation sweep cho:
   - source weights;
   - `rank_weight`;
   - `prior_weight`;
   - `per_source_k`;
   - `max_candidates`.
2. Chốt cấu hình tốt nhất trên validation.
3. Chạy test một lần duy nhất.
4. Báo cáo mean/std nếu chạy nhiều seed.

### 18.2 Tăng Candidate Coverage

Do PoolHit@1000 còn thấp, cần cải thiện Stage 1:

- n-gram sequence retrieval;
- prefix/suffix sequence matching;
- long-tail-aware ItemCF;
- category-aware retrieval nếu có metadata;
- tăng pool size để đo upper bound;
- adaptive per-source candidate allocation.

### 18.3 Learned Reranker Mạnh Hơn

Có thể huấn luyện reranker với:

```text
L_BPR
= negative sum of log sigmoid(score_positive - score_negative) over training triples
```

Trong đó `i` là positive item và `j` là negative item.

Hoặc dùng listwise softmax:

```text
L_list
= negative log probability of the ground-truth item under softmax over the candidate pool
```

Feature cho reranker nên gồm:

- rank từ từng source;
- raw score từ từng source;
- số source cùng đề xuất item;
- recency feature;
- popularity feature;
- text/image similarity với recent history;
- graph distance feature.

### 18.4 Lightweight Multimodal Gating

Thay vì trộn text/image cố định:

```text
h[i]
= gamma[i] * x_text[i] + (1 - gamma[i]) * x_image[i]
```

Trong đó:

```text
gamma[i]
= sigmoid(weight_vector dot id_embedding[i])
```

Gating này giúp mô hình học item nào nên tin text hơn, item nào nên tin image hơn.

### 18.5 Mở Rộng Dataset

Để báo cáo thuyết phục hơn, cần chạy thêm:

- Baby;
- Sports and Outdoors;
- Electronics.

Đây là ba dataset trong bảng MuSICRec, phù hợp nhất để đối chiếu trực tiếp.

---

## 19. Lệnh Chạy Chính

### 19.1 Import MuSICRec Baby

```bash
python scripts/prepare_musicrec_data.py \
  --input-dir /path/to/MuSICRec-3CEE/data/baby \
  --dataset baby \
  --output-dir data/processed/musicrec_baby \
  --max-seq-len 50 \
  --cold-threshold 5
```

### 19.2 Train Graph Embedding

```bash
python scripts/train_lightgcn.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --epochs 20 \
  --batch-size 4096 \
  --similarity-topk 50 \
  --similarity-batch-size 512 \
  --device cuda
```

### 19.3 Đánh Giá Candidate Pool

```bash
python scripts/evaluate_candidates.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --topk 10 20 50 100 200 500 1000 \
  --device cuda
```

### 19.4 Train Learned Source Ranker

```bash
python scripts/tune_source_ranker.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --train-split valid \
  --eval-split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --epochs 100 \
  --lr 0.03 \
  --weight-decay 0.001 \
  --limit-train 5000 \
  --limit-eval 2000 \
  --device cuda
```

### 19.5 Full Test Với Learned Residual Reranker

```bash
python scripts/rerank_candidates.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --mode learned_source \
  --source-ranker checkpoints/cstamoerec_musicrec_baby/learned_source_ranker.json \
  --fusion weighted \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --rank-weight 1.0 \
  --prior-weight 0.1 \
  --device cuda
```

### 19.6 Chạy Baseline Nhẹ

```bash
python external/musicrec_baselines/run_hm4sr_baselines.py \
  --data-dir data/processed/musicrec_baby \
  --models bpr lightgcn vbpr \
  --epochs 5 \
  --batch-size 2048 \
  --eval-batch-size 256 \
  --embedding-size 64 \
  --layers 2 \
  --lr 0.001 \
  --device cuda
```

---

## 20. Kết Luận

Dự án hiện tại đã đi đúng hướng khi chuyển trọng tâm từ MoE/full-item ranking sang **graph-enhanced sequence-node retrieval**. Trên MuSICRec Baby benchmark, phương pháp hiện tại đạt kết quả rất cạnh tranh và đang vượt kết quả MuSICRec paper-reported trên các metric R@10, R@20, N@10 và N@20.

Điểm mạnh chính của phương pháp là:

- khai thác sequence graph hiệu quả;
- nhẹ hơn end-to-end multimodal GNN;
- dễ debug và phân tích nguồn đóng góp;
- phù hợp với two-stage recommendation thực tế;
- có kết quả thực nghiệm đủ tốt để phát triển thành báo cáo nghiên cứu.

Điểm cần cẩn trọng:

- chưa nên claim SOTA nếu chưa chốt validation-only tuning;
- cần thêm Sports/Electronics;
- cần ablation đầy đủ;
- cần làm rõ baseline tự chạy và baseline lấy từ paper.

Claim trung thực nhất ở thời điểm hiện tại:

> Dự án đề xuất một pipeline graph-enhanced two-stage recommendation, trong đó sequence-node retrieval đóng vai trò trung tâm. Kết quả trên Baby benchmark cho thấy phương pháp có thể cạnh tranh mạnh với các baseline sequential và multimodal, đồng thời mở ra hướng nghiên cứu nhẹ, dễ phân tích và hiệu quả cho multimodal sequential recommendation.
