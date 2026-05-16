# Graph-Enhanced Multimodal Sequential Recommendation with Sequence-Node Retrieval

## 1. Tóm Tắt

Dự án này giải quyết bài toán **Multimodal Sequential Recommendation** trong bối cảnh dữ liệu thưa, lịch sử người dùng ngắn và tín hiệu đa phương thức có nhiễu. Thay vì tiếp tục dùng Transformer/MoE để full-item ranking, hướng chính hiện tại là một pipeline **two-stage recommendation**:

1. **Stage 1:** Graph-based candidate generation.
2. **Stage 2:** Candidate ordering bằng weighted source fusion.

Ý tưởng trung tâm là coi lịch sử hành vi của người dùng như một **sequence/context node**. Khi inference, hệ thống truy hồi các sequence tương tự trong tập train, sau đó lấy các item liên quan từ những sequence đó làm candidate. Cách này kế thừa tinh thần **sequences as nodes** của MuSICRec, nhưng triển khai theo hướng retrieval-based, nhẹ hơn và dễ phân tích hơn.

Trên Baby dataset dùng cùng split và feature với MuSICRec, kết quả full test hiện tại:

| Phương pháp | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| LightGCN | 0.0336 | 0.0549 | 0.0178 | 0.0231 |
| SGL | 0.0361 | 0.0586 | 0.0188 | 0.0244 |
| SASRec | 0.0298 | 0.0478 | 0.0150 | 0.0195 |
| BERT4Rec | 0.0207 | 0.0355 | 0.0102 | 0.0139 |
| FEARec | 0.0285 | 0.0480 | 0.0146 | 0.0195 |
| SRGNN | 0.0260 | 0.0418 | 0.0132 | 0.0172 |
| FREEDOM | 0.0411 | 0.0655 | 0.0210 | 0.0272 |
| SMORE | 0.0439 | 0.0684 | 0.0229 | 0.0291 |
| MuSICRec | **0.0455** | **0.0718** | 0.0235 | 0.0300 |
| Dự án hiện tại | 0.0419 | 0.0644 | **0.0246** | **0.0302** |

Kết luận trung thực: phương pháp hiện tại **chưa vượt MuSICRec tổng thể**, vì Recall@10 và Recall@20 còn thấp hơn. Tuy nhiên, dự án đạt **NDCG@10 và NDCG@20 cao hơn nhẹ**, đồng thời vượt nhiều baseline ID-only và sequential truyền thống trên cùng benchmark.

## 2. Động Lực Nghiên Cứu

Trong sequential recommendation, lịch sử người dùng thường được biểu diễn dưới dạng chuỗi:

```math
S_u = [i_1, i_2, \ldots, i_t]
```

Các mô hình như SASRec, BERT4Rec hoặc FEARec học biểu diễn chuỗi này bằng Transformer hoặc attention. Cách tiếp cận này mạnh khi có đủ dữ liệu, nhưng gặp hạn chế với:

- Người dùng có lịch sử ngắn.
- Dữ liệu tương tác thưa.
- Item mới hoặc ít tương tác.
- Text/image feature có nhiễu.

MuSICRec đưa ra insight quan trọng: thay vì chỉ xem sequence là input phẳng, có thể xem mỗi sequence như một node trong graph. Dự án này dùng insight đó cho candidate retrieval: tìm các sequence tương tự, sau đó dùng item trong các sequence đó để mở rộng candidate pool.

## 3. Bài Toán

Với mỗi user `u`, cho lịch sử tương tác:

```math
S_u = [i_1, i_2, \ldots, i_t]
```

nhiệm vụ là dự đoán item tiếp theo:

```math
i_{t+1}
```

Mô hình cần sinh danh sách top-K:

```math
\hat{Y}_u^K = [\hat{i}_1, \hat{i}_2, \ldots, \hat{i}_K]
```

sao cho ground-truth item `i_{t+1}` xuất hiện càng cao trong danh sách càng tốt.

## 4. Phương Pháp

### 4.1 Sequence-Node Retrieval

Mỗi training example được xem như một sequence node:

```math
s_j = (H_j, y_j)
```

trong đó:

- `H_j` là history của training sequence.
- `y_j` là target item của training sequence.

Với query user hiện tại:

```math
q_u = [i_{t-l+1}, \ldots, i_t]
```

hệ thống tìm các sequence node gần nhất với `q_u`. Độ tương tự được tính dựa trên overlap giữa recent items của query và item trong sequence node.

Một dạng Jaccard similarity cơ bản:

```math
\operatorname{sim}_{jac}(q_u, s_j)
=
\frac{|I(q_u) \cap I(s_j)|}{|I(q_u) \cup I(s_j)|}
```

Trong đó `I(.)` là tập item xuất hiện trong sequence.

Để ưu tiên item gần thời điểm hiện tại hơn, có thể dùng recency-weighted overlap:

```math
\operatorname{sim}_{rec}(q_u, s_j)
=
\sum_{i \in I(q_u) \cap I(s_j)} \alpha^{t - pos_q(i)}
```

với:

- `pos_q(i)` là vị trí của item `i` trong query sequence.
- `0 < \alpha \leq 1` điều khiển mức giảm trọng số theo thời gian.

Candidate từ sequence graph được lấy từ target và các item liên quan trong các sequence node có điểm cao:

```math
C_{seq}(u) = \bigcup_{s_j \in \operatorname{TopM}(q_u)} \{y_j\} \cup H_j
```

Đây là thành phần mạnh nhất trong pipeline hiện tại.

### 4.2 Transition Graph

Transition graph học quan hệ chuyển tiếp theo thứ tự thời gian. Với mỗi cặp item liên tiếp trong sequence:

```math
(i_t, i_{t+1})
```

tạo cạnh có hướng hoặc trọng số:

```math
w_{i_t, i_{t+1}} = w_{i_t, i_{t+1}} + 1
```

Candidate transition:

```math
C_{trans}(u) = \operatorname{TopK}(N^+(i_t))
```

trong đó `N^+(i_t)` là tập item thường xuất hiện ngay sau item cuối cùng `i_t`.

### 4.3 ItemCF Graph

ItemCF khai thác đồng xuất hiện item trong cùng history:

```math
w_{ij} = \sum_{u} \mathbb{1}(i \in S_u)\mathbb{1}(j \in S_u)
```

Một phiên bản chuẩn hóa có thể dùng cosine similarity:

```math
\operatorname{sim}_{cf}(i,j)
=
\frac{w_{ij}}{\sqrt{freq(i)freq(j)}}
```

Candidate ItemCF lấy các item gần với những item gần đây của user:

```math
C_{cf}(u) = \bigcup_{i \in q_u} \operatorname{TopK}(\operatorname{sim}_{cf}(i, \cdot))
```

### 4.4 Text/Image Similarity Graph

Dự án dùng feature có sẵn từ MuSICRec:

- `text_feat.npy`: 384 chiều.
- `image_feat.npy`: 4096 chiều.

Với mỗi modality `m`, tính cosine similarity:

```math
\operatorname{sim}_{m}(i,j)
=
\frac{\mathbf{x}_i^{m} \cdot \mathbf{x}_j^{m}}
{\|\mathbf{x}_i^{m}\|_2 \|\mathbf{x}_j^{m}\|_2}
```

Sau đó xây dựng top-k similarity graph:

```math
E_m = \{(i,j) \mid j \in \operatorname{TopK}(\operatorname{sim}_{m}(i,\cdot))\}
```

Thực nghiệm hiện tại cho thấy image signal trên Baby yếu, nên image chỉ được dùng như auxiliary signal với trọng số thấp.

### 4.5 LightGCN Cho Graph Embedding

Với graph item-item, LightGCN lan truyền embedding qua các tầng:

```math
\mathbf{e}_i^{(l+1)}
=
\sum_{j \in \mathcal{N}(i)}
\frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}}
\mathbf{e}_j^{(l)}
```

Embedding cuối cùng là trung bình các layer:

```math
\mathbf{e}_i
=
\frac{1}{L+1}
\sum_{l=0}^{L}
\mathbf{e}_i^{(l)}
```

Trong dự án, LightGCN được dùng để làm giàu graph embedding cho text/image/id graph, không phải claim chính thay cho MuSICRec.

## 5. Candidate Fusion

Các nguồn candidate gồm:

- `sequence_graph`
- `transition`
- `itemcf`
- `text`
- `image`
- `text_graph`
- `image_graph`
- `popularity`

Hệ thống hợp nhất candidate bằng **Weighted Reciprocal Rank Fusion**:

```math
\operatorname{score}(i \mid u)
=
\sum_{m \in \mathcal{M}}
\frac{w_m}{\operatorname{rank}_m(i)}
```

Trong đó:

- `m` là một nguồn candidate.
- `w_m` là trọng số của nguồn.
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

Các trọng số này được chọn theo validation-driven tuning. Không nên diễn giải đây là learned reranker; hiện tại nó vẫn là heuristic fusion.

## 6. Metrics

### 6.1 Recall@K / HR@K

Vì mỗi user trong validation/test có một ground-truth item, Recall@K tương đương Hit Rate@K:

```math
\operatorname{Recall@K}(u)
=
\mathbb{1}(y_u \in \hat{Y}_u^K)
```

Trung bình trên toàn bộ user:

```math
\operatorname{Recall@K}
=
\frac{1}{|\mathcal{U}|}
\sum_{u \in \mathcal{U}}
\mathbb{1}(y_u \in \hat{Y}_u^K)
```

### 6.2 NDCG@K

Nếu ground-truth item nằm ở vị trí `r_u`, với `1 <= r_u <= K`:

```math
\operatorname{NDCG@K}(u)
=
\frac{1}{\log_2(r_u + 1)}
```

Nếu không xuất hiện trong top-K:

```math
\operatorname{NDCG@K}(u) = 0
```

Trung bình:

```math
\operatorname{NDCG@K}
=
\frac{1}{|\mathcal{U}|}
\sum_{u \in \mathcal{U}}
\operatorname{NDCG@K}(u)
```

### 6.3 CandidatePoolHitRate

CandidatePoolHitRate đo tỉ lệ ground-truth item xuất hiện trong candidate pool trước khi lấy top-K:

```math
\operatorname{PoolHit@M}
=
\frac{1}{|\mathcal{U}|}
\sum_{u \in \mathcal{U}}
\mathbb{1}(y_u \in C_u^M)
```

Đây là chỉ số rất quan trọng với two-stage recommendation. Nếu target không nằm trong candidate pool, Stage 2 không thể xếp đúng item đó.

## 7. Thiết Lập Thực Nghiệm

### 7.1 Baby Dataset Theo MuSICRec

Dữ liệu được import từ repo MuSICRec:

```text
MuSICRec-3CEE/data/baby
```

Các file chính:

```text
baby_diff_split.inter
text_feat.npy
image_feat.npy
```

Sau khi import:

| Split | Số lượng |
|---|---:|
| Train | 102,457 |
| Valid | 19,445 |
| Test | 19,445 |
| Users | 19,446 |
| Items | 7,051 |
| Text dim | 384 |
| Image dim | 4096 |

Config:

```text
config/cstamoerec_musicrec_baby.yaml
```

### 7.2 Lệnh Chạy

Cài đặt:

```bash
pip install -r requirements_cstamoerec.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

Import Baby từ MuSICRec:

```bash
python scripts/prepare_musicrec_data.py \
  --input-dir /path/to/MuSICRec-3CEE/data/baby \
  --dataset baby \
  --output-dir data/processed/musicrec_baby \
  --max-seq-len 50 \
  --cold-threshold 5
```

Train graph embedding:

```bash
python scripts/train_lightgcn.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --epochs 20 \
  --batch-size 4096 \
  --similarity-topk 50 \
  --similarity-batch-size 512 \
  --device cuda
```

Đánh giá candidate recall:

```bash
python scripts/evaluate_candidates.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --topk 10 20 50 100 200 500 1000 \
  --device cuda
```

Đánh giá full test candidate-order:

```bash
python scripts/rerank_candidates.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --mode candidate \
  --fusion weighted \
  --split test \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

Chạy nhanh để debug:

```bash
python scripts/rerank_candidates.py \
  --config config/cstamoerec_musicrec_baby.yaml \
  --mode candidate \
  --fusion weighted \
  --split test \
  --limit-users 2000 \
  --per-source-k 500 \
  --max-candidates 1000 \
  --device cuda
```

## 8. Kết Quả Full Test Trên Baby

Kết quả full test hiện tại:

```text
HR@5 / R@5    = 0.0301
HR@10 / R@10  = 0.0419
HR@20 / R@20  = 0.0644
NDCG@5        = 0.0208
NDCG@10       = 0.0246
NDCG@20       = 0.0302
PoolHit@1000  = 0.4812
```

So với MuSICRec:

| Metric | MuSICRec | Dự án hiện tại | Nhận xét |
|---|---:|---:|---|
| R@10 | 0.0455 | 0.0419 | Thấp hơn |
| R@20 | 0.0718 | 0.0644 | Thấp hơn |
| N@10 | 0.0235 | 0.0246 | Cao hơn nhẹ |
| N@20 | 0.0300 | 0.0302 | Cao hơn nhẹ |

Điểm mạnh hiện tại là khả năng đưa một phần target lên vị trí cao, thể hiện qua NDCG cạnh tranh. Điểm yếu chính là candidate pool coverage: `PoolHit@1000 = 0.4812`, nghĩa là hơn một nửa ground-truth item không xuất hiện trong candidate pool.

## 9. So Sánh Với MuSICRec

MuSICRec mạnh hơn ở Recall vì có:

- End-to-end graph representation learning.
- User-sequence contrastive alignment.
- Sequence-item graph.
- ID-guided multimodal gating.

Loss alignment giữa user view và sequence view trong MuSICRec có dạng:

```math
\mathcal{L}_{US}
=
-\frac{1}{|\mathcal{U}|}
\sum_{u \in \mathcal{U}}
\log
\frac{
\exp(\operatorname{sim}(\mathbf{e}^{UI}_u, \mathbf{s}^{SI}_u)/\tau)
}{
\sum_{u' \in \mathcal{U}}
\exp(\operatorname{sim}(\mathbf{e}^{UI}_u, \mathbf{s}^{SI}_{u'})/\tau)
}
```

Dự án hiện tại chưa có contrastive loss này. Vì vậy không nên claim rằng phương pháp hiện tại đã thay thế hoàn toàn MuSICRec. Claim hợp lý hơn là:

> Sequence-node retrieval là một hướng nhẹ, dễ debug và có hiệu quả thực nghiệm tốt, đặc biệt ở NDCG, nhưng vẫn cần cải thiện candidate coverage và learned reranking để vượt MuSICRec tổng thể.

## 10. Đóng Góp Hiện Tại

Các đóng góp có thể trình bày một cách trung thực:

1. Chuyển hướng từ full-item neural ranking yếu sang graph-based two-stage retrieval hiệu quả hơn.
2. Khai thác sequence như context node để truy hồi candidate tương tự.
3. Kết hợp nhiều nguồn tín hiệu: sequence graph, transition, ItemCF, text, image, graph embedding.
4. Đánh giá trên Baby theo cùng split và feature với MuSICRec.
5. Chỉ ra rõ bottleneck thực nghiệm: candidate pool coverage.

## 11. Hạn Chế

1. **Chưa vượt MuSICRec tổng thể.** Recall@10/@20 vẫn thấp hơn.
2. **CandidatePoolHitRate còn thấp.** PoolHit@1000 chỉ đạt 0.4812.
3. **Fusion còn heuristic.** Weighted RRF chưa học tương tác phi tuyến giữa các nguồn.
4. **Image signal yếu trên Baby.** Image feature hiện tại chỉ nên dùng như tín hiệu phụ.
5. **Chưa có learned reranker.** Stage 2 hiện tại là candidate ordering bằng weighted fusion, không phải neural reranking.

## 12. Hướng Cải Thiện Tiếp Theo

Ưu tiên kỹ thuật nên làm tiếp:

1. **Tăng candidate coverage**
   - Thêm n-gram sequence retrieval.
   - Thêm prefix/suffix matching.
   - Tăng candidate pool từ 1000 lên 2000/3000 để đo trần recall.
   - Bổ sung long-tail-aware ItemCF.

2. **Huấn luyện learned reranker**
   - Input feature: rank/score từ sequence graph, transition, ItemCF, text, image, popularity, source count.
   - Loss có thể dùng BCE, BPR hoặc listwise softmax.

3. **Thêm lightweight multimodal gating**
   - Không nên re-embed ảnh nếu đang so sánh công bằng với MuSICRec.
   - Nên học trọng số text/image theo item hoặc theo category.

4. **Mở rộng thực nghiệm**
   - Baby là bước đầu.
   - Cần chạy thêm Sports và Electronics nếu muốn so sánh đầy đủ với bảng MuSICRec.

## 13. Kết Luận

Dự án hiện tại đã đi đúng hướng khi chuyển trọng tâm sang **graph-based sequence-node retrieval**. Trên Baby dataset, phương pháp vượt nhiều baseline truyền thống và đạt NDCG cạnh tranh với MuSICRec. Tuy nhiên, kết quả chưa đủ để claim vượt MuSICRec tổng thể vì Recall còn thấp hơn và candidate pool coverage là nút thắt lớn nhất.

Claim phù hợp nhất ở thời điểm hiện tại:

> Dự án đề xuất một pipeline two-stage nhẹ và dễ phân tích cho multimodal sequential recommendation. Phương pháp sequence-node retrieval cho kết quả cạnh tranh trên Baby, đặc biệt ở NDCG, nhưng cần cải thiện candidate coverage và learned reranking để đạt hoặc vượt MuSICRec một cách toàn diện.
