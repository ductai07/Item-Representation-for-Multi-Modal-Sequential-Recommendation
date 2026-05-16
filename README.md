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

Trong sequential recommendation, lịch sử người dùng được biểu diễn như một chuỗi các item đã tương tác. Các mô hình như SASRec, BERT4Rec hoặc FEARec học biểu diễn chuỗi này bằng Transformer hoặc attention. Cách tiếp cận này mạnh khi có đủ dữ liệu, nhưng gặp hạn chế với:

- Người dùng có lịch sử ngắn.
- Dữ liệu tương tác thưa.
- Item mới hoặc ít tương tác.
- Text/image feature có nhiễu.

MuSICRec đưa ra insight quan trọng: thay vì chỉ xem sequence là input phẳng, có thể xem mỗi sequence như một node trong graph. Dự án này dùng insight đó cho candidate retrieval: tìm các sequence tương tự, sau đó dùng item trong các sequence đó để mở rộng candidate pool.

## 3. Bài Toán

Với mỗi user, hệ thống nhận vào lịch sử tương tác trước đó và cần dự đoán item tiếp theo mà user có khả năng tương tác. Kết quả đầu ra là danh sách top-K item được xếp hạng theo mức độ phù hợp.

Mục tiêu đánh giá là ground-truth item xuất hiện càng cao trong danh sách khuyến nghị càng tốt.

## 4. Phương Pháp

### 4.1 Sequence-Node Retrieval

Mỗi training example được xem như một sequence node. Một sequence node gồm hai phần:

- History của training sequence.
- Target item tương ứng với sequence đó.

Với query user hiện tại, hệ thống lấy một đoạn lịch sử gần nhất của user làm query sequence. Sau đó, hệ thống tìm các sequence node trong tập train có lịch sử gần giống với query sequence.

Độ tương tự giữa hai sequence được tính dựa trên mức độ overlap giữa các item gần đây. Ngoài overlap thông thường, hệ thống có thể ưu tiên các item xuất hiện gần thời điểm hiện tại hơn bằng recency-weighted overlap.

Candidate từ sequence graph được lấy từ target item và các item liên quan trong những sequence node có điểm tương tự cao.

Đây là thành phần mạnh nhất trong pipeline hiện tại.

### 4.2 Transition Graph

Transition graph học quan hệ chuyển tiếp theo thứ tự thời gian. Nếu một item thường xuất hiện ngay sau một item khác trong lịch sử tương tác, hệ thống tạo hoặc tăng trọng số cạnh chuyển tiếp giữa hai item đó.

Khi inference, hệ thống nhìn vào item cuối cùng trong lịch sử user, sau đó lấy các item thường xuất hiện ngay sau item đó làm candidate.

### 4.3 ItemCF Graph

ItemCF khai thác quan hệ đồng xuất hiện giữa các item trong cùng history. Hai item được xem là gần nhau nếu chúng thường xuất hiện trong cùng lịch sử người dùng.

Để tránh việc item quá phổ biến chi phối kết quả, hệ thống có thể chuẩn hóa điểm đồng xuất hiện bằng tần suất của từng item.

Candidate ItemCF được lấy từ các item gần với những item gần đây trong lịch sử user.

### 4.4 Text/Image Similarity Graph

Dự án dùng feature có sẵn từ MuSICRec:

- `text_feat.npy`: 384 chiều.
- `image_feat.npy`: 4096 chiều.

Với mỗi modality, hệ thống tính mức độ tương tự giữa các item dựa trên feature vector. Sau đó, hệ thống xây dựng top-k similarity graph cho từng modality.

Thực nghiệm hiện tại cho thấy image signal trên Baby yếu, nên image chỉ được dùng như auxiliary signal với trọng số thấp.

### 4.5 LightGCN Cho Graph Embedding

Với graph item-item, LightGCN lan truyền embedding giữa các item hàng xóm qua nhiều tầng. Embedding cuối cùng được tổng hợp từ embedding ở các tầng khác nhau.

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
- `sasrec`

Hệ thống hợp nhất candidate bằng **Weighted Reciprocal Rank Fusion**. Ý tưởng là mỗi nguồn candidate đóng góp điểm cho item dựa trên thứ hạng của item trong nguồn đó và trọng số của nguồn.

Nếu một item xuất hiện ở thứ hạng cao trong nhiều nguồn quan trọng, item đó sẽ có điểm tổng hợp cao hơn.

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

Recall@K đo xem ground-truth item có xuất hiện trong top-K recommendation hay không.

Vì mỗi user trong validation/test có một ground-truth item, Recall@K tương đương Hit Rate@K.

### 6.2 NDCG@K

NDCG@K đánh giá vị trí xuất hiện của ground-truth item trong danh sách top-K.

Nếu ground-truth item xuất hiện càng cao trong danh sách, NDCG càng cao. Nếu item không xuất hiện trong top-K, điểm NDCG của user đó bằng 0.

### 6.3 CandidatePoolHitRate

CandidatePoolHitRate đo tỉ lệ ground-truth item xuất hiện trong candidate pool trước khi lấy top-K.

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

| Metric | Value |
|---|---:|
| HR@5 / R@5 | 0.0301 |
| HR@10 / R@10 | 0.0419 |
| HR@20 / R@20 | 0.0644 |
| NDCG@5 | 0.0208 |
| NDCG@10 | 0.0246 |
| NDCG@20 | 0.0302 |
| PoolHit@1000 | 0.4812 |

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

Dự án hiện tại chưa có contrastive loss giữa user view và sequence view như MuSICRec. Vì vậy không nên claim rằng phương pháp hiện tại đã thay thế hoàn toàn MuSICRec.

Claim hợp lý hơn là:

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
