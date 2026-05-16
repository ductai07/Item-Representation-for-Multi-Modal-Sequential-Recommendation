# Graph-Enhanced Multimodal Sequential Recommendation with Sequence-Node Retrieval 

## 1. Tóm tắt (Abstract)

Dự án tập trung giải quyết bài toán **Multimodal Sequential Recommendation** dưới điều kiện dữ liệu thưa thớt và lịch sử người dùng ngắn. Chúng tôi triển khai một pipeline **two-stage recommendation**:

1. **Stage 1**: Graph-based candidate generation  
2. **Stage 2**: Candidate reranking bằng weighted source fusion

Ý tưởng cốt lõi là coi lịch sử hành vi người dùng như một **context node** có thể truy hồi các sequence tương tự trong tập huấn luyện, từ đó khai thác các quan hệ Sequence-Sequence, Sequence-Item, Item-Item và Multimodal similarity để sinh candidate.

Trên benchmark Baby sử dụng cùng split và feature với MuSICRec, kết quả full test của dự án như sau:

**Bảng 1: So sánh hiệu suất trên Baby dataset**

| Phương pháp             | R@10   | R@20   | N@10   | N@20   |
|-------------------------|--------|--------|--------|--------|
| LightGCN                | 0.0336 | 0.0549 | 0.0178 | 0.0231 |
| SGL                     | 0.0361 | 0.0586 | 0.0188 | 0.0244 |
| SASRec                  | 0.0298 | 0.0478 | 0.0150 | 0.0195 |
| BERT4Rec                | 0.0207 | 0.0355 | 0.0102 | 0.0139 |
| FEARec                  | 0.0285 | 0.0480 | 0.0146 | 0.0195 |
| SRGNN                   | 0.0260 | 0.0418 | 0.0132 | 0.0172 |
| FREEDOM                 | 0.0411 | 0.0655 | 0.0210 | 0.0272 |
| SMORE                   | 0.0439 | 0.0684 | 0.0229 | 0.0291 |
| MuSICRec                | **0.0455** | **0.0718** | 0.0235 | 0.0300 |
| **Dự án hiện tại**      | 0.0419 | 0.0644 | **0.0246** | **0.0302** |

Phương pháp hiện tại **chưa vượt MuSICRec về tổng thể** do Recall@10 và Recall@20 vẫn thấp hơn. Tuy nhiên, dự án đạt NDCG@10 và NDCG@20 cao hơn nhẹ so với MuSICRec, đồng thời **vượt rõ rệt phần lớn các baseline truyền thống** (ID-only và sequential) trên cùng benchmark.

## 2. Động lực nghiên cứu

Các mô hình sequential recommendation truyền thống (SASRec, BERT4Rec, FEARec…) thường biểu diễn lịch sử người dùng dưới dạng chuỗi tuyến tính. Cách tiếp cận này gặp nhiều khó khăn với short-history users, dữ liệu thưa và nhiễu multimodal.

MuSICRec đã đề xuất ý tưởng coi sequence như một node trong graph. Dự án này kế thừa triết lý trên nhưng triển khai theo hướng thực dụng hơn: sử dụng sequence graph chủ yếu để **candidate retrieval** thay vì huấn luyện contrastive GNN end-to-end.

## 3. Tổng quan phương pháp

### 3.1 Sequence Graph Retrieval
Đây là thành phần chính. Mỗi sequence được xây dựng thành node và được index bởi tập item. Khi inference, recent items được dùng làm query để truy vấn các sequence tương tự dựa trên recency-weighted overlap và Jaccard similarity.

### 3.2 Transition Graph & ItemCF Graph
- Transition Graph học quan hệ chuyển tiếp theo thứ tự.
- ItemCF Graph học quan hệ đồng xuất hiện.

Sequence Graph vẫn là nguồn tín hiệu mạnh nhất trong thực nghiệm.

### 3.3 Multimodal Graph
Sử dụng feature text (384 chiều) và image (4096 chiều) từ MuSICRec để xây dựng similarity graph và embedding bằng LightGCN. Image signal cho kết quả yếu trên Baby dataset và được coi là auxiliary signal.

## 4. Candidate Fusion

Các candidate được hợp nhất bằng **Weighted Reciprocal Rank Fusion**. Trọng số được tối ưu trên validation set, ưu tiên cao cho Sequence Graph (5.0) và giảm mạnh cho image (0.05).

## 5. Thiết lập thực nghiệm

Dự án sử dụng **đúng split và feature** của MuSICRec trên Baby dataset:
- Train: 102,457 interactions
- Valid/Test: 19,445 interactions
- Features: `text_feat.npy` (384), `image_feat.npy` (4096)

## 6. Kết quả thực nghiệm

### 6.1 Kết quả full test

Như Bảng 1, dự án đạt hiệu suất cạnh tranh trên Baby. Cụ thể:
- **Vượt rõ** LightGCN, SGL và các mô hình sequential truyền thống (SASRec, BERT4Rec, FEARec, SRGNN).
- NDCG@10 và NDCG@20 **nhỉnh hơn nhẹ** so với MuSICRec.
- Recall@10 và Recall@20 **thấp hơn** MuSICRec.

### 6.2 Phân tích

NDCG cao hơn cho thấy hệ thống có khả năng xếp một phần target lên vị trí cao khi chúng nằm trong candidate pool. Tuy nhiên, hiệu quả tổng thể vẫn bị giới hạn bởi **CandidatePoolHitRate@1000 chỉ đạt 0.4812** (khoảng 51.9% target bị miss ở Stage 1).

## 7. So sánh với MuSICRec

MuSICRec vẫn mạnh hơn về Recall nhờ end-to-end learning, contrastive alignment ($\mathcal{L}_{US}$) và ID-guided gating. Dự án hiện tại sử dụng retrieval heuristic + weighted fusion nên đơn giản và dễ debug hơn, nhưng chưa đạt được sự học sâu giữa các view.

## 8. Đóng góp hiện tại

- Chứng minh được rằng cách tiếp cận **sequence-node retrieval** có thể mang lại tín hiệu mạnh mà không cần contrastive GNN phức tạp.
- Xây dựng được pipeline hai giai đoạn nhẹ, interpretable và dễ mở rộng.
- Áp dụng validation-driven weighting hợp lý cho multimodal signals.

## 9. Hạn chế

1. Candidate pool coverage là nút thắt chính (Hit Rate@1000 = 0.4812).
2. Weighted fusion vẫn là heuristic tuyến tính.
3. Chưa áp dụng contrastive learning hay learned reranker.

## 10. Hướng phát triển tiếp theo

- Cải thiện candidate generation (n-gram retrieval, prefix matching…).
- Huấn luyện learned candidate reranker.
- Thêm lightweight multimodal gating.
- Thực nghiệm trên Sports và Electronics dataset.

## 11. Kết luận

Dự án đã chứng minh được giá trị của hướng sequence-node retrieval trên Baby dataset. Phương pháp hiện tại **chưa vượt MuSICRec về mặt tổng thể**, nhưng đạt NDCG cạnh tranh và vượt trội hơn nhiều baseline truyền thống. Kết quả cho thấy đây là hướng đáng tiếp tục khai thác, đặc biệt nếu cải thiện được candidate coverage và bổ sung reranker có giám sát.
