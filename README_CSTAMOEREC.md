# Pipeline CS-TAMoERec

File này tóm tắt nhanh pipeline chạy project. README chính đã mô tả đầy đủ hơn.

## 1. Cài Thư Viện

```bash
pip install -r requirements_cstamoerec.txt
```

## 2. Chuẩn Bị Dữ Liệu

Nhanh, không tải ảnh:

```bash
python scripts/prepare_amazon2023.py --config config/cstamoerec_all_beauty.yaml --skip-images
```

Đầy đủ text + image:

```bash
python scripts/prepare_amazon2023.py --config config/cstamoerec_all_beauty.yaml
```

Debug:

```bash
python scripts/prepare_amazon2023.py --config config/cstamoerec_all_beauty.yaml --skip-images --limit-reviews 20000
```

## 3. Train

```bash
python scripts/train_cstamoerec.py --config config/cstamoerec_all_beauty.yaml
```

## 4. Ablation

```bash
python scripts/run_ablation.py --config config/cstamoerec_all_beauty.yaml --epochs 5
```

## 5. Candidate Generation

```bash
python scripts/evaluate_candidates.py --config config/cstamoerec_all_beauty.yaml --split test
```

## 6. Two-stage Reranking

```bash
python scripts/rerank_candidates.py --config config/cstamoerec_all_beauty.yaml --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt
```

## 7. Perturbation Test

```bash
python scripts/evaluate_perturbation.py --config config/cstamoerec_all_beauty.yaml --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt
```

## 8. Counterfactual Rank Test

```bash
python scripts/evaluate_counterfactual.py --config config/cstamoerec_all_beauty.yaml --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt --limit-users 100
```

## 9. Expert Weight Analysis

```bash
python scripts/analyze_experts.py --config config/cstamoerec_all_beauty.yaml --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt
```

## 10. Export Demo Cache

```bash
python scripts/export_demo_cache.py --config config/cstamoerec_all_beauty.yaml --checkpoint checkpoints/cstamoerec/best_cstamoerec.pt
```

## 11. Demo

```bash
streamlit run demo_streamlit.py
```
