from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


def topk_metrics(scores: torch.Tensor, targets: torch.Tensor, topk: Iterable[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    max_k = min(max(topk), scores.size(1))
    _, indices = torch.topk(scores, k=max_k, dim=1)
    hits = indices.eq(targets.view(-1, 1))
    for k in topk:
        actual_k = min(k, scores.size(1))
        hit_k = hits[:, :actual_k]
        any_hit = hit_k.any(dim=1).float()
        result[f"HR@{k}"] = any_hit.mean().item()
        ranks = torch.where(
            hit_k,
            torch.arange(1, actual_k + 1, device=scores.device).float(),
            torch.zeros_like(hit_k).float(),
        )
        rank = ranks.max(dim=1).values
        mrr = torch.where(rank > 0, 1.0 / rank, torch.zeros_like(rank))
        ndcg = torch.where(rank > 0, 1.0 / torch.log2(rank + 1), torch.zeros_like(rank))
        result[f"MRR@{k}"] = mrr.mean().item()
        result[f"NDCG@{k}"] = ndcg.mean().item()
        result[f"Recall@{k}"] = any_hit.mean().item()
    return result


class MetricAverager:
    def __init__(self) -> None:
        self.sums = defaultdict(float)
        self.count = 0

    def update(self, metrics: dict[str, float], n: int) -> None:
        for key, value in metrics.items():
            self.sums[key] += float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.sums.items()}


def coverage_from_topk(topk_items: list[int], num_items: int) -> float:
    if num_items <= 1:
        return 0.0
    return len(set(topk_items)) / float(num_items - 1)
