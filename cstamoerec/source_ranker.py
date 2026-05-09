from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SOURCE_NAMES = [
    "popularity",
    "transition",
    "itemcf",
    "text_graph",
    "image_graph",
    "text",
    "image",
    "sasrec",
]

GRAPH_NAMES = ["id_graph_score", "text_graph_score", "image_graph_score", "graph_score_sum"]

FEATURE_NAMES = (
    ["bias", "candidate_rrf_rank", "source_count", "log_popularity"]
    + [f"has_{source}" for source in SOURCE_NAMES]
    + [f"score_{source}" for source in SOURCE_NAMES]
    + GRAPH_NAMES
)


@dataclass
class LearnedSourceRanker:
    weights: torch.Tensor
    feature_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor

    def score(self, features: torch.Tensor) -> torch.Tensor:
        features = align_features(features, self.mean, self.std)
        return features @ self.weights.to(features.device)


def _graph_row(graph_scores: list[list[float]] | None, idx: int) -> list[float]:
    if graph_scores is None or idx >= len(graph_scores):
        return [0.0, 0.0, 0.0]
    row = graph_scores[idx]
    if isinstance(row, list):
        values = [float(x) for x in row[:3]]
        return values + [0.0] * (3 - len(values))
    return [float(row), 0.0, 0.0]


def candidate_feature_matrix(
    candidate_ids: list[int],
    sources: dict[int, set[str]],
    source_scores: dict[int, dict[str, float]],
    graph_scores: list[list[float]] | None,
    item_popularity: torch.Tensor,
) -> torch.Tensor:
    rows: list[list[float]] = []
    for rank, item_id in enumerate(candidate_ids, start=1):
        item_sources = sources.get(item_id, set())
        item_source_scores = source_scores.get(item_id, {})
        graph = _graph_row(graph_scores, rank - 1)
        row = [
            1.0,
            1.0 / float(rank),
            float(len(item_sources)),
            float(torch.log1p(item_popularity[item_id].float()).item()) if item_id < item_popularity.numel() else 0.0,
        ]
        row.extend(1.0 if source in item_sources else 0.0 for source in SOURCE_NAMES)
        row.extend(float(item_source_scores.get(source, 0.0)) for source in SOURCE_NAMES)
        row.extend([graph[0], graph[1], graph[2], sum(graph)])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float)


def fit_feature_stats(feature_rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not feature_rows:
        size = len(FEATURE_NAMES)
        return torch.zeros(size), torch.ones(size)
    all_features = torch.cat(feature_rows, dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0).clamp_min(1e-6)
    mean[0] = 0.0
    std[0] = 1.0
    return mean, std


def align_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(features.device)
    std = std.to(features.device)
    return (features - mean) / std


def save_ranker(path: str | Path, ranker: LearnedSourceRanker, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "feature_names": ranker.feature_names,
        "weights": ranker.weights.detach().cpu().tolist(),
        "mean": ranker.mean.detach().cpu().tolist(),
        "std": ranker.std.detach().cpu().tolist(),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_ranker(path: str | Path) -> LearnedSourceRanker:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return LearnedSourceRanker(
        weights=torch.tensor(payload["weights"], dtype=torch.float),
        feature_names=list(payload["feature_names"]),
        mean=torch.tensor(payload["mean"], dtype=torch.float),
        std=torch.tensor(payload["std"], dtype=torch.float),
    )
