from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass
class CandidateResult:
    item_ids: list[int]
    sources: dict[int, set[str]]
    source_scores: dict[int, dict[str, float]]


def build_transition_graph(examples: list[dict[str, Any]], num_items: int, max_neighbors: int = 200) -> dict[int, list[tuple[int, float]]]:
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    for ex in examples:
        seq = [int(x) for x in ex["seq"] if int(x) > 0] + [int(ex["target"])]
        for src, dst in zip(seq[:-1], seq[1:]):
            if src > 0 and dst > 0 and src != dst:
                transitions[src][dst] += 1
    graph = {}
    for src, counter in transitions.items():
        total = sum(counter.values())
        graph[src] = [(dst, count / total) for dst, count in counter.most_common(max_neighbors)]
    return graph


def build_itemcf_graph(examples: list[dict[str, Any]], num_items: int, max_neighbors: int = 200) -> dict[int, list[tuple[int, float]]]:
    co_counts: dict[int, Counter[int]] = defaultdict(Counter)
    item_counts = Counter()
    for ex in examples:
        items = list(dict.fromkeys(int(x) for x in ex["seq"] if int(x) > 0))
        target = int(ex["target"])
        if target > 0:
            items.append(target)
        items = list(dict.fromkeys(items))
        for item in items:
            item_counts[item] += 1
        for i, src in enumerate(items):
            for dst in items[i + 1 :]:
                if src != dst:
                    co_counts[src][dst] += 1
                    co_counts[dst][src] += 1
    graph = {}
    for src, counter in co_counts.items():
        scored = []
        for dst, count in counter.items():
            denom = (item_counts[src] * item_counts[dst]) ** 0.5
            scored.append((dst, count / max(denom, 1.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        graph[src] = scored[:max_neighbors]
    return graph


def top_popularity(item_popularity: torch.Tensor, k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
    exclude = exclude or set()
    scores = item_popularity.float().clone()
    scores[0] = -1
    for item in exclude:
        if 0 <= item < scores.numel():
            scores[item] = -1
    k = min(k, max(scores.numel() - len(exclude) - 1, 1))
    values, indices = torch.topk(scores, k=k)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist()) if int(i) > 0]


def graph_candidates(
    seq: list[int],
    graph: dict[int, list[tuple[int, float]]],
    k: int,
    exclude: set[int] | None = None,
) -> list[tuple[int, float]]:
    exclude = exclude or set()
    scores = Counter()
    recent = [int(x) for x in seq if int(x) > 0][-5:]
    for recency, item in enumerate(reversed(recent), start=1):
        weight = 1.0 / recency
        for dst, score in graph.get(item, [])[:k]:
            if dst not in exclude:
                scores[dst] += score * weight
    return [(item, float(score)) for item, score in scores.most_common(k)]


def feature_similarity_candidates(
    seq: list[int],
    features: torch.Tensor,
    k: int,
    exclude: set[int] | None = None,
    valid_mask: torch.Tensor | None = None,
) -> list[tuple[int, float]]:
    exclude = exclude or set()
    recent = [int(x) for x in seq if int(x) > 0][-5:]
    if not recent:
        return []
    matrix = features.float()
    query = matrix[recent].mean(dim=0, keepdim=True)
    query = F.normalize(query, dim=-1)
    candidates = F.normalize(matrix, dim=-1)
    scores = torch.matmul(query, candidates.t()).squeeze(0)
    scores[0] = -1e9
    for item in exclude:
        if 0 <= item < scores.numel():
            scores[item] = -1e9
    if valid_mask is not None:
        scores = scores.masked_fill(valid_mask <= 0, -1e9)
    actual_k = min(k, max(scores.numel() - 1, 1))
    values, indices = torch.topk(scores, k=actual_k)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist()) if int(i) > 0 and float(v) > -1e8]


def sasrec_candidates(model, batch: dict[str, torch.Tensor], k: int) -> list[tuple[int, float]]:
    with torch.no_grad():
        scores = model(batch)["scores"][0]
        values, indices = torch.topk(scores, k=min(k, scores.numel()))
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist()) if int(i) > 0]


def combine_candidate_sources(named_candidates: dict[str, list[tuple[int, float]]], max_candidates: int) -> CandidateResult:
    sources: dict[int, set[str]] = defaultdict(set)
    source_scores: dict[int, dict[str, float]] = defaultdict(dict)
    aggregate = Counter()
    for source, candidates in named_candidates.items():
        for rank, (item, score) in enumerate(candidates, start=1):
            if item <= 0:
                continue
            sources[item].add(source)
            source_scores[item][source] = float(score)
            aggregate[item] += 1.0 / rank
    ranked = [item for item, _ in aggregate.most_common(max_candidates)]
    return CandidateResult(item_ids=ranked, sources=dict(sources), source_scores=dict(source_scores))


class CandidateGenerator:
    def __init__(
        self,
        artifacts: dict[str, Any],
        per_source_k: int = 100,
        max_candidates: int = 300,
    ) -> None:
        self.artifacts = artifacts
        self.features = artifacts["features"]
        self.meta = artifacts["meta"]
        self.per_source_k = per_source_k
        self.max_candidates = max_candidates
        train_examples = artifacts["examples"]["train"]
        self.transition_graph = build_transition_graph(train_examples, self.meta["num_items"])
        self.itemcf_graph = build_itemcf_graph(train_examples, self.meta["num_items"])

    def generate(
        self,
        seq: list[int],
        model=None,
        batch: dict[str, torch.Tensor] | None = None,
        include_sasrec: bool = False,
    ) -> CandidateResult:
        exclude = {int(x) for x in seq if int(x) > 0}
        named = {
            "popularity": top_popularity(self.features["item_popularity"], self.per_source_k, exclude),
            "transition": graph_candidates(seq, self.transition_graph, self.per_source_k, exclude),
            "itemcf": graph_candidates(seq, self.itemcf_graph, self.per_source_k, exclude),
            "text": feature_similarity_candidates(seq, self.features["text_embeddings"], self.per_source_k, exclude),
            "image": feature_similarity_candidates(
                seq,
                self.features["image_embeddings"],
                self.per_source_k,
                exclude,
                valid_mask=self.features.get("image_mask"),
            ),
        }
        if include_sasrec and model is not None and batch is not None:
            named["sasrec"] = sasrec_candidates(model, batch, self.per_source_k)
        return combine_candidate_sources(named, self.max_candidates)


def candidate_recall(candidate_items: list[int], target: int, topk: list[int]) -> dict[str, float]:
    result = {}
    for k in topk:
        result[f"Recall@{k}"] = 1.0 if int(target) in set(candidate_items[:k]) else 0.0
    return result
