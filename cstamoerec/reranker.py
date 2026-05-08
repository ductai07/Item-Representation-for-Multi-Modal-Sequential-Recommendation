from __future__ import annotations

import torch

from cstamoerec.train import mask_seen_items, move_batch


def source_prior_for_items(
    candidate_ids: list[int],
    sources: dict[int, set[str]] | None = None,
    graph_scores: list[list[float]] | list[float] | None = None,
) -> torch.Tensor:
    source_weights = {
        "image": 1.00,
        "image_graph": 0.95,
        "itemcf": 0.90,
        "text_graph": 0.80,
        "text": 0.75,
        "transition": 0.70,
        "popularity": 0.35,
        "sasrec": 0.50,
    }
    values = []
    for idx, item in enumerate(candidate_ids):
        score = 0.0
        if sources:
            item_sources = sources.get(item, set())
            score += sum(source_weights.get(source, 0.25) for source in item_sources)
            score += 0.1 * len(item_sources)
        if graph_scores is not None and idx < len(graph_scores):
            row = graph_scores[idx]
            if isinstance(row, list):
                score += 0.5 * float(sum(row))
            else:
                score += 0.5 * float(row)
        values.append(score)
    prior = torch.tensor(values, dtype=torch.float)
    if prior.numel() > 1 and float(prior.std()) > 1e-8:
        prior = (prior - prior.mean()) / prior.std()
    return prior


@torch.no_grad()
def rerank_candidates(
    model,
    batch: dict[str, torch.Tensor],
    candidate_ids: list[int],
    device: str,
    topk: int = 10,
    graph_scores: list[float] | None = None,
    sources: dict[int, set[str]] | None = None,
    mode: str = "candidate",
    prior_weight: float = 1.0,
    model_weight: float = 0.05,
) -> dict:
    if not candidate_ids:
        return {"item_ids": [], "scores": [], "expert_weights": []}
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    device_batch = move_batch({key: value.unsqueeze(0) if value.dim() == 0 else value for key, value in batch.items()}, device)
    if device_batch["seq"].dim() == 1:
        device_batch = {key: value.unsqueeze(0) for key, value in device_batch.items()}
    graph_tensor = None
    if graph_scores is not None:
        graph_tensor = torch.tensor(graph_scores, dtype=torch.float, device=device)
    adaptive = model.score_candidates(device_batch, candidate_tensor, graph_scores=graph_tensor)
    weights = adaptive["expert_weights"][0]
    if mode == "candidate":
        scores = -torch.arange(len(candidate_ids), dtype=torch.float, device=device)
    elif mode == "source":
        scores = source_prior_for_items(candidate_ids, sources=sources, graph_scores=graph_scores).to(device)
    elif mode == "adaptive":
        scores = adaptive["scores"][0]
    else:
        full = model(device_batch)["scores"]
        full = mask_seen_items(full, device_batch["seq"])
        scores = full[0, candidate_tensor]
        if mode == "hybrid":
            prior = source_prior_for_items(candidate_ids, sources=sources, graph_scores=graph_scores).to(device)
            if scores.numel() > 1 and float(scores.std()) > 1e-8:
                scores = (scores - scores.mean()) / scores.std()
            scores = model_weight * scores + prior_weight * prior
    k = min(topk, scores.numel())
    values, indices = torch.topk(scores, k=k)
    ranked_items = [candidate_ids[int(i)] for i in indices.detach().cpu().tolist()]
    ranked_weights = weights[indices].detach().cpu().tolist()
    return {
        "item_ids": ranked_items,
        "scores": [float(x) for x in values.detach().cpu().tolist()],
        "expert_weights": ranked_weights,
    }
