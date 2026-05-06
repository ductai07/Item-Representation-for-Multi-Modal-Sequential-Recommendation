from __future__ import annotations

import torch

from cstamoerec.train import move_batch


@torch.no_grad()
def rerank_candidates(model, batch: dict[str, torch.Tensor], candidate_ids: list[int], device: str, topk: int = 10) -> dict:
    if not candidate_ids:
        return {"item_ids": [], "scores": [], "expert_weights": []}
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    device_batch = move_batch({key: value.unsqueeze(0) if value.dim() == 0 else value for key, value in batch.items()}, device)
    if device_batch["seq"].dim() == 1:
        device_batch = {key: value.unsqueeze(0) for key, value in device_batch.items()}
    output = model.score_candidates(device_batch, candidate_tensor)
    scores = output["scores"][0]
    weights = output["expert_weights"][0]
    k = min(topk, scores.numel())
    values, indices = torch.topk(scores, k=k)
    ranked_items = [candidate_ids[int(i)] for i in indices.detach().cpu().tolist()]
    ranked_weights = weights[indices].detach().cpu().tolist()
    return {
        "item_ids": ranked_items,
        "scores": [float(x) for x in values.detach().cpu().tolist()],
        "expert_weights": ranked_weights,
    }
