from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def transition_edges_from_examples(examples: list[dict[str, Any]]) -> list[tuple[int, int]]:
    counts = Counter()
    for ex in examples:
        seq = [int(x) for x in ex["seq"] if int(x) > 0] + [int(ex["target"])]
        for src, dst in zip(seq[:-1], seq[1:]):
            if src > 0 and dst > 0 and src != dst:
                counts[(src, dst)] += 1
                counts[(dst, src)] += 1
    return list(counts.keys())


def similarity_edges_from_features(
    features: torch.Tensor,
    topk: int,
    batch_size: int,
    device: str,
    valid_mask: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    matrix = F.normalize(features.float().to(device), dim=-1)
    num_items = matrix.size(0)
    valid = torch.ones(num_items, dtype=torch.bool, device=device)
    valid[0] = False
    if valid_mask is not None:
        valid &= valid_mask.to(device).bool()
    valid_indices = valid.nonzero(as_tuple=False).view(-1)
    if valid_indices.numel() == 0:
        return []
    edges = set()
    actual_k = min(topk + 1, valid_indices.numel())
    for start in range(0, valid_indices.numel(), batch_size):
        rows = valid_indices[start : start + batch_size]
        scores = matrix[rows] @ matrix.t()
        scores[:, ~valid] = -1e9
        for row_pos, item in enumerate(rows.tolist()):
            scores[row_pos, item] = -1e9
        values, idx = torch.topk(scores, k=actual_k, dim=1)
        for src, neighbors, neighbor_scores in zip(rows.tolist(), idx.detach().cpu().tolist(), values.detach().cpu().tolist()):
            for dst, score in zip(neighbors[:topk], neighbor_scores[:topk]):
                if float(score) > -1e8 and int(dst) > 0 and int(dst) != int(src):
                    edges.add((int(src), int(dst)))
                    edges.add((int(dst), int(src)))
    return list(edges)


def build_norm_adj(num_items: int, edges: list[tuple[int, int]], device: str) -> torch.Tensor:
    if not edges:
        idx = torch.arange(num_items, device=device)
        indices = torch.stack([idx, idx])
        values = torch.ones(num_items, device=device)
        return torch.sparse_coo_tensor(indices, values, (num_items, num_items)).coalesce()
    src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
    self_idx = torch.arange(num_items, device=device)
    row = torch.cat([src, self_idx])
    col = torch.cat([dst, self_idx])
    deg = torch.bincount(row, minlength=num_items).float().clamp_min(1.0)
    values = 1.0 / torch.sqrt(deg[row] * deg[col])
    indices = torch.stack([row, col])
    return torch.sparse_coo_tensor(indices, values, (num_items, num_items)).coalesce()


class EdgeDataset(Dataset):
    def __init__(self, edges: list[tuple[int, int]], num_items: int) -> None:
        self.edges = edges
        self.num_items = num_items

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, pos = self.edges[idx]
        neg = torch.randint(1, self.num_items, (1,)).item()
        while neg == pos:
            neg = torch.randint(1, self.num_items, (1,)).item()
        return torch.tensor(src, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)


class LightGCN(nn.Module):
    def __init__(self, num_items: int, graph_dim: int, n_layers: int, norm_adj: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items, graph_dim, padding_idx=0)
        self.n_layers = n_layers
        self.register_buffer("norm_adj", norm_adj)
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def propagate(self) -> torch.Tensor:
        out = self.embedding.weight
        all_layers = [out]
        for _ in range(self.n_layers):
            out = torch.sparse.mm(self.norm_adj, out)
            all_layers.append(out)
        z = torch.stack(all_layers, dim=0).mean(dim=0)
        z = z.clone()
        z[0].zero_()
        return z

    def bpr_loss(self, src: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        z = self.propagate()
        src_z = z[src]
        pos_z = z[pos]
        neg_z = z[neg]
        pos_score = (src_z * pos_z).sum(dim=-1)
        neg_score = (src_z * neg_z).sum(dim=-1)
        return -F.logsigmoid(pos_score - neg_score).mean()


def graph_transition_scores(seq: torch.Tensor, candidate_ids: torch.Tensor, transition_graph: dict[int, list[tuple[int, float]]] | None) -> torch.Tensor:
    if transition_graph is None:
        return torch.zeros(candidate_ids.shape, dtype=torch.float, device=candidate_ids.device)
    if candidate_ids.dim() == 1:
        candidate_ids = candidate_ids.unsqueeze(0).expand(seq.size(0), -1)
    result = torch.zeros(candidate_ids.shape, dtype=torch.float, device=candidate_ids.device)
    seq_cpu = seq.detach().cpu().tolist()
    cand_cpu = candidate_ids.detach().cpu().tolist()
    for row_idx, (history, candidates) in enumerate(zip(seq_cpu, cand_cpu)):
        scores = Counter()
        recent = [int(x) for x in history if int(x) > 0][-5:]
        for recency, item in enumerate(reversed(recent), start=1):
            decay = 1.0 / recency
            for dst, weight in transition_graph.get(item, []):
                scores[int(dst)] += float(weight) * decay
        for col_idx, item in enumerate(candidates):
            result[row_idx, col_idx] = float(scores.get(int(item), 0.0))
    return result
