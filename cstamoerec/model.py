from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cstamoerec.data import log_time_interval_days, unix_month_weekday


def last_non_pad_indices(item_ids: torch.Tensor) -> torch.Tensor:
    non_pad = item_ids.ne(0)
    positions = torch.arange(item_ids.size(1), device=item_ids.device).unsqueeze(0).expand_as(item_ids)
    return positions.masked_fill(~non_pad, 0).max(dim=1).values


class TimeEncoder(nn.Module):
    def __init__(self, hidden_size: int, time_dim: int) -> None:
        super().__init__()
        self.month_embedding = nn.Embedding(12, time_dim)
        self.weekday_embedding = nn.Embedding(7, time_dim)
        self.interval_projection = nn.Linear(1, time_dim)
        self.out = nn.Linear(time_dim * 3, hidden_size)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        if times.dim() == 1:
            times = times.unsqueeze(0)
        month, weekday = unix_month_weekday(times)
        interval = log_time_interval_days(times).unsqueeze(-1)
        features = torch.cat(
            [
                self.month_embedding(month),
                self.weekday_embedding(weekday),
                self.interval_projection(interval.float()),
            ],
            dim=-1,
        )
        return self.out(features)


class ColdStartTimeAwareMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        router_hidden: int,
        dropout: float,
        use_cross: bool = True,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.use_cross = use_cross
        self.use_graph = use_graph
        self.id_expert = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.text_expert = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.image_expert = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.time_expert = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.cross_expert = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.graph_expert = nn.Sequential(nn.Linear(hidden_size + 3, hidden_size), nn.GELU(), nn.Dropout(dropout))
        router_in = hidden_size * 6 + 6
        self.router = nn.Sequential(
            nn.Linear(router_in, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 6),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        id_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        time_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        graph_score: torch.Tensor,
        popularity: torch.Tensor,
        cold_flags: torch.Tensor,
        image_mask: torch.Tensor,
        user_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if user_context is None:
            user_context = torch.zeros_like(id_emb)
        elif user_context.dim() == 2 and id_emb.dim() == 3:
            user_context = user_context.unsqueeze(1).expand(-1, id_emb.size(1), -1)
        elif user_context.dim() == 2 and id_emb.dim() == 2:
            pass
        log_pop = torch.log1p(popularity.float()).unsqueeze(-1)
        cold_flags = cold_flags.float().unsqueeze(-1)
        image_mask = image_mask.float().unsqueeze(-1)
        if graph_score.dim() == popularity.dim():
            graph_score = graph_score.float().unsqueeze(-1).expand(*graph_score.shape, 3)
        else:
            graph_score = graph_score.float()
        cross_input = torch.cat([id_emb, text_emb, image_emb], dim=-1)
        graph_input = torch.cat([graph_emb, graph_score], dim=-1)
        expert_outputs = torch.stack(
            [
                self.id_expert(id_emb),
                self.text_expert(text_emb),
                self.image_expert(image_emb),
                self.time_expert(time_emb),
                self.cross_expert(cross_input),
                self.graph_expert(graph_input),
            ],
            dim=-2,
        )
        router_input = torch.cat(
            [user_context, id_emb, text_emb, image_emb, time_emb, graph_emb, log_pop, cold_flags, image_mask, graph_score],
            dim=-1,
        )
        logits = self.router(router_input)
        if not self.use_cross:
            logits[..., 4] = -1e9
        if not self.use_graph:
            logits[..., 5] = -1e9
        weights = F.softmax(logits, dim=-1)
        fused = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=-2)
        return self.norm(fused), weights


class CSTAMoERec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_mask: torch.Tensor,
        item_popularity: torch.Tensor,
        graph_embeddings: torch.Tensor | None = None,
        id_graph_embeddings: torch.Tensor | None = None,
        text_graph_embeddings: torch.Tensor | None = None,
        image_graph_embeddings: torch.Tensor | None = None,
        hidden_size: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        text_dim: int = 384,
        image_dim: int = 512,
        time_dim: int = 16,
        router_hidden: int = 256,
        graph_dim: int = 64,
        cold_threshold: int = 5,
        use_text: bool = True,
        use_image: bool = True,
        use_time: bool = True,
        use_cold: bool = True,
        use_cross: bool = True,
        use_graph: bool = True,
        use_id_graph: bool = True,
        use_text_graph: bool = True,
        use_image_graph: bool = True,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.cold_threshold = cold_threshold
        self.use_text = use_text
        self.use_image = use_image
        self.use_time = use_time
        self.use_cold = use_cold
        self.use_cross = use_cross
        self.use_graph = use_graph and any(
            emb is not None for emb in (graph_embeddings, id_graph_embeddings, text_graph_embeddings, image_graph_embeddings)
        )
        self.use_id_graph = use_id_graph
        self.use_text_graph = use_text_graph
        self.use_image_graph = use_image_graph
        self.item_embedding = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.image_projection = nn.Linear(image_dim, hidden_size)
        self.graph_projection = nn.Linear(graph_dim * 3, hidden_size)
        self.time_encoder = TimeEncoder(hidden_size, time_dim)
        self.moe = ColdStartTimeAwareMoE(hidden_size, router_hidden, dropout, use_cross=use_cross, use_graph=self.use_graph)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.category_head = nn.Linear(hidden_size, num_categories)
        item_bias = torch.log1p(item_popularity.float())
        item_bias = item_bias - item_bias[1:].mean().clamp_min(0.0)
        item_bias[0] = 0.0
        self.item_bias = nn.Parameter(item_bias * 0.01)

        self.register_buffer("text_features", text_embeddings.float())
        self.register_buffer("image_features", image_embeddings.float())
        self.register_buffer("image_mask", image_mask.float())
        self.register_buffer("item_popularity", item_popularity.long())
        if id_graph_embeddings is None:
            id_graph_embeddings = graph_embeddings
        if id_graph_embeddings is None:
            id_graph_embeddings = torch.zeros((num_items, graph_dim), dtype=torch.float)
        if text_graph_embeddings is None:
            text_graph_embeddings = torch.zeros_like(id_graph_embeddings)
        if image_graph_embeddings is None:
            image_graph_embeddings = torch.zeros_like(id_graph_embeddings)
        self.register_buffer("id_graph_features", id_graph_embeddings.float())
        self.register_buffer("text_graph_features", text_graph_embeddings.float())
        self.register_buffer("image_graph_features", image_graph_embeddings.float())

    def project_graph_features(self, item_ids: torch.Tensor) -> torch.Tensor:
        id_graph = self.id_graph_features[item_ids]
        text_graph = self.text_graph_features[item_ids]
        image_graph = self.image_graph_features[item_ids]
        if not self.use_id_graph:
            id_graph = torch.zeros_like(id_graph)
        if not self.use_text_graph:
            text_graph = torch.zeros_like(text_graph)
        if not self.use_image_graph:
            image_graph = torch.zeros_like(image_graph)
        return self.graph_projection(torch.cat([id_graph, text_graph, image_graph], dim=-1))

    def prepare_graph_scores(self, graph_scores: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if graph_scores.dim() == like.dim():
            graph_scores = graph_scores.float().unsqueeze(-1).expand(*graph_scores.shape, 3)
        else:
            graph_scores = graph_scores.float()
        if not self.use_id_graph:
            graph_scores[..., 0] = 0.0
        if not self.use_text_graph:
            graph_scores[..., 1] = 0.0
        if not self.use_image_graph:
            graph_scores[..., 2] = 0.0
        return graph_scores

    def represent_sequence(
        self,
        item_ids: torch.Tensor,
        times: torch.Tensor,
        popularity: torch.Tensor,
        cold_flags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        id_emb = self.item_embedding(item_ids)
        text_emb = self.text_projection(self.text_features[item_ids])
        image_emb = self.image_projection(self.image_features[item_ids])
        graph_emb = self.project_graph_features(item_ids)
        time_emb = self.time_encoder(times)
        if not self.use_text:
            text_emb = torch.zeros_like(text_emb)
        if not self.use_image:
            image_emb = torch.zeros_like(image_emb)
        if not self.use_time:
            time_emb = torch.zeros_like(time_emb)
        if not self.use_cold:
            cold_flags = torch.zeros_like(cold_flags)
        if not self.use_graph:
            graph_emb = torch.zeros_like(graph_emb)
        image_mask = self.image_mask[item_ids]
        if not self.use_image:
            image_mask = torch.zeros_like(image_mask)
        graph_score = torch.zeros((*item_ids.shape, 3), dtype=torch.float, device=item_ids.device)
        fused, weights = self.moe(id_emb, text_emb, image_emb, time_emb, graph_emb, graph_score, popularity, cold_flags, image_mask)
        positions = torch.arange(item_ids.size(1), device=item_ids.device).unsqueeze(0)
        fused = fused + self.position_embedding(positions)
        padding_mask = item_ids.eq(0)
        causal_mask = torch.triu(
            torch.full((item_ids.size(1), item_ids.size(1)), float("-inf"), device=item_ids.device),
            diagonal=1,
        )
        encoded = self.encoder(fused, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.output_norm(encoded)
        last_indices = last_non_pad_indices(item_ids)
        context = encoded[torch.arange(encoded.size(0), device=encoded.device), last_indices]
        id_context = self.item_embedding(item_ids)
        id_context = id_context[torch.arange(encoded.size(0), device=encoded.device), last_indices]
        mm_context = (text_emb + image_emb) * 0.5
        mm_context = mm_context[torch.arange(encoded.size(0), device=encoded.device), last_indices]
        return context, weights, id_context, mm_context

    def represent_all_items_static(self) -> torch.Tensor:
        ids = torch.arange(self.num_items, device=self.text_features.device)
        id_emb = self.item_embedding(ids)
        text_emb = self.text_projection(self.text_features)
        image_emb = self.image_projection(self.image_features)
        graph_emb = self.project_graph_features(ids)
        time_emb = torch.zeros_like(id_emb)
        popularity = self.item_popularity.to(ids.device)
        cold_flags = (popularity <= self.cold_threshold).float()
        image_mask = self.image_mask
        if not self.use_text:
            text_emb = torch.zeros_like(text_emb)
        if not self.use_image:
            image_emb = torch.zeros_like(image_emb)
            image_mask = torch.zeros_like(image_mask)
        if not self.use_cold:
            cold_flags = torch.zeros_like(cold_flags)
        if not self.use_graph:
            graph_emb = torch.zeros_like(graph_emb)
        graph_score = torch.zeros((self.num_items, 3), dtype=torch.float, device=ids.device)
        fused, _ = self.moe(id_emb, text_emb, image_emb, time_emb, graph_emb, graph_score, popularity, cold_flags, image_mask)
        fused = fused.clone()
        fused[0].zero_()
        return fused

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, weights, id_context, mm_context = self.represent_sequence(
            batch["seq"],
            batch["times"],
            batch["popularity"],
            batch["cold_flags"],
        )
        item_repr = self.represent_all_items_static()
        scores = torch.matmul(context, item_repr.t()) + self.item_bias.unsqueeze(0)
        scores[:, 0] = -1e9
        target_repr = item_repr[batch["target"]]
        category_logits = self.category_head(target_repr)
        return {
            "scores": scores,
            "expert_weights": weights,
            "context": context,
            "id_context": id_context,
            "mm_context": mm_context,
            "category_logits": category_logits,
            "item_repr": item_repr,
        }

    def score_candidates(
        self,
        batch: dict[str, torch.Tensor],
        candidate_ids: torch.Tensor,
        graph_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context, _, _, _ = self.represent_sequence(
            batch["seq"],
            batch["times"],
            batch["popularity"],
            batch["cold_flags"],
        )
        candidate_repr, weights = self.represent_candidates_adaptive(context, candidate_ids, graph_scores)
        scores = torch.einsum("bh,bnh->bn", context, candidate_repr) + self.item_bias[candidate_ids]
        return {"scores": scores, "expert_weights": weights}

    def represent_candidates_adaptive(
        self,
        user_context: torch.Tensor,
        candidate_ids: torch.Tensor,
        graph_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_ids.dim() == 1:
            candidate_ids = candidate_ids.unsqueeze(0).expand(user_context.size(0), -1)
        if graph_scores is None:
            graph_scores = torch.zeros((*candidate_ids.shape, 3), dtype=torch.float, device=candidate_ids.device)
        elif graph_scores.dim() == 1:
            graph_scores = graph_scores.unsqueeze(0).expand(user_context.size(0), -1).unsqueeze(-1).expand(-1, -1, 3)
        elif graph_scores.dim() == 2 and graph_scores.size(-1) == 3:
            graph_scores = graph_scores.unsqueeze(0).expand(user_context.size(0), -1, -1)
        elif graph_scores.dim() == 2:
            graph_scores = graph_scores.unsqueeze(-1).expand(-1, -1, 3)
        graph_scores = self.prepare_graph_scores(graph_scores, candidate_ids)
        id_emb = self.item_embedding(candidate_ids)
        text_emb = self.text_projection(self.text_features[candidate_ids])
        image_emb = self.image_projection(self.image_features[candidate_ids])
        graph_emb = self.project_graph_features(candidate_ids)
        time_emb = torch.zeros_like(id_emb)
        popularity = self.item_popularity[candidate_ids].to(candidate_ids.device)
        cold_flags = (popularity <= self.cold_threshold).float()
        image_mask = self.image_mask[candidate_ids]
        if not self.use_text:
            text_emb = torch.zeros_like(text_emb)
        if not self.use_image:
            image_emb = torch.zeros_like(image_emb)
            image_mask = torch.zeros_like(image_mask)
        if not self.use_cold:
            cold_flags = torch.zeros_like(cold_flags)
        if not self.use_graph:
            graph_emb = torch.zeros_like(graph_emb)
        fused, weights = self.moe(id_emb, text_emb, image_emb, time_emb, graph_emb, graph_scores, popularity, cold_flags, image_mask, user_context)
        return fused, weights


def alignment_loss(id_context: torch.Tensor, mm_context: torch.Tensor, temperature: float) -> torch.Tensor:
    id_norm = F.normalize(id_context, dim=-1)
    mm_norm = F.normalize(mm_context, dim=-1)
    logits = torch.matmul(id_norm, mm_norm.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5


def router_balance_loss(expert_weights: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    mask = item_ids.ne(0).float().unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    mean_weights = (expert_weights * mask).sum(dim=(0, 1)) / denom
    target = torch.full_like(mean_weights, 1.0 / mean_weights.numel())
    return F.mse_loss(mean_weights, target)
