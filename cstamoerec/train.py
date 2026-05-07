from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts, set_seed
from cstamoerec.metrics import MetricAverager, coverage_from_topk, topk_metrics
from cstamoerec.model import CSTAMoERec, alignment_loss, router_balance_loss


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def make_loaders(data_dir: str, batch_size: int, num_workers: int) -> tuple[dict, dict[str, DataLoader]]:
    artifacts = load_artifacts(data_dir)
    features = artifacts["features"]
    meta = artifacts["meta"]
    datasets = {
        split: SequenceDataset(
            artifacts["examples"][split],
            meta["max_seq_len"],
            features["item_popularity"],
            features["item_categories"],
            meta["cold_threshold"],
        )
        for split in ("train", "valid", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return artifacts, loaders


def build_model(cfg, artifacts: dict, device: str) -> CSTAMoERec:
    features = artifacts["features"]
    meta = artifacts["meta"]
    graph_embeddings = features.get("graph_embeddings")
    graph_dim = int(graph_embeddings.size(1)) if graph_embeddings is not None else cfg.model.graph_dim
    return CSTAMoERec(
        num_items=meta["num_items"],
        num_categories=meta["num_categories"],
        text_embeddings=features["text_embeddings"],
        image_embeddings=features["image_embeddings"],
        image_mask=features["image_mask"],
        item_popularity=features["item_popularity"],
        graph_embeddings=graph_embeddings,
        hidden_size=cfg.model.hidden_size,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        max_seq_len=meta["max_seq_len"],
        text_dim=meta["text_dim"],
        image_dim=meta["image_dim"],
        time_dim=cfg.model.time_dim,
        router_hidden=cfg.model.router_hidden,
        graph_dim=graph_dim,
        cold_threshold=meta["cold_threshold"],
        use_text=cfg.model.use_text,
        use_image=cfg.model.use_image,
        use_time=cfg.model.use_time,
        use_cold=cfg.model.use_cold,
        use_cross=cfg.model.use_cross,
        use_graph=cfg.model.use_graph,
    ).to(device)


def train_one_epoch(model, loader, optimizer, cfg, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        output = model(batch)
        next_loss = F.cross_entropy(output["scores"], batch["target"])
        cat_loss = F.cross_entropy(output["category_logits"], batch["target_category"])
        align = alignment_loss(output["id_context"], output["mm_context"], cfg.train.alignment_temperature)
        balance = router_balance_loss(output["expert_weights"], batch["seq"])
        loss = (
            next_loss
            + cfg.train.category_loss_weight * cat_loss
            + cfg.train.alignment_loss_weight * align
            + cfg.train.router_balance_loss_weight * balance
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch["seq"].size(0)
        total_count += batch["seq"].size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device: str, item_popularity: torch.Tensor, split_name: str) -> dict[str, float]:
    model.eval()
    avg = MetricAverager()
    cold_avg = MetricAverager()
    warm_avg = MetricAverager()
    topk_items: list[int] = []
    item_popularity = item_popularity.to(device)
    for batch in tqdm(loader, desc=f"eval-{split_name}", leave=False):
        batch = move_batch(batch, device)
        output = model(batch)
        scores = output["scores"]
        metrics = topk_metrics(scores, batch["target"], cfg.train.eval_topk)
        avg.update(metrics, batch["seq"].size(0))

        max_k = min(max(cfg.train.eval_topk), scores.size(1))
        _, top_idx = torch.topk(scores, k=max_k, dim=1)
        topk_items.extend(top_idx[:, :10].reshape(-1).detach().cpu().tolist())

        target_pop = item_popularity[batch["target"]]
        cold_mask = target_pop <= cfg.data.cold_threshold
        warm_mask = ~cold_mask
        if cold_mask.any():
            cold_avg.update(topk_metrics(scores[cold_mask], batch["target"][cold_mask], cfg.train.eval_topk), int(cold_mask.sum()))
        if warm_mask.any():
            warm_avg.update(topk_metrics(scores[warm_mask], batch["target"][warm_mask], cfg.train.eval_topk), int(warm_mask.sum()))

    result = avg.compute()
    for key, value in cold_avg.compute().items():
        result[f"cold_{key}"] = value
    for key, value in warm_avg.compute().items():
        result[f"warm_{key}"] = value
    result["Coverage@10"] = coverage_from_topk(topk_items, model.num_items)
    return result


@torch.no_grad()
def popularity_baseline(loader, cfg, item_popularity: torch.Tensor, device: str) -> dict[str, float]:
    scores = item_popularity.float().to(device).unsqueeze(0)
    scores[:, 0] = -1e9
    avg = MetricAverager()
    for batch in loader:
        batch = move_batch(batch, device)
        batch_scores = scores.expand(batch["seq"].size(0), -1)
        avg.update(topk_metrics(batch_scores, batch["target"], cfg.train.eval_topk), batch["seq"].size(0))
    return avg.compute()


def run_training_experiment(cfg, device: str, run_name: str | None = None) -> dict:
    set_seed(cfg.train.seed)
    artifacts, loaders = make_loaders(cfg.train.data_dir, cfg.train.batch_size, cfg.train.num_workers)
    model = build_model(cfg, artifacts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    save_dir = Path(cfg.train.save_dir)
    if run_name:
        save_dir = save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pop_valid = popularity_baseline(loaders["valid"], cfg, artifacts["features"]["item_popularity"], device)
    print("Popularity valid:", pop_valid)
    best_metric = -1.0
    best_path = save_dir / "best_cstamoerec.pt"
    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, cfg, device)
        valid = evaluate(model, loaders["valid"], cfg, device, artifacts["features"]["item_popularity"], "valid")
        ndcg10 = valid.get("NDCG@10", 0.0)
        print(f"epoch={epoch} loss={loss:.4f} valid={valid}")
        history.append({"epoch": epoch, "loss": loss, "valid": valid})
        if ndcg10 > best_metric:
            best_metric = ndcg10
            torch.save({"model": model.state_dict(), "meta": artifacts["meta"], "config": None}, best_path)
            print(f"saved best checkpoint to {best_path}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, loaders["test"], cfg, device, artifacts["features"]["item_popularity"], "test")
    result = {"popularity_valid": pop_valid, "history": history, "test": test, "checkpoint": str(best_path)}
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("test:", test)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CS-TAMoERec.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    result = run_training_experiment(cfg, device)
    checkpoint = torch.load(result["checkpoint"], map_location=device, weights_only=False)
    checkpoint["config"] = args.config
    torch.save(checkpoint, result["checkpoint"])


if __name__ == "__main__":
    main()
