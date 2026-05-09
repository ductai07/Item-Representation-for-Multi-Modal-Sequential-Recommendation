from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.candidate import CandidateGenerator, modal_graph_scores_for_items
from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts, set_seed
from cstamoerec.metrics import MetricAverager, topk_metrics
from cstamoerec.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CS-TAMoERec reranker directly on generated candidate sets.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--checkpoint-in", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--per-source-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-valid", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def make_dataset(artifacts: dict, split: str) -> SequenceDataset:
    features = artifacts["features"]
    meta = artifacts["meta"]
    return SequenceDataset(
        artifacts["examples"][split],
        meta["max_seq_len"],
        features["item_popularity"],
        features["item_categories"],
        meta["cold_threshold"],
    )


def candidate_pack(generator: CandidateGenerator, seq: list[int], target: int, append_target: bool) -> tuple[list[int], list[list[float]], int | None]:
    candidates = generator.generate(seq)
    item_ids = list(candidates.item_ids)
    target_pos = item_ids.index(target) if target in item_ids else None
    if target_pos is None and append_target:
        item_ids.append(target)
        target_pos = len(item_ids) - 1
    graph_scores = modal_graph_scores_for_items(
        seq,
        item_ids,
        generator.transition_graph,
        generator.text_graph,
        generator.image_graph,
    )
    return item_ids, graph_scores, target_pos


def zero_metrics(topk: list[int]) -> dict[str, float]:
    metrics = {}
    for k in topk:
        for name in ("HR", "MRR", "NDCG", "Recall"):
            metrics[f"{name}@{k}"] = 0.0
    return metrics


@torch.no_grad()
def evaluate(model, artifacts: dict, generator: CandidateGenerator, split: str, cfg, device: str, limit: int = 0) -> dict[str, float]:
    model.eval()
    dataset = make_dataset(artifacts, split)
    examples = artifacts["examples"][split]
    total = min(limit, len(dataset)) if limit else len(dataset)
    avg = MetricAverager()
    hit_pool = 0
    for idx in tqdm(range(total), desc=f"candidate-rerank-eval-{split}", leave=False):
        ex = examples[idx]
        seq = [int(x) for x in ex["seq"] if int(x) > 0]
        target = int(ex["target"])
        item_ids, graph_scores, target_pos = candidate_pack(generator, seq, target, append_target=False)
        if target_pos is None:
            avg.update(zero_metrics(cfg.train.eval_topk), 1)
            continue
        hit_pool += 1
        batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
        candidate_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
        graph_tensor = torch.tensor(graph_scores, dtype=torch.float, device=device)
        scores = model.score_candidates(batch, candidate_tensor, graph_scores=graph_tensor)["scores"].detach().cpu()
        avg.update(topk_metrics(scores, torch.tensor([target_pos]), cfg.train.eval_topk), 1)
    result = avg.compute()
    result["CandidatePoolHitRate"] = hit_pool / max(total, 1)
    return result


def train_one_epoch(model, artifacts: dict, generator: CandidateGenerator, cfg, device: str, optimizer, limit: int = 0) -> float:
    model.train()
    dataset = make_dataset(artifacts, "train")
    examples = artifacts["examples"]["train"]
    total = min(limit, len(dataset)) if limit else len(dataset)
    order = torch.randperm(total).tolist()
    total_loss = 0.0
    for idx in tqdm(order, desc="candidate-rerank-train", leave=False):
        ex = examples[idx]
        seq = [int(x) for x in ex["seq"] if int(x) > 0]
        target = int(ex["target"])
        item_ids, graph_scores, target_pos = candidate_pack(generator, seq, target, append_target=True)
        if target_pos is None:
            continue
        batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
        candidate_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
        graph_tensor = torch.tensor(graph_scores, dtype=torch.float, device=device)
        scores = model.score_candidates(batch, candidate_tensor, graph_scores=graph_tensor)["scores"]
        loss = F.cross_entropy(scores, torch.tensor([target_pos], dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(total, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(cfg.train.data_dir)
    model = build_model(cfg, artifacts, device)
    if args.checkpoint_in:
        checkpoint = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    generator = CandidateGenerator(artifacts, per_source_k=args.per_source_k, max_candidates=args.max_candidates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.checkpoint_out) if args.checkpoint_out else Path(cfg.train.save_dir) / "best_candidate_reranker.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    best_ndcg10 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, artifacts, generator, cfg, device, optimizer, args.limit_train)
        valid = evaluate(model, artifacts, generator, "valid", cfg, device, args.limit_valid)
        print(f"epoch={epoch} loss={loss:.4f} valid={valid}")
        history.append({"epoch": epoch, "loss": loss, "valid": valid})
        ndcg10 = valid.get("NDCG@10", 0.0)
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            torch.save({"model": model.state_dict(), "meta": artifacts["meta"], "config": args.config}, output)
            print(f"saved best candidate reranker to {output}")

    with open(output.parent / "candidate_reranker_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "checkpoint": str(output)}, f, indent=2)


if __name__ == "__main__":
    main()
