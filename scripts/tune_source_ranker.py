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
from cstamoerec.data import load_artifacts, set_seed
from cstamoerec.metrics import MetricAverager, topk_metrics
from cstamoerec.source_ranker import (
    FEATURE_NAMES,
    LearnedSourceRanker,
    candidate_feature_matrix,
    fit_feature_stats,
    save_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a lightweight learned source reranker on candidate features.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--train-split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--eval-split", default="test", choices=["valid", "test"])
    parser.add_argument("--per-source-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_rows(artifacts: dict, generator: CandidateGenerator, split: str, limit: int = 0) -> list[tuple[torch.Tensor, int]]:
    features = artifacts["features"]
    examples = artifacts["examples"][split]
    if limit:
        examples = examples[:limit]
    rows = []
    for ex in tqdm(examples, desc=f"source-ranker-build-{split}"):
        seq = [int(x) for x in ex["seq"] if int(x) > 0]
        target = int(ex["target"])
        candidates = generator.generate(seq)
        if target not in candidates.item_ids:
            continue
        graph_scores = modal_graph_scores_for_items(
            seq,
            candidates.item_ids,
            generator.transition_graph,
            generator.text_graph,
            generator.image_graph,
        )
        matrix = candidate_feature_matrix(
            candidates.item_ids,
            candidates.sources,
            candidates.source_scores,
            graph_scores,
            features["item_popularity"],
        )
        rows.append((matrix, candidates.item_ids.index(target)))
    return rows


def evaluate_rows(rows: list[tuple[torch.Tensor, int]], ranker: LearnedSourceRanker, topk: list[int], device: str) -> dict[str, float]:
    avg = MetricAverager()
    for matrix, target_pos in rows:
        matrix = matrix.to(device)
        scores = ranker.score(matrix).view(1, -1).detach().cpu()
        target = torch.tensor([target_pos])
        avg.update(topk_metrics(scores, target, topk), 1)
    return avg.compute()


def train_ranker(
    rows: list[tuple[torch.Tensor, int]],
    mean: torch.Tensor,
    std: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> LearnedSourceRanker:
    weights = torch.zeros(len(FEATURE_NAMES), dtype=torch.float, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weights], lr=lr, weight_decay=weight_decay)
    mean = mean.to(device)
    std = std.to(device)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for matrix, target_pos in rows:
            matrix = matrix.to(device)
            matrix = (matrix - mean) / std
            logits = matrix @ weights
            loss = F.cross_entropy(logits.view(1, -1), torch.tensor([target_pos], dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"epoch={epoch} loss={total / max(len(rows), 1):.4f}")
    return LearnedSourceRanker(weights=weights.detach().cpu(), feature_names=list(FEATURE_NAMES), mean=mean.cpu(), std=std.cpu())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(cfg.train.data_dir)
    generator = CandidateGenerator(artifacts, per_source_k=args.per_source_k, max_candidates=args.max_candidates)

    train_rows = build_rows(artifacts, generator, args.train_split, args.limit_train)
    if not train_rows:
        raise RuntimeError("No train rows with target present in candidate pool. Increase per-source-k/max-candidates.")
    mean, std = fit_feature_stats([row[0] for row in train_rows])
    ranker = train_ranker(train_rows, mean, std, args.epochs, args.lr, args.weight_decay, device)
    train_metrics = evaluate_rows(train_rows, ranker, cfg.train.eval_topk, device)

    eval_rows = build_rows(artifacts, generator, args.eval_split, args.limit_eval)
    eval_metrics = evaluate_rows(eval_rows, ranker, cfg.train.eval_topk, device)
    output = Path(args.output) if args.output else Path(cfg.train.save_dir) / "learned_source_ranker.json"
    metadata = {
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "per_source_k": args.per_source_k,
        "max_candidates": args.max_candidates,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_metrics_candidate_hit_subset": train_metrics,
        "eval_metrics_candidate_hit_subset": eval_metrics,
    }
    save_ranker(output, ranker, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"Saved learned source ranker to {output}")


if __name__ == "__main__":
    main()
