from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.candidate import CandidateGenerator, modal_graph_scores_for_items
from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts
from cstamoerec.metrics import MetricAverager, topk_metrics
from cstamoerec.reranker import source_prior_for_items
from cstamoerec.source_ranker import candidate_feature_matrix, load_ranker
from cstamoerec.train import build_model, mask_seen_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-stage candidate generation + MoE reranking.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec_dense10k/best_cstamoerec.pt")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--per-source-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["candidate", "source", "learned_source", "static", "adaptive", "hybrid"], default="hybrid")
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--model-weight", type=float, default=0.05)
    parser.add_argument("--rank-weight", type=float, default=1.0, help="Weight for the original combined candidate order in hybrid mode.")
    parser.add_argument("--source-ranker", default=None)
    parser.add_argument("--include-model-candidates", action="store_true", help="Add top candidates from the trained CS-TAMoERec model to Stage 1.")
    parser.add_argument(
        "--append-target-for-oracle",
        action="store_true",
        help="Oracle diagnostic only: append the held-out target if candidate generation missed it.",
    )
    return parser.parse_args()


def zero_metrics(topk: list[int]) -> dict[str, float]:
    metrics = {}
    for k in topk:
        for name in ("HR", "MRR", "NDCG", "Recall"):
            metrics[f"{name}@{k}"] = 0.0
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(cfg.train.data_dir)
    features = artifacts["features"]
    meta = artifacts["meta"]
    dataset = SequenceDataset(
        artifacts["examples"][args.split],
        meta["max_seq_len"],
        features["item_popularity"],
        features["item_categories"],
        meta["cold_threshold"],
    )
    model = build_model(cfg, artifacts, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    source_ranker = load_ranker(args.source_ranker) if args.source_ranker else None
    generator = CandidateGenerator(artifacts, per_source_k=args.per_source_k, max_candidates=args.max_candidates)
    avg = MetricAverager()
    total = len(dataset) if not args.limit_users else min(args.limit_users, len(dataset))
    pool_hits = 0
    pool_sizes = []
    for idx in tqdm(range(total), desc="two-stage-eval"):
        raw_ex = artifacts["examples"][args.split][idx]
        seq = [int(x) for x in raw_ex["seq"] if int(x) > 0]
        batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
        candidates = generator.generate(
            seq,
            model=model,
            batch=batch,
            include_sasrec=args.include_model_candidates,
        )
        target = int(raw_ex["target"])
        target_in_pool = target in candidates.item_ids
        if target_in_pool:
            pool_hits += 1
        elif args.append_target_for_oracle:
            candidates.item_ids.append(int(raw_ex["target"]))
        else:
            avg.update(zero_metrics(cfg.train.eval_topk), 1)
            pool_sizes.append(len(candidates.item_ids))
            continue
        pool_sizes.append(len(candidates.item_ids))
        graph_scores = modal_graph_scores_for_items(
            seq,
            candidates.item_ids,
            generator.transition_graph,
            generator.text_graph,
            generator.image_graph,
        )
        candidate_tensor = torch.tensor(candidates.item_ids, dtype=torch.long, device=device)
        graph_tensor = torch.tensor(graph_scores, dtype=torch.float, device=device)
        with torch.no_grad():
            if args.mode == "candidate":
                scores = -torch.arange(len(candidates.item_ids), dtype=torch.float).view(1, -1)
            elif args.mode == "source":
                scores = source_prior_for_items(candidates.item_ids, candidates.sources, graph_scores).view(1, -1)
            elif args.mode == "learned_source":
                if source_ranker is None:
                    raise ValueError("--source-ranker is required when --mode learned_source")
                source_features = candidate_feature_matrix(
                    candidates.item_ids,
                    candidates.sources,
                    candidates.source_scores,
                    graph_scores,
                    features["item_popularity"],
                )
                scores = source_ranker.score(source_features.to(device)).detach().cpu().view(1, -1)
            elif args.mode == "adaptive":
                scores = model.score_candidates(batch, candidate_tensor, graph_scores=graph_tensor)["scores"].detach().cpu()
            else:
                full_scores = mask_seen_items(model(batch)["scores"], batch["seq"])
                scores = full_scores[:, candidate_tensor].detach().cpu()
                if args.mode == "hybrid":
                    prior = source_prior_for_items(candidates.item_ids, candidates.sources, graph_scores).view(1, -1)
                    rank_prior = -torch.arange(len(candidates.item_ids), dtype=torch.float).view(1, -1)
                    if rank_prior.numel() > 1 and float(rank_prior.std()) > 1e-8:
                        rank_prior = (rank_prior - rank_prior.mean()) / rank_prior.std()
                    if scores.numel() > 1 and float(scores.std()) > 1e-8:
                        scores = (scores - scores.mean()) / scores.std()
                    scores = args.model_weight * scores + args.prior_weight * prior + args.rank_weight * rank_prior
        target_pos = torch.tensor([candidates.item_ids.index(target)])
        avg.update(topk_metrics(scores, target_pos, cfg.train.eval_topk), 1)
    summary = avg.compute()
    summary["CandidatePoolHitRate"] = pool_hits / max(total, 1)
    summary["AvgCandidatePoolSize"] = sum(pool_sizes) / max(len(pool_sizes), 1)
    summary["OracleAppendTarget"] = bool(args.append_target_for_oracle)
    suffix = "_oracle" if args.append_target_for_oracle else ""
    out_path = Path(cfg.train.save_dir) / f"two_stage_rerank_{args.mode}_{args.split}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved two-stage metrics to {out_path}")


if __name__ == "__main__":
    main()
