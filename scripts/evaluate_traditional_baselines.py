from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.candidate import CandidateGenerator, graph_score_for_items, modal_graph_scores_for_items
from cstamoerec.config import load_config
from cstamoerec.data import load_artifacts, set_seed
from cstamoerec.metrics import MetricAverager, topk_metrics
from cstamoerec.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traditional baselines with sampled negatives.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sample_negatives(
    num_items: int,
    seq: list[int],
    target: int,
    num_negatives: int,
    generator: torch.Generator,
) -> list[int]:
    forbidden = {0, int(target), *[int(x) for x in seq if int(x) > 0]}
    negatives: list[int] = []
    while len(negatives) < num_negatives:
        draw_size = max((num_negatives - len(negatives)) * 3, 32)
        draws = torch.randint(1, num_items, (draw_size,), generator=generator).tolist()
        for item_id in draws:
            if item_id not in forbidden:
                forbidden.add(int(item_id))
                negatives.append(int(item_id))
                if len(negatives) == num_negatives:
                    break
    return negatives


def cosine_scores(seq: list[int], candidates: list[int], features: torch.Tensor, valid_mask: torch.Tensor | None = None) -> list[float]:
    recent = [int(x) for x in seq if int(x) > 0][-5:]
    if not recent:
        return [0.0 for _ in candidates]
    matrix = features.float()
    query = F.normalize(matrix[recent].mean(dim=0, keepdim=True), dim=-1)
    candidate_ids = torch.tensor(candidates, dtype=torch.long)
    candidate_vectors = F.normalize(matrix[candidate_ids], dim=-1)
    scores = torch.matmul(candidate_vectors, query.squeeze(0)).tolist()
    if valid_mask is not None:
        mask_values = valid_mask[candidate_ids].tolist()
        scores = [float(score) if mask else -1e9 for score, mask in zip(scores, mask_values)]
    return [float(score) for score in scores]


def combined_rank_scores(source_scores: dict[str, list[float]]) -> list[float]:
    num_candidates = len(next(iter(source_scores.values())))
    aggregate = Counter()
    for scores in source_scores.values():
        ranked = sorted(range(num_candidates), key=lambda idx: scores[idx], reverse=True)
        for rank, idx in enumerate(ranked, start=1):
            aggregate[idx] += 1.0 / rank
    return [float(aggregate[idx]) for idx in range(num_candidates)]


def make_batch(ex: dict, features: dict, meta: dict, device: str) -> dict[str, torch.Tensor]:
    seq = [int(x) for x in ex["seq"][-meta["max_seq_len"] :]]
    times = [int(x) for x in ex["times"][-meta["max_seq_len"] :]]
    pad_len = meta["max_seq_len"] - len(seq)
    seq_tensor = torch.tensor([[0] * pad_len + seq], dtype=torch.long, device=device)
    time_tensor = torch.tensor([[0] * pad_len + times], dtype=torch.long, device=device)
    popularity = features["item_popularity"][seq_tensor.cpu()].float().to(device)
    cold_flags = (popularity <= meta["cold_threshold"]).float()
    target = torch.tensor([int(ex["target"])], dtype=torch.long, device=device)
    return {
        "seq": seq_tensor,
        "times": time_tensor,
        "popularity": popularity,
        "cold_flags": cold_flags,
        "target": target,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    artifacts = load_artifacts(cfg.train.data_dir)
    features = artifacts["features"]
    meta = artifacts["meta"]
    examples = artifacts["examples"][args.split]
    if args.limit_users:
        examples = examples[: args.limit_users]

    generator = CandidateGenerator(artifacts, per_source_k=200, max_candidates=500)
    rng = torch.Generator().manual_seed(args.seed + (29 if args.split == "test" else 17))

    averages = {
        "popularity": MetricAverager(),
        "transition": MetricAverager(),
        "itemcf": MetricAverager(),
        "text_knn": MetricAverager(),
        "image_knn": MetricAverager(),
        "combined": MetricAverager(),
    }

    popularity = features["item_popularity"].float()
    text_embeddings = features["text_embeddings"]
    image_embeddings = features["image_embeddings"]
    image_mask = features.get("image_mask")
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    model = None
    if args.checkpoint:
        model = build_model(cfg, artifacts, device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        averages["cstamoerec_adaptive"] = MetricAverager()

    full_rank = args.num_negatives <= 0
    for ex in tqdm(examples, desc="traditional-baselines"):
        seq = [int(x) for x in ex["seq"] if int(x) > 0]
        target = int(ex["target"])
        if full_rank:
            seen = {0, *[int(x) for x in seq if int(x) > 0]}
            others = [i for i in range(1, meta["num_items"]) if i not in seen and i != target]
            candidates = [target] + others
            target_pos = 0
        else:
            negatives = sample_negatives(meta["num_items"], seq, target, args.num_negatives, rng)
            target_pos = int(torch.randint(0, args.num_negatives + 1, (1,), generator=rng).item())
            candidates = negatives[:target_pos] + [target] + negatives[target_pos:]
        target_index = torch.tensor([target_pos], dtype=torch.long)

        source_scores = {
            "popularity": [float(popularity[item]) for item in candidates],
            "transition": graph_score_for_items(seq, candidates, generator.transition_graph),
            "itemcf": graph_score_for_items(seq, candidates, generator.itemcf_graph),
            "text_knn": cosine_scores(seq, candidates, text_embeddings),
            "image_knn": cosine_scores(seq, candidates, image_embeddings, image_mask),
        }
        source_scores["combined"] = combined_rank_scores(source_scores)

        for name, scores in source_scores.items():
            score_tensor = torch.tensor([scores], dtype=torch.float)
            averages[name].update(topk_metrics(score_tensor, target_index, args.topk), 1)

        if model is not None:
            batch = make_batch(ex, features, meta, device)
            candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=device)
            graph_scores = modal_graph_scores_for_items(
                seq,
                candidates,
                generator.transition_graph,
                generator.text_graph,
                generator.image_graph,
            )
            graph_tensor = torch.tensor([graph_scores], dtype=torch.float, device=device)
            with torch.no_grad():
                scores = model.score_candidates(batch, candidate_tensor, graph_tensor)["scores"].detach().cpu()
            averages["cstamoerec_adaptive"].update(topk_metrics(scores, target_index, args.topk), 1)

    summary = {name: avg.compute() for name, avg in averages.items()}
    out_path = Path(cfg.train.save_dir) / f"traditional_baselines_{args.split}_{args.num_negatives}neg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved traditional baseline metrics to {out_path}")


if __name__ == "__main__":
    main()
