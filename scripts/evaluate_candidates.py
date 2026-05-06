from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from cstamoerec.candidate import (
    CandidateGenerator,
    candidate_recall,
    combine_candidate_sources,
    feature_similarity_candidates,
    graph_candidates,
    top_popularity,
)
from cstamoerec.config import load_config
from cstamoerec.data import load_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-generation recall.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--per-source-k", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--topk", nargs="+", type=int, default=[50, 100, 200])
    return parser.parse_args()


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    artifacts = load_artifacts(cfg.train.data_dir)
    generator = CandidateGenerator(artifacts, per_source_k=args.per_source_k, max_candidates=args.max_candidates)
    examples = artifacts["examples"][args.split]
    if args.limit_users:
        examples = examples[: args.limit_users]
    features = artifacts["features"]
    source_metrics = defaultdict(list)

    for ex in tqdm(examples, desc="candidate-eval"):
        seq = [int(x) for x in ex["seq"] if int(x) > 0]
        target = int(ex["target"])
        exclude = set(seq)
        named = {
            "popularity": top_popularity(features["item_popularity"], args.per_source_k, exclude),
            "transition": graph_candidates(seq, generator.transition_graph, args.per_source_k, exclude),
            "itemcf": graph_candidates(seq, generator.itemcf_graph, args.per_source_k, exclude),
            "text": feature_similarity_candidates(seq, features["text_embeddings"], args.per_source_k, exclude),
            "image": feature_similarity_candidates(
                seq,
                features["image_embeddings"],
                args.per_source_k,
                exclude,
                valid_mask=features.get("image_mask"),
            ),
        }
        for source, candidates in named.items():
            source_metrics[source].append(candidate_recall([item for item, _ in candidates], target, args.topk))
        combined = combine_candidate_sources(named, args.max_candidates)
        source_metrics["combined"].append(candidate_recall(combined.item_ids, target, args.topk))

    summary = {source: average(rows) for source, rows in source_metrics.items()}
    out_path = Path(cfg.train.save_dir) / f"candidate_recall_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved candidate recall to {out_path}")


if __name__ == "__main__":
    main()
