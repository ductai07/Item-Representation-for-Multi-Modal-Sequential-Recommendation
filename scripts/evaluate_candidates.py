from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.candidate import (
    CandidateGenerator,
    DEFAULT_SOURCE_WEIGHTS,
    candidate_recall,
    combine_candidate_sources,
    feature_similarity_candidates,
    graph_candidates,
    sasrec_candidates,
    sequence_graph_candidates,
    top_popularity,
)
from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts
from cstamoerec.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-generation recall.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--per-source-k", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--topk", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--include-model-candidates", action="store_true", help="Add top candidates from a trained CS-TAMoERec checkpoint.")
    parser.add_argument("--device", default=None)
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
    model = None
    dataset = None
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    if args.include_model_candidates:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --include-model-candidates is set.")
        model = build_model(cfg, artifacts, device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        meta = artifacts["meta"]
        dataset = SequenceDataset(
            artifacts["examples"][args.split],
            meta["max_seq_len"],
            features["item_popularity"],
            features["item_categories"],
            meta["cold_threshold"],
        )
    source_metrics = defaultdict(list)

    for idx, ex in enumerate(tqdm(examples, desc="candidate-eval")):
        seq = [int(x) for x in ex["seq"] if int(x) > 0]
        target = int(ex["target"])
        exclude = set(seq)
        named = {
            "popularity": top_popularity(features["item_popularity"], args.per_source_k, exclude),
            "transition": graph_candidates(seq, generator.transition_graph, args.per_source_k, exclude),
            "itemcf": graph_candidates(seq, generator.itemcf_graph, args.per_source_k, exclude),
            "sequence_graph": sequence_graph_candidates(seq, generator.sequence_index, args.per_source_k, exclude),
            "text_graph": graph_candidates(seq, generator.text_graph, args.per_source_k, exclude),
            "image_graph": graph_candidates(seq, generator.image_graph, args.per_source_k, exclude),
            "text": feature_similarity_candidates(seq, features["text_embeddings"], args.per_source_k, exclude),
            "image": feature_similarity_candidates(
                seq,
                features["image_embeddings"],
                args.per_source_k,
                exclude,
                valid_mask=features.get("image_mask"),
            ),
        }
        if model is not None and dataset is not None:
            batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
            named["sasrec"] = sasrec_candidates(model, batch, args.per_source_k)
        for source, candidates in named.items():
            source_metrics[source].append(candidate_recall([item for item, _ in candidates], target, args.topk))
        if model is not None and dataset is not None:
            batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
            combined_with_model = generator.generate(seq, model=model, batch=batch, include_sasrec=True)
            source_metrics["combined_with_model"].append(candidate_recall(combined_with_model.item_ids, target, args.topk))
        combined_equal = combine_candidate_sources(named, args.max_candidates)
        source_metrics["combined_equal"].append(candidate_recall(combined_equal.item_ids, target, args.topk))
        combined_weighted = combine_candidate_sources(named, args.max_candidates, DEFAULT_SOURCE_WEIGHTS)
        source_metrics["combined_weighted"].append(candidate_recall(combined_weighted.item_ids, target, args.topk))

    summary = {source: average(rows) for source, rows in source_metrics.items()}
    out_path = Path(cfg.train.save_dir) / f"candidate_recall_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved candidate recall to {out_path}")


if __name__ == "__main__":
    main()
