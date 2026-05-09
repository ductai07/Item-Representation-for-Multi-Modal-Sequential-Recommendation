from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.candidate import CandidateGenerator, modal_graph_scores_for_items
from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts, load_json
from cstamoerec.reranker import rerank_candidates
from cstamoerec.source_ranker import candidate_feature_matrix, load_ranker
from cstamoerec.train import build_model


EXPERTS = ["ID", "Text", "Image", "Time", "Cross", "Graph"]


def expert_names(weights: list[float]) -> list[str]:
    return EXPERTS[: len(weights)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export demo cache for two-stage CS-TAMoERec.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec_dense10k/best_cstamoerec.pt")
    parser.add_argument("--output-dir", default="demo_cache")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--per-source-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--mode", choices=["candidate", "source", "learned_source", "static", "adaptive", "hybrid"], default="hybrid")
    parser.add_argument("--source-ranker", default=None)
    parser.add_argument("--include-model-candidates", action="store_true", help="Add top candidates from the trained CS-TAMoERec model to Stage 1.")
    parser.add_argument("--only-hits", action="store_true", help="Export only cases where the held-out target is in top hit-k.")
    parser.add_argument("--hit-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def card_for_item(meta: dict, item_cards: dict, features: dict, item_id: int) -> dict:
    asin = meta["id2item"][item_id]
    title = meta.get("item_titles", [""])[item_id] if item_id < len(meta.get("item_titles", [])) else asin
    card = item_cards.get(asin, {})
    return {
        "item_id": item_id,
        "asin": asin,
        "title": title,
        "category": card.get("category", "Unknown"),
        "image_url": card.get("image_url"),
        "text": card.get("text", "")[:700],
        "popularity": int(features["item_popularity"][item_id]),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(cfg.train.data_dir)
    item_cards = load_json(Path(cfg.train.data_dir) / "item_cards.json")
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

    out = {"experts": EXPERTS, "users": [], "mode": args.mode}
    needed_topk = max(args.topk, args.hit_k)
    for idx in tqdm(range(len(dataset)), desc="export-demo-cache"):
        if len(out["users"]) >= args.num_users:
            break
        batch = dataset[idx]
        raw_ex = artifacts["examples"][args.split][idx]
        seq = [int(x) for x in raw_ex["seq"] if int(x) > 0]
        target = int(raw_ex["target"])
        device_batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
        candidates = generator.generate(
            seq,
            model=model,
            batch=device_batch,
            include_sasrec=args.include_model_candidates,
        )
        if target not in candidates.item_ids and args.only_hits:
            continue
        graph_scores = modal_graph_scores_for_items(
            seq,
            candidates.item_ids,
            generator.transition_graph,
            generator.text_graph,
            generator.image_graph,
        )
        source_features = None
        if args.mode == "learned_source":
            if source_ranker is None:
                raise ValueError("--source-ranker is required when --mode learned_source")
            source_features = candidate_feature_matrix(
                candidates.item_ids,
                candidates.sources,
                candidates.source_scores,
                graph_scores,
                features["item_popularity"],
            )
        graph_score_by_item = {item: score for item, score in zip(candidates.item_ids, graph_scores)}
        reranked = rerank_candidates(
            model,
            batch,
            candidates.item_ids,
            device,
            topk=needed_topk,
            graph_scores=graph_scores,
            sources=candidates.sources,
            mode=args.mode,
            source_ranker=source_ranker,
            source_features=source_features,
        )
        target_rank = None
        if target in reranked["item_ids"]:
            target_rank = reranked["item_ids"].index(target) + 1
        if args.only_hits and (target_rank is None or target_rank > args.hit_k):
            continue
        history = [card_for_item(meta, item_cards, features, item) for item in seq[-10:]]
        recs = []
        for rank, item_id in enumerate(reranked["item_ids"][: args.topk], start=1):
            weights = reranked["expert_weights"][rank - 1]
            names = expert_names(weights)
            source_names = sorted(candidates.sources.get(item_id, []))
            rec_card = card_for_item(meta, item_cards, features, item_id)
            rec_card.update(
                {
                    "rank": rank,
                    "score": reranked["scores"][rank - 1],
                    "sources": source_names,
                    "main_source": "+".join(source_names[:2]) if source_names else "reranker",
                    "graph_score": float(sum(graph_score_by_item.get(item_id, [0.0, 0.0, 0.0]))),
                    "modal_graph_scores": {
                        "id": float(graph_score_by_item.get(item_id, [0.0, 0.0, 0.0])[0]),
                        "text": float(graph_score_by_item.get(item_id, [0.0, 0.0, 0.0])[1]),
                        "image": float(graph_score_by_item.get(item_id, [0.0, 0.0, 0.0])[2]),
                    },
                    "expert_weights": {names[i]: float(weights[i]) for i in range(len(names))},
                    "main_expert": names[int(torch.tensor(weights).argmax())],
                    "cold": int(features["item_popularity"][item_id]) <= meta["cold_threshold"],
                    "is_target": item_id == target,
                }
            )
            recs.append(rec_card)
        out["users"].append(
            {
                "index": idx,
                "user_id": int(raw_ex["user_id"]),
                "target": card_for_item(meta, item_cards, features, target),
                "target_rank": target_rank,
                "hit_at_k": target_rank is not None and target_rank <= args.hit_k,
                "history": history,
                "recommendations": recs,
                "candidate_count": len(candidates.item_ids),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "recommendations.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print(f"Saved demo cache to {output_dir / 'recommendations.json'}")


if __name__ == "__main__":
    main()
