from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cstamoerec.candidate import CandidateGenerator, graph_score_for_items
from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts, load_json
from cstamoerec.reranker import rerank_candidates
from cstamoerec.train import build_model


EXPERTS = ["ID", "Text", "Image", "Time", "Cross", "Graph"]


def expert_names(weights: list[float]) -> list[str]:
    return EXPERTS[: len(weights)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export demo cache for two-stage CS-TAMoERec.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec/best_cstamoerec.pt")
    parser.add_argument("--output-dir", default="demo_cache")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--per-source-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--topk", type=int, default=10)
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
    generator = CandidateGenerator(artifacts, per_source_k=args.per_source_k, max_candidates=args.max_candidates)

    out = {"experts": EXPERTS, "users": []}
    total = min(args.num_users, len(dataset))
    for idx in tqdm(range(total), desc="export-demo-cache"):
        batch = dataset[idx]
        raw_ex = artifacts["examples"][args.split][idx]
        seq = [int(x) for x in raw_ex["seq"] if int(x) > 0]
        candidates = generator.generate(seq)
        graph_scores = graph_score_for_items(seq, candidates.item_ids, generator.transition_graph)
        graph_score_by_item = {item: score for item, score in zip(candidates.item_ids, graph_scores)}
        reranked = rerank_candidates(model, batch, candidates.item_ids, device, topk=args.topk, graph_scores=graph_scores)
        history = [card_for_item(meta, item_cards, features, item) for item in seq[-10:]]
        recs = []
        for rank, item_id in enumerate(reranked["item_ids"], start=1):
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
                    "graph_score": float(graph_score_by_item.get(item_id, 0.0)),
                    "expert_weights": {names[i]: float(weights[i]) for i in range(len(names))},
                    "main_expert": names[int(torch.tensor(weights).argmax())],
                    "cold": int(features["item_popularity"][item_id]) <= meta["cold_threshold"],
                }
            )
            recs.append(rec_card)
        out["users"].append(
            {
                "index": idx,
                "user_id": int(raw_ex["user_id"]),
                "target": card_for_item(meta, item_cards, features, int(raw_ex["target"])),
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
