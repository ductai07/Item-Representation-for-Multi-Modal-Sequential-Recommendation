from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts, load_json
from cstamoerec.train import build_model


MODES = {
    "full": {},
    "mask_text": {"model.use_text": False},
    "mask_image": {"model.use_image": False},
    "mask_time": {"model.use_time": False},
    "mask_text_image": {"model.use_text": False, "model.use_image": False},
}


def set_nested(obj, dotted_key: str, value) -> None:
    current = obj
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def item_rank(scores: torch.Tensor, item_id: int) -> int:
    item_score = scores[item_id]
    return int((scores > item_score).sum().item()) + 1


def item_card(meta: dict, item_cards: dict, item_id: int) -> dict:
    asin = meta["id2item"][item_id]
    titles = meta.get("item_titles", [])
    return {
        "item_id": int(item_id),
        "asin": asin,
        "title": titles[item_id] if item_id < len(titles) else asin,
        "category": item_cards.get(asin, {}).get("category", "Unknown"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual rank changes for modality removal.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec/best_cstamoerec.pt")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--limit-users", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    device = args.device or (base_cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(base_cfg.train.data_dir)
    item_cards_path = Path(base_cfg.train.data_dir) / "item_cards.json"
    item_cards = load_json(item_cards_path) if item_cards_path.exists() else {}
    features = artifacts["features"]
    meta = artifacts["meta"]
    dataset = SequenceDataset(
        artifacts["examples"][args.split],
        meta["max_seq_len"],
        features["item_popularity"],
        features["item_categories"],
        meta["cold_threshold"],
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    models = {}
    for mode, overrides in MODES.items():
        cfg = copy.deepcopy(base_cfg)
        for key, value in overrides.items():
            set_nested(cfg, key, value)
        model = build_model(cfg, artifacts, device)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()
        models[mode] = model

    total = min(args.limit_users, len(dataset)) if args.limit_users else len(dataset)
    cases = []
    with torch.no_grad():
        for idx in tqdm(range(total), desc="counterfactual"):
            batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
            full_scores = models["full"](batch)["scores"][0]
            full_scores[0] = -1e9
            original_item = int(torch.argmax(full_scores).item())
            target_item = int(batch["target"].item())
            ranks = {}
            for mode, model in models.items():
                scores = model(batch)["scores"][0]
                scores[0] = -1e9
                ranks[mode] = {
                    "original_recommendation_rank": item_rank(scores, original_item),
                    "target_rank": item_rank(scores, target_item),
                    "top_item": item_card(meta, item_cards, int(torch.argmax(scores).item())),
                }
            cases.append(
                {
                    "index": idx,
                    "user_id": int(artifacts["examples"][args.split][idx]["user_id"]),
                    "original_recommendation": item_card(meta, item_cards, original_item),
                    "target": item_card(meta, item_cards, target_item),
                    "ranks": ranks,
                }
            )

    out = {"modes": list(MODES.keys()), "cases": cases}
    out_path = Path(base_cfg.train.save_dir) / f"counterfactual_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print(f"Saved counterfactual cases to {out_path}")


if __name__ == "__main__":
    main()
