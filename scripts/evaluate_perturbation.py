from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from cstamoerec.config import load_config
from cstamoerec.train import build_model, evaluate, make_loaders


MODES = ["full", "mask_text", "mask_image", "shuffle_text", "shuffle_image", "mask_text_image"]


def perturb_features(artifacts: dict, mode: str, seed: int) -> dict:
    perturbed = copy.deepcopy(artifacts)
    features = perturbed["features"]
    generator = torch.Generator().manual_seed(seed)
    if mode == "full":
        return perturbed
    if mode in {"mask_text", "mask_text_image"}:
        features["text_embeddings"] = torch.zeros_like(features["text_embeddings"])
    if mode in {"mask_image", "mask_text_image"}:
        features["image_embeddings"] = torch.zeros_like(features["image_embeddings"])
        features["image_mask"] = torch.zeros_like(features["image_mask"])
    if mode == "shuffle_text":
        perm = torch.randperm(features["text_embeddings"].size(0) - 1, generator=generator) + 1
        shuffled = features["text_embeddings"].clone()
        shuffled[1:] = features["text_embeddings"][perm]
        features["text_embeddings"] = shuffled
    if mode == "shuffle_image":
        perm = torch.randperm(features["image_embeddings"].size(0) - 1, generator=generator) + 1
        shuffled = features["image_embeddings"].clone()
        shuffled_mask = features["image_mask"].clone()
        shuffled[1:] = features["image_embeddings"][perm]
        shuffled_mask[1:] = features["image_mask"][perm]
        features["image_embeddings"] = shuffled
        features["image_mask"] = shuffled_mask
    return perturbed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text/image perturbations for CS-TAMoERec.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec_dense10k/best_cstamoerec.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    base_artifacts, loaders = make_loaders(cfg.train.data_dir, cfg.train.batch_size, cfg.train.num_workers)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    summary = {}
    for mode in args.modes:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}. Available: {MODES}")
        artifacts = perturb_features(base_artifacts, mode, args.seed)
        model = build_model(cfg, artifacts, device)
        model.load_state_dict(checkpoint["model"], strict=False)
        metrics = evaluate(model, loaders[args.split], cfg, device, artifacts["features"]["item_popularity"], args.split)
        summary[mode] = metrics
        print(f"{mode}: {metrics}")

    out_path = Path(cfg.train.save_dir) / f"perturbation_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved perturbation summary to {out_path}")


if __name__ == "__main__":
    main()
