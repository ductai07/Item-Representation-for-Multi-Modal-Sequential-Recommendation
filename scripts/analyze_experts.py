from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts
from cstamoerec.train import build_model, move_batch


EXPERTS = ["ID", "Text", "Image", "Time", "Cross", "Graph"]


class WeightBucket:
    def __init__(self) -> None:
        self.sum = torch.zeros(len(EXPERTS), dtype=torch.float)
        self.count = 0

    def update(self, weights: torch.Tensor) -> None:
        if weights.numel() == 0:
            return
        weights = torch.nan_to_num(weights.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if weights.size(-1) < self.sum.size(0):
            pad = self.sum.size(0) - weights.size(-1)
            weights = torch.nn.functional.pad(weights, (0, pad))
        elif weights.size(-1) > self.sum.size(0):
            weights = weights[..., : self.sum.size(0)]
        self.sum += weights.sum(dim=0)
        self.count += weights.size(0)

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {name: 0.0 for name in EXPERTS}
        values = self.sum / self.count
        return {name: round(float(values[idx]), 6) for idx, name in enumerate(EXPERTS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CS-TAMoERec expert weights.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec/best_cstamoerec.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


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
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    model = build_model(cfg, artifacts, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    buckets = defaultdict(WeightBucket)
    item_pop = features["item_popularity"].to(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="expert-analysis")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            batch = move_batch(batch, device)
            target_ids = batch["target"].view(-1, 1)
            adaptive = model.score_candidates(batch, target_ids)
            target_weights = adaptive["expert_weights"][:, 0, :]

            target_pop = item_pop[batch["target"]]
            cold_mask = target_pop <= meta["cold_threshold"]
            warm_mask = ~cold_mask
            if cold_mask.any():
                buckets["cold_items"].update(target_weights[cold_mask])
            if warm_mask.any():
                buckets["warm_items"].update(target_weights[warm_mask])

            lengths = batch["length"].clamp_min(1)
            last_pos = lengths - 1
            last_time = batch["times"][torch.arange(batch["times"].size(0), device=device), last_pos]
            gap_days = (batch["target_time"] - last_time).float().clamp_min(0) / 86400000.0
            long_gap_mask = gap_days >= 30
            short_gap_mask = gap_days < 7
            if long_gap_mask.any():
                buckets["long_time_gap"].update(target_weights[long_gap_mask])
            if short_gap_mask.any():
                buckets["short_time_gap"].update(target_weights[short_gap_mask])
            buckets["all"].update(target_weights)

    summary = {name: bucket.mean() for name, bucket in buckets.items()}
    out_path = Path(cfg.train.save_dir) / f"expert_weights_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved expert-weight analysis to {out_path}")


if __name__ == "__main__":
    main()
