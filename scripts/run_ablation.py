from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from cstamoerec.config import load_config
from cstamoerec.train import run_training_experiment


VARIANTS = {
    "full": {},
    "id_only": {
        "model.use_text": False,
        "model.use_image": False,
        "model.use_time": False,
        "model.use_cold": False,
        "model.use_cross": False,
        "train.category_loss_weight": 0.0,
        "train.alignment_loss_weight": 0.0,
        "train.router_balance_loss_weight": 0.0,
    },
    "no_text": {"model.use_text": False},
    "no_image": {"model.use_image": False},
    "no_time": {"model.use_time": False},
    "no_cold_router": {"model.use_cold": False},
    "no_cross": {"model.use_cross": False},
    "no_graph": {"model.use_graph": False},
    "no_aux_loss": {
        "train.category_loss_weight": 0.0,
        "train.alignment_loss_weight": 0.0,
        "train.router_balance_loss_weight": 0.0,
    },
    "no_router_balance": {"train.router_balance_loss_weight": 0.0},
}


def set_nested(obj, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    current = obj
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CS-TAMoERec ablation experiments.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "id_only", "no_text", "no_image", "no_time", "no_cold_router", "no_graph"],
    )
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs for quick ablations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    device = args.device or (base_cfg.train.device if torch.cuda.is_available() else "cpu")
    summary = {}
    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Available: {sorted(VARIANTS)}")
        cfg = copy.deepcopy(base_cfg)
        if args.epochs:
            cfg.train.epochs = args.epochs
        for key, value in VARIANTS[variant].items():
            set_nested(cfg, key, value)
        cfg.train.save_dir = str(Path(base_cfg.train.save_dir) / "ablation")
        print(f"\n===== Running ablation: {variant} =====")
        result = run_training_experiment(cfg, device, run_name=variant)
        summary[variant] = result["test"]

    out_dir = Path(base_cfg.train.save_dir) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nAblation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
