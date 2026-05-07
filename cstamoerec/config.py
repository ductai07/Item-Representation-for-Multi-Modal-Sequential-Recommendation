from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    review_config: str = "raw_review_All_Beauty"
    meta_config: str = "raw_meta_All_Beauty"
    output_dir: str = "data/processed/all_beauty"
    max_items: int = 30000
    min_user_interactions: int = 5
    min_item_interactions: int = 2
    cold_item_keep_ratio: float = 0.15
    max_seq_len: int = 50
    cold_threshold: int = 5
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model: str = "openai/clip-vit-base-patch32"
    use_images: bool = True
    max_image_items: int = 8000
    image_timeout: int = 8
    seed: int = 2025


@dataclass
class ModelConfig:
    hidden_size: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    max_seq_len: int = 50
    num_experts: int = 6
    text_dim: int = 384
    image_dim: int = 512
    time_dim: int = 16
    router_hidden: int = 256
    graph_dim: int = 64
    graph_layers: int = 2
    use_text: bool = True
    use_image: bool = True
    use_time: bool = True
    use_cold: bool = True
    use_cross: bool = True
    use_graph: bool = True
    use_id_graph: bool = True
    use_text_graph: bool = True
    use_image_graph: bool = True


@dataclass
class TrainConfig:
    data_dir: str = "data/processed/all_beauty"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 2
    eval_topk: list[int] = field(default_factory=lambda: [5, 10, 20])
    category_loss_weight: float = 0.05
    alignment_loss_weight: float = 0.05
    router_balance_loss_weight: float = 0.01
    alignment_temperature: float = 0.2
    num_eval_negatives: int = 999
    device: str = "cuda"
    save_dir: str = "checkpoints/cstamoerec"
    seed: int = 2025


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> Config:
    cfg = Config()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _merge_dataclass(cfg, raw)
