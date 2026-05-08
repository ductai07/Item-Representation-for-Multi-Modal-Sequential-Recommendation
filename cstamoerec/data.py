from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

PAD = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def text_from_meta(meta: dict[str, Any]) -> str:
    parts = []
    title = meta.get("title")
    store = meta.get("store") or meta.get("brand")
    categories = meta.get("categories") or meta.get("main_category")
    description = meta.get("description")
    features = meta.get("features")
    details = meta.get("details")
    if title:
        parts.append(f"Title: {title}")
    if store:
        parts.append(f"Brand: {store}")
    if categories:
        if isinstance(categories, list):
            flat = []
            for item in categories:
                if isinstance(item, list):
                    flat.extend(str(x) for x in item)
                else:
                    flat.append(str(item))
            parts.append("Category: " + " > ".join(flat[:5]))
        else:
            parts.append(f"Category: {categories}")
    if features:
        if isinstance(features, list):
            parts.append("Features: " + "; ".join(str(x) for x in features[:8]))
        else:
            parts.append(f"Features: {features}")
    if details:
        try:
            parsed = json.loads(details) if isinstance(details, str) else details
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            useful = []
            for key in ("Brand", "Color", "Scent", "Skin Type", "Hair Type", "Item Form", "Item Volume", "Product Benefits"):
                value = parsed.get(key)
                if value:
                    useful.append(f"{key}: {value}")
            if useful:
                parts.append("Details: " + "; ".join(useful[:8]))
    if description:
        if isinstance(description, list):
            desc = " ".join(str(x) for x in description[:4])
        else:
            desc = str(description)
        parts.append("Description: " + desc[:800])
    return ". ".join(parts) or "Unknown product."


def first_image_url(meta: dict[str, Any]) -> str | None:
    images = meta.get("images")
    if not images:
        return None
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            values = images.get(key)
            if isinstance(values, list) and values:
                return values[0]
            if isinstance(values, str):
                return values
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict):
                for key in ("large", "hi_res", "thumb"):
                    if image.get(key):
                        return image[key]
            elif isinstance(image, str):
                return image
    return None


def main_category(meta: dict[str, Any]) -> str:
    if meta.get("main_category"):
        return str(meta["main_category"])
    categories = meta.get("categories")
    if isinstance(categories, list) and categories:
        first = categories[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return "Unknown"


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_examples(sequences: dict[int, list[tuple[int, int]]], max_seq_len: int) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "valid": [], "test": []}
    for user_id, events in sequences.items():
        events = sorted(events, key=lambda x: x[1])
        if len(events) < 5:
            continue
        item_ids = [x[0] for x in events]
        times = [x[1] for x in events]
        for idx in range(1, len(item_ids) - 2):
            splits["train"].append(_make_example(user_id, item_ids, times, idx, max_seq_len))
        splits["valid"].append(_make_example(user_id, item_ids, times, len(item_ids) - 2, max_seq_len))
        splits["test"].append(_make_example(user_id, item_ids, times, len(item_ids) - 1, max_seq_len))
    return splits


def _make_example(user_id: int, item_ids: list[int], times: list[int], target_idx: int, max_seq_len: int) -> dict[str, Any]:
    start = max(0, target_idx - max_seq_len)
    seq = item_ids[start:target_idx]
    seq_times = times[start:target_idx]
    return {
        "user_id": user_id,
        "seq": seq,
        "times": seq_times,
        "target": item_ids[target_idx],
        "target_time": times[target_idx],
    }


class SequenceDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        max_seq_len: int,
        item_popularity: torch.Tensor,
        item_categories: torch.Tensor,
        cold_threshold: int,
    ) -> None:
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.item_popularity = item_popularity
        self.item_categories = item_categories
        self.cold_threshold = cold_threshold

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        seq = ex["seq"][-self.max_seq_len :]
        times = ex["times"][-self.max_seq_len :]
        length = len(seq)
        pad_len = self.max_seq_len - length
        seq_tensor = torch.tensor([PAD] * pad_len + seq, dtype=torch.long)
        time_tensor = torch.tensor([0] * pad_len + times, dtype=torch.long)
        target = int(ex["target"])
        target_time = int(ex["target_time"])
        popularity = self.item_popularity[seq_tensor].float()
        target_popularity = self.item_popularity[target].float()
        cold_flags = (popularity <= self.cold_threshold).float()
        target_cold = (target_popularity <= self.cold_threshold).float()
        return {
            "seq": seq_tensor,
            "times": time_tensor,
            "length": torch.tensor(length, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "target_time": torch.tensor(target_time, dtype=torch.long),
            "popularity": popularity,
            "cold_flags": cold_flags,
            "target_category": self.item_categories[target].long(),
            "target_popularity": target_popularity,
            "target_cold": target_cold,
        }


def load_artifacts(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    examples = torch.load(data_dir / "examples.pt", map_location="cpu", weights_only=False)
    features = torch.load(data_dir / "features.pt", map_location="cpu", weights_only=False)
    meta = load_json(data_dir / "meta.json")
    return {"examples": examples, "features": features, "meta": meta}


def dataset_from_artifacts(data_dir: str | Path, split: str) -> SequenceDataset:
    artifacts = load_artifacts(data_dir)
    features = artifacts["features"]
    return SequenceDataset(
        artifacts["examples"][split],
        artifacts["meta"]["max_seq_len"],
        features["item_popularity"],
        features["item_categories"],
        artifacts["meta"]["cold_threshold"],
    )


def write_artifacts(
    output_dir: str | Path,
    examples: dict[str, list[dict[str, Any]]],
    features: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(examples, output_dir / "examples.pt")
    torch.save(features, output_dir / "features.pt")
    save_json(output_dir / "meta.json", meta)


def log_time_interval_days(times: torch.Tensor) -> torch.Tensor:
    delta = torch.diff(times.float(), dim=1, prepend=times[:, :1].float())
    delta = torch.clamp(delta / 86400000.0, min=0.0)
    return torch.log1p(delta)


def unix_month_weekday(times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Timestamp in Amazon Reviews 2023 is milliseconds. This approximation avoids
    # slow Python datetime calls in the training loop.
    days = torch.div(times, 86400000, rounding_mode="floor")
    weekday = torch.remainder(days + 3, 7)
    month = torch.remainder(torch.div(days, 30, rounding_mode="floor"), 12)
    return month.long(), weekday.long()


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)
