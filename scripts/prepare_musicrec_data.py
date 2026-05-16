from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.data import save_json, set_seed, write_artifacts
from cstamoerec.features import normalize_feature_matrix, zero_image_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MuSICRec-format data into cstamoerec artifacts.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_diff_split.inter and text/image features.")
    parser.add_argument("--dataset", required=True, help="Dataset prefix, e.g. baby, sports, elec.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--cold-threshold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--image-dim-if-missing", type=int, default=512)
    return parser.parse_args()


def _load_feature(path: Path, expected_rows: int, pad: bool = True) -> torch.Tensor:
    array = np.load(path).astype("float32")
    if array.shape[0] < expected_rows:
        raise ValueError(f"{path} has {array.shape[0]} rows, expected at least {expected_rows}.")
    array = array[:expected_rows]
    tensor = torch.from_numpy(array).float()
    if pad:
        tensor = torch.cat([torch.zeros(1, tensor.size(1)), tensor], dim=0)
    return normalize_feature_matrix(tensor)


def _make_example(user_id: int, history: list[tuple[int, int]], target: int, target_time: int, max_seq_len: int) -> dict:
    seq = [item for item, _ in history][-max_seq_len:]
    times = [ts for _, ts in history][-max_seq_len:]
    return {
        "user_id": int(user_id),
        "seq": seq,
        "times": times,
        "target": int(target),
        "target_time": int(target_time),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    inter_path = input_dir / f"{args.dataset}_diff_split.inter"
    if not inter_path.exists():
        raise FileNotFoundError(inter_path)

    df = pd.read_csv(inter_path, sep="\t")
    required = {"userID", "itemID", "timestamp", "x_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {inter_path}: {sorted(missing)}")
    df = df.reset_index(names="_row")
    df["user_id"] = df["userID"].astype(int) + 1
    df["item_id"] = df["itemID"].astype(int) + 1
    df["timestamp_ms"] = df["timestamp"].astype(int) * 1000
    num_users = int(df["user_id"].max()) + 1
    num_items = int(df["item_id"].max()) + 1

    examples: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    train_popularity = torch.zeros(num_items, dtype=torch.long)
    user_groups = df.sort_values(["user_id", "timestamp_ms", "_row"]).groupby("user_id", sort=True)
    for user_id, rows in user_groups:
        history: list[tuple[int, int]] = []
        train_history: list[tuple[int, int]] = []
        for row in rows.itertuples(index=False):
            item = int(row.item_id)
            ts = int(row.timestamp_ms)
            label = int(row.x_label)
            if label == 0:
                if train_history:
                    examples["train"].append(_make_example(int(user_id), train_history, item, ts, args.max_seq_len))
                train_history.append((item, ts))
                history.append((item, ts))
                train_popularity[item] += 1
            elif label == 1:
                if history:
                    examples["valid"].append(_make_example(int(user_id), history, item, ts, args.max_seq_len))
                history.append((item, ts))
            elif label == 2:
                if history:
                    examples["test"].append(_make_example(int(user_id), history, item, ts, args.max_seq_len))
            else:
                raise ValueError(f"Unknown x_label={label}")

    text_path = input_dir / "text_feat.npy"
    if not text_path.exists():
        raise FileNotFoundError(text_path)
    text_embeddings = _load_feature(text_path, expected_rows=num_items - 1, pad=True)

    image_path = input_dir / "image_feat.npy"
    if image_path.exists():
        image_embeddings = _load_feature(image_path, expected_rows=num_items - 1, pad=True)
        image_mask = torch.ones(num_items, dtype=torch.float)
        image_mask[0] = 0.0
    else:
        image_embeddings, image_mask = zero_image_features(num_items, args.image_dim_if_missing)

    item_categories = torch.zeros(num_items, dtype=torch.long)
    features = {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "image_mask": image_mask,
        "item_popularity": train_popularity,
        "item_categories": item_categories,
    }
    item2id = {"[PAD]": 0, **{str(i): i + 1 for i in range(num_items - 1)}}
    user2id = {str(i): i + 1 for i in range(num_users - 1)}
    id2item = ["[PAD]"] + [str(i) for i in range(num_items - 1)]
    item_titles = ["[PAD]"] + [f"{args.dataset}_item_{i}" for i in range(num_items - 1)]
    meta = {
        "dataset": f"musicrec_{args.dataset}",
        "source": "MuSICRec local split",
        "max_seq_len": args.max_seq_len,
        "cold_threshold": args.cold_threshold,
        "min_item_interactions": None,
        "cold_item_keep_ratio": None,
        "popularity_source": "musicrec_train_split_only",
        "num_items": num_items,
        "num_users": num_users,
        "num_categories": 1,
        "text_dim": int(text_embeddings.shape[1]),
        "image_dim": int(image_embeddings.shape[1]),
        "item2id": item2id,
        "user2id": user2id,
        "id2item": id2item,
        "category_names": {"Unknown": 0},
        "item_titles": item_titles,
        "musicrec_item_id_offset": 1,
        "musicrec_user_id_offset": 1,
    }

    output_dir = Path(args.output_dir)
    write_artifacts(output_dir, examples, features, meta)
    item_cards = {
        str(i + 1): {
            "title": item_titles[i + 1],
            "category": "Unknown",
            "image_url": None,
            "text": f"MuSICRec {args.dataset} item {i}",
        }
        for i in range(num_items - 1)
    }
    save_json(output_dir / "item_cards.json", item_cards)
    stats = {
        "train": len(examples["train"]),
        "valid": len(examples["valid"]),
        "test": len(examples["test"]),
        "num_users": num_users,
        "num_items": num_items,
        "text_dim": int(text_embeddings.shape[1]),
        "image_dim": int(image_embeddings.shape[1]),
        "has_image_feat": image_path.exists(),
    }
    with open(output_dir / "musicrec_import_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"Saved MuSICRec artifacts to {output_dir}")


if __name__ == "__main__":
    main()
