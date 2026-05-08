from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import first_image_url, load_artifacts, main_category, save_json, text_from_meta
from cstamoerec.features import encode_texts, normalize_feature_matrix
from scripts.prepare_amazon2023 import category_from_config, load_amazon2023_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh text/category artifacts without re-encoding images.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--data-dir", default=None, help="Artifact directory to refresh. Defaults to data.output_dir.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--keep-stale-text-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir or cfg.data.output_dir)
    if str(cfg.train.data_dir) != str(cfg.data.output_dir) and args.data_dir is None:
        print(
            "Warning: train.data_dir and data.output_dir differ. "
            f"Refreshing data.output_dir={cfg.data.output_dir}. "
            "Pass --data-dir to override."
        )
    artifacts = load_artifacts(data_dir)
    features = artifacts["features"]
    meta = artifacts["meta"]
    id2item = meta["id2item"]

    print(f"Loading metadata: {cfg.data.meta_config}")
    meta_ds = load_amazon2023_config(cfg.data.dataset_name, cfg.data.meta_config)
    needed_items = set(str(x) for x in id2item[1:])
    raw_meta_by_item = {}
    for row in tqdm(meta_ds, desc="Index metadata"):
        item = row.get("parent_asin") or row.get("asin")
        if item in needed_items and item not in raw_meta_by_item:
            raw_meta_by_item[item] = dict(row)

    item_texts = ["[PAD]"]
    category_names = {"Unknown": 0}
    item_categories = torch.zeros(len(id2item), dtype=torch.long)
    item_titles = ["[PAD]"] * len(id2item)
    item_cards = {}

    for idx in range(1, len(id2item)):
        asin = id2item[idx]
        row = raw_meta_by_item.get(asin, {})
        cat = main_category(row)
        if cat == "Unknown":
            cat = category_from_config(cfg.data.meta_config)
            row = {**row, "main_category": cat}
        text = text_from_meta(row)
        if cat not in category_names:
            category_names[cat] = len(category_names)
        item_categories[idx] = category_names[cat]
        item_texts.append(text)
        item_titles[idx] = str(row.get("title") or asin)
        item_cards[asin] = {
            "title": item_titles[idx],
            "category": cat,
            "image_url": first_image_url(row),
            "text": text[:1000],
        }

    print("Encoding refreshed text features")
    text_embeddings = encode_texts(item_texts, cfg.data.text_model, args.text_batch_size, device=device)
    text_embeddings[0].zero_()
    features["text_embeddings"] = normalize_feature_matrix(text_embeddings)
    features["item_categories"] = item_categories

    if not args.keep_stale_text_graph:
        for key in ("text_graph_embeddings", "text_graph_edges"):
            features.pop(key, None)
        print("Dropped stale text graph artifacts. Re-run train_lightgcn.py before main training.")

    meta["num_categories"] = len(category_names)
    meta["category_names"] = category_names
    meta["item_titles"] = item_titles

    torch.save(features, data_dir / "features.pt")
    save_json(data_dir / "meta.json", meta)
    save_json(data_dir / "item_cards.json", item_cards)
    print(f"Refreshed text/category artifacts in {data_dir}; image embeddings were preserved.")


if __name__ == "__main__":
    main()
