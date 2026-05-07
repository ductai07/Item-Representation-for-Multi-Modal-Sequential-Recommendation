from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import (
    build_examples,
    first_image_url,
    main_category,
    save_json,
    set_seed,
    text_from_meta,
    write_artifacts,
)
from cstamoerec.features import encode_images, encode_texts, normalize_feature_matrix, zero_image_features


def load_amazon2023_config(dataset_name: str, config_name: str):
    """Load Amazon Reviews 2023 config without relying on deprecated HF scripts."""
    from datasets import load_dataset

    parquet_glob = f"hf://datasets/{dataset_name}/{config_name}/*.parquet"
    try:
        return load_dataset("parquet", data_files={"full": parquet_glob}, split="full")
    except Exception as parquet_error:
        print(f"Parquet loading failed for {config_name}: {parquet_error}")
        print("Falling back to legacy dataset loading. This may fail on new datasets versions.")
        return load_dataset(dataset_name, config_name, split="full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 data for CS-TAMoERec.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--limit-reviews", type=int, default=0, help="Debug only: keep the first N reviews.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading reviews: {cfg.data.review_config}")
    reviews = load_amazon2023_config(cfg.data.dataset_name, cfg.data.review_config)
    if args.limit_reviews:
        reviews = reviews.select(range(min(args.limit_reviews, len(reviews))))

    raw_events: list[tuple[str, str, int]] = []
    user_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    for row in tqdm(reviews, desc="Scan reviews"):
        user = row.get("user_id")
        item = row.get("parent_asin") or row.get("asin")
        timestamp = row.get("timestamp")
        if user is None or item is None or timestamp is None:
            continue
        raw_events.append((str(user), str(item), int(timestamp)))
        user_counter[str(user)] += 1
        item_counter[str(item)] += 1

    active_users = {u for u, count in user_counter.items() if count >= cfg.data.min_user_interactions}
    eligible_items = {item for item, count in item_counter.items() if count >= cfg.data.min_item_interactions}
    user_events = defaultdict(list)
    for user, item, timestamp in raw_events:
        if user in active_users and item in eligible_items:
            user_events[user].append((item, timestamp))

    valid_test_targets = set()
    for user, events in user_events.items():
        events = sorted(events, key=lambda x: x[1])
        if len(events) >= cfg.data.min_user_interactions:
            valid_test_targets.add(events[-1][0])
            valid_test_targets.add(events[-2][0])

    top_budget = max(cfg.data.max_items - len(valid_test_targets), 0)
    cold_budget = int(cfg.data.max_items * cfg.data.cold_item_keep_ratio)
    popular_items = [item for item, _ in item_counter.most_common(top_budget)]
    cold_items = [
        item
        for item, count in sorted(item_counter.items(), key=lambda x: (x[1], x[0]))
        if item in eligible_items and item not in valid_test_targets
    ][:cold_budget]
    selected_items = []
    seen_items = set()
    for item in list(valid_test_targets) + popular_items + cold_items:
        if item in eligible_items and item not in seen_items:
            selected_items.append(item)
            seen_items.add(item)
        if len(selected_items) >= cfg.data.max_items:
            break
    top_items = selected_items
    item2id = {"[PAD]": 0, **{item: idx + 1 for idx, item in enumerate(top_items)}}
    user2id = {user: idx + 1 for idx, user in enumerate(sorted(active_users))}

    sequences: dict[int, list[tuple[int, int]]] = defaultdict(list)
    train_popularity = torch.zeros(len(item2id), dtype=torch.long)
    for user, item, timestamp in tqdm(raw_events, desc="Build sequences"):
        if user not in user2id or item not in item2id:
            continue
        uid = user2id[user]
        iid = item2id[item]
        sequences[uid].append((iid, timestamp))

    examples = build_examples(sequences, cfg.data.max_seq_len)
    for ex in examples["train"]:
        train_popularity[int(ex["target"])] += 1
        for item in ex["seq"]:
            train_popularity[int(item)] += 1
    used_items = {item for split in examples.values() for ex in split for item in ex["seq"] + [ex["target"]]}
    print(
        f"Prepared examples train={len(examples['train'])}, valid={len(examples['valid'])}, "
        f"test={len(examples['test'])}, users={len(sequences)}, items={len(item2id) - 1}, used_items={len(used_items)}"
    )

    print(f"Loading metadata: {cfg.data.meta_config}")
    meta_ds = load_amazon2023_config(cfg.data.dataset_name, cfg.data.meta_config)
    item_texts = ["[PAD]"]
    item_image_urls: list[str | None] = [None]
    category_names = {"Unknown": 0}
    item_categories = torch.zeros(len(item2id), dtype=torch.long)
    item_titles = ["[PAD]"] * len(item2id)
    item_raw = {}

    raw_meta_by_item = {}
    for row in tqdm(meta_ds, desc="Index metadata"):
        item = row.get("parent_asin") or row.get("asin")
        if item in item2id and item not in raw_meta_by_item:
            raw_meta_by_item[item] = dict(row)

    id2item = [None] * len(item2id)
    for item, idx in item2id.items():
        id2item[idx] = item
    for idx in range(1, len(id2item)):
        asin = id2item[idx]
        meta = raw_meta_by_item.get(asin, {})
        text = text_from_meta(meta)
        cat = main_category(meta)
        if cat not in category_names:
            category_names[cat] = len(category_names)
        item_categories[idx] = category_names[cat]
        item_texts.append(text)
        item_image_urls.append(first_image_url(meta))
        item_titles[idx] = str(meta.get("title") or asin)
        item_raw[asin] = {
            "title": item_titles[idx],
            "category": cat,
            "image_url": item_image_urls[idx],
            "text": text[:1000],
        }

    print("Encoding text features")
    text_embeddings = encode_texts(item_texts, cfg.data.text_model, args.text_batch_size, device=device)
    text_embeddings[0].zero_()
    text_embeddings = normalize_feature_matrix(text_embeddings)

    if cfg.data.use_images and not args.skip_images:
        print("Encoding image features. Missing/failed images receive zero vectors.")
        image_embeddings, image_mask = encode_images(
            item_image_urls,
            cfg.data.image_model,
            cfg.data.max_image_items,
            cfg.data.image_timeout,
            device=device,
        )
        image_embeddings = normalize_feature_matrix(image_embeddings)
    else:
        image_embeddings, image_mask = zero_image_features(len(item2id), cfg.model.image_dim)

    features = {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "image_mask": image_mask,
        "item_popularity": train_popularity,
        "item_categories": item_categories,
    }
    meta = {
        "dataset": cfg.data.review_config,
        "max_seq_len": cfg.data.max_seq_len,
        "cold_threshold": cfg.data.cold_threshold,
        "min_item_interactions": cfg.data.min_item_interactions,
        "cold_item_keep_ratio": cfg.data.cold_item_keep_ratio,
        "popularity_source": "train_split_only",
        "num_items": len(item2id),
        "num_users": len(user2id),
        "num_categories": len(category_names),
        "text_dim": int(text_embeddings.shape[1]),
        "image_dim": int(image_embeddings.shape[1]),
        "item2id": item2id,
        "user2id": user2id,
        "id2item": id2item,
        "category_names": category_names,
        "item_titles": item_titles,
    }

    output_dir = Path(cfg.data.output_dir)
    write_artifacts(output_dir, examples, features, meta)
    save_json(output_dir / "item_cards.json", item_raw)
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
