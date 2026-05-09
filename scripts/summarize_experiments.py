from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cstamoerec.config import load_config
from cstamoerec.data import load_artifacts


METRIC_KEYS = [
    "HR@5",
    "HR@10",
    "HR@20",
    "MRR@10",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "Recall@50",
    "Recall@100",
    "Recall@200",
    "Recall@500",
    "Recall@1000",
    "CandidatePoolHitRate",
    "AvgCandidatePoolSize",
    "Coverage@10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact experiment report from saved JSON outputs.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty_dense10k.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Skip {path}: {exc}")
        return None


def add_metric_rows(rows: list[dict[str, Any]], protocol: str, payload: dict[str, Any], source_path: Path) -> None:
    if "history" in payload and isinstance(payload.get("test"), dict):
        rows.append(make_row(protocol, "cstamoerec_full", payload["test"], source_path))
        return
    if any(key in payload for key in METRIC_KEYS):
        rows.append(make_row(protocol, source_path.stem, payload, source_path))
        return
    for name, metrics in payload.items():
        if isinstance(metrics, dict):
            rows.append(make_row(protocol, str(name), metrics, source_path))


def make_row(protocol: str, method: str, metrics: dict[str, Any], source_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol": protocol,
        "method": method,
        "source": source_path.name,
    }
    for key in METRIC_KEYS:
        value = metrics.get(key)
        row[key] = round(float(value), 6) if isinstance(value, (int, float)) else ""
    return row


def dataset_stats(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    artifacts = load_artifacts(cfg.train.data_dir)
    examples = artifacts["examples"]
    features = artifacts["features"]
    meta = artifacts["meta"]
    popularity = features["item_popularity"]
    num_items = int(meta["num_items"])
    train_examples = len(examples.get("train", []))
    valid_examples = len(examples.get("valid", []))
    test_examples = len(examples.get("test", []))
    nonzero_train_items = int((popularity > 0).sum().item())
    zero_train_items = int((popularity[1:] == 0).sum().item())
    return {
        "data_dir": cfg.train.data_dir,
        "num_users": meta.get("num_users"),
        "num_items": num_items,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "test_examples": test_examples,
        "nonzero_train_items": nonzero_train_items,
        "zero_train_items": zero_train_items,
        "train_examples_per_item": round(train_examples / max(num_items - 1, 1), 6),
    }


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = Path(args.results_dir or cfg.train.save_dir)
    output = Path(args.output or (results_dir / "experiment_report.md"))
    rows: list[dict[str, Any]] = []
    patterns = {
        "model_sampled_negative": ["history.json"],
        "traditional_sampled_negative": ["traditional_baselines_*.json"],
        "candidate_generation_full_pool": ["candidate_recall_*.json"],
        "two_stage_full_pool": ["two_stage_rerank_*.json"],
        "ablation": ["ablation/ablation_summary.json"],
    }
    for protocol, globs in patterns.items():
        for pattern in globs:
            for path in sorted(results_dir.glob(pattern)):
                payload = read_json(path)
                if isinstance(payload, dict):
                    add_metric_rows(rows, protocol, payload, path)

    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["protocol", "method", "source"] + METRIC_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = dataset_stats(args.config)
    metric_keys = ["protocol", "method", "HR@10", "NDCG@10", "HR@20", "CandidatePoolHitRate", "Recall@200", "source"]
    report = [
        "# CS-TAMoERec Experiment Report",
        "",
        "## Dataset",
        "",
        markdown_table([stats], list(stats.keys())),
        "",
        "## Metrics",
        "",
        markdown_table(rows, metric_keys),
        "",
        f"CSV: `{csv_path}`",
    ]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"Saved report to {output}")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
