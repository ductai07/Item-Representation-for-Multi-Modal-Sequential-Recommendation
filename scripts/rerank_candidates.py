from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from cstamoerec.candidate import CandidateGenerator, modal_graph_scores_for_items
from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts
from cstamoerec.metrics import MetricAverager, topk_metrics
from cstamoerec.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-stage candidate generation + MoE reranking.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cstamoerec/best_cstamoerec.pt")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--per-source-k", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--device", default=None)
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
    model = build_model(cfg, artifacts, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = CandidateGenerator(artifacts, per_source_k=args.per_source_k, max_candidates=args.max_candidates)
    avg = MetricAverager()
    total = len(dataset) if not args.limit_users else min(args.limit_users, len(dataset))
    for idx in tqdm(range(total), desc="two-stage-eval"):
        raw_ex = artifacts["examples"][args.split][idx]
        seq = [int(x) for x in raw_ex["seq"] if int(x) > 0]
        candidates = generator.generate(seq)
        if int(raw_ex["target"]) not in candidates.item_ids:
            candidates.item_ids.append(int(raw_ex["target"]))
        graph_scores = modal_graph_scores_for_items(
            seq,
            candidates.item_ids,
            generator.transition_graph,
            generator.text_graph,
            generator.image_graph,
        )
        batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[idx].items()}
        candidate_tensor = torch.tensor(candidates.item_ids, dtype=torch.long, device=device)
        graph_tensor = torch.tensor(graph_scores, dtype=torch.float, device=device)
        with torch.no_grad():
            scores = model.score_candidates(batch, candidate_tensor, graph_scores=graph_tensor)["scores"].detach().cpu()
        target_pos = torch.tensor([candidates.item_ids.index(int(raw_ex["target"]))])
        avg.update(topk_metrics(scores, target_pos, cfg.train.eval_topk), 1)
    summary = avg.compute()
    out_path = Path(cfg.train.save_dir) / f"two_stage_rerank_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved two-stage metrics to {out_path}")


if __name__ == "__main__":
    main()
