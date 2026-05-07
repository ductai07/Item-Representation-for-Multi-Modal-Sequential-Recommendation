from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import load_artifacts, set_seed
from cstamoerec.graph import EdgeDataset, LightGCN, build_norm_adj, transition_edges_from_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGCN item embeddings from train transition graph.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(cfg.train.data_dir)
    meta = artifacts["meta"]
    edges = transition_edges_from_examples(artifacts["examples"]["train"])
    print(f"LightGCN transition edges: {len(edges)}")
    norm_adj = build_norm_adj(meta["num_items"], edges, device)
    model = LightGCN(meta["num_items"], cfg.model.graph_dim, cfg.model.graph_layers, norm_adj).to(device)
    dataset = EdgeDataset(edges, meta["num_items"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for src, pos, neg in tqdm(loader, desc=f"lightgcn-{epoch}", leave=False):
            src, pos, neg = src.to(device), pos.to(device), neg.to(device)
            loss = model.bpr_loss(src, pos, neg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * src.size(0)
            count += src.size(0)
        print(f"epoch={epoch} loss={total / max(count, 1):.4f}")
    with torch.no_grad():
        graph_embeddings = model.propagate().detach().cpu()
    features_path = Path(cfg.train.data_dir) / "features.pt"
    features = artifacts["features"]
    features["graph_embeddings"] = graph_embeddings
    torch.save(features, features_path)
    print(f"Saved graph_embeddings {tuple(graph_embeddings.shape)} to {features_path}")


if __name__ == "__main__":
    main()
