from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cstamoerec.config import load_config
from cstamoerec.data import load_artifacts, set_seed
from cstamoerec.graph import (
    EdgeDataset,
    LightGCN,
    build_norm_adj,
    similarity_edges_from_features,
    transition_edges_from_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train modal-aware LightGCN item embeddings from ID/Text/Image graphs.")
    parser.add_argument("--config", default="config/cstamoerec_all_beauty.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--similarity-topk", type=int, default=50)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def train_graph_embeddings(
    name: str,
    edges: list[tuple[int, int]],
    num_items: int,
    graph_dim: int,
    graph_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> torch.Tensor:
    print(f"LightGCN {name} edges: {len(edges)}")
    if not edges:
        return torch.zeros((num_items, graph_dim), dtype=torch.float)
    norm_adj = build_norm_adj(num_items, edges, device)
    model = LightGCN(num_items, graph_dim, graph_layers, norm_adj).to(device)
    dataset = EdgeDataset(edges, num_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for src, pos, neg in tqdm(loader, desc=f"{name}-lightgcn-{epoch}", leave=False):
            src, pos, neg = src.to(device), pos.to(device), neg.to(device)
            loss = model.bpr_loss(src, pos, neg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * src.size(0)
            count += src.size(0)
        print(f"{name} epoch={epoch} loss={total / max(count, 1):.4f}")
    with torch.no_grad():
        return model.propagate().detach().cpu()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = args.device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    artifacts = load_artifacts(cfg.train.data_dir)
    meta = artifacts["meta"]
    features = artifacts["features"]
    id_edges = transition_edges_from_examples(artifacts["examples"]["train"])
    print("Building text similarity graph...")
    text_edges = similarity_edges_from_features(
        features["text_embeddings"],
        topk=args.similarity_topk,
        batch_size=args.similarity_batch_size,
        device=device,
    )
    print("Building image similarity graph...")
    image_edges = similarity_edges_from_features(
        features["image_embeddings"],
        topk=args.similarity_topk,
        batch_size=args.similarity_batch_size,
        device=device,
        valid_mask=features.get("image_mask"),
    )
    id_graph_embeddings = train_graph_embeddings(
        "id_transition",
        id_edges,
        meta["num_items"],
        cfg.model.graph_dim,
        cfg.model.graph_layers,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
    )
    text_graph_embeddings = train_graph_embeddings(
        "text_similarity",
        text_edges,
        meta["num_items"],
        cfg.model.graph_dim,
        cfg.model.graph_layers,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
    )
    image_graph_embeddings = train_graph_embeddings(
        "image_similarity",
        image_edges,
        meta["num_items"],
        cfg.model.graph_dim,
        cfg.model.graph_layers,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
    )
    features_path = Path(cfg.train.data_dir) / "features.pt"
    features["id_graph_embeddings"] = id_graph_embeddings
    features["text_graph_embeddings"] = text_graph_embeddings
    features["image_graph_embeddings"] = image_graph_embeddings
    features["graph_embeddings"] = id_graph_embeddings
    features["text_graph_edges"] = text_edges
    features["image_graph_edges"] = image_edges
    torch.save(features, features_path)
    print(
        "Saved modal graph embeddings "
        f"id={tuple(id_graph_embeddings.shape)} "
        f"text={tuple(text_graph_embeddings.shape)} "
        f"image={tuple(image_graph_embeddings.shape)} to {features_path}"
    )


if __name__ == "__main__":
    main()
