from __future__ import annotations

import json
from pathlib import Path

import torch
import streamlit as st

from cstamoerec.config import load_config
from cstamoerec.data import SequenceDataset, load_artifacts, load_json
from cstamoerec.train import build_model, move_batch


EXPERTS = ["ID", "Text", "Image", "Time", "Cross", "Graph"]


def expert_names(weights: torch.Tensor) -> list[str]:
    return EXPERTS[: int(weights.numel())]


@st.cache_data
def load_demo_cache(path: str) -> dict | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_json_if_exists(path: str) -> dict | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _protocol_from_filename(name: str) -> str:
    if name.startswith("traditional_baselines"):
        return "sampled-negative"
    if name.startswith("candidate_recall"):
        return "candidate-pool"
    if name.startswith("two_stage"):
        return "two-stage"
    if name.startswith("perturbation"):
        return "perturbation"
    return "metrics"


def _metric_row(protocol: str, method: str, metrics: dict, source: str) -> dict:
    keys = ["HR@5", "HR@10", "HR@20", "MRR@10", "NDCG@10", "Recall@200", "CandidatePoolHitRate"]
    row = {"protocol": protocol, "method": method, "source": source}
    for key in keys:
        value = metrics.get(key)
        row[key] = round(float(value), 4) if isinstance(value, (int, float)) else None
    return row


@st.cache_data
def load_metric_rows(results_dir: str) -> list[dict]:
    root = Path(results_dir)
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        payload = load_json_if_exists(str(path))
        if not isinstance(payload, dict):
            continue
        if path.name == "history.json" and isinstance(payload.get("test"), dict):
            rows.append(_metric_row("model", "cstamoerec_full", payload["test"], path.name))
            continue
        if path.name == "learned_source_ranker.json":
            continue
        protocol = _protocol_from_filename(path.name)
        for method, metrics in payload.items():
            if isinstance(metrics, dict):
                rows.append(_metric_row(protocol, method, metrics, path.name))
    return rows


@st.cache_resource
def load_runtime(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    artifacts = load_artifacts(cfg.train.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, artifacts, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    item_cards = load_json(Path(cfg.train.data_dir) / "item_cards.json")
    return cfg, artifacts, model, item_cards, device


def item_title(meta: dict, item_id: int) -> str:
    titles = meta.get("item_titles", [])
    if item_id < len(titles):
        return titles[item_id]
    return str(item_id)


def render_cache_demo(cache: dict, metric_rows: list[dict] | None = None, counterfactual: dict | None = None) -> None:
    users = cache.get("users", [])
    if not users:
        st.warning("Demo cache is empty.")
        return
    user_idx = st.sidebar.slider("Demo user", 0, len(users) - 1, 0)
    case = users[user_idx]
    tab_rec, tab_exp, tab_cmp, tab_cf = st.tabs(["Recommendation", "Explainability", "Model Comparison", "Counterfactual"])

    with tab_rec:
        left, right = st.columns([1, 1])
        with left:
            st.subheader(f"User {case['user_id']} History")
            st.dataframe(
                [{"product": h["title"], "category": h["category"], "popularity": h["popularity"]} for h in case["history"]],
                use_container_width=True,
            )
        with right:
            st.subheader("Two-stage Top Recommendations")
            target = case.get("target", {})
            target_rank = case.get("target_rank")
            if target:
                st.info(
                    f"Ground-truth next item: {target.get('title', 'Unknown')} | "
                    f"rank: {target_rank if target_rank is not None else 'not in top list'}"
                )
            st.dataframe(
                [
                    {
                        "rank": r["rank"],
                        "target": r.get("is_target", False),
                        "product": r["title"],
                        "score": round(r["score"], 4),
                        "source": r["main_source"],
                        "expert": r["main_expert"],
                        "graph_score": round(float(r.get("graph_score", 0.0)), 4),
                        "cold": r["cold"],
                    }
                    for r in case["recommendations"]
                ],
                use_container_width=True,
            )

    selected_rank = st.sidebar.slider("Explain rank", 1, max(len(case["recommendations"]), 1), 1)
    rec = case["recommendations"][selected_rank - 1]
    with tab_exp:
        cols = st.columns([1, 1])
        with cols[0]:
            st.subheader(rec["title"])
            if rec.get("image_url"):
                st.image(rec["image_url"], width=240)
            st.write("Category:", rec.get("category", "Unknown"))
            st.write("Candidate sources:", ", ".join(rec.get("sources", [])) or "reranker")
            st.write("Total graph score:", round(float(rec.get("graph_score", 0.0)), 4))
            st.write("Modal graph scores:", rec.get("modal_graph_scores", {}))
            st.write("Cold item:", rec.get("cold"))
            st.caption(rec.get("text", "")[:700])
        with cols[1]:
            st.subheader("Expert Weights")
            st.bar_chart(rec["expert_weights"])
            st.write("Explanation:", f"main expert = {rec['main_expert']}, candidate source = {rec['main_source']}.")

    with tab_cmp:
        st.subheader("Model Comparison")
        if metric_rows:
            st.dataframe(metric_rows, use_container_width=True)
        else:
            st.info("Run the evaluation scripts to populate metric JSON files in the checkpoint directory.")
        source_rows = []
        for r in case["recommendations"]:
            for source in r.get("sources", []):
                source_rows.append({"rank": r["rank"], "product": r["title"], "source": source})
        if source_rows:
            st.caption("Candidate sources for this demo case")
            st.dataframe(source_rows, use_container_width=True)

    with tab_cf:
        st.subheader("Counterfactual")
        cf_case = None
        if counterfactual:
            for row in counterfactual.get("cases", []):
                if int(row.get("index", -1)) == int(case.get("index", -2)):
                    cf_case = row
                    break
        if cf_case:
            rank_rows = []
            for mode, values in cf_case.get("ranks", {}).items():
                top_item = values.get("top_item", {})
                rank_rows.append(
                    {
                        "mode": mode,
                        "target_rank": values.get("target_rank"),
                        "original_rec_rank": values.get("original_recommendation_rank"),
                        "top_item": top_item.get("title"),
                    }
                )
            st.dataframe(rank_rows, use_container_width=True)
        else:
            st.info("Run counterfactual evaluation to show rank changes for the selected case.")
        st.caption("Expert weights for selected recommendation")
        st.json(rec["expert_weights"])


def render_live_demo(config_path: str, checkpoint_path: str) -> None:
    if not Path(checkpoint_path).exists():
        st.warning("Train the model first, then point this demo to the checkpoint.")
        return
    cfg, artifacts, model, item_cards, device = load_runtime(config_path, checkpoint_path)
    meta = artifacts["meta"]
    features = artifacts["features"]
    examples = artifacts["examples"]["test"]
    dataset = SequenceDataset(
        examples,
        meta["max_seq_len"],
        features["item_popularity"],
        features["item_categories"],
        meta["cold_threshold"],
    )
    user_idx = st.sidebar.slider("Test user index", 0, max(len(dataset) - 1, 0), 0)
    batch = dataset[user_idx]
    batch = {key: value.unsqueeze(0) for key, value in batch.items()}
    device_batch = move_batch(batch, device)
    with torch.no_grad():
        out = model(device_batch)
        scores = out["scores"]
        top_scores, top_ids = torch.topk(scores, k=10, dim=1)
        adaptive = model.score_candidates(device_batch, top_ids[0])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("User History")
        rows = []
        for item_id, timestamp in zip(batch["seq"][0].tolist(), batch["times"][0].tolist()):
            if item_id == 0:
                continue
            asin = meta["id2item"][item_id]
            card = item_cards.get(asin, {})
            rows.append({"time": timestamp, "product": item_title(meta, item_id), "category": card.get("category", "Unknown")})
        st.dataframe(rows, use_container_width=True)
    with right:
        st.subheader("Top Recommendations")
        rec_rows = []
        for rank, (item_id, score) in enumerate(zip(top_ids[0].tolist(), top_scores[0].tolist()), start=1):
            weights = adaptive["expert_weights"][0, rank - 1].detach().cpu()
            names = expert_names(weights)
            rec_rows.append({"rank": rank, "product": item_title(meta, item_id), "score": round(float(score), 4), "main_expert": names[int(weights.argmax())]})
        st.dataframe(rec_rows, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="CS-TAMoERec++ Demo", layout="wide")
    st.title("CS-TAMoERec++ Two-stage Product Recommendation")
    config_path = st.sidebar.text_input("Config", "config/cstamoerec_all_beauty_dense10k.yaml")
    checkpoint_path = st.sidebar.text_input("Checkpoint", "checkpoints/cstamoerec_dense10k/best_cstamoerec.pt")
    cache_path = st.sidebar.text_input("Demo cache", "demo_cache/recommendations.json")
    results_dir = st.sidebar.text_input("Metrics dir", "checkpoints/cstamoerec_dense10k")
    counterfactual_path = st.sidebar.text_input("Counterfactual JSON", "checkpoints/cstamoerec_dense10k/counterfactual_test.json")
    use_cache = st.sidebar.checkbox("Use demo cache", value=True)
    cache = load_demo_cache(cache_path) if use_cache else None
    if cache is not None:
        render_cache_demo(cache, load_metric_rows(results_dir), load_json_if_exists(counterfactual_path))
    else:
        render_live_demo(config_path, checkpoint_path)


if __name__ == "__main__":
    main()
