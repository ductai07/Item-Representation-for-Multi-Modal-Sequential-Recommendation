from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import requests
import torch
from PIL import Image
from tqdm import tqdm


def encode_texts(texts: list[str], model_name: str, batch_size: int = 128, device: str = "cuda") -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device if torch.cuda.is_available() else "cpu")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    return embeddings.cpu().float()


def encode_images(
    image_urls: list[str | None],
    model_name: str,
    max_image_items: int,
    timeout: int,
    device: str = "cuda",
    cache_path: str | Path | None = None,
    cache_every: int = 250,
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import CLIPImageProcessor, CLIPModel

    actual_device = device if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(actual_device)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model.eval()
    hidden = model.config.projection_dim
    embeddings = torch.zeros((len(image_urls), hidden), dtype=torch.float)
    mask = torch.zeros(len(image_urls), dtype=torch.float)
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        cached_embeddings = cached.get("image_embeddings")
        cached_mask = cached.get("image_mask")
        if (
            isinstance(cached_embeddings, torch.Tensor)
            and isinstance(cached_mask, torch.Tensor)
            and tuple(cached_embeddings.shape) == tuple(embeddings.shape)
            and tuple(cached_mask.shape) == tuple(mask.shape)
        ):
            embeddings = cached_embeddings.float()
            mask = cached_mask.float()
            print(f"Resuming image embeddings from {cache_file} ({int(mask.sum().item())} cached items).")
        else:
            print(f"Ignoring incompatible image cache: {cache_file}")
    encoded = int(mask.sum().item())
    with torch.no_grad():
        for idx, url in tqdm(list(enumerate(image_urls)), desc="Image embedding"):
            if idx == 0 or not url:
                continue
            if mask[idx].item() > 0:
                continue
            if encoded >= max_image_items:
                break
            image = _download_image(url, timeout)
            if image is None:
                continue
            inputs = processor(images=image, return_tensors="pt").to(actual_device)
            output = _clip_image_features(model, inputs).squeeze(0).cpu().float()
            embeddings[idx] = output
            mask[idx] = 1.0
            encoded += 1
            if cache_file and encoded % cache_every == 0:
                _save_image_cache(cache_file, embeddings, mask)
    if cache_file:
        _save_image_cache(cache_file, embeddings, mask)
    return embeddings, mask


def _save_image_cache(path: Path, embeddings: torch.Tensor, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_embeddings": embeddings.cpu(), "image_mask": mask.cpu()}, path)


def _clip_image_features(model, inputs) -> torch.Tensor:
    """Return projected CLIP image features across transformers versions."""
    output = model.get_image_features(**inputs)
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return _project_clip_image_features(model, output.pooler_output)
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                return _project_clip_image_features(model, value)
    vision_output = model.vision_model(**inputs)
    return _project_clip_image_features(model, vision_output.pooler_output)


def _project_clip_image_features(model, features: torch.Tensor) -> torch.Tensor:
    projection_dim = int(model.config.projection_dim)
    if features.shape[-1] == projection_dim:
        return features
    projection_in = int(model.visual_projection.in_features)
    if features.shape[-1] == projection_in:
        return model.visual_projection(features)
    raise ValueError(
        f"Unexpected CLIP image feature shape: got {tuple(features.shape)}, "
        f"expected last dim {projection_dim} or {projection_in}."
    )


def _download_image(url: str, timeout: int) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception:
        return None


def zero_image_features(num_items: int, image_dim: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((num_items, image_dim), dtype=torch.float), torch.zeros(num_items, dtype=torch.float)


def normalize_feature_matrix(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mask = x.abs().sum(dim=1) > 0
    if mask.any():
        mean = x[mask].mean(dim=0, keepdim=True)
        std = x[mask].std(dim=0, keepdim=True).clamp_min(1e-6)
        x = x.clone()
        x[mask] = (x[mask] - mean) / std
    return x
