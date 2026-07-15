"""Encoders for the pilot: BERT target-token contexts, CLIP text, CLIP image.

These follow the conventions of the parent MMEG repo (target-subword averaging,
last-N-layer pooling, L2 normalisation on the unit sphere) but locate the target
via SemCor char offsets rather than a regex, which is exact for multi-word and
repeated tokens.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _pick_device(prefer_cuda: bool = True) -> str:
    return "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"


class BertContextEmbedder:
    """Contextual target-token embedding: mean of target subwords, averaged over
    the last ``pool_last_n_layers`` transformer layers, then L2-normalised."""

    def __init__(self, model_id: str, pool_last_n_layers: int = 4, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or _pick_device()
        self.pool_last_n_layers = pool_last_n_layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        mm = getattr(self.tokenizer, "model_max_length", 512)
        self.max_length = mm if isinstance(mm, int) and 0 < mm <= 4096 else 512

    @torch.no_grad()
    def encode(self, items: list[dict], batch_size: int = 32) -> np.ndarray:
        """Encode occurrences.

        Each item needs keys ``sentence``, ``target_start``, ``target_end``.
        Returns an (N, H) float32 array aligned with ``items``. Rows whose target
        span maps to no subword token (e.g. truncated away) are returned as NaN
        so callers can drop them while keeping alignment.
        """
        out: list[np.ndarray] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            enc = self.tokenizer(
                [it["sentence"] for it in batch],
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            offsets = enc.pop("offset_mapping")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden_states = self.model(**enc, output_hidden_states=True).hidden_states
            stacked = torch.stack(hidden_states[-self.pool_last_n_layers :]).mean(0)  # (B,T,H)

            for b, it in enumerate(batch):
                start, end = it["target_start"], it["target_end"]
                tok_idx = [
                    t
                    for t, (s, e) in enumerate(offsets[b].tolist())
                    if not (s == 0 and e == 0) and s < end and e > start
                ]
                if not tok_idx:
                    out.append(np.full(stacked.shape[-1], np.nan, dtype=np.float32))
                    continue
                vec = stacked[b, tok_idx].mean(0)
                vec = vec / vec.norm(p=2).clamp_min(1e-12)
                out.append(vec.float().cpu().numpy())
        return np.stack(out)


class ClipTextEmbedder:
    """CLIP text encoder producing L2-normalised sentence embeddings."""

    def __init__(self, model_id: str, device: str | None = None):
        from transformers import CLIPModel, CLIPTokenizerFast

        self.device = device or _pick_device()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        # Compute the projected joint-space embedding explicitly. In
        # transformers 5.x get_text_features returns the text-encoder output
        # object (pre-projection), so we apply text_projection ourselves to land
        # in CLIP's shared image/text space.
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            pooled = self.model.text_model(**enc).pooler_output
            feats = self.model.text_projection(pooled)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(feats.float().cpu().numpy())
        return np.vstack(out)


class ClipImageEmbedder:
    """CLIP image encoder producing L2-normalised image embeddings."""

    def __init__(self, model_id: str, device: str | None = None, use_fp16: bool = False):
        from transformers import AutoProcessor, CLIPModel

        self.device = device or _pick_device()
        self.dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            CLIPModel.from_pretrained(model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def encode_paths(self, paths: list, batch_size: int = 128) -> np.ndarray:
        from PIL import Image

        out: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            imgs = []
            for p in paths[i : i + batch_size]:
                try:
                    with open(p, "rb") as fh:
                        imgs.append(Image.open(fh).convert("RGB"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("skip image %s: %s", p, exc)
            if not imgs:
                continue
            enc = self.processor(images=imgs, return_tensors="pt").to(self.device)
            enc = {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in enc.items()}
            pooled = self.model.vision_model(**enc).pooler_output
            feats = self.model.visual_projection(pooled)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(feats.float().cpu().numpy())
        return np.vstack(out) if out else np.empty((0, 512), dtype=np.float32)
