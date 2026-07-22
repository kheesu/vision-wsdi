"""Single multimodal encoder for the pilot: Qwen3-VL-Embedding.

One model embeds both text and images into a shared 4096-d space, so the same
encoder produces the word-in-context text vector t_i (which is also the
clustering base) and the ImageNet image vectors averaged into class prototypes
v_c. Wrapped via sentence-transformers, which handles the instruction prompt and
last-token pooling internally; we always return L2-normalised float32 arrays so
downstream cosine similarities are plain dot products.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _pick_device(prefer_cuda: bool = True) -> str:
    return "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"


class QwenEmbedder:
    """Qwen3-VL-Embedding wrapper exposing text and image encoding.

    ``prompt`` overrides the model's default instruction ("Represent the user's
    input."); ``None`` keeps that default. ``dtype`` is the torch dtype string
    the weights are loaded in (bfloat16 is the model's native format).
    """

    def __init__(self, model_id: str, device: str | None = None,
                 dtype: str = "bfloat16", prompt: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.device = device or _pick_device()
        self.prompt = prompt
        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype},
        )

    def _encode(self, inputs: list, batch_size: int) -> np.ndarray:
        kwargs = {"prompt": self.prompt} if self.prompt else {}
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """(N, D) unit-norm embeddings for a list of strings."""
        return self._encode(list(texts), batch_size)

    def encode_image_paths(self, paths: list, batch_size: int = 8) -> np.ndarray:
        """(N, D) unit-norm embeddings for a list of local image paths."""
        docs = [{"image": str(p)} for p in paths]
        return self._encode(docs, batch_size)
