"""Encoder tiếng Việt thay cho NV-Embed-v2 của repo gốc.

Repo gốc dùng `nvidia/NV-Embed-v2` (7.85B tham số, cần ~16GB VRAM) hoặc GritLM.
Cả hai đều quá nặng so với GPU 8GB và cũng không được huấn luyện cho tiếng Việt.

Mặc định dùng `AITeamVN/Vietnamese_Embedding` — bản fine-tune của BGE-M3 (~568M)
trên ~300k bộ ba truy vấn/tài liệu tiếng Việt, chạy thoải mái trên 8GB hoặc CPU.
"""

from __future__ import annotations

import numpy as np


class VietnameseEncoder:
    """Bọc sentence-transformers, trả về vector đã chuẩn hoá L2.

    Chuẩn hoá L2 để tích vô hướng chính là cosine similarity — dùng thống nhất cho
    cả xếp hạng đoạn văn lẫn rerank triple.
    """

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        device: str | None = None,
        max_length: int = 1024,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.model.max_seq_length = max_length

    @property
    def dim(self) -> int:
        # sentence-transformers 5.x đổi tên hàm; giữ tương thích cả hai phiên bản.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return getter()

    def encode(
        self, texts: list[str], batch_size: int | None = None, show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def encode_queries(self, queries: list[str], **kwargs) -> np.ndarray:
        # BGE-M3 không yêu cầu tiền tố cho truy vấn, giữ đối xứng với encode().
        return self.encode(queries, **kwargs)
