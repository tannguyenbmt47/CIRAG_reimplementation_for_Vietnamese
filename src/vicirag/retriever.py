"""Truy hồi tài liệu và xếp hạng triple, kèm ánh xạ provenance.

Cài đặt phần "retriever R vừa truy hồi tài liệu vừa làm reranker cho triple" mô
tả ở Sec 3.3 của paper.

Về phạm vi triple ứng viên: repo gốc (`rerank_facts_copy` trong CIRAG.py) lọc
triple ứng viên xuống chỉ những triple bắt nguồn từ 10 đoạn văn của CHÍNH câu hỏi
đang xét, trong khi paper mô tả setting corpus toàn cục kiểu IRCoT. Hai cách cho
kết quả rất khác nhau, nên ở đây hỗ trợ cả hai qua `candidate_scope`:
  - "corpus" (mặc định): trung thành với mô tả trong paper.
  - "sample":  tái lập hành vi của mã nguồn được công bố.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .data import Passage, TripleRecord

if TYPE_CHECKING:
    from .embedding.vi_encoder import VietnameseEncoder


def _encode_cached(
    encoder: "VietnameseEncoder", texts: list[str], cache_dir: Path | None, tag: str
) -> np.ndarray:
    """Nhúng văn bản, có cache trên đĩa theo nội dung.

    Một loạt ablation dùng chung đúng corpus và đúng tập triple, nên nhúng lại mỗi
    lần chạy là lãng phí vài phút GPU. Khoá cache là hash của (tên model + toàn bộ
    văn bản), nên đổi model hay đổi corpus là tự động tính lại.
    """
    if cache_dir is None or not texts:
        return encoder.encode(texts, show_progress=True)

    digest = hashlib.sha256(
        (encoder.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")
    ).hexdigest()[:16]
    path = Path(cache_dir) / f"emb_{tag}_{digest}.npy"

    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(texts):
            print(f"[Retriever] Dùng cache embedding: {path.name}")
            return emb

    emb = encoder.encode(texts, show_progress=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def _tokenize_vi(text: str) -> list[str]:
    """Tách từ tiếng Việt cho BM25; lùi về tách theo âm tiết nếu thiếu underthesea."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text.lower(), format="text").split()
    except Exception:
        return text.lower().split()


class Retriever:
    def __init__(
        self,
        corpus: list[Passage],
        triples_by_title: dict[str, list[TripleRecord]],
        encoder: "VietnameseEncoder",
        use_hybrid: bool = False,
        rrf_k: int = 60,
        candidate_scope: str = "corpus",
        cache_dir: Path | None = None,
    ) -> None:
        self.corpus = corpus
        self.encoder = encoder
        self.use_hybrid = use_hybrid
        self.rrf_k = rrf_k
        self.candidate_scope = candidate_scope

        self.titles = [p.title for p in corpus]
        self.title_to_idx = {t: i for i, t in enumerate(self.titles)}
        self.title_to_passage = {p.title: p for p in corpus}

        # --- Danh sách triple phẳng + provenance ---
        self.triple_records: list[TripleRecord] = []
        self.title_to_triple_ids: dict[str, list[int]] = {}
        for title in self.titles:
            ids = []
            for record in triples_by_title.get(title, []):
                ids.append(len(self.triple_records))
                self.triple_records.append(record)
            self.title_to_triple_ids[title] = ids

        print(f"[Retriever] Đánh chỉ mục {len(corpus)} đoạn, {len(self.triple_records)} triple")
        self.passage_emb = _encode_cached(
            encoder, [p.full_text for p in corpus], cache_dir, "passage"
        )
        self.triple_emb = _encode_cached(
            encoder, [r.text for r in self.triple_records], cache_dir, "triple"
        )

        self.bm25 = None
        if use_hybrid:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi([_tokenize_vi(p.full_text) for p in corpus])

    # ------------------------------------------------------------------ đoạn văn
    def retrieve_passages(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Trả về [(chỉ số đoạn, điểm)] đã sắp giảm dần."""
        q_emb = self.encoder.encode_queries([query])[0]
        dense_scores = self.passage_emb @ q_emb

        if self.bm25 is None:
            order = np.argsort(-dense_scores)[:top_k]
            return [(int(i), float(dense_scores[i])) for i in order]

        # Reciprocal Rank Fusion giữa BM25 (khớp từ khoá) và dense (khớp ngữ nghĩa).
        bm25_scores = np.asarray(self.bm25.get_scores(_tokenize_vi(query)))
        dense_rank = np.empty(len(dense_scores), dtype=int)
        dense_rank[np.argsort(-dense_scores)] = np.arange(len(dense_scores))
        bm25_rank = np.empty(len(bm25_scores), dtype=int)
        bm25_rank[np.argsort(-bm25_scores)] = np.arange(len(bm25_scores))
        rrf = 1.0 / (self.rrf_k + dense_rank + 1) + 1.0 / (self.rrf_k + bm25_rank + 1)
        order = np.argsort(-rrf)[:top_k]
        return [(int(i), float(rrf[i])) for i in order]

    # -------------------------------------------------------------------- triple
    def rerank_triples(
        self,
        query: str,
        passage_indices: list[int],
        top_n: int = 30,
        allowed_titles: set[str] | None = None,
        use_reranker: bool = True,
    ) -> list[int]:
        """Gom triple của các đoạn đã truy hồi rồi xếp hạng theo độ tương đồng với truy vấn.

        `use_reranker=False` tương ứng biến thể "w/o reranker" trong Table 2: lấy
        toàn bộ triple của top-K đoạn, không cắt top-N.
        """
        candidate_ids: list[int] = []
        for idx in passage_indices:
            title = self.titles[idx]
            if allowed_titles is not None and title not in allowed_titles:
                continue
            candidate_ids.extend(self.title_to_triple_ids.get(title, []))

        if not candidate_ids:
            return []
        if not use_reranker:
            return candidate_ids

        q_emb = self.encoder.encode_queries([query])[0]
        scores = self.triple_emb[candidate_ids] @ q_emb
        order = np.argsort(-scores)[:top_n]
        return [candidate_ids[i] for i in order]

    def get_candidates(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 30,
        allowed_titles: set[str] | None = None,
        use_reranker: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Một bước Construction Phase: đoạn văn -> triple ứng viên.

        Trả về (chỉ số đoạn top-K, chỉ số triple ứng viên top-N).
        """
        passages = self.retrieve_passages(query, top_k=top_k)
        passage_indices = [i for i, _ in passages]
        scope_titles = allowed_titles if self.candidate_scope == "sample" else None
        triple_ids = self.rerank_triples(
            query,
            passage_indices,
            top_n=top_n,
            allowed_titles=scope_titles,
            use_reranker=use_reranker,
        )
        return passage_indices, triple_ids

    # ------------------------------------------------------------------ tiện ích
    def triple_of(self, triple_id: int) -> TripleRecord:
        return self.triple_records[triple_id]

    def sentences_of(self, triple_id: int) -> list[str]:
        """Ánh xạ M^S: triple -> các câu nguồn (Eq. 3)."""
        record = self.triple_records[triple_id]
        passage = self.title_to_passage.get(record.passage_title)
        if passage is None:
            return []
        return [passage.sentences[i] for i in record.sentence_ids if i < len(passage.sentences)]

    def passage_of(self, triple_id: int) -> Passage | None:
        """Ánh xạ M^D: triple -> đoạn văn nguồn (Eq. 4)."""
        return self.title_to_passage.get(self.triple_records[triple_id].passage_title)
