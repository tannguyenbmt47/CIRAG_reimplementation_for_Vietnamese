"""Cấu hình tập trung cho ViCiRAG.

Siêu tham số mặc định lấy đúng theo Appendix B.4 của paper CIRAG:
K = 10 tài liệu / vòng, N = 30 triple ứng viên / vòng, L = 4 vòng lặp tối đa.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"


@dataclass
class Config:
    # ----- Dữ liệu -----
    dataset: str = "vimqa_sample"
    #  Để None thì suy ra từ tên bộ dữ liệu, nhờ vậy thêm bộ mới (2wiki) không
    #  phải sửa cấu hình. Đặt tay khi dữ liệu nằm chỗ khác.
    data_dir: Path | None = None
    split: str = "dev"

    # ----- LLM (toàn bộ qua OpenRouter) -----
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_name: str = "google/gemini-2.5-flash"
    #  Model riêng cho trích triple hàng loạt. Đây là phần chiếm đa số lời gọi
    #  (2 lời gọi/đoạn cho cả corpus) nhưng lại là tác vụ dễ nhất — trích thông tin
    #  có cấu trúc chứ không phải suy luận nhiều bước. Dùng model rẻ ở đây giảm
    #  chi phí khoảng một nửa mà không đụng tới chất lượng Integration/ACMG.
    openie_llm_name: str = "google/gemini-2.5-flash-lite"
    #  Model teacher dùng cho Trajectory Distillation (Bước 5).
    teacher_llm_name: str = "google/gemini-2.5-pro"
    temperature: float = 0.0
    max_new_tokens: int | None = 2048
    seed: int = 0
    #  Cửa sổ ngữ cảnh của server local, để phát hiện prompt có thể bị cắt
    #  (xem `OpenRouterLLM`). None khi dùng OpenRouter: provider báo lỗi thay vì cắt.
    context_window: int | None = None

    # ----- Embedding / truy hồi -----
    embedding_model_name: str = "AITeamVN/Vietnamese_Embedding"
    embedding_batch_size: int = 8
    embedding_max_length: int = 1024
    embedding_device: str | None = None  # None -> tự chọn cuda nếu có

    retrieval_top_k: int = 10  # K: số tài liệu truy hồi mỗi vòng
    num_candidate_triples: int = 30  # N: số triple ứng viên sau rerank
    max_iterations: int = 4  # L: số vòng ICI tối đa

    #  Truy hồi lai BM25 + dense (cải tiến cho tiếng Việt, tắt để bám sát paper).
    use_hybrid_retrieval: bool = False
    rrf_k: int = 60

    #  w/o reranker: dùng toàn bộ triple của top-K doc thay vì lấy top-N.
    use_triple_reranker: bool = True

    #  "official" = chữ thường và bỏ ký tự đặc biệt khỏi triple như repo gốc. Chỉ dùng cho
    #  tiếng Anh; "none" giữ nguyên văn, bắt buộc với tiếng Việt để không mất dấu.
    triple_norm: str = "none"

    #  Hai cơ chế cứu tập core triple có trong mã nguồn gốc nhưng không có trong bài báo
    #  (xem `IterativeConstructionIntegration._official_rescue`). Tắt = đúng Eq. 1.
    official_rescue: bool = False

    # ----- ACMG -----
    #  Chuỗi mức ngữ cảnh, theo thứ tự mở rộng G = (g_C, g_S, g_D).
    #  Đổi giá trị này để chạy ablation Table 3 (vd: ["fact"], ["fact","sentence"]).
    granularity_cascade: tuple[str, ...] = ("fact", "sentence", "passage")

    #  Số tài liệu tối thiểu mà tập triple lõi phải đến từ thì một mức trong
    #  cascade mới được coi là đủ. 0 = tắt, giữ đúng Eq. 6 của bài báo. Đặt 2 cho
    #  bộ dữ liệu đa bước: câu hỏi cần bằng chứng từ ít nhất hai tài liệu, nên tập
    #  triple lõi chỉ đến từ một tài liệu là chưa đủ dù mô hình vẫn chịu trả lời.
    min_source_docs: int = 0

    #  Mức dự phòng đọc lại toàn bộ đoạn đã truy hồi khi không mức nào trong cascade
    #  trả lời được. Đặt False để lấy đúng hành vi của Eq. 6 trong bài báo gốc.
    passage_fallback: bool = True

    # ----- Hạ tầng -----
    cache_dir: Path = CACHE_DIR
    output_dir: Path = OUTPUT_DIR
    force_openie_from_scratch: bool = False
    num_workers: int = 4  # số luồng gọi API song song

    run_name: str = "default"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_dir is None:
            family = self.dataset.split("_")[0]
            self.data_dir = DATA_DIR / family / "processed"
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_local_llm(self) -> bool:
        return any(h in self.llm_base_url for h in ("localhost", "127.0.0.1", "[::1]"))

    @property
    def api_key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key and self.is_local_llm:
            #  Ollama không kiểm khoá, nhưng SDK OpenAI bắt buộc phải truyền một chuỗi.
            return "local"
        if not key:
            raise RuntimeError(
                "Thiếu OPENROUTER_API_KEY. Tạo file .env từ .env.example "
                "hoặc export OPENROUTER_API_KEY=sk-or-..."
            )
        return key

    @property
    def samples_path(self) -> Path:
        return self.data_dir / f"{self.dataset}_{self.split}.json"

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / f"{self.dataset}_{self.split}_corpus.json"
