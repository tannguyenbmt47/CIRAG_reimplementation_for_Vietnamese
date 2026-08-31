"""IRCoT — baseline truy hồi xen kẽ chuỗi suy luận (Trivedi et al., 2023).

Mỗi vòng: truy hồi tài liệu cho truy vấn hiện tại, sinh MỘT câu suy luận tiếp
theo, rồi dùng chính câu đó làm truy vấn cho vòng sau. Dừng khi câu suy luận chứa
cụm kết luận ("Vậy đáp án là") hoặc hết số vòng.

Khác biệt cốt lõi so với CiRAG: IRCoT tích luỹ nguyên văn ĐOẠN VĂN (nhiễu tích
tụ dần) và mở rộng theo một đường đơn, trong khi ICI chắt lọc thành triple lõi và
giữ nhiều nhánh bằng chứng.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..acmg import extract_answer, reanswer

if TYPE_CHECKING:
    from ..llm.openrouter import OpenRouterLLM
    from ..retriever import Retriever

SYSTEM_VI = (
    "Bạn là trợ lý suy luận nhiều bước. Dựa vào các đoạn văn được cung cấp và chuỗi suy luận "
    "đã có, hãy viết ĐÚNG MỘT câu suy luận tiếp theo, ngắn gọn. "
    'Khi đã đủ thông tin để kết luận, hãy viết câu cuối theo mẫu "Vậy đáp án là: <đáp án>". '
    "Chỉ viết một câu, bằng tiếng Việt, không giải thích thêm."
)

SYSTEM_EN = (
    "You are a multi-step reasoning assistant. Based on the given passages and the reasoning "
    "chain so far, write EXACTLY ONE next reasoning sentence. "
    'When you have enough information, write "So the answer is: <answer>".'
)

FINAL_MARKERS = ("vậy đáp án là", "so the answer is", "đáp án là")


class IRCoT:
    def __init__(
        self,
        retriever: "Retriever",
        llm: "OpenRouterLLM",
        lang: str = "vi",
        top_k: int = 10,
        max_iterations: int = 4,
        use_reanswer: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.lang = lang
        self.top_k = top_k
        self.max_iterations = max_iterations
        # Dùng chung bước chuẩn hoá đáp án với CiRAG để so sánh công bằng.
        self.use_reanswer = use_reanswer
        self.system = SYSTEM_VI if lang == "vi" else SYSTEM_EN

    def run(self, question: str, allowed_titles: set[str] | None = None) -> dict:
        collected: dict[str, None] = {}  # giữ thứ tự, khử trùng lặp
        reasoning: list[str] = []
        all_titles: list[str] = []
        query = question

        for step in range(1, self.max_iterations + 1):
            for idx, _ in self.retriever.retrieve_passages(query, top_k=self.top_k):
                collected[self.retriever.corpus[idx].full_text] = None
                all_titles.append(self.retriever.titles[idx])

            label = "đoạn văn: " if self.lang == "vi" else "passage: "
            context = "\n".join(f"{label}{p}" for p in collected)
            q_label = "Câu hỏi: " if self.lang == "vi" else "Question: "
            r_label = "Suy luận đã có: " if self.lang == "vi" else "Reasoning so far: "
            chain = " ".join(reasoning) if reasoning else "(chưa có)"

            messages = [
                {"role": "system", "content": self.system},
                {
                    "role": "user",
                    "content": f"{context}\n\n{q_label}{question}\n{r_label}{chain}",
                },
            ]
            sentence, _, _ = self.llm.infer(messages, max_tokens=256)
            sentence = sentence.strip()
            reasoning.append(sentence)

            lowered = sentence.lower()
            if any(marker in lowered for marker in FINAL_MARKERS):
                raw = sentence.replace("Vậy đáp án là:", "Đáp án:")
                return {
                    "predicted_answer": (
                        reanswer(self.llm, question, raw) if self.use_reanswer
                        else extract_answer(raw)
                    ),
                    "retrieved_titles": all_titles,
                    "raw_response": " ".join(reasoning),
                    "num_steps": step,
                }
            query = sentence

        last = reasoning[-1] if reasoning else ""
        return {
            "predicted_answer": (
                reanswer(self.llm, question, last) if (self.use_reanswer and last)
                else extract_answer(last)
            ),
            "retrieved_titles": all_titles,
            "raw_response": " ".join(reasoning),
            "num_steps": self.max_iterations,
        }
