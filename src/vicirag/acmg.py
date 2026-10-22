"""Adaptive Cascaded Multi-Granularity Generation (ACMG) — Sec 3.4 của paper.

Sinh câu trả lời lần lượt theo thứ tự mở rộng G = (g_C, g_S, g_D):
triple -> câu -> đoạn văn. Nhận câu trả lời ở MỨC ĐẦU TIÊN thoả điều kiện đủ
Suf(a)=1 (Eq. 6). Nếu không mức nào đủ, mặc định lấy đáp án mức đoạn văn để tối đa
khả năng trả lời được, đúng như mô tả trong paper.

Ý tưởng: triple ít nhiễu nhất nhưng thiếu ngữ cảnh; đoạn văn đủ ngữ cảnh nhất
nhưng nhiều nhiễu nhất. Leo thang theo nhu cầu giúp cân bằng hai thái cực đó.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from string import Template
from typing import TYPE_CHECKING

from .prompts_vi.acmg import (
    REANSWER_PROMPT_EN,
    REANSWER_PROMPT_VI,
    REFUSAL_EN,
    REFUSAL_VI,
    build_acmg_prompt,
)

if TYPE_CHECKING:
    from .ici import ICIResult
    from .llm.openrouter import OpenRouterLLM

GRANULARITY_ORDER = ("fact", "sentence", "passage")


def is_refusal(text: str) -> bool:
    """Suf(a) = 0 khi câu trả lời khớp mẫu từ chối.

    Chấp nhận cả chuỗi tiếng Việt lẫn tiếng Anh: model đôi khi trả về "No answer"
    dù prompt bằng tiếng Việt, và bỏ sót trường hợp này sẽ khiến cascade dừng sai
    mức, làm sai phân bố granularity ở Figure 3.
    """
    if not text or not text.strip():
        return True
    normalized = unicodedata.normalize("NFC", text).lower()
    return REFUSAL_VI.lower() in normalized or REFUSAL_EN.lower() in normalized


def extract_answer(text: str) -> str:
    """Lấy phần sau nhãn 'Đáp án:' / 'Answer:'; nếu không có thì lấy dòng cuối."""
    for label in ("Đáp án:", "Answer:", "đáp án:", "answer:"):
        idx = text.rfind(label)
        if idx != -1:
            return text[idx + len(label) :].strip().strip(".").strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def reanswer(llm: "OpenRouterLLM", question: str, response: str, lang: str = "vi") -> str:
    """Rút gọn 'Suy luận... Đáp án...' thành đúng cụm đáp án ngắn.

    Rất quan trọng với EM: mô hình tiếng Việt hay trả lời thành câu đầy đủ
    ("Đội tuyển Pháp đã vô địch 2 lần") trong khi đáp án vàng là cụm ngắn ("2"),
    khiến EM bằng 0 dù nội dung đúng.

    Hàm này để ở mức module chứ không nằm trong ACMG, để CẢ baseline cũng dùng
    được. Nếu chỉ CiRAG được chuẩn hoá đáp án còn baseline thì không, phần chênh
    lệch EM sẽ lẫn cả công của việc định dạng đáp án — so sánh mất công bằng.

    Chỉ dùng prompt tiếng Anh khi ``lang="en"``. Biến thể ``en-vi`` chạy prompt tiếng
    Anh trên dữ liệu tiếng Việt, nên đáp án vẫn phải rút gọn theo tiếng Việt.
    """
    template = REANSWER_PROMPT_EN if lang == "en" else REANSWER_PROMPT_VI
    messages = [dict(m) for m in template]
    messages[-1]["content"] = Template(messages[-1]["content"]).safe_substitute(
        question=question, response=response
    )
    answer, _, _ = llm.infer(messages, max_tokens=128)
    return answer.strip().strip(".").strip()


@dataclass
class ACMGResult:
    answer: str
    granularity: str  # mức đã dùng: fact | sentence | passage | default
    raw_response: str
    attempts: dict[str, str]  # mức -> câu trả lời thô, phục vụ phân tích lỗi


class AdaptiveCascadedGeneration:
    def __init__(
        self,
        llm: OpenRouterLLM,
        lang: str = "vi",
        cascade: tuple[str, ...] = GRANULARITY_ORDER,
        use_reanswer: bool = True,
        passage_fallback: bool = True,
        min_source_docs: int = 0,
    ) -> None:
        self.llm = llm
        self.lang = lang
        self.cascade = tuple(cascade)
        self.use_reanswer = use_reanswer
        self.passage_fallback = passage_fallback
        self.min_source_docs = min_source_docs
        self.prompts = {g: build_acmg_prompt(g, lang) for g in GRANULARITY_ORDER}
        # Mức dự phòng "retrieved" cũng là đoạn văn, chỉ khác ở cách chọn ngữ cảnh,
        # nên dùng chung prompt mức đoạn.
        self.prompts["retrieved"] = self.prompts["passage"]

    # ------------------------------------------------------------ dựng ngữ cảnh
    def _context_for(self, granularity: str, ici: ICIResult) -> str:
        if granularity == "fact":
            if not ici.core_triples:
                return ""
            label = "triple: " if self.lang == "vi" else "triplets: "
            return label + ", ".join(str(tuple(t)) for t in ici.core_triples)
        if granularity == "sentence":
            label = "câu: " if self.lang == "vi" else "sentence: "
            return "\n".join(f"{label}{s}" for s in ici.core_sentences)
        if granularity == "passage":
            passages = ici.core_passages
            if not passages and self.passage_fallback:
                passages = getattr(ici, "retrieved_passages", [])
            label = "đoạn văn: " if self.lang == "vi" else "passage: "
            return "\n".join(f"{label}{p}" for p in passages)
        if granularity == "retrieved":
            # Mức dự phòng cuối cùng, không nằm trong G của paper: TOÀN BỘ đoạn đã
            # truy hồi, không lọc theo core triple.
            #
            # Vì sao cần: phân tích lỗi cho thấy 71/73 ca rơi vào nhánh mặc định là
            # câu hỏi so sánh hai thực thể, trong đó recall truy hồi bằng 1.0 (cả hai
            # đoạn bằng chứng đều đã lấy về) nhưng Integration Phase chỉ giữ lại
            # triple về MỘT thực thể. Ngữ cảnh do đó thiếu hẳn một vế, và mọi mức đều
            # từ chối dù bằng chứng đã nằm sẵn trong tay. Đây đúng là "bẫy tham lam"
            # mà paper đặt vấn đề. Mức này hiện thực nhánh "mặc định về mức tài liệu
            # để tối đa khả năng trả lời được" của Eq. 6 theo cách thực sự có tác dụng.
            label = "đoạn văn: " if self.lang == "vi" else "passage: "
            return "\n".join(
                f"{label}{p}" for p in getattr(ici, "retrieved_passages", [])
            )
        raise ValueError(f"Mức ngữ cảnh không hợp lệ: {granularity}")

    def _answer_at(self, granularity: str, question: str, ici: ICIResult) -> tuple[str, str]:
        """Trả về (câu trả lời thô, ngữ cảnh đã dùng). Ngữ cảnh rỗng -> coi như từ chối."""
        context = self._context_for(granularity, ici)
        if not context.strip():
            return REFUSAL_VI if self.lang == "vi" else REFUSAL_EN, ""

        q_label = "Câu hỏi: " if self.lang == "vi" else "Question: "
        t_label = "\nSuy luận: " if self.lang == "vi" else "\nThought: "
        prompt_user = f"{context}\n\n{q_label}{question}{t_label}"

        messages = [dict(m) for m in self.prompts[granularity]]
        messages[-1]["content"] = Template(messages[-1]["content"]).safe_substitute(
            prompt_user=prompt_user
        )
        response, _, _ = self.llm.infer(messages)
        return response, context

    def _reanswer(self, question: str, response: str) -> str:
        return reanswer(self.llm, question, response, lang=self.lang)

    # ------------------------------------------------ điều kiện đủ theo số nguồn
    def _spans_enough_docs(self, ici: ICIResult) -> bool:
        """Bằng chứng lõi có đến từ đủ số tài liệu mà câu hỏi đòi hỏi hay không.

        Điều kiện Suf ở Eq. 6 chỉ nhận giá trị 0 khi mô hình chịu trả về chuỗi từ
        chối. Mô hình nền mạnh gần như không bao giờ từ chối, nên cascade chốt ngay
        ở mức đầu tiên kể cả khi ngữ cảnh thiếu. Phép kiểm này thay việc hỏi mô hình
        có chịu thừa nhận thiếu ngữ cảnh không bằng một tính chất kiểm tra được:
        câu hỏi đa bước cần bằng chứng từ ít nhất hai tài liệu, nên tập triple lõi
        chỉ đến từ một tài liệu là chưa đủ, bất kể mô hình có sẵn lòng trả lời.

        Số tài liệu lấy từ ánh xạ provenance M^D (Eq. 3) mà phương pháp vốn đã dựng,
        nên phép kiểm không tốn thêm lời gọi mô hình nào.
        """
        if self.min_source_docs <= 0:
            return True
        return len(ici.core_passages) >= self.min_source_docs

    # ------------------------------------------------------------------ cascade
    def run(self, question: str, ici: ICIResult) -> ACMGResult:
        attempts: dict[str, str] = {}
        last_response = ""

        #  Bằng chứng lõi không trải đủ số tài liệu -> mọi mức trong cascade đều dựng
        #  từ cùng tập triple thiếu đó, nên bỏ qua các mức hẹp và đọc thẳng mức rộng
        #  nhất thay vì chốt sớm ở một câu trả lời không có cơ sở.
        cascade = self.cascade
        if not self._spans_enough_docs(ici) and len(cascade) > 1:
            cascade = cascade[-1:]

        for granularity in cascade:
            response, _ = self._answer_at(granularity, question, ici)
            attempts[granularity] = response
            last_response = response
            if not is_refusal(response):
                answer = extract_answer(response)
                if self.use_reanswer:
                    answer = self._reanswer(question, response)
                if not is_refusal(answer):
                    return ACMGResult(answer, granularity, response, attempts)

        # Không mức nào trong G đủ -> thử nốt toàn bộ đoạn đã truy hồi (Eq. 6, nhánh
        # mặc định "tối đa khả năng trả lời được").
        if self.passage_fallback and getattr(ici, "retrieved_passages", []):
            response, context = self._answer_at("retrieved", question, ici)
            attempts["retrieved"] = response
            if context and not is_refusal(response):
                answer = extract_answer(response)
                if self.use_reanswer:
                    answer = self._reanswer(question, response)
                if not is_refusal(answer):
                    return ACMGResult(answer, "retrieved", response, attempts)
            last_response = response

        fallback = self.cascade[-1]
        response = attempts.get(fallback, last_response)
        answer = extract_answer(response)
        if self.use_reanswer and answer and not is_refusal(answer):
            answer = self._reanswer(question, response)
        return ACMGResult(answer, "default", response, attempts)
