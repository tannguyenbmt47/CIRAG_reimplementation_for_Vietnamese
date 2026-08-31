"""NativeRAG — baseline truy hồi một lần rồi sinh câu trả lời (Lewis et al., 2020).

Đây là baseline yếu nhất trong Table 1 của paper và là mốc so sánh bắt buộc:
nó cho thấy truy hồi một bước không đủ để bắc cầu qua nhiều hop.
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

from ..acmg import extract_answer, reanswer
from ..prompts_vi.acmg import build_acmg_prompt

if TYPE_CHECKING:
    from ..llm.openrouter import OpenRouterLLM
    from ..retriever import Retriever


class NativeRAG:
    def __init__(
        self,
        retriever: "Retriever",
        llm: "OpenRouterLLM",
        lang: str = "vi",
        top_k: int = 10,
        use_reanswer: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.lang = lang
        self.top_k = top_k
        # Dùng chung bước chuẩn hoá đáp án với CiRAG để so sánh công bằng.
        self.use_reanswer = use_reanswer
        # Dùng chung prompt mức đoạn văn của ACMG để so sánh công bằng: khác biệt
        # duy nhất so với CiRAG là cách chọn ngữ cảnh, không phải cách hỏi.
        self.prompt = build_acmg_prompt("passage", lang)

    def run(self, question: str, allowed_titles: set[str] | None = None) -> dict:
        passages = self.retriever.retrieve_passages(question, top_k=self.top_k)
        titles = [self.retriever.titles[i] for i, _ in passages]
        label = "đoạn văn: " if self.lang == "vi" else "passage: "
        context = "\n".join(
            f"{label}{self.retriever.corpus[i].full_text}" for i, _ in passages
        )

        q_label = "Câu hỏi: " if self.lang == "vi" else "Question: "
        t_label = "\nSuy luận: " if self.lang == "vi" else "\nThought: "
        messages = [dict(m) for m in self.prompt]
        messages[-1]["content"] = Template(messages[-1]["content"]).safe_substitute(
            prompt_user=f"{context}\n\n{q_label}{question}{t_label}"
        )

        response, _, _ = self.llm.infer(messages)
        answer = reanswer(self.llm, question, response) if self.use_reanswer else extract_answer(response)
        return {
            "predicted_answer": answer,
            "retrieved_titles": titles,
            "raw_response": response,
            "num_steps": 1,
        }
