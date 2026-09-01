"""Iterative Construction-Integration (ICI) — Sec 3.3 của paper.

Mỗi vòng lặp t:
  Construction: R truy hồi top-K đoạn cho truy vấn a_t, gom triple đã trích sẵn
                của các đoạn đó, rerank theo độ tương đồng với a_t, lấy top-N ->
                tập ứng viên T̂_t.
  Integration:  K_D nhận (x, T̂_t, I, H<t) và sinh ra (r_t, T̃_t, a_{t+1})  [Eq. 1]
                trong đó T̃_t là tập core triple, a_{t+1} là truy vấn hop kế tiếp.

Vòng lặp dừng khi a_{t+1} == "<no question>" hoặc đạt L vòng.

Lịch sử H<t (Eq. 2) được mang theo dưới dạng các lượt hội thoại nối tiếp, đúng
như cách repo gốc làm trong `DSPyFilter_genedata.llm_call` (tham số `old_message`),
chứ không nhồi vào một trường riêng.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .data import Triple
from .prompts_vi.integration import (
    INPUT_TEMPLATE_FOLLOWUP,
    INPUT_TEMPLATE_START,
    INTEGRATION_DEMOS,
    INTEGRATION_SYSTEM,
    NO_QUESTION,
)

if TYPE_CHECKING:  # tách lớp logic khỏi lớp gọi API để test được độc lập
    from .llm.openrouter import OpenRouterLLM
    from .retriever import Retriever

_FIELD_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


def parse_integration_output(response: str) -> tuple[list[Triple], str, str]:
    """Bóc (core triples, truy vấn kế tiếp, suy luận) từ output dạng DSPy.

    Port từ `DSPyFilter_genedata.parse_filter` của repo gốc, thêm phần chịu lỗi:
    model nhỏ hay bọc JSON trong ```...``` hoặc bỏ sót header.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in response.splitlines():
        match = _FIELD_HEADER.match(line.strip())
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)

    facts: list[Triple] = []
    next_query = ""
    thought = ""

    for key, lines in sections:
        value = "\n".join(lines).strip()
        if key == "fact_after_filter":
            fenced = re.search(r"```(?:json)?\s*(.*?)```", value, re.DOTALL)
            if fenced:
                value = fenced.group(1).strip()
            parsed = None
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, dict):
                for item in parsed.get("fact", []) or []:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        facts.append(tuple(str(x) for x in item))  # type: ignore[arg-type]
        elif key == "question":
            next_query = value
        elif key == "thought":
            thought = value

    return facts, next_query, thought


@dataclass
class IterationLog:
    """Ghi lại một vòng ICI — dùng cho phân tích lỗi và cho Trajectory Distillation."""

    step: int
    query: str
    candidate_triples: list[Triple]
    core_triples: list[Triple]
    thought: str
    next_query: str
    retrieved_titles: list[str]
    raw_response: str = ""
    integration_input: str = ""


@dataclass
class ICIResult:
    """Ngữ cảnh đa mức tích luỹ {C, S, D} sau khi vòng lặp kết thúc."""

    question: str
    core_triples: list[Triple] = field(default_factory=list)  # C
    core_sentences: list[str] = field(default_factory=list)  # S
    core_passages: list[str] = field(default_factory=list)  # D
    logs: list[IterationLog] = field(default_factory=list)
    num_steps: int = 0
    terminated_early: bool = False
    #  Toàn bộ đoạn đã truy hồi qua các vòng, KHÔNG lọc theo core triple.
    #  Chỉ dùng làm phương án chót cho ACMG khi tập core rỗng (xem acmg.py).
    retrieved_passages: list[str] = field(default_factory=list)


class IterativeConstructionIntegration:
    def __init__(
        self,
        retriever: "Retriever",
        llm: "OpenRouterLLM",
        lang: str = "vi",
        top_k: int = 10,
        top_n: int = 30,
        max_iterations: int = 4,
        max_core_triples: int = 10,
        use_reranker: bool = True,
        official_rescue: bool = False,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.lang = lang
        self.top_k = top_k
        self.top_n = top_n
        self.max_iterations = max_iterations
        # Repo gốc cắt tập core triple ở 10 (`len_after_rerank=10`).
        self.max_core_triples = max_core_triples
        self.use_reranker = use_reranker
        #  Hai cơ chế cứu của `rerank_facts_gene_data` trong repo gốc, không có trong bài
        #  báo: xem `_official_rescue`. Mặc định tắt để giữ đúng Eq. 1 như bài báo mô tả.
        self.official_rescue = official_rescue

        self.base_messages = [
            {"role": "system", "content": INTEGRATION_SYSTEM[lang]},
            *INTEGRATION_DEMOS[lang],
        ]

    def _match_to_candidates(
        self, generated: list[Triple], candidate_ids: list[int]
    ) -> list[int]:
        """Khớp triple do LLM sinh ra về đúng triple ứng viên để lấy lại provenance.

        LLM thường chép lại triple có sai lệch nhỏ (thiếu dấu, đổi hoa thường), nên
        phải khớp mờ. Repo gốc dùng difflib với cutoff=0.0 — luôn khớp được một cái
        gì đó, kể cả khi triple sinh ra hoàn toàn bịa. Ở đây đặt ngưỡng 0.6 để
        triple bịa bị loại thay vì kéo theo provenance sai.
        """
        candidate_texts = [str(list(self.retriever.triple_of(i).triple)) for i in candidate_ids]
        matched: list[int] = []
        for triple in generated:
            close = difflib.get_close_matches(str(list(triple)), candidate_texts, n=1, cutoff=0.6)
            if close:
                idx = candidate_texts.index(close[0])
                if candidate_ids[idx] not in matched:
                    matched.append(candidate_ids[idx])
        return matched

    def _official_rescue(
        self,
        matched_ids: list[int],
        thought: str,
        candidate_ids: list[int],
        accumulated: dict[Triple, None],
    ) -> list[int]:
        """Bổ sung core triple theo đúng `rerank_facts_gene_data` của repo gốc.

        Mô hình nhỏ chưa chưng cất hay nêu đúng triple bắc cầu trong phần suy luận, kiểu
        `(["a day of roses in august", "directed by", "max mack"])`, rồi vẫn ghi
        `{"fact": []}`. Đo trên 10 câu 2WikiMQA với Qwen2.5-7B, 6-8 câu rỗng ở vòng đầu, và
        mất triple bắc cầu thì mọi mức của ACMG đều thiếu bằng chứng. Repo gốc cứu bằng
        hai bước, giữ nguyên thứ tự ở đây:
          1. Tập lõi rỗng: lấy triple ứng viên xếp hạng cao nhất chưa có trong tập tích luỹ.
          2. Mọi cụm `[...]` trong phần suy luận được khớp mờ về triple ứng viên gần nhất
             (cutoff 0.0 như bản gốc) và thêm vào tập lõi.
        """
        ids = list(matched_ids)
        if not ids:
            for cid in candidate_ids:
                if self.retriever.triple_of(cid).triple not in accumulated:
                    ids.append(cid)
                    break
        candidate_texts = [str(tuple(self.retriever.triple_of(i).triple)) for i in candidate_ids]
        for mention in re.findall(r"\[(.*?)\]", thought):
            close = difflib.get_close_matches(mention, candidate_texts, n=1, cutoff=0.0)
            if close:
                cid = candidate_ids[candidate_texts.index(close[0])]
                if cid not in ids:
                    ids.append(cid)
        return ids

    def run(self, question: str, allowed_titles: set[str] | None = None) -> ICIResult:
        result = ICIResult(question=question)
        query = question
        history: list[dict[str, str]] = []

        # Dùng dict để giữ thứ tự xuất hiện mà vẫn khử trùng lặp.
        acc_triples: dict[Triple, None] = {}
        acc_sentences: dict[str, None] = {}
        acc_passages: dict[str, None] = {}
        all_retrieved: dict[str, None] = {}

        for step in range(1, self.max_iterations + 1):
            # ---------------- Construction Phase ----------------
            passage_indices, candidate_ids = self.retriever.get_candidates(
                query,
                top_k=self.top_k,
                top_n=self.top_n,
                allowed_titles=allowed_titles,
                use_reranker=self.use_reranker,
            )
            candidate_triples = [self.retriever.triple_of(i).triple for i in candidate_ids]
            for idx in passage_indices:
                all_retrieved[self.retriever.corpus[idx].full_text] = None

            if not candidate_triples:
                result.terminated_early = True
                break

            # ---------------- Integration Phase -----------------
            fact_before_filter = json.dumps(
                {"fact": [list(t) for t in candidate_triples]}, ensure_ascii=False
            )
            if history:
                user_content = INPUT_TEMPLATE_FOLLOWUP.format(
                    fact_before_filter=fact_before_filter
                )
            else:
                user_content = INPUT_TEMPLATE_START.format(
                    question=question, fact_before_filter=fact_before_filter
                )

            messages = deepcopy(self.base_messages) + history + [
                {"role": "user", "content": user_content}
            ]
            response, _, _ = self.llm.infer(messages, max_tokens=1024)
            core_triples, next_query, thought = parse_integration_output(response)

            matched_ids = self._match_to_candidates(core_triples, candidate_ids)[
                : self.max_core_triples
            ]
            if self.official_rescue:
                matched_ids = self._official_rescue(
                    matched_ids, thought, candidate_ids, acc_triples
                )

            # ------- Chiếu core triple về câu và đoạn nguồn (Eq. 3-4) -------
            for triple_id in matched_ids:
                record = self.retriever.triple_of(triple_id)
                acc_triples[record.triple] = None
                for sentence in self.retriever.sentences_of(triple_id):
                    acc_sentences[sentence] = None
                passage = self.retriever.passage_of(triple_id)
                if passage is not None:
                    acc_passages[passage.full_text] = None

            result.logs.append(
                IterationLog(
                    step=step,
                    query=query,
                    candidate_triples=candidate_triples,
                    core_triples=[self.retriever.triple_of(i).triple for i in matched_ids],
                    thought=thought,
                    next_query=next_query,
                    retrieved_titles=[self.retriever.titles[i] for i in passage_indices],
                    raw_response=response,
                    integration_input=user_content,
                )
            )
            result.num_steps = step

            # Lịch sử H<t: nối cặp (đầu vào, đầu ra) của vòng này vào hội thoại.
            history.append({"role": "user", "content": user_content})
            history.append({"role": "assistant", "content": response})

            if not next_query or NO_QUESTION in next_query or next_query.strip() == "":
                result.terminated_early = True
                break
            query = next_query

        result.core_triples = list(acc_triples.keys())
        result.core_sentences = list(acc_sentences.keys())
        result.core_passages = list(acc_passages.keys())
        result.retrieved_passages = list(all_retrieved.keys())
        return result
