"""Construction Phase: trích xuất triple tri thức từ corpus (Sec 3.3 của paper).

Chạy OFFLINE một lần cho toàn corpus rồi cache lại. Đây chính là điểm mà paper
nhấn mạnh ở Sec 4.6: nhờ tiền tính triple, độ trễ khi suy luận giảm mạnh so với
biến thể CIRAG (Online) trích triple ngay trong vòng lặp.

Mỗi đoạn văn tốn 2 lời gọi LLM: NER (Figure 9) rồi trích triple có điều kiện trên
danh sách thực thể (Figure 10).
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from .data import Passage, TripleRecord
from .prompts_vi.openie import NER_PROMPT, TRIPLE_FRAME, TRIPLE_PROMPT

if TYPE_CHECKING:
    from .llm.openrouter import OpenRouterLLM


def _fill(template: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    """Thay ${placeholder} ở lượt user cuối cùng của prompt."""
    messages = [dict(m) for m in template]
    messages[-1]["content"] = Template(messages[-1]["content"]).safe_substitute(**kwargs)
    return messages


def _extract_json(text: str) -> dict:
    """Bóc đối tượng JSON đầu tiên trong phản hồi của LLM.

    Model hay bọc JSON trong ```json ... ``` hoặc thêm lời dẫn, nên không thể
    json.loads thẳng.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower().strip()


def map_triple_to_sentences(triple: tuple[str, str, str], sentences: list[str]) -> list[int]:
    """Ánh xạ M^S: xác định các câu là nguồn của một triple.

    Một câu được coi là nguồn nếu nó chứa chủ thể hoặc đối tượng của triple. Nếu
    không câu nào chứa (do LLM diễn đạt lại thực thể), lùi về câu có độ chồng lấp
    token cao nhất để triple không bao giờ mất provenance — mất provenance đồng
    nghĩa với việc ACMG không mở rộng lên mức câu được.
    """
    subj, _, obj = (_norm(x) for x in triple)
    normalized = [_norm(s) for s in sentences]

    hits = [i for i, s in enumerate(normalized) if (subj and subj in s) or (obj and obj in s)]
    if hits:
        return hits

    subj_obj_tokens = set(subj.split()) | set(obj.split())
    if not subj_obj_tokens:
        return []
    scores = [
        (len(subj_obj_tokens & set(s.split())) / len(subj_obj_tokens), i)
        for i, s in enumerate(normalized)
    ]
    best_score, best_idx = max(scores)
    return [best_idx] if best_score > 0 else []


class OpenIEExtractor:
    """Trích triple cho toàn corpus, có cache trên đĩa."""

    def __init__(
        self,
        llm: "OpenRouterLLM",
        lang: str = "vi",
        cache_path: str | Path | None = None,
        num_workers: int = 4,
    ) -> None:
        self.llm = llm
        self.lang = lang
        self.cache_path = Path(cache_path) if cache_path else None
        self.num_workers = num_workers

    def extract_passage(self, passage: Passage) -> list[TripleRecord]:
        """Trích triple cho một đoạn: NER -> triple có điều kiện -> gắn provenance."""
        ner_messages = _fill(NER_PROMPT[self.lang], passage=passage.full_text)
        ner_response, _, _ = self.llm.infer(ner_messages)
        ner_json = _extract_json(ner_response)
        entities = [str(e) for e in ner_json.get("named_entities", []) if str(e).strip()]

        entity_json = json.dumps({"named_entities": entities}, ensure_ascii=False)
        triple_input = TRIPLE_FRAME[self.lang].format(
            passage=passage.full_text, named_entity_json=entity_json
        )
        triple_messages = _fill(TRIPLE_PROMPT[self.lang], triple_input=triple_input)
        triple_response, _, _ = self.llm.infer(triple_messages)
        triple_json = _extract_json(triple_response)

        records: list[TripleRecord] = []
        seen: set[tuple[str, str, str]] = set()
        for item in triple_json.get("triples", []):
            if not isinstance(item, list) or len(item) != 3:
                continue
            triple = tuple(unicodedata.normalize("NFC", str(x)).strip() for x in item)
            if not all(triple) or triple in seen:
                continue
            seen.add(triple)
            records.append(
                TripleRecord(
                    triple=triple,  # type: ignore[arg-type]
                    passage_title=passage.title,
                    sentence_ids=map_triple_to_sentences(triple, passage.sentences),
                )
            )
        return records

    def run(self, corpus: list[Passage], force: bool = False) -> dict[str, list[TripleRecord]]:
        """Trích triple cho toàn corpus, trả về ánh xạ tiêu đề -> danh sách triple."""
        if self.cache_path and self.cache_path.exists() and not force:
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            titles = {p.title for p in corpus}
            if titles.issubset(cached.keys()):
                print(f"[OpenIE] Dùng cache: {self.cache_path}")
                return {
                    title: [
                        TripleRecord(
                            triple=tuple(r["triple"]),  # type: ignore[arg-type]
                            passage_title=title,
                            sentence_ids=r["sentence_ids"],
                        )
                        for r in records
                    ]
                    for title, records in cached.items()
                    if title in titles
                }

        from tqdm import tqdm

        results: dict[str, list[TripleRecord]] = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            for passage, records in tqdm(
                zip(corpus, pool.map(self.extract_passage, corpus)),
                total=len(corpus),
                desc="OpenIE (trích triple)",
            ):
                results[passage.title] = records

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        title: [
                            {"triple": list(r.triple), "sentence_ids": r.sentence_ids}
                            for r in records
                        ]
                        for title, records in results.items()
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(f"[OpenIE] Đã lưu cache: {self.cache_path}")

        total = sum(len(v) for v in results.values())
        empty = sum(1 for v in results.values() if not v)
        print(f"[OpenIE] {total} triple từ {len(corpus)} đoạn ({empty} đoạn không ra triple nào)")
        return results
