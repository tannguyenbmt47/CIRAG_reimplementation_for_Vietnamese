"""Nạp dữ liệu và các kiểu dữ liệu dùng chung."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

Triple = tuple[str, str, str]


@dataclass
class Passage:
    """Một đoạn văn trong corpus truy hồi."""

    title: str
    sentences: list[str]
    text: str

    @property
    def full_text(self) -> str:
        """Dạng đưa vào embedding và vào prompt: tiêu đề + nội dung."""
        return f"{self.title}\n{self.text}"


@dataclass
class Sample:
    """Một câu hỏi kèm bộ đoạn văn gốc của nó."""

    id: str
    question: str
    answer: str
    type: str
    paragraphs: list[dict]
    supporting_facts: list[list]

    @property
    def gold_answers(self) -> list[str]:
        return [self.answer]

    @property
    def gold_titles(self) -> set[str]:
        return {t for t, _ in self.supporting_facts}

    @property
    def own_titles(self) -> list[str]:
        return [p["title"] for p in self.paragraphs]

    @property
    def own_passage_texts(self) -> list[str]:
        return [" ".join(p["sentences"]) for p in self.paragraphs]


@dataclass
class TripleRecord:
    """Một triple kèm nguồn gốc — hiện thực hai ánh xạ M^D và M^S (Eq. 3-4).

    `passage_title` là M^D (triple -> tài liệu), `sentence_ids` là M^S
    (triple -> các câu trong tài liệu đó). Giữ được provenance chính là điều kiện
    để ACMG mở rộng ngữ cảnh từ mức triple lên mức câu rồi mức đoạn.
    """

    triple: Triple
    passage_title: str
    sentence_ids: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Chuỗi hoá triple để tính embedding và đưa vào prompt."""
        return " ".join(self.triple)

    @property
    def key(self) -> str:
        return f"{self.passage_title}||{self.text}"


def load_samples(path: str | Path) -> list[Sample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Sample(
            id=d["id"],
            question=d["question"],
            answer=d.get("answer", ""),
            type=d.get("type", ""),
            paragraphs=d["paragraphs"],
            supporting_facts=d.get("supporting_facts", []),
        )
        for d in data
    ]


def load_corpus(path: str | Path) -> list[Passage]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Passage(title=d["title"], sentences=d["sentences"], text=d["text"]) for d in data]
