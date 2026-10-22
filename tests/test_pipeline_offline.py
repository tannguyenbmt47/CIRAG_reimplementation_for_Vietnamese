"""Integration test cho ICI + ACMG dùng LLM giả và encoder giả — không gọi API.

Mục tiêu: kiểm chứng phần pipeline mà unit test không chạm tới:
  - vòng lặp ICI dừng đúng lúc gặp <no question> và đúng khi hết L vòng,
  - lịch sử H<t được nối đúng vào hội thoại,
  - core triple được chiếu ngược về câu và đoạn nguồn (Eq. 3-4),
  - ACMG mở rộng ngữ cảnh đúng thứ tự triple -> câu -> đoạn và dừng ở mức đầu tiên đủ.

Chạy: .venv/bin/python tests/test_pipeline_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vicirag.acmg import AdaptiveCascadedGeneration  # noqa: E402
from vicirag.data import Passage, TripleRecord  # noqa: E402
from vicirag.ici import IterativeConstructionIntegration  # noqa: E402
from vicirag.retriever import Retriever  # noqa: E402

# --------------------------------------------------------------- đồ giả (fake)


class FakeEncoder:
    """Encoder tất định: vector đếm ký tự, đủ để kiểm tra thứ tự xếp hạng."""

    DIM = 64

    def encode(self, texts, batch_size=None, show_progress=False):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in text.lower():
                out[i, ord(ch) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    def encode_queries(self, queries, **kwargs):
        return self.encode(queries)


class ScriptedLLM:
    """LLM giả trả lời theo kịch bản định sẵn, có ghi lại lời gọi."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def infer(self, messages, **kwargs):
        self.calls.append(messages)
        response = self.responses.pop(0) if self.responses else ""
        return response, {"prompt_tokens": 0, "completion_tokens": 0}, False


# ------------------------------------------------------------------- dữ liệu

CORPUS = [
    Passage(
        title="Truyện Kiều",
        sentences=[
            "Truyện Kiều là truyện thơ Nôm của Nguyễn Du.",
            "Tác phẩm gồm 3254 câu thơ lục bát.",
        ],
        text="Truyện Kiều là truyện thơ Nôm của Nguyễn Du. Tác phẩm gồm 3254 câu thơ lục bát.",
    ),
    Passage(
        title="Nguyễn Du",
        sentences=[
            "Nguyễn Du sinh năm 1765 tại Thăng Long.",
            "Ông mất năm 1820.",
        ],
        text="Nguyễn Du sinh năm 1765 tại Thăng Long. Ông mất năm 1820.",
    ),
]

TRIPLES = {
    "Truyện Kiều": [
        TripleRecord(("Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"), "Truyện Kiều", [0]),
        TripleRecord(("Truyện Kiều", "có", "3254 câu thơ lục bát"), "Truyện Kiều", [1]),
    ],
    "Nguyễn Du": [
        TripleRecord(("Nguyễn Du", "sinh năm", "1765"), "Nguyễn Du", [0]),
        TripleRecord(("Nguyễn Du", "mất năm", "1820"), "Nguyễn Du", [1]),
    ],
}


def build_retriever() -> Retriever:
    return Retriever(CORPUS, TRIPLES, FakeEncoder())


def step(thought: str, facts: str, question: str) -> str:
    return (
        f"[[ ## thought ## ]]\n{thought}\n\n"
        f"[[ ## fact_after_filter ## ]]\n{facts}\n\n"
        f"[[ ## question ## ]]\n{question}"
    )


# --------------------------------------------------------------------- tests


def test_ici_two_hop_then_terminate():
    """Hai vòng: vòng 1 xác định tác giả, vòng 2 tìm năm sinh rồi dừng."""
    llm = ScriptedLLM(
        [
            step(
                "Cần biết tác giả.",
                '{"fact": [["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"]]}',
                "Nguyễn Du sinh năm nào?",
            ),
            step(
                "Đã có năm sinh.",
                '{"fact": [["Nguyễn Du", "sinh năm", "1765"]]}',
                "<no question>",
            ),
        ]
    )
    ici = IterativeConstructionIntegration(build_retriever(), llm, max_iterations=4)
    result = ici.run("Tác giả của Truyện Kiều sinh năm nào?")

    assert result.num_steps == 2, f"kỳ vọng dừng sau 2 vòng, thực tế {result.num_steps}"
    assert result.terminated_early is True
    assert ("Truyện Kiều", "được sáng tác bởi", "Nguyễn Du") in result.core_triples
    assert ("Nguyễn Du", "sinh năm", "1765") in result.core_triples


def test_provenance_projection():
    """Core triple phải kéo theo đúng câu nguồn (M^S) và đoạn nguồn (M^D)."""
    llm = ScriptedLLM(
        [step("x", '{"fact": [["Nguyễn Du", "sinh năm", "1765"]]}', "<no question>")]
    )
    result = IterativeConstructionIntegration(build_retriever(), llm).run("Nguyễn Du sinh năm nào?")

    assert "Nguyễn Du sinh năm 1765 tại Thăng Long." in result.core_sentences
    assert any("Nguyễn Du" in p for p in result.core_passages)
    # Chỉ triple được chọn mới kéo câu của nó vào, câu của triple khác không lọt vào.
    assert "Ông mất năm 1820." not in result.core_sentences


def test_ici_respects_max_iterations():
    """Model không bao giờ nói <no question> -> phải bị chặn bởi L."""
    never_stops = [step("tiếp", '{"fact": []}', "câu hỏi tiếp theo") for _ in range(10)]
    ici = IterativeConstructionIntegration(build_retriever(), ScriptedLLM(never_stops),
                                           max_iterations=3)
    result = ici.run("Câu hỏi gì đó?")
    assert result.num_steps == 3


def test_history_accumulates_in_conversation():
    """H<t phải được nối vào hội thoại: lời gọi vòng 2 dài hơn vòng 1."""
    llm = ScriptedLLM(
        [
            step("vòng 1", '{"fact": []}', "câu hỏi hop 2"),
            step("vòng 2", '{"fact": []}', "<no question>"),
        ]
    )
    IterativeConstructionIntegration(build_retriever(), llm).run("Câu hỏi gốc?")

    assert len(llm.calls) == 2
    assert len(llm.calls[1]) == len(llm.calls[0]) + 2, "phải thêm đúng 1 cặp user/assistant"
    # Vòng đầu mang truy vấn gốc; các vòng sau chỉ mang tập dữ kiện mới.
    assert "[[ ## query ## ]]" in llm.calls[0][-1]["content"]
    assert "[[ ## query ## ]]" not in llm.calls[1][-1]["content"]


def test_hallucinated_triple_is_rejected():
    """Triple LLM bịa ra (không có trong ứng viên) không được nhận."""
    llm = ScriptedLLM(
        [step("x", '{"fact": [["Paris", "là thủ đô của", "Pháp"]]}', "<no question>")]
    )
    result = IterativeConstructionIntegration(build_retriever(), llm).run("Câu hỏi?")
    assert result.core_triples == [], f"đã nhận nhầm triple bịa: {result.core_triples}"


def test_official_rescue_recovers_triple_cited_in_thought():
    """Mô hình trích triple trong suy luận nhưng ghi danh sách rỗng: bản gốc vớt lại được."""
    response = step(
        'Tác giả là Nguyễn Du (["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"]).',
        '{"fact": []}',
        "<no question>",
    )
    plain = IterativeConstructionIntegration(build_retriever(), ScriptedLLM([response]))
    assert plain.run("Ai viết Truyện Kiều?").core_triples == []

    rescued = IterativeConstructionIntegration(
        build_retriever(), ScriptedLLM([response]), official_rescue=True
    ).run("Ai viết Truyện Kiều?")
    assert ("Truyện Kiều", "được sáng tác bởi", "Nguyễn Du") in rescued.core_triples


def test_official_rescue_falls_back_to_top_candidate():
    """Tập lõi rỗng và suy luận không trích gì: lấy đúng một ứng viên xếp hạng cao nhất."""
    llm = ScriptedLLM([step("Không thấy gì.", '{"fact": []}', "<no question>")])
    result = IterativeConstructionIntegration(
        build_retriever(), llm, official_rescue=True
    ).run("Nguyễn Du sinh năm nào?")
    assert len(result.core_triples) == 1


# ------------------------------------------------------------------- ACMG


class _StubICI:
    def __init__(self, triples, sentences, passages, retrieved=None):
        self.core_triples = triples
        self.core_sentences = sentences
        self.core_passages = passages
        self.retrieved_passages = retrieved or []


FULL_CTX = _StubICI(
    [("Nguyễn Du", "sinh năm", "1765")],
    ["Nguyễn Du sinh năm 1765 tại Thăng Long."],
    ["Nguyễn Du\nNguyễn Du sinh năm 1765 tại Thăng Long."],
)


def test_acmg_stops_at_first_sufficient_level():
    """Mức triple đã đủ -> chỉ tốn 2 lời gọi (trả lời + reanswer), không mở rộng ngữ cảnh."""
    llm = ScriptedLLM(["Suy luận: có dữ kiện.\nĐáp án: 1765", "1765"])
    result = AdaptiveCascadedGeneration(llm).run("Nguyễn Du sinh năm nào?", FULL_CTX)

    assert result.granularity == "fact"
    assert result.answer == "1765"
    assert len(llm.calls) == 2


def test_acmg_escalates_on_refusal():
    """Mức triple từ chối -> phải leo lên mức câu."""
    llm = ScriptedLLM(
        [
            "Suy luận: thiếu dữ kiện.\nĐáp án: Không có câu trả lời",
            "Suy luận: câu văn có nói.\nĐáp án: 1765",
            "1765",
        ]
    )
    result = AdaptiveCascadedGeneration(llm).run("Nguyễn Du sinh năm nào?", FULL_CTX)

    assert result.granularity == "sentence"
    assert result.answer == "1765"
    assert "fact" in result.attempts and "sentence" in result.attempts


def test_acmg_falls_back_to_default():
    """Mọi mức đều từ chối và không còn đoạn dự phòng -> nhãn 'default'."""
    refusal = "Đáp án: Không có câu trả lời"
    result = AdaptiveCascadedGeneration(ScriptedLLM([refusal] * 4)).run("Câu hỏi?", FULL_CTX)
    assert result.granularity == "default"


def test_acmg_retrieved_rescue_when_core_context_incomplete():
    """Ba mức đều từ chối vì core triple thiếu một vế -> cứu bằng toàn bộ đoạn đã truy hồi.

    Tái hiện dạng lỗi chiếm 71/73 ca 'default' trên ViMQA: câu hỏi so sánh hai thực
    thể, retriever đã lấy về cả hai đoạn nhưng integration chỉ giữ triple của một bên.
    """
    ctx = _StubICI(
        [("Roy Makaay", "sinh năm", "1975")],
        ["Roy Makaay sinh năm 1975."],
        ["Roy Makaay\nRoy Makaay sinh năm 1975."],
        retrieved=[
            "Roy Makaay\nRoy Makaay sinh năm 1975.",
            "Gary Lineker\nGary Lineker sinh năm 1960.",
        ],
    )
    refusal = "Đáp án: Không có câu trả lời"
    llm = ScriptedLLM([refusal, refusal, refusal,
                       "Suy luận: Makaay 1975, Lineker 1960.\nĐáp án: đúng", "đúng"])
    result = AdaptiveCascadedGeneration(llm).run("Roy Makaay nhỏ tuổi hơn Gary Lineker?", ctx)

    assert result.granularity == "retrieved"
    assert result.answer == "đúng"


def test_acmg_empty_context_counts_as_refusal():
    """Ngữ cảnh rỗng ở một mức không được tính là câu trả lời hợp lệ."""
    empty = _StubICI([], [], ["Nguyễn Du\nNguyễn Du sinh năm 1765."])
    llm = ScriptedLLM(["Suy luận: đoạn văn có nói.\nĐáp án: 1765", "1765"])
    result = AdaptiveCascadedGeneration(llm).run("Nguyễn Du sinh năm nào?", empty)

    assert result.granularity == "passage"
    # Hai mức đầu ngữ cảnh rỗng nên bị bỏ qua mà không tốn lời gọi nào.
    assert len(llm.calls) == 2


def test_acmg_passage_fallback_when_no_core_triples():
    """Không có core triple nào -> phải lùi về các đoạn đã truy hồi, không từ chối thẳng."""
    empty = _StubICI([], [], [], retrieved=["Guinea\nGuinea là thành viên của IMF và FAO."])
    llm = ScriptedLLM(["Suy luận: đoạn văn có nói.\nĐáp án: IMF và FAO", "IMF và FAO"])
    result = AdaptiveCascadedGeneration(llm).run("Guinea là thành viên tổ chức nào?", empty)

    assert result.granularity == "passage"
    assert "IMF" in result.answer


def test_acmg_fallback_can_be_disabled():
    """Tắt phương án chót -> quay lại hành vi bám sát paper."""
    empty = _StubICI([], [], [], retrieved=["Guinea\nGuinea là thành viên của IMF."])
    acmg = AdaptiveCascadedGeneration(ScriptedLLM([]), passage_fallback=False)
    assert acmg.run("Câu hỏi?", empty).granularity == "default"


def test_acmg_min_source_docs_skips_narrow_levels():
    """Triple lõi chỉ từ MỘT đoạn -> bỏ qua mức triple và mức câu, đọc thẳng mức đoạn.

    Đây là điều kiện đủ theo số nguồn thay cho Eq. 6. Mô hình vẫn trả lời bình thường
    ở mức triple (không hề từ chối), nên Eq. 6 gốc sẽ chốt luôn ở đó. Câu hỏi đa bước
    cần bằng chứng từ hai tài liệu, mà tập lõi chỉ đến từ một, nên mức triple bị
    coi là chưa đủ dù mô hình sẵn lòng trả lời.
    """
    llm = ScriptedLLM(["Suy luận: từ đoạn văn.\nĐáp án: 1765", "1765"])
    acmg = AdaptiveCascadedGeneration(llm, min_source_docs=2)
    result = acmg.run("Nguyễn Du sinh năm nào?", FULL_CTX)  # FULL_CTX chỉ có 1 đoạn

    assert result.granularity == "passage", result.granularity
    assert "fact" not in result.attempts
    assert len(llm.calls) == 2  # không tốn thêm lời gọi nào so với dừng ở mức triple


def test_acmg_min_source_docs_keeps_cascade_when_evidence_spans_enough():
    """Triple lõi đến từ hai đoạn -> điều kiện thoả, cascade chạy như cũ."""
    ctx = _StubICI(
        [("Nguyễn Du", "sinh năm", "1765"), ("Nguyễn Du", "quê ở", "Hà Tĩnh")],
        ["Nguyễn Du sinh năm 1765.", "Nguyễn Du quê ở Hà Tĩnh."],
        ["Nguyễn Du\nNguyễn Du sinh năm 1765.", "Hà Tĩnh\nNguyễn Du quê ở Hà Tĩnh."],
    )
    llm = ScriptedLLM(["Suy luận: có dữ kiện.\nĐáp án: 1765", "1765"])
    result = AdaptiveCascadedGeneration(llm, min_source_docs=2).run("Sinh năm nào?", ctx)
    assert result.granularity == "fact"


def test_acmg_min_source_docs_off_by_default():
    """Mặc định tắt, giữ đúng Eq. 6 của bài báo để bản tái lập không bị đổi hành vi."""
    llm = ScriptedLLM(["Suy luận: có dữ kiện.\nĐáp án: 1765", "1765"])
    assert AdaptiveCascadedGeneration(llm).run("Câu hỏi?", FULL_CTX).granularity == "fact"


def test_acmg_single_granularity_ablation():
    """Cấu hình cascade=('fact',) tái lập ablation một mức trong Table 3."""
    llm = ScriptedLLM(["Suy luận: x.\nĐáp án: 1765", "1765"])
    acmg = AdaptiveCascadedGeneration(llm, cascade=("fact",))
    assert acmg.run("Câu hỏi?", FULL_CTX).granularity == "fact"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                import traceback

                failed += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    print(f"\n{'THẤT BẠI: ' + str(failed) if failed else 'TẤT CẢ ĐỀU PASS'}")
    raise SystemExit(1 if failed else 0)
