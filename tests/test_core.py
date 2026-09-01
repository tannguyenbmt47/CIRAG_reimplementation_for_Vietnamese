"""Kiểm thử các thành phần logic thuần (không cần GPU, không gọi API).

Chạy: .venv/bin/python -m pytest tests/ -v      hoặc      python tests/test_core.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vicirag.acmg import extract_answer, is_refusal, reanswer  # noqa: E402
from vicirag.eval.vi_metrics import (  # noqa: E402
    evaluate,
    exact_match,
    f1_score,
    normalize_answer,
)
from vicirag.ici import parse_integration_output  # noqa: E402
from vicirag.data import TripleRecord  # noqa: E402
from vicirag.openie import (  # noqa: E402
    _extract_json,
    map_triple_to_sentences,
    normalize_triples_official,
    official_text_processing,
)


# ------------------------------------------------------------ chuẩn hoá đáp án
def test_normalize_nfc_vs_nfd():
    """Hai cách mã hoá dấu phải quy về cùng một chuỗi."""
    nfc = unicodedata.normalize("NFC", "Việt Nam")
    nfd = unicodedata.normalize("NFD", "Việt Nam")
    assert nfc != nfd, "tiền đề: hai dạng khác nhau về byte"
    assert normalize_answer(nfc) == normalize_answer(nfd)


def test_normalize_removes_function_words():
    assert normalize_answer("là 1824") == normalize_answer("1824")
    assert normalize_answer("của Hà Nội") == normalize_answer("Hà Nội")
    assert normalize_answer("vào 1824") == normalize_answer("1824")


def test_normalize_keeps_content_words():
    """Giới hạn đã biết: 'năm' KHÔNG bị coi là hư từ.

    'năm' vừa là danh từ chỉ thời gian vừa là số đếm (5), xoá đi sẽ làm sai những
    đáp án như "năm cuốn sách". Hệ quả: "năm 1824" vs "1824" trượt EM nhưng vẫn
    được F1 tính điểm một phần, và bước reanswer trong ACMG có nhiệm vụ rút gọn
    đáp án về đúng cụm ngắn.
    """
    assert normalize_answer("năm 1824") != normalize_answer("1824")
    assert 0 < f1_score("năm 1824", "1824") < 1


def test_normalize_keeps_pure_function_word_answer():
    """Đáp án chỉ gồm hư từ thì không được xoá sạch thành chuỗi rỗng."""
    assert normalize_answer("có") != ""
    assert normalize_answer("là") != ""


def test_yes_no_canonicalization():
    assert exact_match("có", "đúng") == 1.0
    assert exact_match("phải", "đúng") == 1.0
    assert exact_match("sai", "không") == 1.0
    assert exact_match("không", "đúng") == 0.0


def test_case_and_punctuation():
    assert exact_match("HÀ NỘI.", "Hà Nội") == 1.0
    assert exact_match("Amy Winehouse", "amy winehouse") == 1.0


def test_f1_partial_overlap():
    score = f1_score("thành phố Hà Nội", "Hà Nội")
    assert 0.0 < score < 1.0
    assert f1_score("Hà Nội", "Hà Nội") == 1.0
    assert f1_score("Sài Gòn", "Hà Nội") == 0.0


def test_f1_word_level_differs_from_syllable():
    """F1 theo từ và theo âm tiết là hai con số khác nhau -> báo cáo cả hai."""
    syllable = f1_score("Hà Nội Việt Nam", "Hà Nội", word_level=False)
    word = f1_score("Hà Nội Việt Nam", "Hà Nội", word_level=True)
    assert syllable > 0 and word > 0


def test_evaluate_aggregates():
    overall, per_example = evaluate(["1824", "Hà Nội"], [["1824"], ["Sài Gòn"]])
    assert overall["em"] == 50.0
    assert len(per_example) == 2


# --------------------------------------------------- parser Integration Phase
INTEGRATION_RESPONSE = """[[ ## thought ## ]]
Truy vấn hỏi về năm sinh của tác giả Truyện Kiều. Dữ kiện cho biết tác giả là Nguyễn Du.

[[ ## fact_after_filter ## ]]
{"fact": [["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"]]}

[[ ## question ## ]]
Nguyễn Du sinh năm nào?"""


def test_parse_integration_output():
    facts, next_query, thought = parse_integration_output(INTEGRATION_RESPONSE)
    assert facts == [("Truyện Kiều", "được sáng tác bởi", "Nguyễn Du")]
    assert next_query == "Nguyễn Du sinh năm nào?"
    assert "Nguyễn Du" in thought


def test_parse_no_question_terminates():
    response = INTEGRATION_RESPONSE.replace("Nguyễn Du sinh năm nào?", "<no question>")
    _, next_query, _ = parse_integration_output(response)
    assert next_query == "<no question>"


def test_parse_empty_fact_list():
    response = """[[ ## thought ## ]]
Không có dữ kiện nào liên quan.

[[ ## fact_after_filter ## ]]
{"fact": []}

[[ ## question ## ]]
Ai là tác giả Truyện Kiều?"""
    facts, next_query, _ = parse_integration_output(response)
    assert facts == []
    assert next_query == "Ai là tác giả Truyện Kiều?"


def test_parse_json_in_code_fence():
    """Model nhỏ hay bọc JSON trong ```json ... ```."""
    response = """[[ ## thought ## ]]
Phân tích.

[[ ## fact_after_filter ## ]]
```json
{"fact": [["Hà Nội", "là thủ đô của", "Việt Nam"]]}
```

[[ ## question ## ]]
<no question>"""
    facts, next_query, _ = parse_integration_output(response)
    assert facts == [("Hà Nội", "là thủ đô của", "Việt Nam")]
    assert next_query == "<no question>"


def test_parse_malformed_does_not_crash():
    facts, next_query, thought = parse_integration_output("hoàn toàn rác, không có header")
    assert facts == [] and next_query == "" and thought == ""


# ------------------------------------------------------------ provenance M^S
def test_map_triple_to_sentences_direct_hit():
    sentences = [
        "Chùa Một Cột là một ngôi chùa nằm ở trung tâm thành phố Hà Nội.",
        "Chùa được vua Lý Thái Tông cho xây dựng vào mùa đông năm 1049.",
    ]
    ids = map_triple_to_sentences(("Chùa Một Cột", "nằm ở", "Hà Nội"), sentences)
    assert 0 in ids


def test_map_triple_to_sentences_fallback_never_loses_provenance():
    """Thực thể bị diễn đạt lại vẫn phải giữ được ít nhất một câu nguồn."""
    sentences = ["Ngôi chùa này được vua Lý Thái Tông cho dựng năm 1049."]
    ids = map_triple_to_sentences(("Chùa Một Cột", "được xây dựng bởi", "Lý Thái Tông"), sentences)
    assert ids == [0]


def test_official_triple_norm_matches_paper_and_keeps_provenance():
    records = {
        "God's Gift to Women": [
            TripleRecord(("God's Gift to Women", "directed by", "Michael Curtiz"),
                         "God's Gift to Women", [0]),
            # Trùng với triple trên sau chuẩn hoá -> bị bỏ.
            TripleRecord(("god's gift to women", "Directed By", "michael curtiz"),
                         "God's Gift to Women", [1]),
            # Thành phần rỗng sau chuẩn hoá -> bị bỏ.
            TripleRecord(("God's Gift to Women", "is", "!!!"), "God's Gift to Women", [0]),
        ]
    }
    out = normalize_triples_official(records)["God's Gift to Women"]
    assert [r.triple for r in out] == [("god s gift to women", "directed by", "michael curtiz")]
    assert out[0].sentence_ids == [0]  # provenance tính từ triple gốc, không đổi
    assert official_text_processing("Aldri annet enn bråk") == "aldri annet enn br k"


def test_map_triple_unrelated_returns_empty():
    sentences = ["Hoàn toàn không liên quan gì cả."]
    ids = map_triple_to_sentences(("Paris", "là thủ đô của", "Pháp"), sentences)
    assert ids == []


# --------------------------------------------------------- ACMG / bóc JSON
def test_is_refusal():
    assert is_refusal("Không có câu trả lời")
    assert is_refusal("Suy luận: thiếu dữ kiện. Đáp án: Không có câu trả lời")
    assert is_refusal("No answer")  # model đôi khi trả tiếng Anh
    assert is_refusal("")
    assert not is_refusal("Đáp án: Hà Nội")


def test_extract_answer():
    assert extract_answer("Suy luận: abc.\nĐáp án: Hà Nội.") == "Hà Nội"
    assert extract_answer("Thought: xyz\nAnswer: 1824") == "1824"
    assert extract_answer("Hà Nội") == "Hà Nội"


def test_reanswer_prompt_follows_data_language():
    class _CaptureLLM:
        def infer(self, messages, **kwargs):
            self.messages = messages
            return "x", {}, False

    llm = _CaptureLLM()
    reanswer(llm, "Who directed it?", "Answer: Michael Curtiz", lang="en")
    assert llm.messages[-1]["content"].startswith("Question: Who directed it?")
    # en-vi = prompt tiếng Anh trên dữ liệu tiếng Việt: đáp án vẫn rút gọn theo tiếng Việt.
    reanswer(llm, "Ai đạo diễn?", "Đáp án: Michael Curtiz", lang="en-vi")
    assert llm.messages[-1]["content"].startswith("Câu hỏi: Ai đạo diễn?")


def test_extract_json_from_fence():
    assert _extract_json('```json\n{"named_entities": ["Hà Nội"]}\n```') == {
        "named_entities": ["Hà Nội"]
    }
    assert _extract_json('Đây là kết quả: {"triples": [["a","b","c"]]}') == {
        "triples": [["a", "b", "c"]]
    }
    assert _extract_json("không có json") == {}


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
                failed += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'THẤT BẠI: ' + str(failed) if failed else 'TẤT CẢ ĐỀU PASS'}")
    raise SystemExit(1 if failed else 0)
