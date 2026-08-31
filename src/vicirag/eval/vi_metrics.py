"""EM / F1 chuẩn hoá cho tiếng Việt.

Hàm `normalize_answer` chuẩn (MRQA / HotpotQA / MuSiQue, cũng là hàm mà repo gốc
dùng trong `src/cirag/utils/eval_utils.py`) loại bỏ mạo từ tiếng Anh a/an/the.
Với tiếng Việt việc đó vô nghĩa, trong khi tiếng Việt lại có ba vấn đề riêng:

1. **Mã hoá dấu.** "Việt" có thể được mã hoá dạng dựng sẵn (NFC) hoặc tổ hợp
   (NFD). Hai chuỗi hiển thị giống hệt nhau nhưng khác byte -> EM sai âm tính.
2. **Hư từ.** Vai trò tương tự mạo từ tiếng Anh: "là", "của", "một", "các",
   "những", "ở"... Mô hình hay trả "là năm 1824" trong khi đáp án vàng là "1824".
3. **Ranh giới từ.** Tiếng Việt không đánh dấu ranh giới từ bằng khoảng trắng:
   "Hà Nội" là MỘT từ gồm hai âm tiết. Tính F1 theo âm tiết hay theo từ cho ra
   hai con số khác nhau, nên báo cáo cả hai.

Riêng câu hỏi dạng đúng/sai của ViMQA, đáp án vàng là "đúng"/"không" còn mô hình
hay trả "có"/"phải"/"sai" -> cần quy về cùng một dạng chuẩn.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

# Hư từ tiếng Việt, vai trò tương đương mạo từ tiếng Anh trong hàm chuẩn hoá gốc.
VIETNAMESE_FUNCTION_WORDS = frozenset(
    {"là", "của", "một", "các", "những", "ở", "tại", "vào", "và", "có", "được", "thì", "mà"}
)

#  Chuẩn SQuAD/MuSiQue mà bài báo gốc theo có bỏ mạo từ trước khi so khớp. Không bỏ
#  thì EM của ta khắt khe hơn của họ, và phép tái lập sẽ cho số thấp một cách giả tạo.
ENGLISH_ARTICLES = frozenset({"a", "an", "the"})

_YES = frozenset({"đúng", "có", "phải", "vâng", "yes", "true", "đúng vậy", "chính xác"})
_NO = frozenset({"không", "sai", "no", "false", "không phải", "không đúng"})

_PUNCT = set(string.punctuation) | {"“", "”", "‘", "’", "–", "—", "…", "«", "»"}


def _strip_punct(text: str) -> str:
    return "".join(" " if ch in _PUNCT else ch for ch in text)


def normalize_answer(
    text: str,
    remove_function_words: bool = True,
    canonicalize_yes_no: bool = True,
    lang: str = "vi",
) -> str:
    """Chuẩn hoá đáp án trước khi so khớp.

    Với ``lang="en"`` thì bỏ mạo từ theo chuẩn SQuAD/MuSiQue; với tiếng Việt thì bỏ
    hư từ, vì tiếng Việt không có mạo từ tương đương.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _strip_punct(text)
    text = re.sub(r"\s+", " ", text).strip()

    if canonicalize_yes_no:
        if text in _YES:
            return "đúng"
        if text in _NO:
            return "không"

    if remove_function_words:
        stop = ENGLISH_ARTICLES if lang == "en" else VIETNAMESE_FUNCTION_WORDS
        tokens = [t for t in text.split() if t not in stop]
        # Nếu đáp án CHỈ gồm hư từ (vd đáp án vàng đúng là "có") thì giữ nguyên,
        # bỏ hết sẽ thành chuỗi rỗng và khớp bừa với mọi đáp án rỗng khác.
        if tokens:
            text = " ".join(tokens)

    return re.sub(r"\s+", " ", text).strip()


def tokenize_syllables(text: str) -> list[str]:
    """Tách theo âm tiết — cách tách mặc định, tương đương split() của bản gốc."""
    return text.split()


def tokenize_words(text: str) -> list[str]:
    """Tách theo từ bằng underthesea; lùi về âm tiết nếu thư viện không có."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").split()
    except Exception:
        return text.split()


def exact_match(prediction: str, gold: str, **kwargs) -> float:
    return float(normalize_answer(prediction, **kwargs) == normalize_answer(gold, **kwargs))


def f1_score(prediction: str, gold: str, word_level: bool = False, **kwargs) -> float:
    """F1 theo token, công thức giống MuSiQue mà paper dùng."""
    tokenizer = tokenize_words if word_level else tokenize_syllables
    pred_tokens = tokenizer(normalize_answer(prediction, **kwargs))
    gold_tokens = tokenizer(normalize_answer(gold, **kwargs))

    if not pred_tokens or not gold_tokens:
        # Quy ước chuẩn: cả hai rỗng thì coi như khớp hoàn toàn.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(
    predictions: list[str], gold_answers: list[list[str]], **kwargs
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Tính EM/F1 tổng hợp; nhiều đáp án vàng thì lấy max, giống bản gốc."""
    assert len(predictions) == len(gold_answers), "Số dự đoán và số đáp án vàng phải bằng nhau"

    per_example = []
    for pred, golds in zip(predictions, gold_answers):
        golds = golds or [""]
        per_example.append(
            {
                "em": max(exact_match(pred, g, **kwargs) for g in golds),
                "f1": max(f1_score(pred, g, word_level=False, **kwargs) for g in golds),
                "f1_word": max(f1_score(pred, g, word_level=True, **kwargs) for g in golds),
            }
        )

    n = len(per_example) or 1
    overall = {
        key: round(100 * sum(e[key] for e in per_example) / n, 2)
        for key in ("em", "f1", "f1_word")
    }
    return overall, per_example


def retrieval_recall(retrieved_titles: list[str], gold_titles: set[str]) -> float:
    """Tỉ lệ đoạn vàng nằm trong danh sách đã truy hồi — dùng cho phân tích lỗi."""
    if not gold_titles:
        return 0.0
    return len(set(retrieved_titles) & gold_titles) / len(gold_titles)
