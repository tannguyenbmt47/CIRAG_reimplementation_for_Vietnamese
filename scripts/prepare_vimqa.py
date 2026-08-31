#!/usr/bin/env python3
"""Chuyển ViMQA sang định dạng ViCiRAG: file mẫu + corpus truy hồi.

ViMQA có cấu trúc giống HotpotQA:
    {_id, question, answer, type, context: [[title, [câu...]], ...],
     supporting_facts: [[title, sent_id], ...]}

Script sinh ra hai file:
  1. `{ds}_{split}.json`   — danh sách mẫu, mỗi mẫu giữ nguyên đoạn văn đã tách câu.
  2. `{ds}_{split}_corpus.json` — corpus truy hồi, gộp TOÀN BỘ đoạn supporting và
     distractor của mọi câu hỏi được chọn rồi khử trùng lặp theo tiêu đề. Đây đúng
     là cách dựng cơ sở dữ liệu truy hồi mà paper mô tả (theo setting IRCoT).

Lưu ý quan trọng: KHÔNG tách lại câu. Chỉ số `sent_id` trong `supporting_facts`
tham chiếu trực tiếp vào danh sách câu gốc của ViMQA; tách lại bằng underthesea sẽ
làm lệch toàn bộ nhãn supporting facts.

Hai nguồn dữ liệu, script tự nhận dạng cấu trúc:
  - `--hf nguyenlab/vimqa` : tải thẳng bộ đầy đủ 10.047 mẫu từ HuggingFace (nhóm tác giả
    tự phát hành, giấy phép CC-BY-NC-SA-4.0). Đây là cách nên dùng.
  - `--input <file.json>`  : file JSON theo cấu trúc của repo GitHub nguyenlab/vmqa.

Hai nguồn khác nhau ở cách biểu diễn `context` và `supporting_facts`: bản GitHub dùng danh
sách các cặp `[title, [câu...]]`, bản HuggingFace dùng từ điển các danh sách song song
`{"title": [...], "sentences": [...]}`. Hàm `normalize_raw_sample` quy cả hai về một dạng.

Cách dùng:
    python scripts/prepare_vimqa.py --hf nguyenlab/vimqa --split validation \\
        --dataset vimqa --limit 500
    python scripts/prepare_vimqa.py --input data/vimqa/raw/sample_dev.json \\
        --split dev --dataset vimqa_sample
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "vimqa" / "processed"


def normalize_text(text: str) -> str:
    """Chuẩn hoá Unicode NFC + gọn khoảng trắng.

    Tiếng Việt có hai cách mã hoá dấu (tổ hợp NFD và dựng sẵn NFC). Nếu corpus và
    truy vấn không cùng dạng thì so khớp chuỗi và cả tokenizer đều lệch, nên phải
    chuẩn hoá ngay từ khâu nạp dữ liệu.
    """
    text = unicodedata.normalize("NFC", text)
    # ViMQA tách token sẵn nên hay có khoảng trắng thừa trước dấu câu.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_raw_sample(raw: dict) -> dict:
    """Quy cấu trúc HuggingFace về cấu trúc GitHub.

    HuggingFace lưu context dạng từ điển các danh sách song song
    ``{"title": [...], "sentences": [...]}``; GitHub lưu dạng danh sách các cặp
    ``[[title, [câu...]], ...]``. Tương tự với supporting_facts. Chuẩn về một dạng
    ngay từ đây để phần còn lại của script không phải phân nhánh.
    """
    out = dict(raw)
    out["_id"] = raw.get("_id") or raw.get("id", "")

    context = raw.get("context")
    if isinstance(context, dict):
        out["context"] = list(zip(context["title"], context["sentences"]))

    facts = raw.get("supporting_facts")
    if isinstance(facts, dict):
        out["supporting_facts"] = list(zip(facts["title"], facts["sent_id"]))

    return out


def build_sample(raw: dict) -> dict:
    """Chuyển một mẫu ViMQA sang định dạng nội bộ."""
    raw = normalize_raw_sample(raw)
    supporting = {}
    for title, sent_id in raw.get("supporting_facts", []):
        supporting.setdefault(normalize_text(title), set()).add(int(sent_id))

    paragraphs = []
    for title, sentences in raw["context"]:
        title_n = normalize_text(title)
        sents = [normalize_text(s) for s in sentences]
        paragraphs.append(
            {
                "title": title_n,
                "sentences": sents,
                "text": " ".join(sents),
                "is_supporting": title_n in supporting,
                "supporting_sent_ids": sorted(supporting.get(title_n, set())),
            }
        )

    return {
        "id": raw["_id"],
        "question": normalize_text(raw["question"]),
        "answer": normalize_text(raw.get("answer", "")),
        "type": raw.get("type", ""),
        "paragraphs": paragraphs,
        "supporting_facts": [
            [normalize_text(t), int(i)] for t, i in raw.get("supporting_facts", [])
        ],
    }


def build_corpus(samples: list[dict]) -> list[dict]:
    """Gộp mọi đoạn văn của mọi câu hỏi, khử trùng lặp theo tiêu đề."""
    corpus: dict[str, dict] = {}
    collisions = 0
    for sample in samples:
        for para in sample["paragraphs"]:
            title = para["title"]
            if title in corpus:
                # Cùng tiêu đề nhưng nội dung khác nhau -> giữ bản dài hơn để
                # không mất câu supporting của câu hỏi khác.
                if corpus[title]["text"] != para["text"]:
                    collisions += 1
                    if len(para["sentences"]) > len(corpus[title]["sentences"]):
                        corpus[title] = {
                            "title": title,
                            "sentences": para["sentences"],
                            "text": para["text"],
                        }
                continue
            corpus[title] = {
                "title": title,
                "sentences": para["sentences"],
                "text": para["text"],
            }
    if collisions:
        print(f"  [!] {collisions} đoạn trùng tiêu đề nhưng khác nội dung (đã giữ bản dài hơn)")
    return list(corpus.values())


def compute_stats(samples: list[dict], corpus: list[dict]) -> dict:
    """Thống kê phục vụ mục Dataset của báo cáo."""
    n_sup = [len(s["supporting_facts"]) for s in samples]
    para_lens = [len(p["sentences"]) for s in samples for p in s["paragraphs"]]
    q_lens = [len(s["question"].split()) for s in samples]
    a_lens = [len(s["answer"].split()) for s in samples]

    yes_no = sum(1 for s in samples if s["answer"].lower() in {"đúng", "không", "có", "sai"})
    # Đáp án có xuất hiện nguyên văn trong câu supporting hay không -> cho biết
    # bài toán là trích rút (extractive) hay phải suy luận/diễn đạt lại.
    extractive = 0
    for s in samples:
        sup_text = " ".join(
            p["sentences"][i]
            for p in s["paragraphs"]
            for i in p["supporting_sent_ids"]
            if i < len(p["sentences"])
        ).lower()
        if s["answer"] and s["answer"].lower() in sup_text:
            extractive += 1

    def avg(xs: list) -> float:
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "num_questions": len(samples),
        "num_corpus_passages": len(corpus),
        "num_corpus_sentences": sum(len(c["sentences"]) for c in corpus),
        "question_types": dict(Counter(s["type"] for s in samples)),
        "avg_supporting_facts": avg(n_sup),
        "avg_paragraphs_per_question": avg([len(s["paragraphs"]) for s in samples]),
        "avg_sentences_per_paragraph": avg(para_lens),
        "avg_question_len_tokens": avg(q_lens),
        "avg_answer_len_tokens": avg(a_lens),
        "num_yes_no_answers": yes_no,
        "pct_answer_in_supporting": round(100 * extractive / len(samples), 1) if samples else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf", metavar="REPO", help="Nạp từ HuggingFace, vd nguyenlab/vimqa")
    source.add_argument("--input", type=Path, help="File ViMQA gốc (.json) từ repo GitHub")
    parser.add_argument("--split", required=True,
                        choices=["train", "dev", "validation", "test"])
    parser.add_argument("--dataset", default="vimqa", help="Tên bộ dữ liệu dùng để đặt tên file")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ lấy N câu hỏi đầu (sau khi xáo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.hf:
        from datasets import load_dataset

        # ViMQA trên HuggingFace đặt tên split là "validation", không phải "dev".
        hf_split = {"dev": "validation"}.get(args.split, args.split)
        raw_data = [dict(r) for r in load_dataset(args.hf, split=hf_split)]
        print(f"Đọc {len(raw_data)} mẫu từ HuggingFace {args.hf} (split={hf_split})")
    else:
        if not args.input.exists():
            print(f"Không tìm thấy {args.input}", file=sys.stderr)
            return 1
        raw_data = json.loads(args.input.read_text(encoding="utf-8"))
        print(f"Đọc {len(raw_data)} mẫu từ {args.input}")

    if args.limit is not None and args.limit < len(raw_data):
        rng = random.Random(args.seed)
        raw_data = rng.sample(raw_data, args.limit)
        print(f"Lấy ngẫu nhiên {args.limit} mẫu (seed={args.seed})")

    samples = [build_sample(r) for r in raw_data]
    corpus = build_corpus(samples)
    stats = compute_stats(samples, corpus)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.split}"
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (args.out_dir / f"{stem}_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (args.out_dir / f"{stem}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nĐã ghi vào {args.out_dir}/")
    print(f"  {stem}.json            ({len(samples)} mẫu)")
    print(f"  {stem}_corpus.json     ({len(corpus)} đoạn)")
    print(f"  {stem}_stats.json")
    print("\nThống kê:")
    for k, v in stats.items():
        print(f"  {k:34s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
