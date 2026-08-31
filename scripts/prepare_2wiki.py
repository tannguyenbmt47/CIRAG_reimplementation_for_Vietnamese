#!/usr/bin/env python3
"""Chuẩn bị 2WikiMultihopQA cho phép thử tái lập trên bộ dữ liệu gốc.

Dùng cùng định dạng nội bộ với ViMQA nên toàn bộ pipeline chạy lại được không
cần sửa gì. Mục đích: kiểm tra bản cài đặt có tái hiện được con số của bài báo
gốc trên chính bộ dữ liệu và mô hình nền mà họ dùng hay không.

Dùng: python scripts/prepare_2wiki.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "2wiki" / "processed"


def build_sample(raw: dict) -> dict:
    ctx = raw["context"]
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    facts = raw["supporting_facts"]
    if isinstance(facts, str):
        facts = json.loads(facts)

    supporting: dict[str, set[int]] = {}
    for title, sent_id in facts:
        supporting.setdefault(title, set()).add(int(sent_id))

    paragraphs = []
    for title, sentences in ctx:
        sentences = [unicodedata.normalize("NFC", s) for s in sentences]
        ids = sorted(supporting.get(title, ()))
        paragraphs.append({
            "title": unicodedata.normalize("NFC", title),
            "sentences": sentences,
            "text": " ".join(sentences),
            "is_supporting": bool(ids),
            "supporting_sent_ids": ids,
        })
    return {
        "id": raw["_id"],
        "question": unicodedata.normalize("NFC", raw["question"]),
        "answer": unicodedata.normalize("NFC", raw["answer"]),
        "type": raw.get("type", ""),
        "paragraphs": paragraphs,
        #  Chỉ giữ nhãn trỏ tới đoạn thực sự có trong context: 2WikiMQA có một số
        #  supporting_facts trỏ tới tiêu đề không nằm trong mười đoạn kèm câu hỏi.
        "supporting_facts": [[t, int(i)] for t, i in facts
                             if t in {p["title"] for p in paragraphs}],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    import pandas as pd

    url = f"hf://datasets/xanhho/2WikiMultihopQA/{args.split}.parquet"
    df = pd.read_parquet(url)
    print(f"Đọc {len(df)} mẫu từ {url}")
    df = df.sample(args.limit, random_state=args.seed)
    print(f"Lấy ngẫu nhiên {args.limit} mẫu (seed={args.seed})")

    samples = [build_sample(r) for _, r in df.iterrows()]

    #  Kho ngữ liệu: hợp nhất đoạn của mọi câu hỏi rồi khử trùng lặp theo tiêu đề,
    #  đúng thiết lập distractor mà bài báo gốc mô tả (Sec 4.1).
    corpus: dict[str, dict] = {}
    for s in samples:
        for p in s["paragraphs"]:
            prev = corpus.get(p["title"])
            if prev is None or len(p["sentences"]) > len(prev["sentences"]):
                corpus[p["title"]] = {
                    "title": p["title"],
                    "sentences": p["sentences"],
                    "text": p["text"],
                }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"2wiki_{args.split}"
    (OUT_DIR / f"{tag}.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / f"{tag}_corpus.json").write_text(
        json.dumps(list(corpus.values()), ensure_ascii=False, indent=1), encoding="utf-8")

    n_yesno = sum(1 for s in samples if s["answer"].lower() in ("yes", "no"))
    print(f"\nĐã ghi vào {OUT_DIR}/")
    print(f"  {len(samples)} câu hỏi | {len(corpus)} đoạn sau khử trùng lặp")
    print(f"  câu hỏi đúng/sai: {n_yesno} | loại: "
          f"{ {t: sum(1 for s in samples if s['type']==t) for t in {s['type'] for s in samples}} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
