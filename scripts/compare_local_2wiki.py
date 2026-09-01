#!/usr/bin/env python3
"""Đối chiếu các lần chạy local trên 2WikiMQA (scripts/run_local_2wiki.sh) với bài báo gốc.

In ba bảng Markdown: mốc chính CIRAG w/o TD so với IRCoT, bóc tách cascade của Bảng 3, và
phân bố mức ngữ cảnh. Kiểm định McNemar chính xác (nhị thức hai phía) trên các cặp câu bất
đồng, vì hai phương pháp chạy trên cùng 100 câu.

Dùng: python scripts/compare_local_2wiki.py [--suffix _smoke]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

#  Số của bài báo gốc trên 2WikiMQA với Qwen2.5-7B-Instruct, dạng (EM, F1).
PAPER = {
    "cirag_wo_td": (49.8, 59.7),  # Bảng 2, dòng w/o TD
    "ircot": (36.4, 45.7),  # Bảng 1
    "ircot_bge_small": (35.9, 44.8),  # Bảng 6: retriever bge-small-en-v1.5
}

#  Bảng 3 của bài báo dùng CIRAG đầy đủ (có TD), nên chỉ so được thứ tự và độ chênh so
#  với cấu hình đầy đủ, không so được số tuyệt đối.
TABLE3 = [
    # (tên lần chạy, nhãn trong Bảng 3, EM bài báo, F1 bài báo)
    ("cirag_eq6", "CIRAG (đầy đủ cascade)", 58.9, 68.1),
    ("ab_fact_sentence", "w/o Passage", 57.5, 67.4),
    ("ab_sentence_passage", "w/o Triple", 58.8, 67.5),
    ("ab_fact", "w/o Sentence + Passage (chỉ triple)", 54.7, 64.1),
    ("ab_sentence", "w/o Triple + Passage (chỉ câu)", 56.5, 64.7),
    ("ab_passage", "w/o Triple + Sentence (chỉ đoạn)", 53.2, 61.9),
]

#  Hình 5, cột w/o TD của 2WikiMQA. Đọc bằng mắt từ biểu đồ cột chồng nên sai số
#  khoảng ±3 điểm phần trăm.
FIG5_WO_TD = {"fact": 39, "sentence": 18, "passage": 23, "default": 20}


def load(name: str) -> tuple[dict, dict[str, dict]] | None:
    run_dir = OUTPUTS / name
    if not (run_dir / "summary.json").exists():
        return None
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    preds = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    return summary, {p["id"]: p for p in preds}


def mcnemar(a: dict[str, dict], b: dict[str, dict]) -> tuple[int, int, float]:
    """Trả về (số câu chỉ A đúng, số câu chỉ B đúng, p hai phía)."""
    ids = a.keys() & b.keys()
    only_a = sum(1 for i in ids if a[i]["em"] == 1 and b[i]["em"] == 0)
    only_b = sum(1 for i in ids if a[i]["em"] == 0 and b[i]["em"] == 1)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="", help="Hậu tố tên lần chạy, vd _smoke")
    args = ap.parse_args()

    def run(name: str):
        return load(f"local2w_{name}{args.suffix}")

    cirag, ircot = run("cirag_eq6"), run("ircot")

    # ------------------------------------------------------------ mốc chính
    print("## CIRAG w/o TD so với IRCoT\n")
    print("| | EM | F1 | EM bài báo | F1 bài báo |")
    print("|---|---:|---:|---:|---:|")
    rows = [
        ("IRCoT", ircot, PAPER["ircot"]),
        ("CIRAG đúng Eq. 6", cirag, PAPER["cirag_wo_td"]),
        ("CIRAG + mức dự phòng", run("cirag_fallback"), None),
        ("CIRAG + điều kiện đủ theo số nguồn", run("fix_hop"), None),
    ]
    for label, res, ref in rows:
        if res is None:
            continue
        m = res[0]["metrics"]
        print(f"| {label} | {m['em']:.1f} | {m['f1']:.1f} | "
              f"{fmt(ref[0] if ref else None)} | {fmt(ref[1] if ref else None)} |")
    print(f"\nMốc IRCoT với đúng retriever bge-small-en-v1.5 (Bảng 6): "
          f"{PAPER['ircot_bge_small'][0]} EM / {PAPER['ircot_bge_small'][1]} F1.")

    if cirag and ircot:
        gap_em = cirag[0]["metrics"]["em"] - ircot[0]["metrics"]["em"]
        gap_f1 = cirag[0]["metrics"]["f1"] - ircot[0]["metrics"]["f1"]
        ref_em = PAPER["cirag_wo_td"][0] - PAPER["ircot"][0]
        ref_f1 = PAPER["cirag_wo_td"][1] - PAPER["ircot"][1]
        a, b, p = mcnemar(cirag[1], ircot[1])
        print(f"\n**CIRAG − IRCoT:** {gap_em:+.1f} EM / {gap_f1:+.1f} F1 "
              f"(bài báo {ref_em:+.1f} / {ref_f1:+.1f}). "
              f"McNemar: {a} câu chỉ CIRAG đúng, {b} câu chỉ IRCoT đúng, p = {p:.3f}.")

    # ------------------------------------------------------------ Bảng 3
    print("\n## Bảng 3: bóc tách cascade\n")
    print("Bài báo dùng CIRAG có TD nên chỉ so được độ chênh so với cascade đầy đủ.\n")
    print("| Biến thể | EM | Δ so với đầy đủ | Δ bài báo | p (so với đầy đủ) |")
    print("|---|---:|---:|---:|---:|")
    full = cirag
    for name, label, ref_em, _ in TABLE3:
        res = run(name)
        if res is None:
            continue
        em = res[0]["metrics"]["em"]
        if name == "cirag_eq6":
            print(f"| {label} | {em:.1f} | | | |")
            continue
        delta = em - full[0]["metrics"]["em"] if full else None
        p = mcnemar(res[1], full[1])[2] if full else None
        print(f"| {label} | {em:.1f} | {fmt(delta)} | {ref_em - TABLE3[0][2]:+.1f} | "
              f"{'—' if p is None else f'{p:.3f}'} |")

    # ------------------------------------------------- phân bố mức ngữ cảnh
    print("\n## Phân bố mức ngữ cảnh (CIRAG đúng Eq. 6)\n")
    if cirag:
        counts = Counter(p["granularity"] for p in cirag[1].values())
        n = sum(counts.values())
        print("| Mức | Của chúng tôi | Hình 5, w/o TD (≈) |")
        print("|---|---:|---:|")
        for g in ("fact", "sentence", "passage", "default"):
            print(f"| {g} | {100 * counts.get(g, 0) / n:.0f}% | {FIG5_WO_TD[g]}% |")

    # ------------------------------------------------------- kiểm tra cắt cụt
    print("\n## Kiểm tra cửa sổ ngữ cảnh\n")
    for path in sorted(OUTPUTS.glob(f"local2w_*{args.suffix}/summary.json")):
        if not args.suffix and path.parent.name.endswith("_smoke"):
            continue
        usage = json.loads(path.read_text(encoding="utf-8")).get("token_usage", {})
        print(f"- `{path.parent.name}`: prompt dài nhất {usage.get('max_prompt_tokens', '?')} "
              f"token, {usage.get('num_context_at_risk', '?')} lời gọi có nguy cơ bị cắt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
