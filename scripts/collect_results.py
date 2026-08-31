#!/usr/bin/env python3
"""Gom mọi outputs/*/summary.json thành bảng Markdown + LaTeX cho báo cáo.

Dùng: python scripts/collect_results.py [--out report/results.tex]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#  Nhãn hiển thị cho từng lần chạy, theo thứ tự muốn trình bày trong báo cáo.
LABELS = {
    "main_native": "NativeRAG",
    "main_ircot": "IRCoT",
    "main_cirag": "ViCiRAG (đầy đủ)",
    "ab_fact": "chỉ mức triple",
    "ab_sentence": "chỉ mức câu",
    "ab_passage": "chỉ mức đoạn",
    "ab_fact_sentence": "triple + câu",
    "ab_fact_passage": "triple + đoạn",
    "ab_no_reranker": "không dùng reranker",
    "it1": "L = 1",
    "it2": "L = 2",
    "it8": "L = 8",
    "ext_hybrid": "truy hồi lai BM25 + dense",
    "ext_scope_sample": "phạm vi ứng viên theo mã nguồn gốc",
}


def load_runs(output_dir: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(output_dir.glob("*/summary.json")):
        try:
            runs[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  [!] Bỏ qua file hỏng: {path}")
    return runs


def error_breakdown(run_dir: Path) -> Counter:
    """Phân loại lỗi theo khâu phát sinh lỗi đầu tiên trong pipeline."""
    path = run_dir / "predictions.json"
    if not path.exists():
        return Counter()

    counts = Counter()
    for p in json.loads(path.read_text(encoding="utf-8")):
        if p.get("em") == 1.0:
            counts["đúng"] += 1
        elif p.get("f1", 0) > 0.6:
            counts["sai do chuẩn hoá/diễn đạt"] += 1
        elif p.get("retrieval_recall", 1.0) < 1.0:
            counts["truy hồi trượt"] += 1
        elif p.get("num_core_triples", 1) == 0:
            counts["trích triple sót"] += 1
        elif p.get("granularity") == "default":
            counts["ACMG không mức nào đủ"] += 1
        elif p.get("granularity") == "retrieved":
            # Cascade trong G hụt nhưng phương án chót đã trả lời được (dù sai).
            counts["cascade hụt, cứu bằng đoạn truy hồi"] += 1
        else:
            counts["integration/suy luận sai"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=None, help="Ghi bảng LaTeX ra file")
    args = parser.parse_args()

    runs = load_runs(args.output_dir)
    if not runs:
        print("Chưa có kết quả nào trong outputs/")
        return 1

    order = [k for k in LABELS if k in runs] + [k for k in runs if k not in LABELS]

    print(f"{'Lần chạy':<36} {'n':>5} {'EM':>7} {'F1':>7} {'F1từ':>7} {'recall':>7} {'hop':>5}")
    print("-" * 78)
    for key in order:
        s = runs[key]
        m = s["metrics"]
        print(
            f"{LABELS.get(key, key):<36} {s['num_questions']:>5} {m['em']:>7.2f} {m['f1']:>7.2f} "
            f"{m['f1_word']:>7.2f} {s.get('avg_retrieval_recall', 0):>7.3f} {s.get('avg_steps', 0):>5.2f}"
        )

    # ---- Phân bố mức ngữ cảnh (dựng lại Figure 3 của paper) ----
    if "main_cirag" in runs:
        dist = runs["main_cirag"].get("granularity_distribution", {})
        total = sum(dist.values()) or 1
        print("\nPhân bố mức ngữ cảnh ACMG (main_cirag):")
        for level in ("fact", "sentence", "passage", "retrieved", "default"):
            n = dist.get(level, 0)
            print(f"  {level:<10} {n:>4}  {100 * n / total:>5.1f}%  {'█' * int(40 * n / total)}")

    # ---- Phân tích lỗi ----
    errors = error_breakdown(args.output_dir / "main_cirag")
    if errors:
        total = sum(errors.values())
        print("\nPhân tích lỗi (main_cirag):")
        for label, n in errors.most_common():
            print(f"  {label:<34} {n:>4}  {100 * n / total:>5.1f}%")

    # ---- Chi phí ----
    tot_in = sum(s.get("token_usage", {}).get("prompt_tokens", 0) for s in runs.values())
    tot_out = sum(s.get("token_usage", {}).get("completion_tokens", 0) for s in runs.values())
    print(f"\nTổng token: {tot_in:,} vào / {tot_out:,} ra")

    if args.out:
        lines = [
            "% Sinh tự động bởi scripts/collect_results.py",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "\\textbf{Phương pháp} & \\textbf{EM} & \\textbf{F1} & \\textbf{F1$_{\\text{từ}}$} \\\\",
            "\\midrule",
        ]
        for key in order:
            m = runs[key]["metrics"]
            label = LABELS.get(key, key).replace("_", "\\_")
            lines.append(f"{label} & {m['em']:.2f} & {m['f1']:.2f} & {m['f1_word']:.2f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nĐã ghi bảng LaTeX: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
