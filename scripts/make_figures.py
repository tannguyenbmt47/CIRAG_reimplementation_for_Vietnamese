#!/usr/bin/env python3
"""Sinh hình cho báo cáo từ outputs/ — không có số nào viết tay.

Mọi giá trị đọc trực tiếp từ summary.json / predictions.json, nên hình luôn khớp
với bảng số trong RESULTS.md. Chạy: python scripts/make_figures.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#  Nhãn theo ngôn ngữ. Bản tiếng Anh dùng cho report/main_en.tex; hình phải tự giải
#  thích được nên nhãn không thể để tiếng Việt trong bài tiếng Anh.
L = {
    "vi": {
        "weak": "Qwen2.5-7B (mô hình nền yếu)", "strong": "Gemini 2.5 Flash (mô hình nền mạnh)",
        "em": "Exact Match (%)", "triple": "triple", "sent": "câu", "psg": "đoạn",
        "fallback": "dự phòng", "insuf": "không đủ", "ratio": "Tỷ lệ câu hỏi (%)",
        "orig": "CIRAG gốc", "plus": "+ cơ chế dự phòng",
        "err": {"integ": "Tích hợp sai", "para": "Lệch cách diễn đạt",
                "none": "Không có mức nào đủ", "ret": "Truy hồi trượt",
                "trip": "Trích triple sót", "fb": "Dự phòng vẫn sai"},
        "nwrong": "Số câu hỏi sai (trên {n})",
    },
    "en": {
        "weak": "Qwen2.5-7B (weaker backbone)", "strong": "Gemini 2.5 Flash (stronger backbone)",
        "em": "Exact Match (%)", "triple": "triples", "sent": "sentences", "psg": "passages",
        "fallback": "fallback", "insuf": "no level", "ratio": "Share of questions (%)",
        "orig": "CIRAG as published", "plus": "+ fallback level",
        "err": {"integ": "Integration error", "para": "Wording mismatch",
                "none": "No level sufficient", "ret": "Retrieval miss",
                "trip": "Triple extraction miss", "fb": "Fallback still wrong"},
        "nwrong": "Wrong answers (out of {n})",
    },
}
LANG = "vi"
OUT = ROOT / "report" / "figures"

#  Bảng màu phân loại đã qua kiểm định (xem scripts/validate_palette.js của skill
#  dataviz): 3 slot đầu đạt ngưỡng all-pairs cho cả người nhìn thường lẫn mù màu.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#ffffff"
GRID = "#e5e4e0"

plt.rcParams.update({
    "font.family": "DejaVu Serif",     # phủ đủ dấu tiếng Việt
    "font.size": 8.5,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(run: str) -> dict:
    return json.loads((ROOT / "outputs" / run / "summary.json").read_text(encoding="utf-8"))


def preds(run: str) -> list[dict]:
    return json.loads((ROOT / "outputs" / run / "predictions.json").read_text(encoding="utf-8"))


def strip_axes(ax, keep_left=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


# ═════════════════════════════════ Hình 1: hiệu ứng tương tác với mô hình nền
def fig_backbone():
    methods = [("NativeRAG", "main_native", "qwen_native", C_BLUE),
               ("IRCoT", "main_ircot", "qwen_ircot", C_ORANGE),
               ("CIRAG", "main_cirag", "qwen_cirag", C_AQUA)]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, (title, idx) in zip(axes, [(L[LANG]["weak"], 2),
                                       (L[LANG]["strong"], 1)]):
        vals = []
        for i, (name, gem, qwen, color) in enumerate(methods):
            run = gem if idx == 1 else qwen
            em = load(run)["metrics"]["em"]
            vals.append(em)
            ax.bar(i, em, width=0.62, color=color, zorder=3)
            #  Nhãn số trực tiếp — vừa là relief cho cảnh báo tương phản của màu
            #  aqua, vừa để người đọc không phải dò theo trục.
            ax.text(i, em + 1.2, f"{em:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")

        #  Thông điệp của hình là khoảng cách ViCiRAG - NativeRAG CO LẠI khi mô
        #  hình nền mạnh lên. Vẽ đường chuẩn ở mức NativeRAG rồi đo khoảng hở dọc
        #  lên tới đỉnh cột ViCiRAG: như vậy độ lớn của Δ được mã hoá bằng CHIỀU
        #  DÀI, đọc được ngay bằng mắt, thay vì chỉ nằm trong con chữ.
        base, target = vals[0], vals[2]
        ax.axhline(base, color=INK3, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.annotate("", xy=(2.52, target), xytext=(2.52, base),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
        #  Δ nhỏ thì mũi tên rất ngắn, đặt nhãn ngang tâm sẽ đè lên đầu mũi tên;
        #  khi đó đẩy nhãn lên trên một chút.
        gap = target - base
        label_y = (base + target) / 2 if gap >= 4 else target + 2.0
        ax.text(2.72, label_y, f"Δ = +{gap:.1f}",
                ha="left", va="center", fontsize=8.5, color=INK, fontweight="bold")

        ax.set_title(title, fontsize=8.5, color=INK, pad=16)
        ax.set_xticks(range(3))
        ax.set_xticklabels([m[0] for m in methods], color=INK2)
        ax.set_xlim(-0.62, 3.45)
        ax.set_ylim(0, 68)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        strip_axes(ax)

    axes[0].set_ylabel(L[LANG]["em"], color=INK2)
    fig.savefig(OUT / "fig_backbone.pdf")
    plt.close(fig)
    print("  fig_backbone.pdf")


# ═══════════════════════ Hình 2: phân bố mức ngữ cảnh dịch chuyển theo mô hình nền
def fig_granularity():
    """Khổ một cột (~3,2 inch) — xem ghi chú ở fig_rescue về lý do."""
    levels = [(L[LANG]["triple"], C_BLUE), (L[LANG]["sent"], C_ORANGE), (L[LANG]["psg"], C_AQUA),
              (L[LANG]["fallback"], C_YELLOW), (L[LANG]["insuf"], C_MAGENTA)]
    keys = ["fact", "sentence", "passage", "retrieved", "default"]
    rows = [("Gemini 2.5 Flash", "main_cirag"), ("Qwen2.5-7B", "qwen_cirag")]

    fig, ax = plt.subplots(figsize=(3.25, 1.6))
    for r, (label, run) in enumerate(rows):
        dist = load(run)["granularity_distribution"]
        total = sum(dist.values())
        left = 0.0
        for key, (name, color) in zip(keys, levels):
            pct = 100 * dist.get(key, 0) / total
            if pct <= 0:
                continue
            ax.barh(r, pct - 0.5, left=left, height=0.42, color=color, zorder=3)
            #  Khổ hẹp nên chỉ nhãn được đoạn đủ rộng; phần còn lại dựa vào chú giải.
            if pct >= 12:
                ax.text(left + pct / 2, r, f"{pct:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
            left += pct

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel(L[LANG]["ratio"], color=INK2, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.invert_yaxis()
    strip_axes(ax)
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, c in levels],
              loc="upper center", bbox_to_anchor=(0.45, -0.52), ncol=3,
              frameon=False, fontsize=7, labelcolor=INK2, handlelength=1.0,
              columnspacing=1.0, handletextpad=0.5)
    fig.savefig(OUT / "fig_granularity.pdf")
    plt.close(fig)
    print("  fig_granularity.pdf")


# ═════════════════════════════════════ Hình 3: phân tích lỗi theo khâu hỏng
def fig_errors():
    p = preds("main_cirag")
    counts = Counter()
    for x in p:
        if x.get("em") == 1.0:
            continue
        if x.get("f1", 0) > 0.6:
            counts[L[LANG]["err"]["para"]] += 1
        elif x.get("retrieval_recall", 1.0) < 1.0:
            counts[L[LANG]["err"]["ret"]] += 1
        elif x.get("num_core_triples", 1) == 0:
            counts[L[LANG]["err"]["trip"]] += 1
        elif x.get("granularity") == "default":
            counts[L[LANG]["err"]["none"]] += 1
        elif x.get("granularity") == "retrieved":
            counts[L[LANG]["err"]["fb"]] += 1
        else:
            counts[L[LANG]["err"]["integ"]] += 1

    items = counts.most_common()
    n_total = len(p)
    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    ys = range(len(items))
    ax.barh(list(ys), [c for _, c in items], height=0.6, color=C_BLUE, zorder=3)
    for y, (_, c) in zip(ys, items):
        ax.text(c + 1.5, y, f"{c}  ({100*c/n_total:.1f}%)", va="center",
                fontsize=7.5, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _ in items], color=INK2, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(c for _, c in items) * 1.55)
    ax.set_xlabel(L[LANG]["nwrong"].format(n=n_total), color=INK2)
    ax.set_xticks([])
    strip_axes(ax)
    ax.spines["bottom"].set_visible(False)
    fig.savefig(OUT / "fig_errors.pdf")
    plt.close(fig)
    print("  fig_errors.pdf")


# ══════════════════════════ Hình 4: tác động của cơ chế dự phòng đề xuất
def fig_rescue():
    """Hình này đặt trong MỘT CỘT của bài ACL (~3,2 inch), nên phải thiết kế đúng
    khổ đó: vẽ rộng rồi để LaTeX thu nhỏ sẽ làm chữ bé đi một nửa."""
    before, after = load("main_cirag_norescue")["metrics"], load("main_cirag")["metrics"]

    labels = ["EM", "F1"]
    vals_b = [before["em"], before["f1"]]
    vals_a = [after["em"], after["f1"]]

    fig, ax = plt.subplots(figsize=(3.25, 1.85))
    xs = [0, 1]
    w = 0.34
    #  Khe 2px giữa cặp cột: lệch tâm w/2 + 0.015.
    ax.bar([x - w / 2 - 0.008 for x in xs], vals_b, width=w, color=INK3,
           label=L[LANG]["orig"], zorder=3)
    ax.bar([x + w / 2 + 0.008 for x in xs], vals_a, width=w, color=C_AQUA,
           label=L[LANG]["plus"], zorder=3)
    for x, b, a in zip(xs, vals_b, vals_a):
        ax.text(x - w / 2 - 0.008, b + 1.5, f"{b:.1f}", ha="center", va="bottom",
                fontsize=8, color=INK)
        ax.text(x + w / 2 + 0.008, a + 1.5, f"{a:.1f}", ha="center", va="bottom",
                fontsize=8, color=INK, fontweight="bold")
        ax.text(x, a + 9.5, f"+{a - b:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=C_AQUA, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 25, 50, 75])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2,
              frameon=False, fontsize=7.5, labelcolor=INK2, handlelength=1.1,
              columnspacing=1.2)
    fig.savefig(OUT / "fig_rescue.pdf")
    plt.close(fig)
    print("  fig_rescue.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    LANG = ap.parse_args().lang
    OUT = ROOT / "report" / ("figures" if LANG == "vi" else "figures_en")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Sinh hình ({LANG}) vào {OUT.relative_to(ROOT)}/:")
    fig_backbone()
    fig_granularity()
    fig_errors()
    fig_rescue()
