#!/usr/bin/env python3
"""Sinh notebook Colab từ mã nguồn Python.

Viết notebook bằng script thay vì soạn JSON tay để nội dung luôn khớp với API
thật của package (đổi tên hàm là script này sẽ báo lỗi ngay), và để diff trên git
đọc được.

Chạy: python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "CiRAG_Vietnamese.ipynb"

cells: list[dict] = []


def md(source: str) -> None:
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}
    )


def code(source: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().splitlines(True),
        }
    )


# ════════════════════════════════════════════════════════════════════ Mở đầu
md(
    """
# ViCiRAG — Hỏi đáp Đa bước tiếng Việt với CiRAG

Cài đặt lại paper **CIRAG: Construction-Integration Retrieval and Adaptive Generation for
Multi-hop Question Answering** ([arXiv:2601.06799](https://arxiv.org/abs/2601.06799)) và chuyển
sang tiếng Việt trên bộ dữ liệu **ViMQA**.

Notebook chạy được trên Google Colab hoặc Kaggle, không phụ thuộc đường dẫn cục bộ.

**Kiến trúc.** CiRAG có hai module chính:

1. **ICI** (Iterative Construction-Integration) — mỗi vòng lặp: truy hồi top-K đoạn văn cho truy
   vấn hiện tại, gom các triple đã trích sẵn, rerank lấy top-N ứng viên, rồi để LLM lọc ra *core
   triple* và sinh truy vấn cho hop kế tiếp — có điều kiện trên toàn bộ lịch sử. Nhờ giữ nhiều
   triple thay vì một triple tốt nhất, module này tránh được bẫy tham lam (greedy trap) của các
   phương pháp trước.
2. **ACMG** (Adaptive Cascaded Multi-Granularity) — trả lời lần lượt ở ba mức ngữ cảnh
   triple → câu → đoạn văn, nhận đáp án ở **mức đầu tiên đủ thông tin**. Triple ít nhiễu nhưng
   thiếu ngữ cảnh; đoạn văn đủ ngữ cảnh nhưng nhiều nhiễu, nên mở rộng ngữ cảnh theo nhu cầu giúp cân bằng hai thái cực đó.

**Mục lục**
1. Cài đặt thư viện và cấu hình môi trường
2. Tải và đọc bộ dữ liệu
3. Khám phá và tiền xử lý dữ liệu
4. Cài đặt mô hình
5. Đánh giá trên tập kiểm thử
6. So sánh với mô hình cơ sở
7. Phân tích lỗi
8. Demo trên dữ liệu tiếng Việt mới
"""
)

# ═══════════════════════════════════════════════ 1. Cài đặt & cấu hình
md(
    """
---
## 1. Cài đặt thư viện và cấu hình môi trường
"""
)

code(
    '''
import os, sys, subprocess

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle")
print(f"Colab: {IN_COLAB} | Kaggle: {IN_KAGGLE}")

REPO_URL = "https://github.com/<user>/NLP_CiRAG.git"   # <-- đổi thành repo của bạn
REPO_DIR = "NLP_CiRAG"

if IN_COLAB or IN_KAGGLE:
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)
else:
    # Chạy local: notebook nằm trong notebooks/ nên lùi một cấp.
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")

print("Thư mục làm việc:", os.getcwd())
'''
)

code(
    '''
%pip install -q openai tenacity filelock python-dotenv sentence-transformers underthesea rank-bm25 tqdm
'''
)

code(
    '''
import sys
sys.path.insert(0, "src")

# --- Khoá API ---
# Toàn bộ lời gọi LLM đi qua OpenRouter (giao thức OpenAI-compatible), nên chỉ cần một khoá.
# Lấy tại https://openrouter.ai/keys
import os
from getpass import getpass

if not os.environ.get("OPENROUTER_API_KEY"):
    try:  # Colab: ưu tiên lấy từ Secrets để không lộ khoá trong output
        from google.colab import userdata
        os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
    except Exception:
        os.environ["OPENROUTER_API_KEY"] = getpass("OPENROUTER_API_KEY: ")

print("Đã có khoá API:", bool(os.environ.get("OPENROUTER_API_KEY")))
'''
)

code(
    '''
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có (dùng CPU)")

# Model sinh: rẻ và tốt tiếng Việt. Có thể đổi sang bất kỳ model nào trên OpenRouter.
LLM_NAME = "google/gemini-2.5-flash"

# Model embedding. Bản đầy đủ (BGE-M3 fine-tune tiếng Việt, ~6GB tải về) cho chất lượng tốt nhất;
# nếu Colab chật đĩa/VRAM, đổi sang bản nhẹ bên dưới.
EMBEDDING_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# EMBEDDING_NAME = "bkai-foundation-models/vietnamese-bi-encoder"   # PhoBERT, ~540MB

# Số câu hỏi đánh giá. Để nhỏ khi chạy thử, tăng dần khi đã thông pipeline.
NUM_EVAL_QUESTIONS = 500
'''
)

# ═══════════════════════════════════════════════════ 2. Tải dữ liệu
md(
    """
---
## 2. Tải hoặc đọc bộ dữ liệu

**ViMQA** (Le et al., LREC 2022) gồm 10.047 cặp hỏi–đáp đa bước tiếng Việt do người viết, dựa
trên Wikipedia tiếng Việt. Cấu trúc giống HotpotQA: mỗi câu hỏi kèm 10 đoạn văn (2 đoạn chứa
bằng chứng + 8 đoạn nhiễu) và nhãn *supporting facts* ở mức câu.

Nhóm tác giả phát hành toàn bộ dữ liệu trên HuggingFace tại
[`nguyenlab/vimqa`](https://huggingface.co/datasets/nguyenlab/vimqa), giấy phép CC-BY-NC-SA-4.0
(nghiên cứu phi thương mại, được phép phát hành lại kèm ghi công).
"""
)

code(
    '''
import json

from datasets import load_dataset

raw = load_dataset("nguyenlab/vimqa")
print(raw)

ex = raw["validation"][0]
print("\\nMột mẫu:")
print("  câu hỏi:", ex["question"])
print("  đáp án :", ex["answer"])
print("  số đoạn:", len(ex["context"]["title"]))
print("  bằng chứng:", ex["supporting_facts"])
'''
)

code(
    '''
DATASET = "vimqa"
SPLIT = "validation"

# Số câu hỏi đánh giá. Corpus truy hồi lớn dần theo số câu hỏi (500 câu -> ~1.969 đoạn)
# và chi phí trích triple tỉ lệ thuận với cỡ corpus, nên hãy để nhỏ khi chạy thử.
NUM_EVAL_QUESTIONS = 500
'''
)

# ═════════════════════════════════════ 3. Khám phá & tiền xử lý
md(
    """
---
## 3. Khám phá và tiền xử lý dữ liệu

Bước tiền xử lý làm hai việc:

1. **Chuẩn hoá Unicode NFC.** Tiếng Việt có hai cách mã hoá dấu — dựng sẵn (NFC) và tổ hợp (NFD).
   Hai chuỗi hiển thị giống hệt nhau nhưng khác byte; không quy chuẩn thì cả so khớp chuỗi lẫn
   tokenizer đều lệch.
2. **Dựng corpus truy hồi.** Gộp *toàn bộ* đoạn supporting và distractor của mọi câu hỏi rồi khử
   trùng lặp theo tiêu đề — đúng setting IRCoT mà paper sử dụng. Đây là điểm khiến bài toán khó
   hơn hẳn so với việc chỉ đọc 10 đoạn của riêng câu hỏi đó.

**Lưu ý:** không tách lại câu bằng `underthesea`. Chỉ số `sent_id` trong `supporting_facts` trỏ
trực tiếp vào danh sách câu gốc của ViMQA; tách lại sẽ làm lệch toàn bộ nhãn.
"""
)

code(
    '''
!python scripts/prepare_vimqa.py --hf nguyenlab/vimqa --split {SPLIT} --dataset {DATASET} --limit {NUM_EVAL_QUESTIONS}
'''
)

code(
    '''
from vicirag.data import load_samples, load_corpus

samples = load_samples(f"data/vimqa/processed/{DATASET}_{SPLIT}.json")
corpus  = load_corpus(f"data/vimqa/processed/{DATASET}_{SPLIT}_corpus.json")
stats   = json.load(open(f"data/vimqa/processed/{DATASET}_{SPLIT}_stats.json", encoding="utf-8"))

print(f"{len(samples)} câu hỏi | corpus {len(corpus)} đoạn\\n")
for k, v in stats.items():
    print(f"  {k:32s} {v}")
'''
)

code(
    '''
# Xem một mẫu cụ thể: câu hỏi, đáp án, và các câu là bằng chứng.
s = samples[0]
print("Câu hỏi:", s.question)
print("Đáp án :", s.answer)
print("Loại   :", s.type)
print("\\nCác đoạn bằng chứng:")
for p in s.paragraphs:
    if p["is_supporting"]:
        print(f"  [{p['title']}]")
        for i in p["supporting_sent_ids"]:
            print(f"     -> {p['sentences'][i]}")

print("\\nCác đoạn nhiễu:", [p["title"] for p in s.paragraphs if not p["is_supporting"]])
'''
)

code(
    '''
# Phân bố độ dài câu hỏi và đáp án -> cho biết bài toán thiên về trích rút hay diễn đạt lại.
import statistics

q_len = [len(s.question.split()) for s in samples]
a_len = [len(s.answer.split()) for s in samples]
p_len = [len(p["sentences"]) for s in samples for p in s.paragraphs]

for name, xs in [("Độ dài câu hỏi (âm tiết)", q_len), ("Độ dài đáp án (âm tiết)", a_len),
                 ("Số câu mỗi đoạn", p_len)]:
    print(f"{name:28s} TB={statistics.mean(xs):5.1f}  min={min(xs):3d}  max={max(xs):3d}")
'''
)

# ═══════════════════════════════════════════════ 4. Cài đặt mô hình
md(
    """
---
## 4. Cài đặt mô hình

Ba bước, chạy theo thứ tự:

1. **Construction Phase (offline).** Trích triple tri thức cho *toàn bộ* corpus bằng hai lời gọi
   LLM mỗi đoạn (NER → trích triple có điều kiện trên danh sách thực thể), rồi cache lại. Paper
   nhấn mạnh ở Sec 4.6 rằng chính bước tiền tính này giúp độ trễ khi suy luận thấp hơn hẳn biến
   thể trích triple trực tuyến.
2. **Đánh chỉ mục.** Nhúng cả đoạn văn lẫn triple bằng encoder tiếng Việt.
3. **Chạy ICI + ACMG.**

Trong lúc trích triple, mỗi triple được ghi kèm *provenance*: nó đến từ đoạn nào (ánh xạ M^D) và
từ những câu nào (ánh xạ M^S). Provenance chính là thứ cho phép ACMG mở rộng ngữ cảnh từ mức triple lên
mức câu rồi mức đoạn — mất nó là hỏng toàn bộ cascade.
"""
)

code(
    '''
from vicirag.config import Config
from vicirag.pipeline import ViCiRAG

config = Config(
    dataset=DATASET,
    split=SPLIT,
    llm_name=LLM_NAME,
    embedding_model_name=EMBEDDING_NAME,
    max_iterations=4,        # L: số vòng ICI tối đa (Appendix B.4)
    retrieval_top_k=10,      # K: số đoạn truy hồi mỗi vòng
    num_candidate_triples=30,# N: số triple ứng viên sau rerank
    num_workers=4,
    run_name=f"cirag_{DATASET}_{SPLIT}",
)

# Lần chạy đầu sẽ tốn thời gian vì phải trích triple cho cả corpus; các lần sau dùng cache.
cirag = ViCiRAG(config)
'''
)

code(
    '''
# Xem thử vòng lặp ICI hoạt động trên một câu hỏi: từng hop truy vấn gì, giữ lại triple nào.
sample = samples[0]
ici_result = cirag.ici.run(sample.question, allowed_titles=set(sample.own_titles))

print(f"CÂU HỎI: {sample.question}")
print(f"ĐÁP ÁN VÀNG: {sample.answer}\\n")
for log in ici_result.logs:
    print(f"--- Vòng {log.step} ---")
    print(f"Truy vấn      : {log.query}")
    print(f"Triple ứng viên: {len(log.candidate_triples)}")
    print(f"Suy luận      : {log.thought[:220]}...")
    print(f"Core triple   : {log.core_triples}")
    print(f"Truy vấn kế   : {log.next_query}\\n")

print(f"Tích luỹ: {len(ici_result.core_triples)} triple, "
      f"{len(ici_result.core_sentences)} câu, {len(ici_result.core_passages)} đoạn")
'''
)

code(
    '''
# ACMG: mở rộng ngữ cảnh cho tới khi đủ thông tin trả lời.
acmg_result = cirag.acmg.run(sample.question, ici_result)

print(f"Mức ngữ cảnh đã dùng: {acmg_result.granularity}")
print(f"Đáp án dự đoán      : {acmg_result.answer}")
print(f"Đáp án vàng         : {sample.answer}\\n")
for level, response in acmg_result.attempts.items():
    print(f"[{level}] {response[:180]}...")
'''
)

# ═══════════════════════════════════════ 5. Đánh giá tập kiểm thử
md(
    """
---
## 5. Đánh giá trên tập kiểm thử

Hai độ đo chuẩn cho hỏi đáp đa bước là **EM** (Exact Match) và **F1** theo token, nhưng hàm
chuẩn hoá gốc được thiết kế cho tiếng Anh (loại mạo từ `a/an/the`). Bản dùng ở đây xử lý ba
đặc thù của tiếng Việt:

| Vấn đề | Cách xử lý |
|---|---|
| Dấu mã hoá hai kiểu (NFC/NFD) | Quy chuẩn về NFC |
| Hư từ (`là`, `của`, `một`, `các`...) | Loại bỏ, tương tự mạo từ tiếng Anh |
| Ranh giới từ (`Hà Nội` = 1 từ, 2 âm tiết) | Báo cáo F1 theo **cả** âm tiết và từ |
| Câu hỏi đúng/sai | Quy `có/phải/vâng → đúng`, `sai/không → không` |
"""
)

code(
    '''
summary = cirag.run(limit=NUM_EVAL_QUESTIONS)
'''
)

code(
    '''
import json

preds = json.load(open(f"outputs/{config.run_name}/predictions.json", encoding="utf-8"))

print(f"{'EM':>4} {'F1':>6}  {'mức':<9} {'hop':>3}  câu hỏi / dự đoán / vàng")
print("-" * 100)
for p in preds:
    print(f"{p['em']:>4.0f} {p['f1']:>6.2f}  {p['granularity']:<9} {p['num_steps']:>3}  "
          f"{p['question'][:42]}")
    print(f"{'':>16} dự đoán: {p['predicted_answer'][:60]}")
    print(f"{'':>16} vàng   : {p['gold_answer'][:60]}")
'''
)

# ═════════════════════════════════════════ 6. So sánh baseline
md(
    """
---
## 6. So sánh với mô hình cơ sở

Hai baseline, **dùng chung retriever, chung corpus và chung backbone LLM** với CiRAG — đúng
nguyên tắc so sánh công bằng mà paper đặt ra ở Sec 4.2:

- **NativeRAG** (Lewis et al., 2020): truy hồi một lần rồi trả lời. Cho thấy truy hồi một bước
  không đủ để bắc cầu qua nhiều hop.
- **IRCoT** (Trivedi et al., 2023): truy hồi xen kẽ chuỗi suy luận. Khác biệt cốt lõi so với
  CiRAG là IRCoT tích luỹ nguyên văn **đoạn văn** (nhiễu dồn dần qua các vòng) và chỉ mở rộng
  theo một đường, trong khi ICI chắt lọc thành triple lõi.
"""
)

code(
    '''
results = {"CiRAG": summary}

for method, label in [("native", "NativeRAG"), ("ircot", "IRCoT")]:
    cfg = Config(
        dataset=DATASET, split=SPLIT, llm_name=LLM_NAME,
        embedding_model_name=EMBEDDING_NAME, max_iterations=4,
        num_workers=4, run_name=f"{method}_{DATASET}_{SPLIT}",
    )
    results[label] = ViCiRAG(cfg, method=method).run(limit=NUM_EVAL_QUESTIONS)
'''
)

code(
    '''
print(f"{'Phương pháp':<12} {'EM':>7} {'F1':>7} {'F1(từ)':>8} {'hop TB':>7} {'giây/câu':>9}")
print("-" * 56)
for name, s in results.items():
    m = s["metrics"]
    print(f"{name:<12} {m['em']:>7.2f} {m['f1']:>7.2f} {m['f1_word']:>8.2f} "
          f"{s['avg_steps']:>7.2f} {s['seconds_per_question']:>9.2f}")
'''
)

code(
    '''
# Phân bố mức ngữ cảnh mà ACMG thực sự dùng — tái lập Figure 3 của paper cho tiếng Việt.
# Paper quan sát triple chiếm đa số (76.3% trên 2WikiMQA), nghĩa là phần lớn câu hỏi đa bước
# giải quyết được bằng triple gọn, tránh được nhiễu không cần thiết.
dist = summary["granularity_distribution"]
total = sum(dist.values())
for level in ["fact", "sentence", "passage", "retrieved", "default"]:
    n = dist.get(level, 0)
    bar = "█" * int(40 * n / max(total, 1))
    print(f"{level:<9} {n:>3} ({100*n/max(total,1):5.1f}%) {bar}")
'''
)

# ══════════════════════════════════════════════ 7. Phân tích lỗi
md(
    """
---
## 7. Phân tích lỗi

Phân loại lỗi theo **khâu phát sinh lỗi đầu tiên trong pipeline**, vì mỗi loại cần một cách sửa khác nhau:

| Loại lỗi | Dấu hiệu | Hướng khắc phục |
|---|---|---|
| Truy hồi trượt | recall < 1: đoạn bằng chứng không nằm trong top-K | encoder tốt hơn, tăng K, bật hybrid BM25 |
| Trích triple sót | truy hồi đúng nhưng không có core triple nào | prompt OpenIE tốt hơn, tăng few-shot |
| Integration lọc nhầm | có core triple nhưng không chứa bằng chứng cần thiết | chưng cất quỹ đạo (Trajectory Distillation) |
| ACMG dừng sai mức | rơi vào nhãn `default`, mọi mức đều từ chối | chỉnh ngưỡng từ chối, mở rộng ngữ cảnh |
| Sai do chuẩn hoá | F1 cao nhưng EM = 0 | chuẩn hoá đáp án tiếng Việt, bước reanswer |
"""
)

code(
    '''
def classify_error(p):
    """Quy lỗi về khâu phát sinh lỗi đầu tiên trong pipeline."""
    if p["em"] == 1.0:
        return "đúng"
    if p["f1"] > 0.6:
        return "sai do chuẩn hoá/diễn đạt"      # nội dung đúng, dạng chuỗi lệch
    if p.get("retrieval_recall", 1.0) < 1.0:
        return "truy hồi trượt"
    if p.get("num_core_triples", 0) == 0:
        return "trích triple sót"
    if p["granularity"] == "default":
        return "ACMG không mức nào đủ"
    if p["granularity"] == "retrieved":
        return "cascade hụt, cứu bằng đoạn truy hồi"
    return "integration/suy luận sai"

from collections import Counter
errors = Counter(classify_error(p) for p in preds)

print(f"{'Loại':<30} {'Số câu':>7} {'Tỷ lệ':>7}")
print("-" * 46)
for label, n in errors.most_common():
    print(f"{label:<30} {n:>7} {100*n/len(preds):>6.1f}%")
'''
)

code(
    '''
# Xem chi tiết một ca sai: lần theo quỹ đạo để biết hỏng ở hop nào.
wrong = [p for p in preds if p["em"] == 0.0]
if wrong:
    p = wrong[0]
    print(f"Câu hỏi : {p['question']}")
    print(f"Vàng    : {p['gold_answer']}")
    print(f"Dự đoán : {p['predicted_answer']}")
    print(f"Loại lỗi: {classify_error(p)}")
    print(f"Recall  : {p.get('retrieval_recall')}  |  mức: {p['granularity']}\\n")
    for log in p.get("trajectory", []):
        print(f"--- Vòng {log['step']} | truy vấn: {log['query']}")
        print(f"    core triple: {log['core_triples']}")
        print(f"    truy vấn kế: {log['next_query']}")
else:
    print("Không có câu nào sai trong tập con này.")
'''
)

code(
    '''
# Khoảng cách EM vs F1 cho biết có bao nhiêu câu "đúng nội dung nhưng lệch cách diễn đạt".
m = summary["metrics"]
gap = [p for p in preds if p["em"] == 0.0 and p["f1"] > 0.5]
print(f"EM = {m['em']:.2f} | F1 = {m['f1']:.2f} | chênh lệch = {m['f1'] - m['em']:.2f}")
print(f"Số câu F1>0.5 nhưng EM=0: {len(gap)}/{len(preds)}")
for p in gap[:5]:
    print(f"  vàng: {p['gold_answer']!r}  <->  dự đoán: {p['predicted_answer']!r}")
'''
)

# ═════════════════════════════════════ 8. Demo dữ liệu mới
md(
    """
---
## 8. Demo trên dữ liệu tiếng Việt mới

Chạy CiRAG trên corpus tự soạn, hoàn toàn nằm ngoài ViMQA, để kiểm tra hệ thống hoạt động
trên văn bản tiếng Việt bất kỳ. Câu hỏi bên dưới cần **hai hop**: trước hết xác định tác giả
Truyện Kiều, sau đó tra năm sinh của người đó — không đoạn văn nào chứa sẵn cả hai thông tin.
"""
)

code(
    '''
from vicirag.data import Passage
from vicirag.openie import OpenIEExtractor
from vicirag.retriever import Retriever
from vicirag.ici import IterativeConstructionIntegration
from vicirag.acmg import AdaptiveCascadedGeneration

demo_corpus = [
    Passage(
        title="Truyện Kiều",
        sentences=[
            "Truyện Kiều là truyện thơ Nôm nổi tiếng nhất của văn học Việt Nam.",
            "Tác phẩm do Nguyễn Du sáng tác, gồm 3254 câu thơ lục bát.",
            "Truyện Kiều còn có tên gọi khác là Đoạn trường tân thanh.",
        ],
        text="",
    ),
    Passage(
        title="Nguyễn Du",
        sentences=[
            "Nguyễn Du là một nhà thơ lớn của Việt Nam, hiệu là Thanh Hiên.",
            "Ông sinh năm 1765 và mất năm 1820.",
            "Quê ông ở làng Tiên Điền, huyện Nghi Xuân, tỉnh Hà Tĩnh.",
        ],
        text="",
    ),
    Passage(
        title="Hồ Xuân Hương",
        sentences=[
            "Hồ Xuân Hương là nữ thi sĩ nổi tiếng sống vào cuối thế kỷ 18 đầu thế kỷ 19.",
            "Bà được mệnh danh là Bà chúa thơ Nôm.",
        ],
        text="",
    ),
    Passage(
        title="Chữ Nôm",
        sentences=[
            "Chữ Nôm là hệ thống chữ viết ghi âm tiếng Việt dựa trên chữ Hán.",
            "Chữ Nôm được dùng rộng rãi từ thế kỷ 13 đến đầu thế kỷ 20.",
        ],
        text="",
    ),
]
for p in demo_corpus:
    p.text = " ".join(p.sentences)

print(f"Corpus demo: {len(demo_corpus)} đoạn")
'''
)

code(
    '''
# Trích triple cho corpus demo (dùng chung LLM và encoder đã khởi tạo ở trên).
demo_triples = OpenIEExtractor(cirag.llm, lang="vi", num_workers=2).run(demo_corpus)

for title, records in demo_triples.items():
    print(f"\\n[{title}]")
    for r in records[:5]:
        print(f"   {r.triple}   <- câu {r.sentence_ids}")
'''
)

code(
    '''
demo_retriever = Retriever(demo_corpus, demo_triples, cirag.retriever.encoder)
demo_ici = IterativeConstructionIntegration(demo_retriever, cirag.llm, max_iterations=3)
demo_acmg = AdaptiveCascadedGeneration(cirag.llm)

for question in [
    "Tác giả của Truyện Kiều sinh năm nào?",
    "Người sáng tác Đoạn trường tân thanh quê ở tỉnh nào?",
    "Bà chúa thơ Nôm là ai?",
]:
    r = demo_ici.run(question)
    a = demo_acmg.run(question, r)
    print(f"\\nHỎI : {question}")
    print(f"ĐÁP : {a.answer}   [mức: {a.granularity}, {r.num_steps} hop]")
    for log in r.logs:
        print(f"   hop {log.step}: {log.query}  ->  {log.core_triples}")
'''
)

md(
    """
---
## Tổng kết

Notebook đã chạy trọn vẹn: nạp và tiền xử lý dữ liệu ViMQA, trích triple offline cho corpus,
chạy vòng lặp ICI, sinh câu trả lời qua cascade ACMG, đánh giá bằng EM/F1 đã chuẩn hoá cho tiếng
Việt, so sánh với hai baseline, phân tích lỗi theo khâu hỏng, và demo trên corpus tiếng Việt mới.

**Các bước tiếp theo**
- Xin dữ liệu ViMQA đầy đủ (10.047 mẫu) qua EULA và chạy lại với `DATASET = "vimqa"`.
- Chạy các ablation: `--ab fact|sentence|passage` (Table 3), `--no-reranker` (Table 2),
  `--it 2|4|8` (Figure 8).
- Trajectory Distillation: xem `src/vicirag/distill/` và mục 4 trong README.
"""
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Đã ghi {OUT} ({len(cells)} ô: "
      f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
