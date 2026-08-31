#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm cho báo cáo.
#
# Thứ tự có chủ đích: lần chạy đầu tiên trích triple cho cả corpus và nhúng embedding;
# mọi lần sau đọc lại từ cache nên chỉ tốn lời gọi LLM cho phần suy luận.
#
# Dùng:  bash scripts/run_experiments.sh [DATASET] [SPLIT] [LIMIT]

set -euo pipefail

DATASET="${1:-vimqa}"
SPLIT="${2:-validation}"
WORKERS="${WORKERS:-8}"
RUN="PYTHONPATH=src .venv/bin/python -m vicirag.pipeline --dataset $DATASET --split $SPLIT --workers $WORKERS"

run() {
  local name="$1"; shift
  if [ -f "outputs/$name/summary.json" ]; then
    echo "== BỎ QUA $name (đã có kết quả)"
    return
  fi
  echo "== CHẠY $name"
  eval "$RUN --run-name $name $*"
}

# ---- Bảng chính: CiRAG so với baseline (dùng chung retriever + backbone) ----
run main_cirag  --method cirag  --it 4
run main_native --method native
run main_ircot  --method ircot  --it 4

# ---- Table 3: mức ngữ cảnh đơn lẻ và cascade hai mức ----
run ab_fact             --ab fact             --it 4
run ab_sentence         --ab sentence         --it 4
run ab_passage          --ab passage          --it 4
run ab_fact_sentence    --ab fact-sentence    --it 4
run ab_fact_passage     --ab fact-passage     --it 4

# ---- Cải tiến: điều kiện đủ theo số tài liệu nguồn (mục 8 của báo cáo) ----
# Bản gốc chỉ dừng khi mô hình trả về câu từ chối. Cờ dưới đây đòi thêm tập triple
# lõi phải đến từ ít nhất 2 tài liệu, đúng số tài liệu một câu hỏi đa bước cần.
run fix_hop --method cirag --it 4 --min-source-docs 2

# ---- Table 2: bỏ reranker triple ----
run ab_no_reranker --no-reranker --it 4

# ---- Bảng 1: bản đúng theo Eq. 6 của paper, chưa có mức dự phòng ----
run main_cirag_norescue --method cirag --it 4 --no-passage-fallback

# ---- Figure 8: ảnh hưởng của số vòng lặp L ----
run it1 --it 1
run it2 --it 2
run it8 --it 8

# ---- Bổ sung: truy hồi lai, và phạm vi triple ứng viên theo mã nguồn gốc ----
run ext_hybrid          --hybrid --it 4
run ext_scope_sample    --candidate-scope sample --it 4

# ---- Phép thử tái lập trên bộ dữ liệu gốc ----
# Chạy chính bản cài đặt này trên 2WikiMultihopQA với Qwen2.5-7B, đúng bộ dữ liệu và mô
# hình nền mà bài báo gốc dùng, rồi đối chiếu với con số họ công bố (w/o TD: 49,8 EM;
# IRCoT: 36,4 EM). Đây là phép kiểm trực tiếp cho tính trung thực của bản cài đặt.
if [ ! -f data/2wiki/processed/2wiki_dev.json ]; then
  .venv/bin/python scripts/prepare_2wiki.py --limit 100
fi
R2W="PYTHONPATH=src .venv/bin/python -m vicirag.pipeline --dataset 2wiki --split dev \
     --lang en --workers $WORKERS --llm-name qwen/qwen-2.5-7b-instruct \
     --openie-llm-name google/gemini-2.5-flash-lite --it 4"
for m in cirag ircot; do
  n="repro_2wiki_$m"
  [ -f "outputs/$n/summary.json" ] || eval "$R2W --method $m --run-name $n"
done

echo
echo "===== TỔNG HỢP ====="
PYTHONPATH=src .venv/bin/python scripts/collect_results.py
