#!/usr/bin/env bash
# Tái lập trên 2WikiMultihopQA hoàn toàn local, không tốn API.
#
# Thiết lập gần bài báo gốc nhất mà một GPU 12GB chạy được: Qwen2.5-7B-Instruct qua
# Ollama làm mô hình nền cho mọi khâu (kể cả trích triple), bge-small-en-v1.5 làm
# retriever (đúng retriever ở Bảng 6 của bài báo), triple chuẩn hoá như repo gốc. Khác
# bài báo ở lượng tử hoá Q8_0.
#
# Bám mã nguồn gốc ở hai chỗ bài báo không nói, vì con số trong bài báo do mã nguồn đó
# sinh ra (`rerank_facts_gene_data` trong src/cirag/CIRAG.py):
#   --official-rescue  tập core triple rỗng thì lấy ứng viên đầu bảng, và vớt mọi triple
#                      mô hình trích trong phần suy luận. Không có nó, Qwen2.5-7B ghi tập
#                      rỗng ở 6-8/10 câu ngay vòng đầu (đo cả với Q4, Q8 và đúng prompt gốc).
#   --top-n 10         mã nguồn đưa 10 triple ứng viên vào LLM (`candidate_facts[:10]`),
#                      dù phụ lục B.4 ghi N = 30.
#
# Chuẩn bị một lần:
#   ollama pull qwen2.5:7b-instruct-q8_0
#   ollama create qwen2.5-7b-instruct-q8-greedy -f scripts/ollama/qwen2.5-7b-q8-greedy.Modelfile
#   OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 ollama serve
#     (mỗi luồng song song cần thêm bộ nhớ ngữ cảnh; 1 luồng mới nằm trọn 12GB VRAM)
#   python scripts/prepare_2wiki.py --limit 100
#
# Dùng:  PY=/đường/dẫn/python bash scripts/run_local_2wiki.sh
#        LIMIT=10 bash scripts/run_local_2wiki.sh     # chạy thử, ghi vào *_smoke
#
# Không sửa file này khi nó đang chạy: bash đọc script dần theo vị trí byte.

set -euo pipefail

PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-2}"
MODEL="${MODEL:-qwen2.5-7b-instruct-q8-greedy}"
LIMIT="${LIMIT:-}"

RUN="PYTHONPATH=src $PY -m vicirag.pipeline --dataset 2wiki --split dev --lang en \
  --llm-base-url http://localhost:11434/v1 --llm-name $MODEL --openie-llm-name $MODEL \
  --context-window 16384 --embedding-name BAAI/bge-small-en-v1.5 --embedding-max-length 512 \
  --triple-norm official --official-rescue --top-n 10 --workers $WORKERS --it 4"
SUFFIX=""
if [ -n "$LIMIT" ]; then
  RUN="$RUN --limit $LIMIT"
  SUFFIX="_smoke"
fi

run() {
  local name="local2w_$1$SUFFIX"; shift
  if [ -f "outputs/$name/summary.json" ]; then
    echo "== BỎ QUA $name (đã có kết quả)"
    return
  fi
  echo "== CHẠY $name"
  eval "$RUN --run-name $name $*"
}

# ---- Mốc chính: CIRAG đúng Eq. 6 (không mức dự phòng) so với IRCoT ----
# Đối chiếu: CIRAG w/o TD 49,8 EM / 59,7 F1 (Bảng 2), IRCoT 36,4 / 45,7 (Bảng 1).
run cirag_eq6 --method cirag --no-passage-fallback
run ircot     --method ircot

# ---- Cùng cấu hình với phép tái lập cũ (có mức dự phòng "retrieved") ----
# Đo riêng phần EM mà mức dự phòng, vốn không có trong bài báo, đóng góp trên 2Wiki.
run cirag_fallback --method cirag

# ---- Bảng 3: bóc tách cascade, cùng quỹ đạo ICI nên chỉ tốn lời gọi ACMG ----
# Tắt mức dự phòng để mọi biến thể đúng Eq. 6 như bài báo.
run ab_fact_sentence   --ab fact-sentence    --no-passage-fallback   # w/o Passage
run ab_sentence_passage --ab sentence-passage --no-passage-fallback  # w/o Triple
run ab_fact            --ab fact             --no-passage-fallback   # w/o Sentence + Passage
run ab_sentence        --ab sentence         --no-passage-fallback   # w/o Triple + Passage
run ab_passage         --ab passage          --no-passage-fallback   # w/o Triple + Sentence

# ---- Điều kiện dừng đề xuất, trên chính thiết lập của bài báo ----
# Mọi câu 2WikiMQA đều cần ít nhất 2 tài liệu. So với cirag_fallback.
run fix_hop --method cirag --min-source-docs 2

echo
PYTHONPATH=src $PY scripts/compare_local_2wiki.py ${LIMIT:+--suffix _smoke}
