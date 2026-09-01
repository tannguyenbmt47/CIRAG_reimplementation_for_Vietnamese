# Kết quả thực nghiệm

Dữ liệu: ViMQA split `validation`, 500 câu hỏi lấy ngẫu nhiên (seed 42), corpus truy hồi 1.969
đoạn khử trùng lặp từ 5.000 đoạn supporting + distractor — đúng setting IRCoT mà paper dùng.

| Thành phần | Cấu hình |
|---|---|
| Suy luận (Integration + ACMG) | `google/gemini-2.5-flash` qua OpenRouter, temperature 0 |
| Trích triple | `google/gemini-2.5-flash-lite` (rẻ hơn ~6× ở token đầu ra) |
| Nhúng | `AITeamVN/Vietnamese_Embedding` (BGE-M3 fine-tune tiếng Việt) |
| Siêu tham số | K=10, N=30, L=4 (theo Appendix B.4) |
| Triple trích được | **22.697** từ 1.969 đoạn (~11,5/đoạn); 6 đoạn không ra triple nào |
| Tổng chi phí API | ~$4,5 cho toàn bộ 11 thí nghiệm |

Mọi phương pháp dùng **chung retriever, chung corpus, chung mô hình nền và chung bước chuẩn hoá
đáp án** — điều kiện so sánh công bằng mà paper đặt ra ở Sec 4.2.

---

## 1. Bảng kết quả đầy đủ

| Cấu hình | EM | F1 (âm tiết) | F1 (từ) | recall | hop TB |
|---|---:|---:|---:|---:|---:|
| **Đường cơ sở** | | | | | |
| NativeRAG | 55.20 | 67.95 | 67.23 | 0.920 | 1.00 |
| IRCoT | **59.00** | 73.27 | 72.27 | 0.974 | 2.72 |
| **ViCiRAG** | | | | | |
| ViCiRAG (đầy đủ) | 57.00 | 73.05 | 72.41 | 0.973 | 2.08 |
| ViCiRAG *không có cơ chế dự phòng* | 50.80 | 66.66 | 66.11 | 0.973 | 2.08 |
| **Mức ngữ cảnh đơn lẻ** | | | | | |
| chỉ mức triple | 57.40 | 72.99 | 72.39 | 0.973 | 2.08 |
| chỉ mức câu | 60.80 | **75.38** | 74.54 | 0.973 | 2.08 |
| chỉ mức đoạn | **61.20** | 75.16 | 74.52 | 0.973 | 2.08 |
| **Cascade hai mức** | | | | | |
| triple + câu | 57.00 | 73.03 | 72.38 | 0.973 | 2.08 |
| triple + đoạn | 56.80 | 72.78 | 72.16 | 0.973 | 2.08 |
| **Số vòng lặp L** | | | | | |
| L = 1 | 52.60 | 67.52 | 66.70 | 0.920 | 1.00 |
| L = 2 | 57.40 | 73.18 | 72.70 | 0.969 | 1.45 |
| L = 4 (mặc định) | 57.00 | 73.05 | 72.41 | 0.973 | 2.08 |

---

## 2. Bốn phát hiện chính

### 2.1. Không tái lập được lợi thế mà paper báo cáo

ViCiRAG (57.00 EM) vượt NativeRAG (55.20) nhưng **thua IRCoT** (59.00). Trong khi đó paper báo
cáo CIRAG vượt IRCoT tới 23,8 điểm EM trên 2WikiMQA với Qwen2.5-7B.

Đây **không phải lỗi cài đặt**, và có ba bằng chứng cho điều đó:

1. Recall truy hồi của ViCiRAG là 0.973 — cao nhất trong ba phương pháp.
2. Trích triple chỉ gây 2,0% số lỗi (xem §3).
3. Phân bố mức ngữ cảnh khớp sát Figure 3 của paper: triple 72,2% so với 76,3% trên 2WikiMQA,
   câu 10,8% so với 10,2%. Cơ chế cascade chạy đúng như thiết kế.

Các bộ phận đều hoạt động đúng. Vấn đề nằm ở chỗ khác — xem §2.3 và §2.4.

### 2.2. Cơ chế dự phòng thu hẹp phần lớn khoảng cách: +6,2 EM

Bản cài đặt bám sát Eq. 6 để nhánh mặc định trả về đáp án mức tài liệu dựng từ *core triple*.
Nhưng khi Integration Phase không giữ được triple nào liên quan, tập tài liệu đó rỗng và hệ thống
chắc chắn từ chối — dù retriever **đã** lấy về đúng đoạn chứa bằng chứng.

Chúng tôi bổ sung một mức dự phòng cuối: dùng lại **toàn bộ đoạn đã truy hồi**, không lọc theo
core triple.

| | EM | F1 | nhóm `default` | lỗi "ACMG không mức nào đủ" |
|---|---:|---:|---:|---:|
| Không có dự phòng | 50.80 | 66.66 | 73 (14,6%) | 56 (11,2%) |
| Có dự phòng | **57.00** | **73.05** | 35 (7,0%) | 18 (3,6%) |

### 2.3. Nguyên nhân gốc: pha tích hợp đánh mất nhánh bằng chứng song song

Phân tích 73 ca rơi vào nhánh mặc định (bản chưa có dự phòng): **71/73 dùng hết cả 4 hop**, và
phần lớn là **câu hỏi so sánh hai thực thể** ("Roy Makaay nhỏ tuổi hơn Gary Lineker phải không?").

Recall truy hồi của nhóm này bằng **1.0** — cả hai đoạn bằng chứng đều đã nằm trong tay — nhưng
Integration Phase chỉ giữ lại triple về **một** thực thể. ACMG nhận được ngữ cảnh thiếu hẳn một
vế nên từ chối, và từ chối là hành xử đúng trên ngữ cảnh đó.

Đây chính là **greedy single-path expansion** mà paper đặt vấn đề và tuyên bố khắc phục. Kết quả
cho thấy việc điều kiện hoá theo lịch sử chưa đủ để đảm bảo giữ nhánh bằng chứng song song, ít
nhất khi prompt few-shot chỉ chứa ví dụ dạng bắc cầu chứ không có ví dụ so sánh.

### 2.4. Cascade gây hại: tiêu chí thoát sớm quá dễ dãi

Ngược hẳn Table 3 của paper, **mọi biến thể một mức đều thắng mọi biến thể cascade**:

| | EM |
|---|---:|
| chỉ mức đoạn | **61.20** |
| chỉ mức câu | 60.80 |
| chỉ mức triple | 57.40 |
| triple + câu | 57.00 |
| triple + đoạn | 56.80 |
| cascade đầy đủ | 57.00 |

Giải thích: hàm đủ-thông-tin `Suf(a)` ở Eq. 6 giả định mô hình biết tự nhận là thiếu dữ kiện và
trả về mẫu từ chối. Giả định đó **không đúng với mô hình nền mạnh** — `gemini-2.5-flash` có xu
hướng cứ trả lời từ triple thay vì nói không đủ thông tin. Cascade do đó kẹt lại ở mức triple với
đáp án sai, không bao giờ leo lên mức câu hay đoạn vốn có ngữ cảnh đầy đủ hơn.

Đây là cơ chế hỏng **thứ hai và độc lập** với §2.3, nhưng cùng chỉ về một kết luận: các thành
phần của CiRAG được thiết kế quanh giả định mô hình nền yếu, và giả định đó không còn đúng.

### 2.5. Kiểm chứng giả thuyết: hiệu ứng tương tác giữa phương pháp và mô hình nền

Các mục trên đưa ra giả thuyết rằng lợi ích của CiRAG phụ thuộc vào năng lực xử lý ngữ cảnh dài
của mô hình nền. Để kiểm chứng, chúng tôi lặp lại toàn bộ ba phương pháp với
`qwen/qwen-2.5-7b-instruct` — **đúng mô hình nền mà paper sử dụng**. Chỉ đổi mô hình suy luận;
mô hình trích triple giữ nguyên `gemini-2.5-flash-lite` để chất lượng triple không trở thành
biến gây nhiễu.

| Phương pháp | EM (Gemini) | EM (Qwen) | F1 (Gemini) | F1 (Qwen) |
|---|---:|---:|---:|---:|
| NativeRAG | 55.20 | 29.80 | 67.95 | 43.51 |
| IRCoT | 59.00 | 39.80 | 73.27 | 52.87 |
| ViCiRAG | 57.00 | 38.60 | 73.05 | 54.01 |
| **Chênh lệch ViCiRAG − NativeRAG** | **+1.80** | **+8.80** | **+5.10** | **+10.50** |

**Giả thuyết được xác nhận.** Lợi thế của ViCiRAG so với NativeRAG **co lại 5 lần** khi đổi từ mô
hình nền yếu sang mạnh (+8,80 xuống +1,80 EM). Với mô hình nền yếu, việc nén ngữ cảnh thành triple
mang lại lợi ích ròng lớn; với mô hình đọc tốt ngữ cảnh dài, phần thông tin mất đi khi nén gần như
triệt tiêu lợi ích đó. Về F1, ViCiRAG thậm chí **vượt IRCoT** ở mô hình nền yếu (54.01 so với
52.87) trong khi thua ở mô hình mạnh (73.05 so với 73.27).

#### Bằng chứng độc lập thứ hai: phân bố mức ngữ cảnh dịch chuyển đúng hướng

| Mức | Gemini 2.5 Flash | Qwen2.5-7B |
|---|---:|---:|
| triple | 72,2% | 46,2% |
| câu | 10,8% | 19,0% |
| đoạn | 2,4% | 8,2% |
| dự phòng `retrieved` | 7,6% | 8,6% |
| default | 7,0% | 18,0% |

Với mô hình nền yếu, cascade mở rộng lên mức câu và đoạn nhiều gấp hai đến ba lần. Nghĩa là
ACMG **đang hoạt động đúng thiết kế**: nó nhận ra triple không đủ và mở rộng ngữ cảnh. Với mô hình
mạnh, cơ chế này gần như không kích hoạt vì mô hình cứ trả lời từ triple thay vì báo thiếu thông
tin — xác nhận trực tiếp chẩn đoán ở §2.4 về việc hàm `Suf()` quá dễ dãi.

#### Loại trừ nhiễu định dạng

Một cách giải thích cạnh tranh là Qwen chỉ đơn giản không tuân thủ được định dạng output
`[[ ## ... ## ]]` của DSPy. Chúng tôi bác bỏ khả năng này bằng đo đạc:

| Mô hình nền | Số bước tích hợp | Parse đúng định dạng | Sinh `<no question>` |
|---|---:|---:|---:|
| Gemini 2.5 Flash | 1.039 | 100,0% | 33,5% |
| Qwen2.5-7B | 964 | **99,9%** | 11,3% |

Qwen tuân thủ định dạng gần như tuyệt đối. Khác biệt nằm ở **năng lực suy luận**, không phải ở
khả năng tuân thủ chỉ dẫn. (Đáng chú ý: Qwen chỉ sinh `<no question>` ở 11,3% số bước so với 33,5%
của Gemini — mô hình yếu kém hẳn trong việc tự nhận biết khi nào đã đủ dữ kiện để dừng.)

---

## 3. Phân tích lỗi theo khâu hỏng (500 câu, bản đầy đủ)

| Loại | Số câu | Tỷ lệ |
|---|---:|---:|
| Đúng | 285 | 57,0% |
| Integration/suy luận sai | 95 | 19,0% |
| Sai do chuẩn hoá/diễn đạt | 76 | 15,2% |
| ACMG không mức nào đủ | 18 | 3,6% |
| Truy hồi trượt | 11 | 2,2% |
| Trích triple sót | 10 | 2,0% |
| Cascade hụt, cứu bằng đoạn truy hồi nhưng vẫn sai | 5 | 1,0% |

Phân bố mức ngữ cảnh sau khi thêm cơ chế dự phòng: triple 361 (72,2%), câu 54 (10,8%),
đoạn 12 (2,4%), **dự phòng `retrieved` 38 (7,6%)**, default 35 (7,0%). Trong 38 ca được cơ chế
dự phòng xử lý, **31 ca cho đáp án đúng** — tỷ lệ cứu thành công 82%. Tỷ lệ cao như vậy xác nhận
chẩn đoán ở §2.3: bằng chứng vốn đã nằm sẵn trong các đoạn đã truy hồi, chỉ là pha tích hợp đã
loại bỏ mất.

**Truy hồi và trích triple gộp lại chỉ gây 4,2% lỗi.** Encoder tiếng Việt và prompt OpenIE làm tốt
phần việc của chúng; nút thắt nằm ở Integration Phase (20,0%) và ở chuẩn hoá đáp án (15,2%).

### Số vòng lặp là triệu chứng, không phải nguyên nhân

Đo trên bản chưa có dự phòng, EM theo số hop thực tế: 1 hop 55.8 — 2 hop **68.7** — 4 hop **33.8**.
Câu nào chạy hết 4 vòng là câu vòng lặp không hội tụ (mô hình không bao giờ sinh `<no question>`),
và gần như chắc chắn sai.

Điều này khớp với bảng L ở §1: L=1 (52.60) → L=2 (57.40) → L=4 (57.00), tức **bão hoà ngay từ
L=2**, đúng như Appendix C.3 của paper nhận định. Tăng L không giúp gì vì nút thắt là chất lượng
tích hợp chứ không phải số vòng.

---

## 4. Chất lượng truy hồi

Đo riêng, không tốn lời gọi LLM.

| Cấu hình | recall@5 | recall@10 | recall@20 |
|---|---:|---:|---:|
| dense | 0.880 | 0.920 | 0.947 |
| hybrid BM25 + dense | **0.889** | **0.924** | **0.959** |

Tỷ lệ câu hỏi lấy đủ **cả hai** đoạn vàng ở k=10: 85,8% (dense), 86,2% (hybrid). Truy hồi lai hơn
dense ổn định ở mọi mức k, rõ nhất ở k=20 (+1,2 điểm) — hợp lý vì BM25 bắt được khớp từ khoá chính
xác mà embedding bỏ sót, đặc biệt với tên riêng tiếng Việt.

Lưu ý: recall của ViCiRAG trong bảng §1 (0.973) cao hơn recall@10 một lần (0.920) vì ICI truy hồi
nhiều lần qua các hop và tích luỹ đoạn.

---

## 5. Phát hiện phương pháp luận: chuẩn hoá đáp án chi phối kết quả

Ở lần chạy thử ban đầu, chỉ ViCiRAG có bước chuẩn hoá đáp án (`reanswer`) còn đường cơ sở thì
không:

| Phương pháp | EM (không reanswer) | EM (có reanswer) | Chênh |
|---|---:|---:|---:|
| ViCiRAG | 40.00 | 60.00 | +20,0 |
| IRCoT | 10.00 | 60.00 | **+50,0** |

Giữ nguyên thiết lập thiếu công bằng đó, ViCiRAG sẽ "thắng" IRCoT 60 so với 10 — một kết luận
hoàn toàn sai. Riêng việc rút đáp án về cụm ngắn đã dịch chuyển EM tới 50 điểm.

Hệ quả: với tiếng Việt, phần lớn khoảng cách EM giữa các phương pháp có thể chỉ đến từ **định
dạng đáp án** chứ không phải chất lượng suy luận. Mọi so sánh bắt buộc phải dùng chung quy trình
chuẩn hoá, và nên báo cáo cả F1 bên cạnh EM. Trong bảng §1, chênh lệch EM−F1 khoảng 16 điểm cho
thấy còn nhiều đáp án đúng nội dung nhưng lệch cách diễn đạt.

---

## 6. Lưu ý khi đọc lại số

- Cột "thời gian/câu" trong `summary.json` **không dùng được** khi `num_cache_hits` gần bằng
  `num_calls` — lúc đó pipeline chỉ đọc cache SQLite. Muốn đo độ trễ thật (dựng lại Figure 6 của
  paper), phải xoá `.cache/llm_cache_*.sqlite` trước khi chạy.
- Thư mục `outputs/_smoketest_10q/` chứa các lần chạy thử trên 10 mẫu công khai. **Không dùng
  những số đó trong báo cáo** — ở n=10 một câu hỏi đáng 10 điểm EM.

## 7. Việc còn lại

- [ ] Trajectory Distillation: sinh quỹ đạo teacher, LoRA fine-tune student, đo ablation w/ TD.
- [ ] Thí nghiệm kiểm chứng giả thuyết §2.4: chạy lại với mô hình nền yếu (`qwen-2.5-7b-instruct`,
      đúng mô hình paper dùng). Nếu CiRAG **thắng** đường cơ sở ở mô hình yếu và **thua** ở mô hình
      mạnh thì giả thuyết được chứng minh trực tiếp thay vì chỉ là suy đoán.
- [ ] Ablation `--no-reranker`, `--hybrid`, `--candidate-scope sample` (mỗi cái ~$5 vì phải chạy
      lại ICI từ đầu).
- [ ] Đo độ trễ thật với cache đã xoá.
- [ ] Điền số vào `report/main.tex`.

---

# Phụ lục: phép thử tái lập trên bộ dữ liệu gốc

## Vì sao cần

Toàn bộ kết luận của báo cáo dựa trên tiền đề "bản cài đặt của chúng tôi trung thực với
bài báo gốc". Cho tới đây, tiền đề đó chỉ được chống đỡ bằng **bằng chứng gián tiếp**:
recall truy hồi cao, phân bố mức ngữ cảnh khớp Figure 3, trích triple chỉ gây 2% lỗi.

Phép kiểm trực tiếp là chạy chính bản cài đặt này trên **bộ dữ liệu và mô hình nền mà bài
báo gốc dùng**, rồi so với con số họ công bố.

## Thiết lập

| | |
|---|---|
| Dữ liệu | 2WikiMultihopQA, 100 câu lấy ngẫu nhiên từ dev (seed 42) |
| Kho ngữ liệu | 770 đoạn, hợp nhất từ 100 câu rồi khử trùng lặp |
| Mô hình nền | `qwen-2.5-7b-instruct` — đúng mô hình bài báo gốc dùng |
| Prompt | tiếng Anh, lấy từ `reference/CIRAG-official/` |
| Mốc đối chiếu | CIRAG **w/o TD** trên 2WikiMQA: **49,8 EM** (Table 2 của bài báo gốc) |

Dùng mốc `w/o TD` vì bản cài đặt của chúng tôi chưa có module Chưng cất Quỹ đạo.

## Phép thử này kết luận được gì

Thiết lập của chúng tôi **dễ hơn** của bài báo gốc: họ dựng kho ngữ liệu từ 1.000 câu
(khoảng 6.000 đoạn), chúng tôi từ 100 câu (770 đoạn), nên truy hồi ít nhiễu hơn hẳn. Do đó
phép thử chỉ có giá trị **một chiều**:

- Nếu EM **xấp xỉ hoặc cao hơn 49,8**: bản cài đặt vận hành đúng như mô tả. Đây không phải
  bằng chứng tuyệt đối vì thiết lập dễ hơn, nhưng đủ để loại bỏ nghi ngờ có lỗi nghiêm trọng.
- Nếu EM **thấp hơn 49,8 đáng kể**: bản cài đặt có vấn đề. Được ưu thế về độ khó mà vẫn
  thua thì không thể đổ cho thiết lập.

Với $n = 100$, sai số chuẩn của EM khoảng 5 điểm, nên chỉ khoảng cách lớn hơn mức đó mới
đáng diễn giải.

## Kết quả

Chạy ngày hoàn tất, chi phí $0,24 cho cả hai lần chạy.

| | Của chúng tôi (100 câu) | Bài báo gốc (1.000 câu) | Chênh |
|---|---:|---:|---:|
| CIRAG w/o TD, EM | 54.00 | 49,8 | +4.2 |
| CIRAG w/o TD, F1 | 57.97 | 59,7 | -1.7 |
| IRCoT, EM | 40.00 | 36,4 | +3.6 |
| IRCoT, F1 | 44.63 | 45,7 | -1.1 |

Cả bốn con số nằm trong khoảng ±4,2, dưới sai số chuẩn khoảng 5 điểm ở $n = 100$. EM của
chúng tôi nhỉnh hơn đúng như dự đoán từ thiết lập dễ hơn.

**Chỉ số quyết định là khoảng cách giữa hai phương pháp**, vì nó triệt tiêu ảnh hưởng của
cỡ kho ngữ liệu:

| | Của chúng tôi | Bài báo gốc | Lệch |
|---|---:|---:|---:|
| CIRAG − IRCoT, EM | **+14.0** | **+13,4** | 0.6 |
| CIRAG − IRCoT, F1 | +13.3 | +14,0 | 0.7 |

Lệch dưới 1 điểm ở cả hai chỉ số. Bản cài đặt tái lập được lợi thế mà bài báo gốc công bố.

> **Lưu ý bổ sung.** Lần chạy CIRAG này bật mức dự phòng `retrieved` (20% số câu trả lời đến
> từ mức đó, vốn không có trong bài báo) và rút gọn đáp án tiếng Anh bằng prompt tiếng Việt,
> nên con số 54,00 không phải CIRAG đúng Eq. 6. Phép so sánh đúng cấu hình nằm ở phụ lục
> *Tái lập local, bám mã nguồn gốc* cuối file.

## Ý nghĩa cho toàn bộ báo cáo

Cùng một dòng mã, hai thiết lập:

| Thiết lập | CIRAG − IRCoT |
|---|---:|
| 2WikiMQA, tiếng Anh, Qwen2.5-7B | **+14,0 EM** |
| ViMQA, tiếng Việt, Gemini 2.5 Flash | **−2,0 EM** |

Lợi thế 14 điểm biến mất hoàn toàn. Kết luận của báo cáo do đó không còn là "chúng tôi cài
lại và không thu được kết quả như bài báo gốc", vốn dễ bị quy cho lỗi cài đặt, mà là
"chúng tôi **tái lập được** lợi thế đó rồi chỉ ra nó biến mất khi đổi thiết lập".

**Giới hạn.** Hai thiết lập khác nhau ở cả ngôn ngữ lẫn mô hình nền, nên chưa tách được
phần đóng góp của từng yếu tố. Thí nghiệm Qwen trên ViMQA (§2.5) cho thấy ngay với mô hình
nền yếu, trên tiếng Việt CIRAG cũng chỉ hơn NativeRAG 8,8 điểm chứ không đạt mức của tiếng
Anh, gợi ý cả hai yếu tố cùng góp phần.

## Phân bố mức ngữ cảnh

Trên 2WikiMQA với Qwen2.5-7B: triple 41%, câu 13%, đoạn 5%, dự phòng 20%, không mức nào đủ
21%. Tỷ lệ dùng mức triple (41%) thấp hơn hẳn con số 76,3% mà Figure 3 của bài báo gốc ghi
nhận, phù hợp với quan sát ở §2.5 rằng mô hình nền yếu mở rộng ngữ cảnh nhiều hơn.

---

# Phụ lục: prompt tiếng Việt có phải nút thắt không?

## Mối đe doạ cần loại trừ

Prompt tiếng Việt trong báo cáo do chúng tôi tự viết, còn prompt tiếng Anh là của nhóm tác
giả. Nếu prompt của chúng tôi kém hơn, thì cái gọi là "bất lợi của tiếng Việt" thực chất
là "bất lợi của prompt chưa tối ưu", và kết luận của báo cáo mất hiệu lực.

## Thiết lập

Chỉ đổi **một** biến. Giữ nguyên dữ liệu (500 câu ViMQA), mô hình nền (Gemini 2.5 Flash),
siêu tham số, prompt trích triple (tiếng Việt, dùng lại cache) và bước rút gọn câu trả lời.
Chỉ prompt của ICI và ACMG đổi sang bản tiếng Anh gốc, kèm một câu buộc trả lời bằng
tiếng Việt.

## Kết quả

| | EM | F1 | Nhánh mặc định | Mức triple |
|---|---:|---:|---:|---:|
| Prompt tiếng Việt (chúng tôi viết) | 57,00 | 73,05 | 35 | 361 |
| Prompt tiếng Anh (của tác giả) | 57,40 | 73,38 | 37 | 353 |

Kiểm định McNemar ghép cặp trên cùng 500 câu: 18 câu chỉ prompt tiếng Anh đúng, 16 câu
chỉ prompt tiếng Việt đúng, **p = 0.864**. Chênh lệch không phân biệt được với ngẫu
nhiên, và phân bố mức ngữ cảnh gần như không đổi.

**Kết luận: prompt tiếng Việt không phải nút thắt.**

Một quan sát phụ: prompt tiếng Anh làm tăng số đáp án thuần ASCII từ 154 lên 183, tức mô
hình trả lời lẫn tiếng Anh nhiều hơn dù đã có chỉ dẫn. Phần lớn là tên riêng nên EM không
bị ảnh hưởng, nhưng đây là lý do chúng tôi giữ prompt tiếng Việt cho cấu hình chính.

## Bốn cách giải thích đã bị loại trừ

| Giả thuyết | Bằng chứng bác bỏ |
|---|---|
| Lỗi cài đặt | Tái lập được +14,0 EM trên 2WikiMQA so với +13,4 của bài báo gốc |
| Prompt tiếng Việt kém | $p = 0.864$ |
| Một khâu pipeline chạy sai | Recall 0,96, tỷ lệ giữ triple 75%, mức mở rộng của ACMG đều tương đương giữa hai ngôn ngữ |
| Năng lực mô hình nền | Đổi Qwen sang Gemini chỉ dịch chuyển khoảng cách 0,8 điểm |

Điều duy nhất còn lại và đạt ý nghĩa thống kê là nhóm **câu hỏi mở** (−4,9 điểm,
$p = 0{,}031$), tức nhóm phải sinh ra một cụm từ cụ thể thay vì chọn giữa hai đáp án.

---

# Cải tiến đề xuất: thay điều kiện đủ bằng phép kiểm bằng chứng

## Chẩn đoán

Phương trình 6 định nghĩa `Suf(a) = 1 - 1[a = "Unanswerable"]`. Nó hỏi **mô hình có chịu
trả lời không**, chứ không hỏi **bằng chứng có đủ để trả lời không**. Hai thứ này khác nhau,
và sự lẫn lộn đó sinh ra đúng hai dạng hỏng, đối xứng nhau:

| | Mô hình từ chối | Mô hình trả lời |
|---|---|---|
| **Bằng chứng đủ** | (hiếm) | đúng, không cần can thiệp |
| **Bằng chứng thiếu** | Eq. 6 bắt được, nhưng cascade hết mức thì bó tay | **Eq. 6 không bắt được** |

Ô dưới bên phải là chỗ hỏng lớn nhất: `gemini-2.5-flash` gần như không bao giờ từ chối, nên
cascade chốt ngay ở mức triple với câu trả lời sai. 361/500 câu dừng ở mức triple.

## Vì sao không phải là ảo giác

Giả thuyết đầu tiên của chúng tôi là mô hình bịa đáp án ngoài ngữ cảnh. Kiểm tra trên
361 câu dừng ở mức triple cho thấy giả thuyết này **sai**:

| Mức bám của đáp án vào tập triple lõi | n | EM |
|---|---|---|
| bám hoàn toàn | 252 | 46,8 |
| bám một phần | 5 | 20,0 |
| không bám | 5 | 0,0 |
| câu đúng/sai (không kiểm được bằng chuỗi) | 99 | 92,9 |

Chỉ 10/361 câu không bám. Đáp án **có** nằm trong triple, nhưng là triple sai. Vấn đề nằm ở
tập triple chứ không ở bước sinh.

## Tín hiệu thay thế: số tài liệu nguồn

Tách 361 câu đó theo số đoạn nguồn của tập triple lõi (lấy từ ánh xạ provenance `M^D` ở
Eq. 3, phương pháp vốn đã dựng sẵn):

| Số đoạn nguồn của tập triple lõi | n | EM |
|---|---|---|
| 1 đoạn | 115 | **35,7** |
| 2 đoạn | 204 | 71,6 |
| 3 đoạn trở lên | 42 | 57,1 |

ViMQA là bộ đa bước: mọi câu hỏi đều cần bằng chứng từ ít nhất hai đoạn. Tập triple lõi chỉ
đến từ một đoạn thì về mặt cấu trúc là chưa đủ, **bất kể mô hình có sẵn lòng trả lời**.
Đây là tính chất kiểm tra được, khác hẳn việc chờ mô hình chịu thừa nhận.

## Điều kiện đủ mới

```
Suf(a, T) = (a không phải chuỗi từ chối)  AND  (|M^D(T)| >= h)
```

với `h` là số tài liệu mà câu hỏi đòi hỏi (`h = 2` cho bộ đa bước). Không thoả thì bỏ qua các
mức hẹp và đọc thẳng mức rộng nhất. Phép kiểm **không tốn thêm lời gọi mô hình nào**.

Cài đặt: `--min-source-docs 2`. Mặc định 0 (tắt) để bản tái lập giữ đúng Eq. 6.

## Kết quả (ViMQA, 500 câu, Gemini 2.5 Flash)

| Cấu hình | EM | F1 | so với CIRAG gốc |
|---|---|---|---|
| CIRAG đúng Eq. 6 như công bố | 50,80 | — | |
| + mức dự phòng (ô dưới bên **trái**) | 57,00 | 73,05 | +6,2 |
| + điều kiện đủ theo số nguồn (ô dưới bên **phải**) | **60,40** | **75,08** | **+9,6** |
| IRCoT | 59,00 | 73,27 | |
| NativeRAG | 55,20 | 67,95 | |
| ablation: luôn đọc mức đoạn | 61,20 | 75,16 | |

Kiểm định McNemar ghép cặp trên cùng 500 câu:

| So sánh | Δ EM | chỉ A đúng | chỉ B đúng | p |
|---|---|---|---|---|
| đề xuất vs CIRAG gốc | **+3,40** | 20 | 3 | **0,0005** |
| đề xuất vs NativeRAG | **+5,20** | 45 | 19 | **0,0016** |
| đề xuất vs IRCoT | +1,40 | 34 | 27 | 0,44 |
| đề xuất vs luôn đọc mức đoạn | −0,80 | 11 | 15 | 0,56 |
| *(để đối chiếu)* CIRAG gốc vs IRCoT | −2,00 | 31 | 41 | 0,29 |

Phân bố mức sau cải tiến: `fact` 246, `passage` 130, `retrieved` 46, `sentence` 42,
`default` 36. Số câu phải đọc ngữ cảnh rộng giảm còn 212/500 thay vì 500/500 của ablation
"luôn đọc mức đoạn".

## Giới hạn phải nói rõ

Cải tiến **không** vượt được ablation "luôn đọc mức đoạn" (60,40 so với 61,20, p = 0,56).
Nghĩa là: điều kiện đủ sửa lại giúp cascade thôi gây hại và đạt ngang mức đơn giản nhất,
nhưng **chưa chứng minh được cơ chế cascade đáng với độ phức tạp của nó** trên bộ dữ liệu này.
Lợi ích còn lại chỉ là chi phí: 212/500 câu phải đọc ngữ cảnh rộng thay vì 500/500, tương ứng
tổng token đầu vào thấp hơn 2,3%. Mức tiết kiệm nhỏ vì giai đoạn ICI mới là phần chiếm phần
lớn token, không phải bước sinh câu trả lời.

Ngưỡng `h = 2` lấy từ cấu trúc đã biết của bộ dữ liệu (mọi câu ViMQA đều 2 bước), **không**
dò trên tập đánh giá, nên không phải siêu tham số được tinh chỉnh trên chính tập báo cáo.

## Chi phí

Lần chạy này tốn **$0**: 2166/2166 lời gọi đều trúng cache, vì các câu được mở rộng ngữ cảnh đã được
tính sẵn ở lần chạy ablation "chỉ mức đoạn".

## Cải tiến có giữ được trên mô hình nền khác không?

Chạy lại cả ba cấu hình với `qwen/qwen3-30b-a3b-instruct-2507`, giữ nguyên mô hình trích
triple (`gemini-2.5-flash-lite`) để chỉ đổi đúng một biến.

| Mô hình nền | IRCoT | CIRAG gốc | CIRAG + cải tiến | Δ cải tiến | p |
|---|---|---|---|---|---|
| Gemini 2.5 Flash | 59,00 | 57,00 | **60,40** | +3,40 | **0,0005** |
| Qwen3-30B-A3B | 55,60 | 55,40 | **57,00** | +1,60 | 0,12 |

Ba điều đọc được:

1. **Cùng chiều trên cả hai mô hình nền**, nhưng chỉ đạt ý nghĩa thống kê trên Gemini. Trên
   Qwen3 chưa đủ bằng chứng để loại giả thuyết không có hiệu ứng.
2. **Điểm đến giống hệt nhau**: sau cải tiến, CIRAG hơn IRCoT đúng 1,4 điểm trên **cả hai**
   mô hình nền ($p = 0{,}44$ và $p = 0{,}49$). Trước cải tiến, CIRAG ở mức bằng hoặc dưới
   IRCoT trên cả hai (−2,0 và −0,2).
3. **Phân bố mức dịch chuyển mạnh hơn trên Qwen3**: số câu chốt ở mức triple giảm 400 → 212
   (Gemini: 361 → 246), tức quy tắc mới kích hoạt trên gần một nửa số câu, nhưng phần được
   cứu lại ít hơn. Với mô hình nền yếu hơn, mở rộng ngữ cảnh không giúp được nhiều vì chỗ
   nghẽn đã chuyển sang khâu suy luận.

Chi phí ba lần chạy Qwen3: **$0,28**.

---

# Phụ lục: tái lập local, bám mã nguồn gốc

## Mục đích

Phép thử tái lập ở phụ lục đầu có hai lỗ hổng: CIRAG chạy kèm mức dự phòng không có trong bài
báo, và đáp án tiếng Anh được rút gọn bằng prompt tiếng Việt. Phụ lục này chạy lại trên đúng
100 câu đó với cấu hình gần bài báo nhất có thể, hoàn toàn local, và đối chiếu thêm Bảng 3.

## Thiết lập

| | |
|---|---|
| Dữ liệu | 2WikiMultihopQA, đúng 100 câu của phép thử cũ (seed 42), kho 770 đoạn |
| Mô hình nền | Qwen2.5-7B-Instruct, lượng tử hoá Q8_0, qua Ollama, giải mã tham lam; dùng cho mọi khâu kể cả trích triple |
| Retriever | `BAAI/bge-small-en-v1.5`, retriever ở Bảng 6 của bài báo |
| Bám mã nguồn gốc | triple chữ thường, bỏ ký tự đặc biệt; N = 10 triple ứng viên; cơ chế cứu tập core triple |
| Prompt | prompt tiếng Anh của tác giả; rút gọn đáp án bằng prompt tiếng Anh |
| CIRAG | đúng Eq. 6, không mức dự phòng (`--no-passage-fallback`) |
| Chi phí | $0, một GPU 12GB |

Chạy lại: `scripts/run_local_2wiki.sh`, đối chiếu: `scripts/compare_local_2wiki.py`.

## Năm chỗ phải sửa trước khi con số có nghĩa

Lần chạy thử đầu cho CIRAG 0/10 câu đúng. Lần theo nguyên nhân thì gặp năm chỗ, ba chỗ thuộc
về chính bài báo và mã nguồn gốc:

| | Phát hiện | Bằng chứng |
|---|---|---|
| 1 | **Mã nguồn gốc có hai cơ chế cứu tập core triple mà bài báo không nhắc tới.** Hàm `rerank_facts_gene_data` vớt mọi triple mô hình trích trong phần suy luận, và lấy ứng viên đầu bảng khi tập lõi vẫn rỗng | Qwen2.5-7B ghi `{"fact": []}` ở 6/10 câu ngay vòng đầu, thường sau khi đã nêu đúng triple bắc cầu trong suy luận. Trên 10 câu thử: không có cơ chế này (Q4, N = 30), 9/10 câu rơi vào default và EM = 0; có cơ chế này và N = 10 (Q8), còn 5/10 câu default và EM = 40 |
| 2 | **Mã nguồn đưa 10 triple ứng viên vào LLM**, dù phụ lục B.4 ghi N = 30 | `candidate_facts[:10]` trong cùng hàm |
| 3 | **Bản port rút gọn demo của prompt Integration** so với `filter.json`, dù chú thích ghi là chép nguyên văn | Chạy đúng prompt gốc không tốt hơn: 8/10 câu rỗng ở vòng đầu, so với 6/10 |
| 4 | **Ollama mặc định phạt lặp 1.1**, áp cả khi temperature = 0. Bước Integration phải chép triple, vốn lặp tên thực thể | Vòng ICI đầu trên 10 câu: tổng 7 triple khi phạt, 40 khi tắt |
| 5 | **Bước rút gọn đáp án luôn dùng prompt tiếng Việt**, dặn trả "đúng"/"không" cho câu hỏi đúng/sai, kể cả với dữ liệu tiếng Anh | Đã thêm prompt tiếng Anh, chỉ dùng khi `--lang en` |

Lượng tử hoá **không** phải nguyên nhân: Q4_K_M và Q8_0 cho đúng cùng 6/10 câu rỗng ở vòng
ICI đầu. Cơ chế cứu và N = 10 là cờ riêng (`--official-rescue`, `--top-n 10`), mặc định tắt,
nên các kết quả ViMQA ở trên không đổi.

## Kết quả

| | Của chúng tôi | Bài báo | Chênh |
|---:|---:|---:|---:|
| CIRAG *w/o TD*, EM | **48,0** | 49,8 | −1,8 |
| CIRAG *w/o TD*, F1 | 52,9 | 59,7 | −6,8 |
| IRCoT, EM | 28,0 | 36,4 (35,9 với bge-small, Bảng 6) | −8,4 |
| IRCoT, F1 | 31,6 | 45,7 (44,8) | −14,1 |
| CIRAG − IRCoT, EM | +20,0 | +13,4 | |

**EM của CIRAG lệch 1,8 điểm, dưới sai số chuẩn khoảng 5 điểm ở n = 100.** Lần này là CIRAG
đúng Eq. 6, không có mức dự phòng.

IRCoT thấp hơn bài báo 8,4 điểm, nên khoảng cách +20,0 (McNemar: 28 câu chỉ CIRAG đúng, 8 câu
chỉ IRCoT đúng, p = 0,001) bị phóng đại và **không** nên dùng làm phép so với +13,4. Nguyên
nhân nằm ở IRCoT của repo: bản zero-shot rút gọn, trong khi bài báo dùng bản của DualRAG có
few-shot. Qwen2.5-7B viết "So the answer is: the passage does not provide…" ngay bước đầu ở
62/100 câu, cụm "so the answer is" kích hoạt điều kiện dừng, và 49 câu ra "No answer". EM của
IRCoT ở các câu dừng sau 1 bước chỉ 16,1, so với 45,2 ở các câu chạy 2 bước.

Theo loại câu hỏi, CIRAG hơn IRCoT ở cả bốn loại:

| Loại | n | CIRAG | IRCoT | CIRAG rơi vào default |
|---|---:|---:|---:|---:|
| bridge_comparison | 31 | 54,8 | 32,3 | 10 |
| comparison | 14 | 92,9 | 85,7 | 1 |
| compositional | 45 | 33,3 | 13,3 | 14 |
| inference | 10 | 30,0 | 0,0 | 2 |

## Bảng 3 tái lập được về thứ tự

Bảng 3 của bài báo dùng CIRAG có TD nên chỉ so được thứ tự và độ chênh.

| Biến thể | EM | Δ so với đầy đủ | p | Δ bài báo (có TD) |
|---|---:|---:|---:|---:|
| Cascade đầy đủ | **48,0** | | | |
| w/o Triple (câu → đoạn) | 44,0 | −4,0 | 0,39 | −0,1 |
| w/o Passage (triple → câu) | 41,0 | −7,0 | 0,016 | −1,4 |
| chỉ đoạn | 40,0 | −8,0 | 0,12 | −5,7 |
| chỉ câu | 36,0 | −12,0 | 0,012 | −2,4 |
| chỉ triple | 34,0 | −14,0 | <0,001 | −4,2 |

Hai nhận định chính của bài báo về Bảng 3 đúng trên thiết lập này: cascade đầy đủ tốt nhất,
và mọi cascade hai mức (41, 44) hơn mọi biến thể một mức (34, 36, 40). Độ chênh lớn hơn bài
báo vì không có TD thì nhiều câu rơi vào default hơn.

Nhận định thứ ba **không** tái lập được: bài báo viết rằng trên 2WikiMQA, dùng riêng mức đoạn
kém nhất vì ngữ cảnh dài gây nhiễu (53,2 so với 54,7 và 56,5). Ở đây chỉ mức đoạn lại tốt nhất
trong ba biến thể một mức (40 so với 36 và 34), dù chênh lệch với chỉ mức triple không có ý
nghĩa thống kê (p = 0,36).

Kết quả này **ngược hẳn** ViMQA với Gemini 2.5 Flash (§2.4), nơi mọi biến thể một mức đều
thắng mọi biến thể cascade. Cùng một mã nguồn, cascade có lợi khi mô hình nền yếu và chịu từ
chối, và có hại khi mô hình nền mạnh và hiếm khi từ chối. Đây là bằng chứng trực tiếp cho
chẩn đoán ở §2.4 rằng điều kiện đủ dựa trên lời từ chối chỉ đúng với mô hình nền yếu.

## Mức dự phòng và điều kiện đủ đề xuất trên thiết lập gốc

| Cấu hình | EM | so với cấu hình trên |
|---|---:|---|
| CIRAG đúng Eq. 6 | 48,0 | |
| + mức dự phòng | 54,0 | +6,0 (6 câu chỉ bản này đúng, 0 câu ngược lại, p = 0,031) |
| + điều kiện đủ theo số nguồn | 54,0 | 0 (không câu nào khác kết quả) |

Mức dự phòng được dùng ở 8 câu, cứu đúng 6. Điều kiện đủ theo số nguồn **không tác động gì**
trên thiết lập này. Điều đó khớp với chẩn đoán của phần cải tiến: phép kiểm bằng chứng sửa
trường hợp mô hình trả lời dù bằng chứng thiếu. Qwen2.5-7B vốn sẵn sàng từ chối, nên `Suf(a)`
gốc đã làm đúng việc của nó, và phép kiểm mới không còn gì để sửa.

## Phân bố mức ngữ cảnh

| Mức | Của chúng tôi | Hình 5, *w/o TD* (đọc từ biểu đồ, ±3) |
|---|---:|---:|
| triple | 53% | ≈39% |
| câu | 12% | ≈18% |
| đoạn | 8% | ≈23% |
| default | 27% | ≈20% |

Cùng hình dạng (triple chiếm đa số, default đáng kể), nhưng của chúng tôi nghiêng về triple
và ít dùng mức đoạn hơn.

## Giới hạn

- n = 100, sai số chuẩn của EM khoảng 5 điểm.
- Q8_0 thay cho bản không lượng tử hoá; bge-small-en-v1.5 là retriever của Bảng 6, còn Bảng 2
  nhiều khả năng dùng NV-Embed-v2.
- N = 10 và cơ chế cứu theo mã nguồn, không theo bài báo. Mã nguồn được công bố có thể không
  trùng với phiên bản đã sinh ra con số trong bài báo.
- Phép so với IRCoT chưa dùng được cho tới khi có IRCoT few-shot như bài báo.
