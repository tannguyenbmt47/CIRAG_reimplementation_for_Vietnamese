# Báo cáo

## Hai bản báo cáo

| File | Ngôn ngữ | Hình dùng |
|---|---|---|
| `main.tex` | tiếng Việt | `figures/` |
| `main_en.tex` | tiếng Anh, dịch từ bản tiếng Việt, từ vựng mức B2 | `figures_en/` |

## Biên dịch

```bash
cd report
pdflatex main    && bibtex main    && pdflatex main    && pdflatex main
pdflatex main_en && bibtex main_en && pdflatex main_en && pdflatex main_en
```

Hoặc tải cả thư mục này lên Overleaf.

Nếu thiếu gói LaTeX (bản TeX Live tối giản thường thiếu):

```bash
tlmgr update --self
tlmgr install lineno vntex babel-vietnamese grfext courier psnfss
# vntex cần cho cả hai bản: bản tiếng Anh dùng mã hoá T5 để in ví dụ
# tiếng Việt ("học sinh"), T1 không có ký tự tiếng Việt.
```

`vntex` và `babel-vietnamese` là bắt buộc để gõ tiếng Việt có dấu với `pdflatex`.

## Hình

Bốn hình trong `figures/` được **sinh tự động từ `outputs/`**, không có con số nào
nhập thủ công:

```bash
python scripts/make_figures.py            # nhãn tiếng Việt  -> figures/
python scripts/make_figures.py --lang en  # nhãn tiếng Anh   -> figures_en/
```

Chạy lại lệnh trên sau mỗi lần có kết quả thực nghiệm mới; hình sẽ luôn khớp với
`RESULTS.md`.

| Hình | Nội dung |
|---|---|
| `fig_rescue.pdf` | Tác động của cơ chế dự phòng đề xuất |
| `fig_backbone.pdf` | Hiệu ứng tương tác giữa phương pháp và mô hình nền (kết quả trung tâm) |
| `fig_granularity.pdf` | Phân bố mức chi tiết ngữ cảnh theo mô hình nền |
| `fig_errors.pdf` | Phân loại lỗi theo khâu phát sinh |

Bảng màu đã qua kiểm định về độ phân biệt cho người mù màu; mọi cột đều kèm nhãn số
trực tiếp nên hình vẫn đọc được khi in đen trắng.

## Còn phải điền

- Tên học viên, MSHV, email (đầu `main.tex`)
- Đường dẫn repo (trong `\begin{abstract}` và mục Kết luận)
