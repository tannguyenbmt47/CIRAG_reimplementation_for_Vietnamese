---
name: review-vi-paper
description: Rà soát và viết lại bản thảo khoa học tiếng Việt. Bắt từ sai bản chất, câu thiếu chủ ngữ, thuật ngữ dùng trước khi định nghĩa, số liệu sai mẫu số, và câu không đóng góp cho lập luận. Dùng khi người dùng nói "review bài", "check lại bài", "câu này k rõ", "viết chưa tốt", "sai bản chất", "gom ý lại", "vừa đủ N trang".
---

# Rà soát bản thảo khoa học tiếng Việt

Rà bằng **phép đối chiếu**, không bằng cảm nhận văn chương. Mọi phát hiện phải chỉ ra
được nó sai so với cái gì: paper gốc, dữ liệu thật, mã nguồn, hay người đọc lần đầu.

**Tra `references/tu-dien-loi.md` trước khi tự nghĩ ra cách diễn đạt mới.** File đó là
danh mục lỗi thật đã bị bắt trên bản thảo, kèm bản sửa đã được chấp nhận, chia theo:
từ sai bản chất, Việt hoá quá đà, cấu trúc câu, thuật ngữ chưa đặt tên, số liệu, cắt
gì, định dạng, và bẫy quy trình.

## Nguyên nhân gốc của gần như mọi lỗi diễn đạt

Người viết với tay lấy **một danh từ trừu tượng** thay vì nói thẳng **chủ thể nào làm gì**.

| Viết trừu tượng | Viết thẳng |
|---|---|
| "Cái chúng tôi đo được là lợi thế 14 điểm không còn" | "Số liệu chỉ cho phép nói CIRAG không còn hơn IRCoT, chứ chưa cho phép nói CIRAG thua" |
| "Phép đếm này bỏ qua vị trí token" | "F1 chỉ xét token nào xuất hiện, không xét chúng nằm ở đâu" |
| "Chỗ lẫn lộn này sinh ra hai dạng hỏng" | "Vì hai câu hỏi đó khác nhau, ACMG sai theo hai chiều" |
| "Phần này trình bày các thành phần" | "CIRAG chạy hai module nối tiếp: ICI…, ACMG…" |

Quy tắc: **mỗi câu phải trả lời được _cái gì làm gì_.**

## Bảy phép thử, chạy trên từng câu

1. **Chủ thể cụ thể.** Chủ ngữ có phải vật/khái niệm có thật trong hệ thống không?
   Câu có mở đầu bằng mệnh đề không chủ ngữ ("Chạy trên…", "Cộng cả hai…") không?
2. **Thuật ngữ đã đặt tên chưa.** Đọc tuần tự, không nhảy cóc. Gọi tên một tập hợp
   ("hai dạng lỗi", "bốn cách giải thích") thì phải liệt kê ngay.
3. **Ẩn dụ có thuộc lĩnh vực này không.** "Leo thang" từ xung đột, "kích hoạt" từ tâm
   lý học, "trải trên" tự chế. Hỏi: mã nguồn thực sự làm thao tác gì?
4. **So sánh có mốc chưa.** "Gọn nhất", "tốt hơn" phải kèm mốc hoặc con số.
5. **Động từ có mạnh hơn bằng chứng không.** "Chống", "khắc phục", "chứng minh",
   "loại bỏ" thường vượt quá điều đã đo. Đối chiếu với chính số liệu trong bài.
6. **Số có kèm mẫu số không**, và mọi số trong một đoạn có cùng mẫu số không.
7. **Câu có nói gì mới không.** Không thì bỏ.

## Khi người dùng nói "câu này chưa rõ"

Đừng chỉ đổi từ. Chạy bốn câu hỏi theo thứ tự:

1. Câu này **nêu kết luận trước dữ kiện** phải không? Đảo lại: cơ chế trước, hệ quả sau.
2. Nó **gọi tên một thứ chưa được định nghĩa** phải không? Định nghĩa tại chỗ.
3. Nó **chồng quá hai mệnh đề** phải không? Tách, mỗi câu một việc.
4. Nó có **nói ra được người đọc phải làm gì với thông tin đó** không?
   (ví dụ: giới thiệu $p$-value thì phải nói $p$ nhỏ nghĩa là gì)

## Khi phải gom cho vừa N trang

Cắt theo thứ tự này, dừng ngay khi đủ:

1. Câu bênh vực một điểm yếu vừa thừa nhận
2. Nội dung mô tả nhưng không dùng lại ở đâu
3. Biện minh cho lựa chọn không ảnh hưởng kết quả
4. Danh sách chỉ đường trùng với phần đóng góp
5. Bác bỏ nghi phạm chưa bao giờ nêu
6. Câu nói lại điều số liệu ngay trước đã hàm ý
7. Nhận xét về bảng → chuyển xuống **chú thích bảng**, giữ bảng

**Không cắt** bước lập luận, con số chống đỡ cho một khẳng định, hay lời thừa nhận
hạn chế. Sau khi cắt phải đếm lại số liệu then chốt — đã có lần nén làm mất
$p = 0{,}0005$, con số ý nghĩa thống kê của chính cải tiến.

## Quy trình sửa

Sửa **từng mục một**, mỗi mục biên dịch và kiểm lại ngay. Sửa gộp nhiều mục rồi mới
biên dịch thì không biết chỗ nào làm tràn trang.

```bash
cd report && pdflatex -interaction=nonstopmode main >/tmp/t.log 2>&1
echo "trang $(pdfinfo main.pdf|awk '/Pages/{print $2}') | overfull $(grep -ac Overfull /tmp/t.log)"
# thân bài kết thúc ở trang nào
for i in $(seq 1 6); do pdftotext -f $i -l $i main.pdf - | grep -q '^Hạn chế' && echo "Hạn chế bắt đầu t$i"; done
```

## Kiểm tra máy móc — chạy trên PDF

```bash
cd report && pdftotext main.pdf - | tr -s '[:space:]' ' ' > /tmp/v.txt

# 1. Từ đã bị loại khỏi bản thảo
for w in 'hỏng' 'trải qua' 'trải trên' 'leo thang' 'đường ống' 'trùng khít' \
         'chịu nhận' 'tụt xuống' 'vô hiệu' 'mổ xẻ' 'mô-đun' 'nhúng'; do
  c=$(grep -coF "$w" /tmp/v.txt); [ "$c" != 0 ] && echo "CÒN: $w ($c)"
done

# 2. Phủ định thiếu "có" — BẮT BUỘC có -i, nhãn hình viết hoa
grep -oiE 'không [a-zA-ZÀ-ỹ]+ nào' /tmp/v.txt | grep -vi 'không có'

# 3. Dấu chấm phẩy và gạch dài
echo "; $(grep -oF ';' /tmp/v.txt|wc -l) | — $(grep -oF '—' /tmp/v.txt|wc -l)"

# 4. Monospace lạm dụng trong văn xuôi
grep -on '\\texttt{[^}]*}' main.tex

# 5. Dấu phẩy thập phân trong chế độ toán (in ra "−3, 8")
grep -on '\$-\?[0-9]\+,[0-9]\+\$' main.tex

# 6. Thuật ngữ dùng lẫn hai tên
for a in 'pipeline|đường ống' 'cascade|thác|xếp tầng' 'embedding|nhúng' \
         'Trajectory Distillation|Chưng cất Quỹ đạo'; do
  echo "== $a"; grep -oiE "$a" /tmp/v.txt | sort | uniq -c
done

# 7. Mệnh đề mở đầu không chủ ngữ
grep -oE '(^|\. )(Chạy|Chuyển|Đổi mô|Lấy|Tách|Cộng|Bỏ hẳn|Đặt|Nhìn)[^.]{0,55}' /tmp/v.txt
```

## Đối chiếu hai bản Việt / Anh

Bản dịch phải khớp nội dung và khớp số liệu:

```bash
pdftotext main.pdf - | tr -s '[:space:]' ' ' > /tmp/v.txt
pdftotext main_en.pdf - | tr -s '[:space:]' ' ' > /tmp/e.txt
for k in 60.40 50.80 0.0005 0.29 35.7 71.6; do
  vi=$(echo "$k"|sed 's/\./,/')
  a=$(grep -coF "$k" /tmp/e.txt); b=$(grep -coF "$vi" /tmp/v.txt)
  [ "$a" -gt 0 ] && [ "$b" -gt 0 ] || echo "LỆCH $k (EN=$a VI=$b)"
done
```

Mỗi lần sửa bản tiếng Việt phải port ngay sang bản tiếng Anh, nếu không hai bản trôi
xa nhau. Đã có lần bản tiếng Anh còn giữ sáu chỗ mà bản tiếng Việt đã cắt.

## Định dạng báo cáo kết quả

Mỗi phát hiện gồm bốn phần: **trích nguyên văn**, **sai so với cái gì**, **bản sửa đề
xuất**, **vì sao bản sửa đúng hơn**. Xếp theo mức nghiêm trọng: sai bản chất và sai số
liệu trước, diễn đạt sau. Không khen. Mục nào không tìm được lỗi thì nói thẳng là đạt.
