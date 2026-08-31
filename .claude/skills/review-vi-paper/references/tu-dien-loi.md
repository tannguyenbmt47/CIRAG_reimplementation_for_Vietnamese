# Từ điển lỗi — tra khi rà bản thảo

Mỗi dòng là một lỗi thật đã bị bắt trên bản thảo, kèm bản sửa đã được chấp nhận.
Tra bảng này trước khi tự nghĩ ra cách diễn đạt mới.

## A. Từ sai bản chất

Hỏi: *mã nguồn thực sự làm thao tác gì?* rồi viết đúng thao tác đó.

| Sai | Đúng | Vì sao |
|---|---|---|
| tập triple lõi **trải qua** / **trải trên** hai đoạn | **đến từ** hai đoạn | Tập triple không đi qua hay phủ lên đoạn văn. Mỗi triple lưu vết đoạn đã sinh ra nó, nên quan hệ là *đến từ* |
| một khâu **hỏng** trong pipeline | một khâu **chạy sai** | "hỏng" là đồ vật bị vỡ |
| hai **dạng hỏng** / **cách hỏng** | hai **kiểu lỗi** | như trên |
| 3 câu bị **làm hỏng** | 3 câu bị **làm sai** | như trên |
| khâu truy hồi **hỏng ở** 21 câu | **gây ra** 21 câu | Hình xếp câu sai theo khâu phát sinh lỗi đầu tiên, nên quan hệ là *gây ra* |
| cascade **leo thang** sang mức câu | **mở rộng ngữ cảnh** | "leo thang" là ẩn dụ xung đột |
| $\nu(\hat a)$ **trùng khít** $\nu(a)$ | **là hai chuỗi giống hệt nhau** | Đây là phép so chuỗi, không có chuyện "khít" |
| EM **tụt xuống** 0 | EM **nhận giá trị** 0 | văn nói |
| mô hình **chịu nhận** / **chịu thừa nhận** ngữ cảnh thiếu | mô hình **trả về câu từ chối** | Hệ thống chỉ quan sát được chuỗi đầu ra, không quan sát được "sự thừa nhận" |
| điều kiện dừng **mất hiệu lực** / cơ chế **vô hiệu** | **luôn thoả ngay ở mức đầu**, các mức rộng hơn **không bao giờ được dùng tới** | Cơ chế vẫn chạy đúng định nghĩa, chỉ là không bao giờ được kích hoạt |
| pha kiến tạo **kích hoạt** các triple | **gom** triple rồi xếp hạng | *Spreading activation* là ẩn dụ tâm lý học |
| Để **chống** mở rộng tham lam | Nhằm **hạn chế** | Paper nói *mitigate* |
| nhằm **loại bỏ** nhiễu | nhằm **giảm** nhiễu | Paper nói *reduce* |
| §8 đề xuất cách **khắc phục** | đề xuất một **điều kiện dừng thay thế** | §8 tự nhận chưa vượt được biến thể đơn giản |
| hai mục dưới đây **mổ xẻ** | **xét** | văn nói |
| CIRAG **nhắm đúng hai điểm này** | CIRAG **khác hai nhánh trên ở đúng hai điểm đó** | văn nói, và "nhắm" hàm ý đã giải quyết được |
| **Sửa đổi** mang lại 6,2 EM | **Mức dự phòng** này mang lại | Không sửa cơ chế nào, chỉ thêm một mức |
| chọn mức chi tiết **gọn nhất** | thử từ mức triple, rồi mức câu, rồi mức đoạn | So sánh không mốc |
| chúng tôi **chứng minh** | kết quả **cho thấy** | Thực nghiệm không phải chứng minh hình thức |

## B. Việt hoá quá đà

Giữ thuật ngữ tiếng Anh đã ổn định, chú giải một lần rồi dùng nhất quán.

| Sai | Đúng |
|---|---|
| **đường ống** | `pipeline` |
| **mô hình nhúng** | `embedding` |
| **mô-đun** | `module` |
| **Chưng cất Quỹ đạo** | `Trajectory Distillation` |
| K đoạn văn có thứ hạng cao nhất | top-$K$ đoạn |

Kiểm chéo: một thứ không được mang hai tên trong cùng bài. Đã có lần
`Trajectory Distillation` ở §6 nhưng "Chưng cất Quỹ đạo" ở mục Hạn chế.

## C. Cấu trúc câu

| Lỗi | Ví dụ đã bị bắt | Sửa |
|---|---|---|
| **Mệnh đề mở đầu không chủ ngữ** | "Chạy trên chính bộ dữ liệu của họ, bản cài đặt tái lập được..." | "Chúng tôi chạy nó trên..., và nó tái lập được..." |
| | "Cộng cả hai thay đổi, CIRAG chuyển từ..." | "Với cả hai thay đổi, CIRAG..." |
| | "Tách 361 câu theo số đoạn..., nhóm một đoạn đạt 35,7" | "Chúng tôi tách 361 câu... Nhóm một đoạn đạt 35,7" |
| | "Lấy NativeRAG làm mốc thì..." | "Nếu lấy NativeRAG làm mốc thì..." |
| | "Đổi mô hình nền sang Qwen3, cải tiến vẫn cùng chiều" | "Khi chúng tôi đổi mô hình nền sang Qwen3, cải tiến..." |
| **Cụm danh từ cụt** làm gạch đầu dòng | "Một điều kiện dừng thay thế, đòi bằng chứng phải..." | "Chúng tôi thay dấu hiệu đó bằng một phép kiểm... Phép kiểm nâng EM..." |
| **Nêu kết luận trước dữ kiện** | "Hai dạng hỏng quy về cùng một gốc: ..." (chưa nói hai dạng là gì) | Nêu cơ chế trước, kết luận sau |
| **Câu chồng bốn mệnh đề** | "CIRAG kém 2,0 điểm, và McNemar cho $p$, với 31 câu..., nên ngay cả..." | Tách: nêu số → nêu phép kiểm → nêu kết luận |
| **Liên từ sai chỗ** | "...so sánh là có nghĩa. **Nhưng** CIRAG kém IRCoT 2,0 điểm" | Hai câu này không đối lập. Chỗ đối lập là *có khoảng cách* với *khoảng cách không đủ bằng chứng* |
| **Danh từ trừu tượng tự đặt làm chủ ngữ** | "Cái chúng tôi đo được là...", "Chỗ lẫn lộn này sinh ra..." | Chủ ngữ phải là vật/khái niệm có thật trong hệ thống |
| **Lặp cụm trong một câu** | "đọc đúng **thứ nó được đưa**, chỉ là **thứ nó được đưa** chưa đủ" | "đọc đúng những triple nó nhận được, chỉ là tập triple đó chưa đủ" |
| **Phủ định thiếu "có"** | "khi **không mức nào** trả lời được" | "khi **không có** mức nào" |
| **Dấu hai chấm sai chức năng** | "mở rộng tham lam theo một đường: các phương pháp trước chỉ giữ..." | Sau dấu hai chấm phải là lời giải thích cho vế trước, không phải một chuỗi nhân quả mới |
| | "$p = 0{,}0005$: 20 câu sửa đúng và 3 câu hỏng" | "$p = 0{,}0005$, **trong đó** 20 câu..." |

## D. Thuật ngữ dùng trước khi đặt tên

Đọc tuần tự từ đầu. Mỗi tên gọi phải được định nghĩa tại chỗ nó sinh ra.

Đã bị bắt:

- **"nhánh mặc định"** dùng ở §7.2 mà §3 chỉ nói "lùi về câu trả lời mức tài liệu"
- **"Trajectory Distillation"** dùng ở Bảng 1 và §6, giải thích mãi ở mục Hạn chế
- **kiểm định McNemar** dùng 8 lần khắp §6–§9 mà không chỗ nào giới thiệu
- **$M^D$** được §8 dẫn "(§3)" nhưng §3 chưa hề đặt tên ký hiệu đó
- **"hai dạng hỏng"**, **"bốn cách giải thích"** gọi tên mà không kể ra là những gì

Quy tắc: gọi tên một tập hợp thì phải liệt kê ngay, hoặc trỏ tới chỗ liệt kê.

## E. Số liệu

| Lỗi | Ví dụ |
|---|---|
| **Trộn mẫu số** | "21 trong 500 **câu hỏi**" rồi "95 **câu sai**" — hai mẫu số khác nhau trong một đoạn |
| **Khái quát không kiểm được** | "99 câu đáp án dài, chỉ 10 đạt EM 1, **phần còn lại** nêu đủ nhưng đảo thứ tự" — 89 câu kia chưa kiểm từng câu. Sửa: "**Dạng trượt điển hình là**..." |
| **Khẳng định vượt phạm vi đo** | "lợi thế biến mất **trên tiếng Việt**" khi chỉ đo trên **một** bộ dữ liệu. Sửa: "**trên ViMQA**" |
| **Kết luận vượt bằng chứng** | Khoảng cách âm không có ý nghĩa thống kê **không** cho phép kết luận chiều ngược lại. Chỉ được nói "*không còn hơn*", không được nói "*thua*" |
| **p-value chỉ có trong văn xuôi** | Người đọc thấy $p = 0{,}29$ mà không truy được về bảng nào. Thêm cột $p$ vào bảng |
| **Chép số từ bảng vào văn xuôi** | Bảng đã có số, văn xuôi nêu quy luật |
| **Số không dùng lại** | "156 câu (31,2%) đúng/sai" ở §4 trong khi §9 đã nêu $n = 156$ đúng chỗ cần |

## F. Cắt gì

Hỏi từng câu: *câu này có đẩy mạch lập luận đi không?* Không thì bỏ.

Đã bỏ khỏi bản thảo:

- Câu bênh vực một điểm yếu vừa thừa nhận ("chỉ 2,3% tổng token" trong đoạn nói cải tiến **không** vượt được biến thể đơn giản)
- Mô tả dữ liệu không dùng tới ("nhãn chỉ rõ câu nào mang bằng chứng")
- Biện minh cho lựa chọn không ảnh hưởng kết quả ("dùng model rẻ vì tác vụ dễ hơn")
- Danh sách chỉ đường trùng với phần đóng góp ở §1
- Bác bỏ một nghi phạm chưa bao giờ nêu ("số vòng lặp không phải chỗ nghẽn")
- Câu nói lại điều số liệu ngay trước đã hàm ý
- Câu hắng giọng ("Phần này trình bày…"), câu phòng thủ, câu bù nhìn

Nội dung tái lập vẫn nên giữ trong **bảng**, chỉ chuyển nhận xét xuống **chú thích bảng**
để không chiếm chỗ trong mạch chính.

## G. Định dạng

- **`\texttt` chỉ dành cho chuỗi mã thật** (`<no question>`, email). Tên mô hình viết
  thường như trong bảng: "Gemini 2.5 Flash", không phải `gemini-2.5-flash`
- Định danh kỹ thuật đặt trong bảng cấu hình, đừng để cắt ngang câu văn
- **Không dấu chấm phẩy, không gạch dài** trong văn xuôi
- Chú thích bảng **không lặp lại** điều văn xuôi vừa nói
- Dấu phẩy thập phân trong chế độ toán phải viết `$-3{,}8$`, viết `$-3,8$` sẽ in ra "−3, 8"
- Mục cuối chỉ cần **"Hạn chế"**, bỏ phần đạo đức nếu nơi nộp không yêu cầu

## H. Bẫy quy trình

- **Kiểm trên PDF, không trên `.tex`.** LaTeX ngắt dòng giữa cụm từ nên `grep` trên
  mã nguồn bỏ sót. Đã lọt lỗi ba lần theo đúng cách này.
- **`grep` phải có `-i`.** Nhãn hình viết hoa ("Không mức nào đủ") lọt qua phép kiểm
  phân biệt hoa thường.
- **`grep -c` trên log LaTeX phải có `-a`**, không thì grep coi log là nhị phân và không in số.
- **`pdflatex` không báo lỗi khi lệnh toán không tồn tại** (`\mathbb{1}` trong `amssymb`),
  nó thay bằng ký tự sai và biên dịch sạch. Mọi công thức phải nhìn tận mắt trên bản kết xuất.
- **Script sửa nhiều chỗ: ghi file sau từng chỗ.** Nếu gom tất cả rồi mới ghi, một chỗ
  không khớp là mất hết các chỗ đã sửa.
- **T1 không có ký tự tiếng Việt.** Bản tiếng Anh trích ví dụ tiếng Việt phải nạp
  `[T5,T1]{fontenc}` và bọc `{\fontencoding{T5}\selectfont ...}`, nếu không "học sinh"
  in ra thành "hc sinh" mà log vẫn sạch. `pdftotext` không rút được chữ T5 nên phải
  phóng to ảnh render để xác nhận.
