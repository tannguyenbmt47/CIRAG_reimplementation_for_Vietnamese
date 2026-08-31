"""Prompt cho Integration Phase (Eq. 1) — bản tiếng Việt và bản gốc tiếng Anh.

Tương ứng Figure 11 / Appendix A.2 của paper. Bản `en` chép nguyên văn từ
`reference/CIRAG-official/src/cirag/prompt/prompts/filter.json` (prompt đã được
DSPy biên dịch, gồm 2 demo của ví dụ Lothair II trong Figure 2).

Format output `[[ ## ... ## ]]` giữ nguyên như repo gốc — đây là hợp đồng giữa
prompt và hàm `parse_integration_output` trong `vicirag/ici.py`, đồng thời cho
phép so sánh trực tiếp quỹ đạo với bản cài đặt gốc.
"""

from __future__ import annotations

INPUT_TEMPLATE_START = """[[ ## query ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}"""
INPUT_TEMPLATE_FOLLOWUP = """[[ ## fact_before_filter ## ]]\n{fact_before_filter}"""
OUTPUT_TEMPLATE = """[[ ## thought ## ]]\n{thought}\n\n[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## question ## ]]\n{question}"""

NO_QUESTION = "<no question>"


# ---------------------------------------------------------------- tiếng Việt

INTEGRATION_SYSTEM_VI = """Bạn là thành phần cốt lõi của một hệ thống hỏi đáp đòi hỏi độ chính xác cao.
Nhiệm vụ của bạn là lặp lại việc thu thập dữ kiện cho tới khi đủ dữ kiện để trả lời truy vấn gốc.
Bạn phải tiến hành theo từng bước, mỗi bước có dạng:

[[ ## fact_before_filter ## ]]
{fact_before_filter}

[[ ## thought ## ]]
{thought}

[[ ## fact_after_filter ## ]]
{fact_after_filter}

[[ ## question ## ]]
{question}

Trong đó:
- Ở phần [[ ## fact_before_filter ## ]] là tập dữ kiện ứng viên cần được lọc.
- Ở phần [[ ## thought ## ]], CHỈ dùng thông tin trong các dữ kiện trước khi lọc. Phân tích ngắn gọn
  đóng góp của từng dữ kiện đối với truy vấn gốc và liệt kê những dữ kiện hữu ích. Dựa trên phân tích
  đó, xác định xem truy vấn gốc đã có thể trả lời được chưa. Nếu chưa, hãy phân tích còn thiếu thông
  tin gì và đặt ra câu hỏi mới. Thông tin trong các dữ kiện được coi là hoàn toàn chính xác.
- Ở phần [[ ## fact_after_filter ## ]], dùng đúng định dạng JSON, ví dụ
  {"fact": [["chủ thể 1", "quan hệ 1", "đối tượng 1"], ["chủ thể 2", "quan hệ 2", "đối tượng 2"]]}.
  Nếu không có dữ kiện nào liên quan, trả về danh sách rỗng {"fact": []}.
  Chỉ được chép lại nguyên văn các dữ kiện xuất hiện trong [[ ## fact_before_filter ## ]], không tự bịa.
- Ở phần [[ ## question ## ]], dùng các dữ kiện đã lọc để đặt ra câu hỏi tiếp theo cần giải quyết.
  Nếu các dữ kiện thu thập được ở những vòng trước đã đủ để trả lời truy vấn gốc, hãy xuất ra
  <no question> để kết thúc vòng lặp.

Toàn bộ phần suy luận và câu hỏi mới phải viết bằng tiếng Việt.

Dưới đây là một ví dụ:
"""

_DEMO1_FACTS_VI = (
    '{"fact": [["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"], '
    '["Truyện Kiều", "là", "truyện thơ Nôm"], '
    '["Truyện Kiều", "có", "3254 câu thơ lục bát"], '
    '["Truyện Kiều", "còn có tên là", "Đoạn trường tân thanh"], '
    '["Nguyễn Du", "có hiệu là", "Thanh Hiên"], '
    '["Truyện Kiều", "dựa trên", "Kim Vân Kiều truyện"]]}'
)

_DEMO1_THOUGHT_VI = (
    "Truy vấn hỏi: Tác giả của Truyện Kiều sinh năm nào? Từ các dữ kiện, ta xác định được tác giả của "
    'Truyện Kiều là Nguyễn Du (["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"]). Đây là dữ kiện then '
    "chốt cần giữ lại. Tuy nhiên, không có dữ kiện nào cho biết năm sinh của Nguyễn Du; các dữ kiện còn "
    "lại chỉ mô tả thể loại, độ dài và nguồn gốc của tác phẩm nên không đóng góp cho truy vấn. Do đó cần "
    "thêm thông tin về năm sinh của Nguyễn Du."
)

_DEMO1_FILTERED_VI = '{"fact": [["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"]]}'
_DEMO1_NEXT_VI = "Nguyễn Du sinh năm nào?"

_DEMO2_FACTS_VI = (
    '{"fact": [["Nguyễn Du", "sinh năm", "1765"], '
    '["Nguyễn Du", "mất năm", "1820"], '
    '["Nguyễn Du", "quê ở", "làng Tiên Điền"], '
    '["Nguyễn Du", "là", "nhà thơ"], '
    '["làng Tiên Điền", "thuộc", "huyện Nghi Xuân"], '
    '["Truyện Kiều", "được sáng tác bởi", "Nguyễn Du"]]}'
)

_DEMO2_THOUGHT_VI = (
    "Truy vấn gốc là: Tác giả của Truyện Kiều sinh năm nào? Ở bước trước ta đã xác định tác giả của "
    'Truyện Kiều là Nguyễn Du. Trong tập dữ kiện hiện tại, dữ kiện ["Nguyễn Du", "sinh năm", "1765"] '
    "trả lời trực tiếp cho truy vấn. Các dữ kiện khác về năm mất, quê quán và nghề nghiệp chỉ là thông "
    "tin tiểu sử bổ sung, không cần thiết để xác định năm sinh. Như vậy đã đủ dữ kiện để trả lời."
)

_DEMO2_FILTERED_VI = '{"fact": [["Nguyễn Du", "sinh năm", "1765"]]}'

INTEGRATION_DEMOS_VI = [
    {
        "role": "user",
        "content": INPUT_TEMPLATE_START.format(
            question="Tác giả của Truyện Kiều sinh năm nào?",
            fact_before_filter=_DEMO1_FACTS_VI,
        ),
    },
    {
        "role": "assistant",
        "content": OUTPUT_TEMPLATE.format(
            thought=_DEMO1_THOUGHT_VI,
            fact_after_filter=_DEMO1_FILTERED_VI,
            question=_DEMO1_NEXT_VI,
        ),
    },
    {
        "role": "user",
        "content": INPUT_TEMPLATE_FOLLOWUP.format(fact_before_filter=_DEMO2_FACTS_VI),
    },
    {
        "role": "assistant",
        "content": OUTPUT_TEMPLATE.format(
            thought=_DEMO2_THOUGHT_VI,
            fact_after_filter=_DEMO2_FILTERED_VI,
            question=NO_QUESTION,
        ),
    },
]


# ------------------------------------------------- tiếng Anh (từ repo gốc)

INTEGRATION_SYSTEM_EN = (
    "You are a key component of the high-risk question answering system used by top researchers "
    "and decision-makers worldwide. Your task is to loop through collecting facts to solve the "
    "query until enough facts are collected to solve the query. You must plan to continue in a "
    "series of steps, looping as follows: [[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
    "[[ ## thought## ]]\n{thought}\n\n[[ ## fact_after_filter ## ]]\n{fact_after_filter}  "
    '# note: the value you produce must be pareseable according to the following JSON schema: '
    '{"type": "object", "properties": {"fact": {"type": "array", "description": "A list of facts, '
    'each fact is a list of 3 strings: [subject, predicate, object]", "items": {"type": "array", '
    '"items": {"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}\n\n'
    "[[ ## question ## ]]\n{question} sequence.\n\n In each step, in the "
    "[[ ## fact_before_filter ## ]] sequence, this is the collection of facts that need to be "
    "filtered to solve the previous problem. \n in the [[ ## thought## ]] sequence, only use the "
    "information from the facts before filtering, briefly analyze the contribution of the "
    "information in the facts to the problem, and list useful facts. Based on the analysis of "
    "useful facts, determine whether the original problem can be solved. If it cannot be solved, "
    "analyze what additional information is needed and raise new questions. The information "
    "contained in the facts is absolutely correct.\n in the [[ ## fact_after_filter ## ]] "
    'sequence, use JSON format, such as {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}. '
    'If there are no relevant facts, return an empty list {"fact": []}. \n'
    "In the [[ ## question ## ]] sequence, You need to use the filtered facts to propose the next "
    "problem that still needs to be solved. If the facts collected in the previous rounds are "
    "sufficient to answer the original query, output<no question>as the end to terminate the "
    "loop. \n Here is an example:\n"
)

_DEMO1_FACTS_EN = (
    '{"fact": [["lothair ii", "is the son of", "emperor lothair i"], '
    '["lothair ii", "is the son of", "ermengarde of tours"], '
    '["lothair ii", "was married to", "teutberga"], '
    '["lothair ii", "was king of", "lotharingia"], '
    '["bertha", "daughter of", "lothair ii"], '
    '["lothair ii", "born in", "835"]]}'
)

_DEMO1_THOUGHT_EN = (
    "The query asks: When did Lothair II's mother die? From the facts, we identify that Lothair "
    'II\'s mother is Ermengarde of Tours (["lothair ii", "is the son of", "ermengarde of tours"]). '
    "This is the key fact to isolate. However, none of the provided facts give the death date of "
    "Ermengarde of Tours. Thus, the useful fact is only the identification of the mother. To "
    "proceed, we need additional information: specifically, the death date of Ermengarde of Tours."
)

_DEMO2_FACTS_EN = (
    '{"fact": [["ermengarde of tours", "death date", "20 march 851"], '
    '["ermengarde of tours", "married in", "october 821"], '
    '["ermengarde of tours", "daughter of", "hugh of tours"], '
    '["lothair ii", "is the son of", "ermengarde of tours"], '
    '["teutberga", "died in", "875"]]}'
)

_DEMO2_THOUGHT_EN = (
    "The question is: When did Lothair II's mother die? From the previous step, we identified that "
    "Lothair II's mother is Ermengarde of Tours. In the current facts, we see that Ermengarde of "
    'Tours died on 20 March 851 (["ermengarde of tours", "death date", "20 march 851"]). This '
    "directly answers the query. Other facts provide biographical context but are not necessary. "
    "Therefore, the required fact is the death date of Ermengarde of Tours."
)

INTEGRATION_DEMOS_EN = [
    {
        "role": "user",
        "content": INPUT_TEMPLATE_START.format(
            question="When did Lothair II's mother die?",
            fact_before_filter=_DEMO1_FACTS_EN,
        ),
    },
    {
        "role": "assistant",
        "content": OUTPUT_TEMPLATE.format(
            thought=_DEMO1_THOUGHT_EN,
            fact_after_filter='{"fact":[["lothair ii", "is the son of", "ermengarde of tours"]]}',
            question="When did Ermengarde of Tours die?",
        ),
    },
    {
        "role": "user",
        "content": INPUT_TEMPLATE_FOLLOWUP.format(fact_before_filter=_DEMO2_FACTS_EN),
    },
    {
        "role": "assistant",
        "content": OUTPUT_TEMPLATE.format(
            thought=_DEMO2_THOUGHT_EN,
            fact_after_filter='{"fact":[["ermengarde of tours", "death date", "20 march 851"]]}',
            question=NO_QUESTION,
        ),
    },
]


INTEGRATION_SYSTEM = {"vi": INTEGRATION_SYSTEM_VI, "en": INTEGRATION_SYSTEM_EN}
INTEGRATION_DEMOS = {"vi": INTEGRATION_DEMOS_VI, "en": INTEGRATION_DEMOS_EN}

#  en-vi: prompt tiếng Anh của bài báo gốc, chạy trên dữ liệu tiếng Việt. Dùng để kiểm
#  chứng xem prompt tiếng Việt do chúng tôi viết có phải nút thắt hay không.
INTEGRATION_SYSTEM["en-vi"] = INTEGRATION_SYSTEM["en"]
INTEGRATION_DEMOS["en-vi"] = INTEGRATION_DEMOS["en"]
