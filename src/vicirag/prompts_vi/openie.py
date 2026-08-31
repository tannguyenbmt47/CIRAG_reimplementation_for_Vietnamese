"""Prompt trích xuất tri thức (Construction Phase) — bản tiếng Việt và bản gốc tiếng Anh.

Tương ứng Figure 9 (NER) và Figure 10 (trích triple) trong Appendix A.1 của paper.
Bản `en` chép nguyên văn từ `reference/CIRAG-official/src/cirag/prompt/templates/`
để có thể đối chiếu hành vi giữa hai bản cài đặt.

Vì sao phải Việt hoá few-shot demo: nếu giữ demo tiếng Anh (Radio City / India),
LLM có xu hướng trả về triple lẫn tiếng Anh hoặc dịch thực thể tiếng Việt sang
tiếng Anh, phá hỏng hoàn toàn bước so khớp triple với truy vấn tiếng Việt ở sau.
"""

from __future__ import annotations

# ---------------------------------------------------------------- tiếng Việt

NER_SYSTEM_VI = """Nhiệm vụ của bạn là trích xuất các thực thể có tên từ đoạn văn được cho.
Chỉ trả lời bằng một đối tượng JSON chứa danh sách thực thể, không giải thích thêm.
Giữ nguyên dạng viết của thực thể như trong đoạn văn, bao gồm dấu tiếng Việt và chữ hoa.
"""

NER_DEMO_PARAGRAPH_VI = """Chùa Một Cột
Chùa Một Cột là một ngôi chùa nằm ở trung tâm thành phố Hà Nội.
Chùa được vua Lý Thái Tông cho xây dựng vào mùa đông năm 1049.
Ngôi chùa có kiến trúc độc đáo với một điện thờ đặt trên một cột đá giữa hồ nước, mô phỏng hình dáng một bông sen."""

NER_DEMO_OUTPUT_VI = """{"named_entities":
    ["Chùa Một Cột", "Hà Nội", "Lý Thái Tông", "năm 1049"]
}
"""

NER_PROMPT_VI = [
    {"role": "system", "content": NER_SYSTEM_VI},
    {"role": "user", "content": NER_DEMO_PARAGRAPH_VI},
    {"role": "assistant", "content": NER_DEMO_OUTPUT_VI},
    {"role": "user", "content": "${passage}"},
]


TRIPLE_SYSTEM_VI = """Nhiệm vụ của bạn là xây dựng một đồ thị RDF từ đoạn văn và danh sách thực thể được cho.
Chỉ trả lời bằng một đối tượng JSON chứa danh sách các triple, mỗi triple là một danh sách 3 chuỗi
[chủ thể, quan hệ, đối tượng], không giải thích thêm.

Các yêu cầu bắt buộc:
- Mỗi triple phải chứa ít nhất một, tốt nhất là hai, thực thể trong danh sách thực thể của đoạn văn.
- Thay đại từ và cách gọi tắt bằng tên đầy đủ của thực thể để câu triple tự nó rõ nghĩa.
- Viết toàn bộ triple bằng tiếng Việt, giữ nguyên dấu tiếng Việt. Tuyệt đối không dịch sang tiếng Anh.
- Quan hệ phải là một cụm động từ ngắn gọn bằng tiếng Việt, ví dụ "nằm ở", "được xây dựng vào", "là".
"""

TRIPLE_FRAME_VI = """Chuyển đoạn văn sau thành một đối tượng JSON gồm danh sách thực thể và danh sách triple.
Đoạn văn:
```
{passage}
```

{named_entity_json}
"""

TRIPLE_DEMO_OUTPUT_VI = """{"triples": [
        ["Chùa Một Cột", "nằm ở", "Hà Nội"],
        ["Chùa Một Cột", "là", "ngôi chùa"],
        ["Chùa Một Cột", "được xây dựng bởi", "Lý Thái Tông"],
        ["Chùa Một Cột", "được xây dựng vào", "năm 1049"],
        ["Chùa Một Cột", "có", "kiến trúc độc đáo"],
        ["Chùa Một Cột", "mô phỏng", "bông sen"]
    ]
}
"""

TRIPLE_PROMPT_VI = [
    {"role": "system", "content": TRIPLE_SYSTEM_VI},
    {
        "role": "user",
        "content": TRIPLE_FRAME_VI.format(
            passage=NER_DEMO_PARAGRAPH_VI, named_entity_json=NER_DEMO_OUTPUT_VI
        ),
    },
    {"role": "assistant", "content": TRIPLE_DEMO_OUTPUT_VI},
    {"role": "user", "content": "${triple_input}"},
]


# ------------------------------------------------- tiếng Anh (từ repo gốc)

NER_SYSTEM_EN = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
"""

NER_DEMO_PARAGRAPH_EN = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

NER_DEMO_OUTPUT_EN = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

NER_PROMPT_EN = [
    {"role": "system", "content": NER_SYSTEM_EN},
    {"role": "user", "content": NER_DEMO_PARAGRAPH_EN},
    {"role": "assistant", "content": NER_DEMO_OUTPUT_EN},
    {"role": "user", "content": "${passage}"},
]

TRIPLE_SYSTEM_EN = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

TRIPLE_FRAME_EN = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

TRIPLE_DEMO_OUTPUT_EN = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

TRIPLE_PROMPT_EN = [
    {"role": "system", "content": TRIPLE_SYSTEM_EN},
    {
        "role": "user",
        "content": TRIPLE_FRAME_EN.format(
            passage=NER_DEMO_PARAGRAPH_EN, named_entity_json=NER_DEMO_OUTPUT_EN
        ),
    },
    {"role": "assistant", "content": TRIPLE_DEMO_OUTPUT_EN},
    {"role": "user", "content": "${triple_input}"},
]


TRIPLE_FRAME = {"vi": TRIPLE_FRAME_VI, "en": TRIPLE_FRAME_EN}
NER_PROMPT = {"vi": NER_PROMPT_VI, "en": NER_PROMPT_EN}
TRIPLE_PROMPT = {"vi": TRIPLE_PROMPT_VI, "en": TRIPLE_PROMPT_EN}
