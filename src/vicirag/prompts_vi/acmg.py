"""Prompt cho ACMG (Adaptive Cascaded Multi-Granularity) — 3 mức ngữ cảnh.

Tương ứng Figure 12 / Appendix A.3. Paper viết mức câu và mức đoạn "chỉ khác mức
triple ở phần ví dụ", nên ở đây ba mức dùng chung khung, khác nhau ở demo và ở
mô tả loại ngữ cảnh.

Chuỗi từ chối: repo gốc dùng chuỗi `No answer` (không phải `Unanswerable` như
trong paper). Bản tiếng Việt dùng `Không có câu trả lời`; hàm kiểm tra
`is_refusal` trong `vicirag/acmg.py` chấp nhận cả hai để không phụ thuộc ngôn ngữ
model lỡ trả về.
"""

from __future__ import annotations

REFUSAL_VI = "Không có câu trả lời"
REFUSAL_EN = "No answer"


def _system(context_desc: str, refusal: str, lang: str) -> str:
    if lang == "vi":
        return (
            "Bạn là một trợ lý đọc hiểu nâng cao. Nhiệm vụ của bạn là phân tích kỹ "
            f"{context_desc} được cung cấp và rút ra thông tin hữu ích để trả lời câu hỏi tương ứng. "
            'Câu trả lời của bạn bắt đầu sau "Suy luận: ", nơi bạn trình bày từng bước lập luận dẫn '
            "đến kết luận. "
            f'Nếu thông tin được cung cấp không đủ để trả lời câu hỏi, hãy xuất ra "{refusal}". '
            'Kết thúc bằng "Đáp án: " kèm một câu trả lời ngắn gọn, dứt khoát, không giải thích thêm. '
            "Toàn bộ câu trả lời phải bằng tiếng Việt."
        )
    return (
        "As an advanced reading comprehension assistant, your task is to carefully analyze "
        f"{context_desc} and extract useful information from them to answer corresponding questions."
        'Your response start after "Thought: ", where you will methodically break down the '
        "reasoning process, illustrating how you arrive at conclusions. "
        f"If the provided information cannot answer the question, output {refusal}."
        'Conclude with "Answer: " to present a concise, definitive response, devoid of additional '
        "elaborations."
        #  Biến thể en-vi: prompt tiếng Anh nhưng dữ liệu và đáp án vàng là tiếng Việt.
        #  Không có câu này thì mô hình trả lời bằng tiếng Anh và EM bằng 0 hết.
        + (" The context and the question are in Vietnamese. Write both the reasoning and the "
           "answer in Vietnamese." if lang == "en-vi" else "")
    )


# ---------------------------------------------------------------- tiếng Việt
# Demo 2-hop: Chùa Một Cột -> Hà Nội -> thủ đô Việt Nam.

_DEMO_TRIPLES_VI = (
    "triple: ('Chùa Một Cột', 'nằm ở', 'Hà Nội'), ('Hà Nội', 'là thủ đô của', 'Việt Nam'), "
    "('Chùa Một Cột', 'được xây dựng vào', 'năm 1049')"
)
_DEMO_SENTENCES_VI = (
    "câu: Chùa Một Cột là một ngôi chùa nằm ở trung tâm thành phố Hà Nội.\n"
    "câu: Hà Nội là thủ đô của Việt Nam, đồng thời là thành phố lớn thứ hai cả nước về dân số.\n"
    "câu: Văn Miếu - Quốc Tử Giám được xem là trường đại học đầu tiên của Việt Nam.\n"
)
_DEMO_PASSAGES_VI = (
    "đoạn văn: Chùa Một Cột\nChùa Một Cột là một ngôi chùa nằm ở trung tâm thành phố Hà Nội. "
    "Chùa được vua Lý Thái Tông cho xây dựng vào mùa đông năm 1049 với kiến trúc độc đáo gồm một "
    "điện thờ đặt trên một cột đá giữa hồ nước.\n"
    "đoạn văn: Hà Nội\nHà Nội là thủ đô của Việt Nam. Thành phố nằm ở vùng đồng bằng sông Hồng và "
    "là trung tâm chính trị, văn hoá của cả nước.\n"
)

_DEMO_QUESTION_VI = "Chùa Một Cột nằm ở thủ đô của nước nào?"
_DEMO_THOUGHT_VI = (
    "Dữ kiện cho biết Chùa Một Cột nằm ở Hà Nội. Dữ kiện tiếp theo cho biết Hà Nội là thủ đô của "
    "Việt Nam. Vậy Chùa Một Cột nằm ở thủ đô của Việt Nam."
)
_DEMO_ANSWER_VI = "Việt Nam"

# --------------------------------------------------------------- tiếng Anh

_DEMO_TRIPLES_EN = (
    "triplets: ('erik hort', 'born in', 'montebello'), ('erik hort', 'is', 'american'), "
    "('montebello', 'is located in', 'rockland county')"
)
_DEMO_SENTENCES_EN = (
    "sentence: Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the "
    "University of Southampton.\n"
    "sentence: The University of Southampton, which was founded in 1862 and received its Royal "
    "Charter as a university in 1952, has over 22,000 students.\n"
)
_DEMO_PASSAGES_EN = _DEMO_SENTENCES_EN
_DEMO_QUESTION_EN = "What county is Erik Hort's birthplace a part of?"
_DEMO_THOUGHT_EN = (
    "The triplet ('erik hort', 'born in', 'montebello') tells us that Erik Hort was born in "
    "Montebello. The triplet ('montebello', 'is located in', 'rockland county') tells us that "
    "Montebello is located in Rockland County. Therefore, Erik Hort's birthplace is part of "
    "Rockland County."
)
_DEMO_ANSWER_EN = "Rockland County."


_CONTEXT_DESC = {
    "vi": {"fact": "các triple tri thức", "sentence": "các câu văn", "passage": "các đoạn văn"},
    "en": {"fact": "triplets", "sentence": "text sentences", "passage": "text passages"},
}
_DEMO_CONTEXT = {
    "vi": {
        "fact": _DEMO_TRIPLES_VI,
        "sentence": _DEMO_SENTENCES_VI,
        "passage": _DEMO_PASSAGES_VI,
    },
    "en": {
        "fact": _DEMO_TRIPLES_EN,
        "sentence": _DEMO_SENTENCES_EN,
        "passage": _DEMO_PASSAGES_EN,
    },
}
_LABELS = {
    "vi": {"question": "Câu hỏi: ", "thought": "\nSuy luận: ", "answer": "\nĐáp án: "},
    "en": {"question": "Question: ", "thought": "\nThought: ", "answer": "\nAnswer: "},
}


#  en-vi: dùng nguyên prompt tiếng Anh của bài báo gốc, chỉ thêm yêu cầu trả lời bằng
#  tiếng Việt. Dùng để kiểm chứng xem prompt tiếng Việt do chúng tôi viết có phải là
#  nút thắt hay không.
_CONTEXT_DESC["en-vi"] = _CONTEXT_DESC["en"]
_DEMO_CONTEXT["en-vi"] = _DEMO_CONTEXT["en"]
_LABELS["en-vi"] = _LABELS["en"]


def build_acmg_prompt(granularity: str, lang: str = "vi") -> list[dict[str, str]]:
    """Dựng prompt 1-shot cho một mức ngữ cảnh ('fact' | 'sentence' | 'passage')."""
    if granularity not in ("fact", "sentence", "passage"):
        raise ValueError(f"Mức ngữ cảnh không hợp lệ: {granularity}")

    refusal = REFUSAL_VI if lang == "vi" else REFUSAL_EN
    labels = _LABELS[lang]
    demo_ctx = _DEMO_CONTEXT[lang][granularity]
    question = _DEMO_QUESTION_VI if lang == "vi" else _DEMO_QUESTION_EN
    thought = _DEMO_THOUGHT_VI if lang == "vi" else _DEMO_THOUGHT_EN
    answer = _DEMO_ANSWER_VI if lang == "vi" else _DEMO_ANSWER_EN

    return [
        {"role": "system", "content": _system(_CONTEXT_DESC[lang][granularity], refusal, lang)},
        {
            "role": "user",
            "content": f"{demo_ctx}\n\n{labels['question']}{question}{labels['thought']}",
        },
        {"role": "assistant", "content": f"{thought}{labels['answer']}{answer}"},
        {"role": "user", "content": "${prompt_user}"},
    ]


# ---- Chuẩn hoá câu trả lời cuối (repo gốc gọi là `reanswer`) ----------------
# Bước này rút gọn phần "Suy luận: ... Đáp án: ..." thành đúng chuỗi đáp án ngắn,
# rất quan trọng với EM: model tiếng Việt hay trả lời thành câu đầy đủ
# ("Chùa Một Cột nằm ở Việt Nam") thay vì cụm danh từ ("Việt Nam").

REANSWER_PROMPT_VI = [
    {
        "role": "system",
        "content": (
            "Bạn nhận được một câu hỏi và phần trả lời dài dòng của một mô hình khác. "
            "Hãy trích ra ĐÚNG cụm từ ngắn nhất trả lời cho câu hỏi.\n"
            "Quy tắc:\n"
            "- Không lặp lại chủ ngữ hay bất kỳ phần nào của câu hỏi.\n"
            "- Câu hỏi \"bao nhiêu\" thì chỉ trả về CON SỐ, bỏ danh từ đơn vị đi kèm "
            '(ví dụ trả "2" chứ không phải "2 lần", trả "55" chứ không phải "55 liên đoàn").\n'
            "- Câu hỏi đúng/sai thì chỉ trả về đúng một từ: \"đúng\" hoặc \"không\".\n"
            "- Với tên riêng, giữ NGUYÊN VĂN dạng viết trong ngữ cảnh, kể cả hậu tố như "
            '"F.C.", "Inc.", và giữ nguyên dấu tiếng Việt cùng cách viết hoa.\n'
            "- Không thêm dấu câu ở cuối, không giải thích.\n"
            f'- Nếu phần trả lời cho biết không có câu trả lời, xuất ra đúng "{REFUSAL_VI}".'
        ),
    },
    {
        "role": "user",
        "content": (
            "Câu hỏi: Chùa Một Cột nằm ở thủ đô của nước nào?\n"
            "Trả lời: Suy luận: Chùa Một Cột nằm ở Hà Nội, mà Hà Nội là thủ đô của Việt Nam. "
            "Đáp án: Chùa Một Cột nằm ở thủ đô của Việt Nam."
        ),
    },
    {"role": "assistant", "content": "Việt Nam"},
    {
        "role": "user",
        "content": (
            "Câu hỏi: Đội tuyển Pháp đã vô địch thế giới bao nhiêu lần?\n"
            "Trả lời: Suy luận: Pháp vô địch World Cup năm 1998 và 2018. "
            "Đáp án: Đội tuyển Pháp đã vô địch 2 lần."
        ),
    },
    {"role": "assistant", "content": "2"},
    {
        "role": "user",
        "content": (
            "Câu hỏi: Đĩa đơn đó do hãng nào phát hành?\n"
            "Trả lời: Suy luận: Dữ kiện cho biết đĩa đơn do Columbia Records Inc. phát hành. "
            "Đáp án: Đĩa đơn được phát hành bởi hãng Columbia Records Inc."
        ),
    },
    {"role": "assistant", "content": "Columbia Records Inc."},
    {"role": "user", "content": "Câu hỏi: ${question}\nTrả lời: ${response}"},
]

#  Bản tiếng Anh, dùng khi dữ liệu là tiếng Anh (--lang en). Dùng bản tiếng Việt cho câu
#  hỏi tiếng Anh thì mô hình có thể dịch đáp án sang tiếng Việt, và quy tắc đúng/sai lại đòi
#  "đúng"/"không", trong khi bài báo gốc rút gọn đáp án bằng prompt tiếng Anh.
REANSWER_PROMPT_EN = [
    {
        "role": "system",
        "content": (
            "You are given a question and a long-winded response from another model. "
            "Extract EXACTLY the shortest phrase that answers the question.\n"
            "Rules:\n"
            "- Do not repeat the subject or any part of the question.\n"
            '- For "how many" questions, return only the NUMBER, without the unit noun '
            '(for example "2", not "2 times"; "55", not "55 federations").\n'
            '- For yes/no questions, return exactly one word: "yes" or "no".\n'
            "- For proper names, keep the EXACT spelling used in the context, including "
            'suffixes such as "F.C." and "Inc.", and keep the capitalisation.\n'
            "- Do not add trailing punctuation or any explanation.\n"
            f'- If the response says there is no answer, output exactly "{REFUSAL_EN}".'
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: The One Pillar Pagoda is located in the capital of which country?\n"
            "Response: Thought: The One Pillar Pagoda is in Hanoi, and Hanoi is the capital "
            "of Vietnam. Answer: The One Pillar Pagoda is located in the capital of Vietnam."
        ),
    },
    {"role": "assistant", "content": "Vietnam"},
    {
        "role": "user",
        "content": (
            "Question: How many times has the French national team won the World Cup?\n"
            "Response: Thought: France won the World Cup in 1998 and 2018. "
            "Answer: The French national team has won it 2 times."
        ),
    },
    {"role": "assistant", "content": "2"},
    {
        "role": "user",
        "content": (
            "Question: Which label released that single?\n"
            "Response: Thought: The facts say the single was released by Columbia Records "
            "Inc. Answer: The single was released by the label Columbia Records Inc."
        ),
    },
    {"role": "assistant", "content": "Columbia Records Inc."},
    {"role": "user", "content": "Question: ${question}\nResponse: ${response}"},
]
