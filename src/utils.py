import config

def is_reasoning_question(question: str, choices: list[str]) -> bool:
    text = (question + "\n" + "\n".join(choices)).lower()
    if config.COMBINED_MATH_REGEX.search(text): 
        return True
    if any(kw in text for kw in config.REASONING_KEYWORDS): 
        return True
    return False

def format_prompt(question: str, choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_text = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Answer the following multiple choice question.\n"
        f"Question:\n{question}\n"
        f"Choices:\n{choice_text}\n"
        f"Answer:"
    )

def format_cot_prompt(question: str, choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_text = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Bạn là một chuyên gia giải đề trắc nghiệm. Hãy giải câu hỏi sau một cách cực kỳ ngắn gọn theo cấu trúc mẫu.\n\n"
        f"CẤU TRÚC MẪU BẮT BUỘC:\n"
        f"1. Công thức/Lý do: [Viết 1 dòng ngắn gọn]\n"
        f"2. Thay số/Phân tích: [Viết 1 dòng ngắn gọn]\n"
        f"3. Tính toán/Kết luận: [Viết 1 dòng ngắn gọn]\n"
        f"Đáp án: [Chữ cái nhãn đúng]\n\n"
        f"--- BÀI THI ---\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_text}\n\n"
        f"Hãy bắt đầu giải trực tiếp ngay lập tức, không viết dài dòng, không suy nghĩ lan man."
    )

def get_shuffled_variants(choices: list[str], num_variants=3) -> list[dict]:
    n = len(choices)
    variants = []
    original_labels = [chr(65 + i) for i in range(n)]
    actual_variants = min(num_variants, n)

    for i in range(actual_variants):
        shuffled_indices = [(j + i) % n for j in range(n)]
        shuffled_choices = [choices[idx] for idx in shuffled_indices]
        mapping = {chr(65 + new_idx): original_labels[old_idx]
                   for new_idx, old_idx in enumerate(shuffled_indices)}
        variants.append({"choices": shuffled_choices, "mapping": mapping})
    return variants