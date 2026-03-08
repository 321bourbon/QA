import json
import re
from pathlib import Path


def parse_sub_questions(response: str, fallback_q: str, subq_num: int = 5) -> list:
    """从 VLM 回复中解析 Sub-Q 列表，不足 subq_num 个时用 fallback_q 补齐。"""
    if not response.strip():
        return [fallback_q] * subq_num

    out = []
    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^Output\s*\d+\s*[:.]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^\d+\s*[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if len(line.split()) < 4:
            continue
        if not line.endswith("?"):
            line = line.rstrip(" .;:") + "?"
        out.append(line)

    deduped, seen = [], set()
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            deduped.append(q)
    if not deduped:
        deduped = [fallback_q]
    while len(deduped) < subq_num:
        deduped.append(fallback_q)
    return deduped[:subq_num]


def extract_result(text):
    """Extract final Yes/No from model response."""
    if not text:
        return "No"

    m = re.search(r"-?\s*Result\s*:\s*(Yes|No)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1].lower()
        if "yes" in last and "no" not in last:
            return "Yes"
        if "no" in last and "yes" not in last:
            return "No"

    lowered = text.lower()
    yes_count = len(re.findall(r"\byes\b", lowered))
    no_count = len(re.findall(r"\bno\b", lowered))
    return "Yes" if yes_count > no_count else "No"


def parse_questions(text):
    """Parse generated questions from VLM output."""
    if not text:
        return []

    questions = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"^\(?\s*[Qq]\d+\s*\)?\s*[:.]\s*", "", line)
        line = re.sub(r"^Question\s*\d+\s*[:.]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^Output\s*\d+\s*[:.]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^\d+\s*[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)

        if len(line.split()) < 4:
            continue
        if not line.endswith("?"):
            line = line.rstrip(" .;:") + "?"
        questions.append(line)

    deduped = []
    seen = set()
    for q in questions:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped


def save_main_questions(class_name, questions, save_dir="./saved_questions"):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    out_file = save_path / f"{class_name}.json"
    payload = {
        "class_name": class_name,
        "num_questions": len(questions),
        "main_questions": questions,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved main questions: {out_file}")
    return out_file


def load_main_questions(class_name, save_dir="./saved_questions"):
    in_file = Path(save_dir) / f"{class_name}.json"
    if not in_file.exists():
        raise FileNotFoundError(f"Main questions file not found: {in_file}")
    with in_file.open("r", encoding="utf-8") as f:
        return json.load(f)["main_questions"]


def save_sub_questions(class_name, sub_questions_dict, save_dir="./saved_questions"):
    """
    sub_questions_dict: dict[int, list[str]]
        key   = Main-Q 索引 (0-based)
        value = Sub-Q 字符串列表
    保存为 {save_dir}/{class_name}_subq.json
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    out_file = save_path / f"{class_name}_subq.json"
    serializable = {str(k): v for k, v in sub_questions_dict.items()}
    payload = {
        "class_name":    class_name,
        "num_main_q":    len(sub_questions_dict),
        "sub_questions": serializable,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved sub-questions cache: {out_file}")
    return out_file


def load_sub_questions(class_name, save_dir="./saved_questions"):
    """
    返回 dict[int, list[str]]，key 恢复为 int。
    文件不存在时返回 None。
    """
    in_file = Path(save_dir) / f"{class_name}_subq.json"
    if not in_file.exists():
        return None
    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["sub_questions"].items()}


def get_normality_definition(class_name):
    """Appendix C.2 normality definitions."""
    definitions = {
        "breakfast_box": (
            "- The breakfast box always contain exactly two tangerines and one nectarine that are always "
            "located on the left-hand side of the box.\n"
            "- The ratio and relative position of the cereals and the mix of banana chips and almonds on "
            "the right-hand side are fixed."
        ),
        "screw_bag": (
            "- A screw bag contains exactly two washers, two nuts, one long screw, and one short screw.\n"
            "- All bolts (screws) are longer than 3 times the diameter of the washer."
        ),
        "pushpins": (
            "- Each compartment of the box of pushpins contains exactly one pushpin."
        ),
        "splicing_connectors": (
            "- Exactly two splicing connectors with the same number of cable clamps are linked by exactly one cable.\n"
            "- In addition, the number of clamps has a one-to-one correspondence to the cable color.\n"
            "- The cable must be connected to the same position on both connectors to maintain mirror symmetry.\n"
            "- The cable length is roughly longer than the length of the splicing connector terminal block."
        ),
        "juice_bottle": (
            "- The juice bottle is filled with one fruit type of juice and carries exactly two labels.\n"
            "- The first label is attached to the center of the bottle, with the fruit icon positioned exactly "
            "at the center of the label, clearly indicating the type of fruit juice.\n"
            "- The second is attached to the lower part of the bottle with the text \"100% Juice\" written on it.\n"
            "- The fill level is the same for each bottle.\n"
            "- The bottle is filled with at least 90% of its capacity with juice, but not 100%."
        ),
    }
    return definitions.get(class_name, "")
