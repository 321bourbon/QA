import os
import random
from pathlib import Path

from api.gpt_api import VLM
from utils.logger import setup_runtime_logger, shutdown_runtime_logger
from utils.utils import (
    extract_result, get_normality_definition, parse_questions,
    parse_sub_questions, save_main_questions, save_sub_questions,
)
from api.vlm_prompts import p_describe, p_gen_mainq, p_gen_subq, p_summarize, p_test


def _env_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


CONFIG = {
    "mvtec_root": os.environ.get("LOGICQA_MVTEC_ROOT", r"E:\dataSet\MVtec LOCO AD\preprocessed\grounded_sam"),
    "api_key": "",
    "mode": os.environ.get("LOGICQA_MODE", "api"),
    "model": os.environ.get("LOGICQA_MODEL", "gpt-4o"),  # Appendix B.1
    "temperature": 1.0,  # Appendix B.1
    "max_new_tokens": None,  # GPT-4o default
    # ── 成本简化开关（论文值见注释，按需恢复）──
    "normal_shot": 3,             # 论文 = 3（Section 5.1）；不建议修改
    "mainq_acc_threshold": 0.8,   # 论文 = 0.8（Section 3.4）；不建议修改
    "filter_shot": 5,             # 论文 = 50；数量越大过滤越稳定，成本越高
    "subq_num": 5,                # 论文 = 5（Section 3.5）；不建议修改
    "seed": 42,
    "class_list": _env_list(
        "LOGICQA_CLASSES",
        ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
    ),
}


def load_normal_images(class_name):
    folder = Path(CONFIG["mvtec_root"]) / class_name / "train" / "good"
    if not folder.exists():
        print(f"  missing folder: {folder}")
        return []
    images = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    return [str(p) for p in images]


def generate_main_questions(vlm, class_name, fewshot_imgs):
    normal_def = get_normality_definition(class_name)
    if not normal_def:
        print(f"  missing normality definition for {class_name}")
        return []

    print("\n[Stage 1] Describe normal images")
    descriptions = []
    for idx, img_path in enumerate(fewshot_imgs, 1):
        print(f"  {idx}/{len(fewshot_imgs)} {os.path.basename(img_path)}")
        prompt = p_describe(class_name, normal_def)
        response, _ = vlm.ask(img_path, prompt)
        if len(response.strip()) >= 20:
            descriptions.append(response.strip())

    if len(descriptions) < 3:
        print("  failed to collect 3 valid normal descriptions")
        return []

    print("\n[Stage 2] Summarize normal context")
    summary_prompt = p_summarize(class_name, descriptions[:3])
    summary, _ = vlm.ask("", summary_prompt)
    if len(summary.strip()) < 20:
        print("  summary generation failed")
        return []

    print("\n[Stage 3] Generate main questions")
    mainq_prompt = p_gen_mainq(class_name, summary, normal_def)
    raw_questions, _ = vlm.ask("", mainq_prompt)
    questions = parse_questions(raw_questions)
    print(f"  generated {len(questions)} candidate questions")
    return questions


def filter_main_questions(vlm, class_name, candidate_qs, filter_imgs):
    print("\n[Stage 4] Filter main questions by normal-image consistency")
    print(f"  threshold: {CONFIG['mainq_acc_threshold']:.0%}")
    kept = []

    for idx, q in enumerate(candidate_qs, 1):
        yes = 0
        for img_path in filter_imgs:
            prompt = p_test(q, class_name)
            response, _ = vlm.ask(img_path, prompt)
            if extract_result(response) == "Yes":
                yes += 1
        acc = yes / len(filter_imgs)
        mark = "keep" if acc >= CONFIG["mainq_acc_threshold"] else "drop"
        print(f"  Q{idx}: {acc:.1%} -> {mark}")
        if acc >= CONFIG["mainq_acc_threshold"]:
            kept.append(q)

    print(f"  kept {len(kept)}/{len(candidate_qs)}")
    return kept


def main():
    setup_runtime_logger(log_dir="./runtime_logs", file_prefix="GPT", redirect_stdout=True)
    random.seed(CONFIG["seed"])
    try:
        print("=" * 80)
        print("LogicQA Training")
        print("=" * 80)
        print(f"config.class_list: {CONFIG['class_list']}")
        print(f"config.normal_shot: {CONFIG['normal_shot']}")
        print(f"config.filter_shot: {CONFIG['filter_shot']}")
        print(f"config.mainq_acc_threshold: {CONFIG['mainq_acc_threshold']}")

        vlm = VLM(CONFIG)
        class_list = CONFIG["class_list"]

        for cls_idx, class_name in enumerate(class_list, 1):
            print("\n" + "=" * 80)
            print(f"[{cls_idx}/{len(class_list)}] {class_name}")
            print("=" * 80)

            all_normals = load_normal_images(class_name)
            if len(all_normals) < CONFIG["normal_shot"] + 1:
                print("  not enough normal images, skip")
                continue

            fewshot_imgs = random.sample(all_normals, CONFIG["normal_shot"])
            remaining = [p for p in all_normals if p not in set(fewshot_imgs)]
            filter_k = min(CONFIG["filter_shot"], len(remaining))
            filter_imgs = random.sample(remaining, filter_k) if filter_k > 0 else []
            print(f"  total_normals: {len(all_normals)}")
            print(f"  fewshot_selected: {len(fewshot_imgs)}")
            print(f"  filter_selected: {len(filter_imgs)}")

            candidates = generate_main_questions(vlm, class_name, fewshot_imgs)
            if not candidates:
                print("  no candidate questions, skip")
                continue

            final_qs = filter_main_questions(vlm, class_name, candidates, filter_imgs) if filter_imgs else candidates

            if final_qs:
                save_main_questions(class_name, final_qs)
                print(f"  final_main_questions: {len(final_qs)}")

                # ── Stage 5: Generate and cache Sub-Questions ──
                print("\n[Stage 5] Generate sub-questions for each Main-Q")
                sub_q_dict = {}
                for q_idx, main_q in enumerate(final_qs):
                    prompt = p_gen_subq(main_q)
                    response, _ = vlm.ask("", prompt)
                    sub_qs = parse_sub_questions(
                        response, fallback_q=main_q, subq_num=CONFIG["subq_num"]
                    )
                    sub_q_dict[q_idx] = sub_qs
                    print(f"  Main-Q{q_idx + 1}: {len(sub_qs)} sub-Qs generated")
                save_sub_questions(class_name, sub_q_dict)
            else:
                print("  all questions filtered out")

        print("\nDone.")
    finally:
        shutdown_runtime_logger()


if __name__ == "__main__":
    main()
