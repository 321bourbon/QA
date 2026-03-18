import os
import sys
import random
import faulthandler
import signal
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import setup_runtime_logger, shutdown_runtime_logger
from utils.utils import (
    extract_result, get_normality_definition, parse_questions,
    parse_sub_questions, save_main_questions, save_sub_questions,
)
from api.vlm_prompts import p_describe, p_gen_mainq, p_gen_subq, p_summarize, p_test

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

# ──────────────────────────────────────────────────────────────────────────────
# Dual-backend design
#   1) generation backend: describe / summarize / generate main-q / generate sub-q
#   2) filter backend: validate candidate main-q on normal images
#
# Env vars:
#   LOGICQA_GEN_VLM     = gpt / gemini / internvl   (default: gpt)
#   LOGICQA_FILTER_VLM  = gpt / gemini / internvl   (default: internvl)
# ──────────────────────────────────────────────────────────────────────────────
_GEN_BACKEND = os.environ.get("LOGICQA_GEN_VLM", "gpt").lower()
_FILTER_BACKEND = os.environ.get("LOGICQA_FILTER_VLM", "internvl").lower()

_RUNTIME_LOG_DIR = "/home/liuxinyi/projects/QA/runtime_logs"


def _import_vlm_by_name(name: str):
    if name == "gemini":
        from api.gemini_api import APIQuotaExceededError, VLM
        return APIQuotaExceededError, VLM, "Gemini"
    elif name == "internvl":
        from api.internvl_api import APIQuotaExceededError, VLM
        return APIQuotaExceededError, VLM, "InternVL"
    else:
        from api.gpt_api import APIQuotaExceededError, VLM
        return APIQuotaExceededError, VLM, "GPT"


GEN_APIQuotaExceededError, GenVLM, _GEN_NAME = _import_vlm_by_name(_GEN_BACKEND)
FILTER_APIQuotaExceededError, FilterVLM, _FILTER_NAME = _import_vlm_by_name(_FILTER_BACKEND)

_LOG_PREFIX = f"{_GEN_NAME}(gen)+{_FILTER_NAME}(filter)+train_mix"


def _env_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


CONFIG = {
    # 共用配置
    "mode": os.environ.get("LOGICQA_MODE", "api"),
    "temperature": 1.0,
    "max_new_tokens": None,

    # 生成端配置
    "gen_api_key": os.environ.get("OPENAI_API_KEY", os.environ.get("openai_api_key", "")),
    "gen_model": os.environ.get("LOGICQA_GEN_MODEL", "gpt-5"),

    # 过滤端配置（InternVL 本地）
    "filter_model": os.environ.get("LOGICQA_FILTER_MODEL", "internvl-local"),

    # 训练流程配置
    "normal_shot": 3,
    "mainq_acc_threshold": 0.8,
    "filter_shot": 20,
    "subq_num": 5,
    "seed": int(os.environ.get("LOGICQA_SEED", 42)),
    "class_list": _env_list(
        "LOGICQA_CLASSES",
        ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
    ),
}

CLASS_PATHS = {
    "breakfast_box": {
        "train_good": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection/breakfast_box/train/good",
        "test_good": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection/breakfast_box/test/good",
        "test_anomaly": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection/breakfast_box/test/logical_anomalies",
    },
    "juice_bottle": {
        "train_good": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection/juice_bottle/train/good",
        "test_good": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection/juice_bottle/test/good",
        "test_anomaly": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection/juice_bottle/test/logical_anomalies",
    },
    "pushpins": {
        "train_good": "/home/liuxinyi/datasets/MVtec_preprocessed/sam3/pushpins/train/good/viz",
        "test_good": "/home/liuxinyi/datasets/MVtec_preprocessed/sam3/pushpins/test/good/viz",
        "test_anomaly": "/home/liuxinyi/datasets/MVtec_preprocessed/sam3/pushpins/test/logical_anomalies",
    },
    "screw_bag": {
        "train_good": "/home/liuxinyi/datasets/MVtec_preprocessed/bpm/screw_bag/train/good",
        "test_good": "/home/liuxinyi/datasets/MVtec_preprocessed/bpm/screw_bag/test/good",
        "test_anomaly": "/home/liuxinyi/datasets/MVtec_preprocessed/bpm/screw_bag/test/logical_anomalies",
    },
    "splicing_connectors": {
        "train_good": "/home/liuxinyi/datasets/MVtec_preprocessed/full/grounded_sam/splicing_connectors/train/good/viz",
        "test_good": "/home/liuxinyi/datasets/MVtec_preprocessed/full/grounded_sam/splicing_connectors/test/good/viz",
        "test_anomaly": "/home/liuxinyi/datasets/MVtec_preprocessed/full/grounded_sam/splicing_connectors/test/logical_anomalies/viz",
    },
}


def _validate_config():
    if _GEN_BACKEND == "gpt":
        gm = CONFIG["gen_model"].lower()
        if not any(x in gm for x in ["gpt-5", "gpt-5o", "gpt-4o"]):
            raise ValueError(
                f"生成端模型不合法: {CONFIG['gen_model']}。"
                f"请使用 gpt-5 / gpt-5o / gpt-4o。"
            )


def build_vlm_cfg(role: str, backend: str):
    """
    尽量保持和原始 train.py 一致的配置风格：
    api_key / mode / model / temperature / max_new_tokens
    """
    cfg = {
        "api_key": "",
        "mode": CONFIG["mode"],
        "model": "",
        "temperature": CONFIG["temperature"],
        "max_new_tokens": CONFIG["max_new_tokens"],
    }

    if role == "gen":
        cfg["model"] = CONFIG["gen_model"]
        if backend in {"gpt", "gemini"}:
            cfg["api_key"] = CONFIG["gen_api_key"]
    else:
        cfg["model"] = CONFIG["filter_model"]
        if backend in {"gpt", "gemini"}:
            raise ValueError("filter 端当前应为本地 InternVL，不应设置为在线 API 后端")

    return cfg


def load_normal_images(class_name):
    class_cfg = CLASS_PATHS.get(class_name)
    if class_cfg is None:
        print(f"  [错误] CLASS_PATHS 未配置类别: {class_name}")
        return []

    folder = Path(class_cfg["train_good"])
    print(f"  [DEBUG] class={class_name}")
    print(f"  [DEBUG] train_good={folder}")

    if not folder.exists():
        print(f"  [警告] 找不到正常图像文件夹: {folder}")
        return []

    images = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    print(f"  [调试] 发现正常图像文件数: {len(images)}")
    return [str(p) for p in images]


def generate_main_questions(vlm, class_name, fewshot_imgs):
    normal_def = get_normality_definition(class_name)
    if not normal_def:
        print(f"  [警告] {class_name} 缺少正常性定义 (normality definition)，跳过")
        return []

    print("\n[阶段1] 描述正常图像 (Describe Normal Images)")
    descriptions = []
    for idx, img_path in enumerate(fewshot_imgs, 1):
        print(f"  {idx}/{len(fewshot_imgs)}  {os.path.basename(img_path)}")
        prompt = p_describe(class_name, normal_def)
        response, _ = vlm.ask(img_path, prompt)

        text = (response or "").strip()
        if text.lower().startswith("api error:"):
            print(f"    [警告] API 返回错误文本，丢弃: {text}")
            continue

        if len(text) >= 20:
            descriptions.append(text)
            preview = text[:160].replace("\n", " ")
            ellipsis = "..." if len(text) > 160 else ""
            print(f"    描述预览: {preview}{ellipsis}")
        else:
            print(f"    [警告] 描述过短 ({len(text)} 字符)，丢弃")

    if len(descriptions) < 3:
        print(f"  [失败] 仅收集到 {len(descriptions)} 条有效描述（需要3条），跳过")
        return []

    print("\n[阶段2] 汇总正常特征 (Summarize Normal Context)")
    summary_prompt = p_summarize(class_name, descriptions[:3])
    summary, _ = vlm.ask("", summary_prompt)
    summary = (summary or "").strip()

    if summary.lower().startswith("api error:"):
        print(f"  [失败] 摘要生成返回 API 错误文本: {summary}")
        return []

    if len(summary) < 20:
        print("  [失败] 摘要生成失败（回复过短），跳过")
        return []

    preview = summary[:200].replace("\n", " ")
    print(f"  摘要预览: {preview}{'...' if len(summary) > 200 else ''}")

    print("\n[阶段3] 生成主问题 (Generate Main Questions)")
    mainq_prompt = p_gen_mainq(class_name, summary, normal_def)
    raw_questions, _ = vlm.ask("", mainq_prompt)
    raw_questions = (raw_questions or "").strip()

    if raw_questions.lower().startswith("api error:"):
        print(f"  [失败] 主问题生成返回 API 错误文本: {raw_questions}")
        return []

    questions = parse_questions(raw_questions)
    print(f"  共生成 {len(questions)} 个候选主问题 (candidate main questions):")
    for k, q in enumerate(questions, 1):
        print(f"    [{k}] {q}")

    if questions:
        print(f"  [调试] 主问题生成成功：{questions[:3]}")
    return questions


def filter_main_questions(vlm, class_name, candidate_qs, filter_imgs):
    print("\n[阶段4] 过滤主问题（正常图像一致性检验）")
    print(f"  保留阈值 (threshold): {CONFIG['mainq_acc_threshold']:.0%}  "
          f"（正常图像回答 Yes 的比例须达到此值才保留）")
    kept = []

    for idx, q in enumerate(candidate_qs, 1):
        yes = 0
        for img_path in filter_imgs:
            prompt = p_test(q, class_name)
            response, _ = vlm.ask(img_path, prompt)
            if extract_result(response) == "Yes":
                yes += 1
        acc = yes / len(filter_imgs)
        passed = acc >= CONFIG["mainq_acc_threshold"]
        mark = "保留 ✓ (keep)" if passed else "丢弃 ✗ (drop)"
        print(f"  Q{idx}: {acc:.1%}  →  {mark}")
        print(f"       问题: {q}")
        if passed:
            kept.append(q)

    print(f"\n  过滤完成: 保留 {len(kept)}/{len(candidate_qs)} 个主问题")
    return kept


def main():
    Path(_RUNTIME_LOG_DIR).mkdir(parents=True, exist_ok=True)
    setup_runtime_logger(log_dir=_RUNTIME_LOG_DIR, file_prefix=_LOG_PREFIX, redirect_stdout=True)
    random.seed(CONFIG["seed"])

    try:
        _validate_config()

        print("=" * 80)
        print("  LogicQA 训练流程 (Training Pipeline)")
        print("=" * 80)
        print(f"  生成后端 gen_vlm          : {_GEN_BACKEND}")
        print(f"  过滤后端 filter_vlm       : {_FILTER_BACKEND}")
        print(f"  生成模型 gen_model        : {CONFIG['gen_model']}")
        print(f"  过滤模型 filter_model     : {CONFIG['filter_model']}")
        if _GEN_BACKEND in {"gpt", "gemini"}:
            print(f"  gen_api_key 已设置        : {'Yes' if bool(CONFIG['gen_api_key']) else 'No'}")
        else:
            print(f"  gen_api_key 已设置        : N/A (backend={_GEN_BACKEND})")
        print(f"  filter_api_key 已设置     : N/A (backend={_FILTER_BACKEND})")
        print(f"  配置 class_list           : {CONFIG['class_list']}")
        print(f"  配置 normal_shot          : {CONFIG['normal_shot']}")
        print(f"  配置 filter_shot          : {CONFIG['filter_shot']}")
        print(f"  配置 mainq_acc_threshold  : {CONFIG['mainq_acc_threshold']}")
        print(f"  配置 subq_num             : {CONFIG['subq_num']}")
        print(f"  配置 seed                 : {CONFIG['seed']}")
        print("=" * 80)

        gen_vlm = GenVLM(build_vlm_cfg("gen", _GEN_BACKEND))
        filter_vlm = FilterVLM(build_vlm_cfg("filter", _FILTER_BACKEND))

        class_list = CONFIG["class_list"]

        for cls_idx, class_name in enumerate(class_list, 1):
            print("\n" + "=" * 80)
            print(f"  [{cls_idx}/{len(class_list)}]  类别 (class): {class_name}")
            print("=" * 80)

            all_normals = load_normal_images(class_name)
            if len(all_normals) < CONFIG["normal_shot"] + 1:
                print(f"  [跳过] 正常图像不足 ({len(all_normals)} 张，至少需要 {CONFIG['normal_shot'] + 1} 张)")
                continue

            fewshot_imgs = random.sample(all_normals, CONFIG["normal_shot"])
            remaining = [p for p in all_normals if p not in set(fewshot_imgs)]
            filter_k = min(CONFIG["filter_shot"], len(remaining))
            filter_imgs = random.sample(remaining, filter_k) if filter_k > 0 else []

            print(f"  正常图像总数  (total_normals)  : {len(all_normals)}")
            print(f"  描述用样本数  (fewshot_selected): {len(fewshot_imgs)}")
            print(f"  过滤用样本数  (filter_selected) : {len(filter_imgs)}")

            candidates = generate_main_questions(gen_vlm, class_name, fewshot_imgs)
            if not candidates:
                print("  [跳过] 无候选主问题")
                continue

            final_qs = filter_main_questions(filter_vlm, class_name, candidates, filter_imgs) if filter_imgs else candidates

            if final_qs:
                save_main_questions(class_name, final_qs, vlm_name=_GEN_BACKEND)
                print(f"  最终保留主问题数 (final_main_questions): {len(final_qs)}")

                print("\n[阶段5] 为每个主问题生成子问题 (Generate Sub-Questions)")
                sub_q_dict = {}
                for q_idx, main_q in enumerate(final_qs):
                    prompt = p_gen_subq(main_q)
                    response, _ = gen_vlm.ask("", prompt)
                    text = (response or "").strip()

                    if text.lower().startswith("api error:"):
                        print(f"  主问题 {q_idx + 1}/{len(final_qs)}: {main_q}")
                        print(f"    [警告] 子问题生成返回 API 错误文本，跳过: {text}")
                        continue

                    sub_qs = parse_sub_questions(
                        text, fallback_q=main_q, subq_num=CONFIG["subq_num"]
                    )
                    sub_q_dict[q_idx] = sub_qs
                    print(f"  主问题 {q_idx + 1}/{len(final_qs)}: {main_q}")
                    for k, sq in enumerate(sub_qs, 1):
                        prefix = "  └──" if k == len(sub_qs) else "  ├──"
                        print(f"    {prefix} 子问题 {k}: {sq}")
                save_sub_questions(class_name, sub_q_dict, vlm_name=_GEN_BACKEND)
            else:
                print("  [跳过] 所有候选主问题均被过滤")

        print("\n" + "=" * 80)
        print("  训练完成 (Training Done).")
        print("=" * 80)
    finally:
        shutdown_runtime_logger()


if __name__ == "__main__":
    main()