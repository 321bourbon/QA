import os
import sys
import json
import math
import re
import faulthandler
import signal
from datetime import datetime
from pathlib import Path
from statistics import median

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.utils import (
    display_sep, display_box,
    extract_result, load_main_questions,
    load_sub_questions, save_sub_questions, parse_sub_questions,
)

from utils.logger import setup_runtime_logger, shutdown_runtime_logger

from api.vlm_prompts import p_gen_subq, p_test

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def append_result_jsonl(path, record):
    """追加写入单张图结果，供断点续跑使用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

def load_partial_results(path):
    """
    从 jsonl 读取已完成结果。
    返回:
      - finished_map: {unique_key: result_dict}
      - finished_order: [result_dict, ...]
    """
    finished_map = {}
    finished_order = []

    path = Path(path)
    if not path.exists():
        return finished_map, finished_order

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            unique_key = rec.get("unique_key")
            if not unique_key:
                display_name = rec.get("display_name")
                subfolder = rec.get("subfolder", "")
                if not display_name:
                    continue
                unique_key = f"{subfolder}/{display_name}"

            result = {
                "image_path": rec.get("image_path"),
                "display_name": rec.get("display_name"),
                "true_label": rec.get("true_label"),
                "prediction": rec.get("prediction"),
                "anomaly_score": rec.get("anomaly_score"),
                "score_median": rec.get("score_median"),
                "score_components": rec.get("score_components", []),
                "triggered_questions": rec.get("triggered_questions", []),
                "subfolder": rec.get("subfolder"),
                "main_q_results": rec.get("main_q_results", []),
                "main_q_logprobs": rec.get("main_q_logprobs", []),
                "num_segments": rec.get("num_segments"),
                "segment_paths": rec.get("segment_paths", []),
                "per_segment_results": rec.get("per_segment_results"),
                "unique_key": unique_key,
            }
            finished_map[unique_key] = result

    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            unique_key = rec.get("unique_key")
            if not unique_key:
                display_name = rec.get("display_name")
                subfolder = rec.get("subfolder", "")
                if not display_name:
                    continue
                unique_key = f"{subfolder}/{display_name}"

            if unique_key in seen:
                continue
            if unique_key in finished_map:
                finished_order.append(finished_map[unique_key])
                seen.add(unique_key)

    return finished_map, finished_order

# ── Dynamic VLM selection via LOGICQA_VLM env var ────────────────────────────
# Supported: gpt (default), gemini, internvl
_VLM_BACKEND = os.environ.get("LOGICQA_VLM", "internvl").lower()
if _VLM_BACKEND == "gemini":
    from api.gemini_api import APIQuotaExceededError, VLM
    _LOG_PREFIX = "Gemini+test"
elif _VLM_BACKEND == "internvl":
    from api.internvl_api import APIQuotaExceededError, VLM
    _LOG_PREFIX = "InternVL+test"
else:
    from api.gpt_api import APIQuotaExceededError, VLM
    _LOG_PREFIX = "GPT+test"


def _env_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


CONFIG = {
    "api_key": "",
    "mode": os.environ.get("LOGICQA_MODE", "api"),
    "model": os.environ.get("LOGICQA_MODEL", "gpt-4o"),  # Appendix B.1
    "temperature": 1.0,  # Appendix B.1
    "max_new_tokens": None,
    "subq_num": 5,                # 论文 = 5（Section 3.5）
    "results_dir": os.environ.get("LOGICQA_RESULTS_DIR", "./test_results"),
    "max_good_images_per_class": 30,   # 论文 = 0（全量）
    "max_anomaly_images_per_class": 30, 
    "max_images_per_class": 0,
    "max_segments_per_query": 0,       # 论文 = 全部
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
        "test_anomaly": "/home/liuxinyi/datasets/MVtec_preprocessed/sam3/pushpins/test/logical_anomalies/viz",
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

# ── Image loading ─────────────────────────────────────────────────────────────

def _parse_segment_name(path_obj):
    m = re.match(r"^(.*)_(\d+)$", path_obj.stem)
    if m:
        return m.group(1), int(m.group(2))
    return path_obj.stem, None

def _uniform_sample(lst, n):
    """从 lst 中均匀取 n 个，n=0 或 n>=len 时返回全部。"""
    if n <= 0 or n >= len(lst):
        return lst
    indices = [int(i * len(lst) / n) for i in range(n)]
    return [lst[i] for i in indices]

def load_test_images(class_name):
    class_cfg = CLASS_PATHS.get(class_name)
    if class_cfg is None:
        print(f"  [错误] CLASS_PATHS 未配置类别: {class_name}")
        return []

    good_dir = Path(class_cfg["test_good"])
    anomaly_dir = Path(class_cfg["test_anomaly"])

    print(f"  [DEBUG] class={class_name}")
    print(f"  [DEBUG] test_good={good_dir}")
    print(f"  [DEBUG] test_anomaly={anomaly_dir}")

    if not good_dir.exists():
        print(f"  [警告] 正常测试目录不存在: {good_dir}")
        return []
    if not anomaly_dir.exists():
        print(f"  [警告] 异常测试目录不存在: {anomaly_dir}")
        return []

    def collect_images_from_dir(img_dir, label, subfolder_name):
        bucket = []
        groups = {}

        images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        print(f"  [DEBUG] {subfolder_name} 图像数: {len(images)}")

        for img in images:
            base, seg_idx = _parse_segment_name(img)
            rec = groups.setdefault(base, {"seg": [], "raw": []})
            if seg_idx is None:
                rec["raw"].append(img)
            else:
                rec["seg"].append((seg_idx, img))

        for base, rec in groups.items():
            if rec["seg"]:
                seg_paths = [str(p) for _, p in sorted(rec["seg"], key=lambda x: x[0])]
                max_seg = CONFIG["max_segments_per_query"]
                if max_seg > 0:
                    seg_paths = seg_paths[:max_seg]
                display_name = f"{base}.png ({len(seg_paths)} 个分片)"
            else:
                seg_paths = [str(sorted(rec["raw"])[0])]
                display_name = Path(seg_paths[0]).name

            bucket.append((seg_paths, label, subfolder_name, display_name))

        return bucket

    good_images = collect_images_from_dir(good_dir, 0, good_dir.name)
    anomaly_images = collect_images_from_dir(anomaly_dir, 1, anomaly_dir.name)

    good_images = _uniform_sample(good_images, CONFIG["max_good_images_per_class"])
    anomaly_images = _uniform_sample(anomaly_images, CONFIG["max_anomaly_images_per_class"])

    images = good_images + anomaly_images
    if CONFIG["max_images_per_class"] > 0:
        images = images[:CONFIG["max_images_per_class"]]

    return images

# ── Sub-question generation ───────────────────────────────────────────────────

def generate_sub_questions(vlm, class_name, main_questions):
    cached = load_sub_questions(class_name, vlm_name=_VLM_BACKEND)
    if cached is not None and len(cached) == len(main_questions):
        print()
        print(display_sep("-"))
        print(f"  [子问题] 从缓存加载 (loaded from cache)  ·  共 {len(cached)} 个主问题")
        for i, main_q in enumerate(main_questions):
            sub_qs = cached.get(i, [])
            print(f"  主问题 {i+1}/{len(main_questions)}: {main_q}")
            for k, sq in enumerate(sub_qs, 1):
                prefix = "  └──" if k == len(sub_qs) else "  ├──"
                print(f"    {prefix} 子问题 {k}: {sq}")
        print(display_sep("-"))
        return cached

    print()
    print(display_sep("-"))
    print(f"  [子问题] 无缓存，正在生成 (cache miss, generating) for {class_name} ...")
    all_sub_q = {}
    for i, main_q in enumerate(main_questions):
        prompt   = p_gen_subq(main_q)
        response, _ = vlm.ask("", prompt)
        sub_qs   = parse_sub_questions(response, fallback_q=main_q, subq_num=CONFIG["subq_num"])
        all_sub_q[i] = sub_qs
        print(f"  主问题 {i+1}/{len(main_questions)}: {main_q}")
        for k, sq in enumerate(sub_qs, 1):
            prefix = "  └──" if k == len(sub_qs) else "  ├──"
            print(f"    {prefix} 子问题 {k}: {sq}")
    save_sub_questions(class_name, all_sub_q, vlm_name=_VLM_BACKEND)
    print(display_sep("-"))
    return all_sub_q


# ── Single-segment inference ──────────────────────────────────────────────────

def _test_one_segment(vlm, image_path, class_name, main_questions, sub_questions,
                      image_display_name=None):
    main_q_results  = []
    main_q_logprobs = []
    display_name    = image_display_name or os.path.basename(image_path)
    segment_name    = os.path.basename(image_path)

    for i, main_q in enumerate(main_questions):
        sub_qs       = sub_questions.get(i, [main_q] * CONFIG["subq_num"])
        sub_answers  = []
        sub_logprobs = []

        print(f"\n    ◆ 主问题 (Main-Q) {i+1}/{len(main_questions)}: {main_q}")

        for j, sub_q in enumerate(sub_qs, 1):
            prompt = p_test(sub_q, class_name)
            try:
                print(f"        [ENTER_SUBQ] main={i+1} sub={j}/{len(sub_qs)} seg={segment_name}")
                print(f"        [BEFORE_VLM] question={sub_q}")

                response, logprob = vlm.ask1(image_path, prompt)

                print(f"        [AFTER_VLM] main={i+1} sub={j}/{len(sub_qs)}")
                result_text = extract_result(response)
                print(f"        [AFTER_PARSE] result={result_text} logprob={logprob}")

                q_ij        = 0 if result_text == "Yes" else 1
                sub_answers.append(q_ij)
                sub_logprobs.append(logprob)
                lp_display  = f"{logprob:.4f}" if logprob is not None else "N/A"
                print(
                    f"        子问题 (Sub-Q) {j}/{len(sub_qs)}: {sub_q}\n"
                    f"                   → 回答 (Answer): {result_text:<3}  logprob: {lp_display}"
                    f"  [分片={segment_name}]"
                )
            except APIQuotaExceededError:
                raise
            except Exception as e:
                sub_answers.append(1)
                sub_logprobs.append(None)
                print(
                    f"        子问题 (Sub-Q) {j}/{len(sub_qs)}: {sub_q}\n"
                    f"                   → 回答 (Answer): No   logprob: N/A  [警告: {e}]"
                    f"  [分片={segment_name}]"
                )

        yes_count = sub_answers.count(0)
        no_count  = sub_answers.count(1)
        Q_i       = 0 if yes_count > no_count else 1
        main_q_results.append(Q_i)

        matching_lps = [
            sub_logprobs[j]
            for j in range(len(sub_answers))
            if sub_answers[j] == Q_i and sub_logprobs[j] is not None
        ]
        best_lp = max(matching_lps) if matching_lps else None
        main_q_logprobs.append(best_lp)

        vote_label = "YES (正常)" if Q_i == 0 else "NO  (异常)"
        lp_str     = f"{best_lp:.4f}" if best_lp is not None else "N/A"
        print(
            f"        {'─' * 62}\n"
            f"        投票结果 (Vote): {vote_label}  "
            f"({yes_count if Q_i==0 else no_count}/{len(sub_qs)} 票)  "
            f"→ Q_{i+1} = {'Normal' if Q_i==0 else 'Anomaly'}  "
            f"|  best_logprob: {lp_str}"
        )

    prediction = "Normal" if sum(main_q_results) == 0 else "Anomaly"

    valid_lps = [lp for lp in main_q_logprobs if lp is not None]
    score_components = [math.exp(lp) for lp in valid_lps] if valid_lps else []
    score_median = median(score_components) if score_components else 0.5
    anomaly_score = -score_median if prediction == "Normal" else score_median

    triggered = [main_questions[i] for i, q in enumerate(main_q_results) if q == 1]
    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "score_median": score_median,
        "score_components": score_components,
        "main_q_results": main_q_results,
        "main_q_logprobs": main_q_logprobs,
        "triggered_questions": triggered,
    }

def test_single_image_multi_seg(vlm, segment_paths, class_name, main_questions,
                                sub_questions, image_display_name=None):
    """逐片轮询策略 — 论文 Section 4 & Appendix F."""
    if len(segment_paths) == 1:
        return _test_one_segment(
            vlm, segment_paths[0], class_name,
            main_questions, sub_questions, image_display_name
        )

    all_seg_results = []
    for seg_idx, seg_path in enumerate(segment_paths):
        seg_display = f"{image_display_name}_分片{seg_idx+1}"
        print()
        print(f"    {'─'*60}")
        print(f"    [分片 {seg_idx+1}/{len(segment_paths)}  (Segment)] {os.path.basename(seg_path)}")
        print(f"    {'─'*60}")
        seg_result = _test_one_segment(
            vlm, seg_path, class_name,
            main_questions, sub_questions, seg_display
        )
        all_seg_results.append(seg_result)

    any_anomaly = any(r["prediction"] == "Anomaly" for r in all_seg_results)
    prediction = "Anomaly" if any_anomaly else "Normal"

    best_seg = max(all_seg_results, key=lambda r: r["anomaly_score"])
    anomaly_score = best_seg["anomaly_score"]
    score_median = best_seg.get("score_median")
    score_components = best_seg.get("score_components", [])

    triggered = []
    seen_q = set()
    for r in all_seg_results:
        for q in r["triggered_questions"]:
            if q not in seen_q:
                seen_q.add(q)
                triggered.append(q)

    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "score_median": score_median,
        "score_components": score_components,
        "main_q_results": best_seg.get("main_q_results", []),
        "main_q_logprobs": best_seg.get("main_q_logprobs", []),
        "triggered_questions": triggered,
        "num_segments": len(segment_paths),
        "per_segment_results": all_seg_results,
    }

# ── Per-image result display ──────────────────────────────────────────────────

def _print_image_result(result, true_label):
    prediction = result["prediction"]
    score = result["anomaly_score"]
    score_median = result.get("score_median")
    score_components = result.get("score_components", [])
    true_str = "Normal" if true_label == 0 else "Anomaly"
    correct = (prediction == "Anomaly") == (true_label == 1)
    verdict = "正确 ✓ (CORRECT)" if correct else "错误 ✗ (WRONG)"

    box_lines = [
        f"最终预测 (Pred)  : {prediction:<10}  |  异常分数 (Score) : {score:.4f}",
        f"真实标签 (True)  : {true_str:<10}  |  判断结果 (Eval)  : {verdict}",
    ]

    if score_median is not None:
        box_lines.append(f"score_median      : {score_median:.4f}")
    if score_components:
        box_lines.append(f"score_components  : {[round(x, 4) for x in score_components]}")

    triggered = result.get("triggered_questions", [])
    if triggered:
        box_lines.append("")
        box_lines.append("触发异常的问题 (Triggered Questions — anomaly explanation):")
        for k, q in enumerate(triggered, 1):
            box_lines.append(f"  [{k}] {q}")
    else:
        box_lines.append("")
        box_lines.append("无异常触发 — 所有主问题均通过 (No anomaly triggered, all passed).")

    num_segs = result.get("num_segments")
    if num_segs and num_segs > 1:
        box_lines.append("")
        box_lines.append(f"处理分片数 (Segments processed): {num_segs}")

    print()
    print(display_box(box_lines))

# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(results, test_images):
    y_true   = [item[1] for item in test_images]
    y_scores = [r["anomaly_score"] for r in results]
    y_pred   = [1 if r["prediction"] == "Anomaly" else 0 for r in results]

    if len(results) != len(test_images):
        print(f"  [警告] results 数量({len(results)}) 与 test_images 数量({len(test_images)}) 不一致")

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        print(f"  [警告] 当前类别只有单一标签，AUROC 不可靠: pos={n_pos}, neg={n_neg}")

    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    try:
        from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
        try:
            auroc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auroc = 0.0
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            f1_values = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_max    = float(max(f1_values))
        except Exception:
            f1_max = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        auroc  = accuracy
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_max = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "auroc": auroc, "f1_max": f1_max, "accuracy": accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": len(y_true),
    }


def _print_class_metrics(class_name, metrics):
    print()
    print(display_sep("="))
    print(f"  类别指标汇总 (CLASS METRICS) : {class_name}")
    print(display_sep("="))
    print(
        f"  AUROC    : {metrics['auroc']:.4f}    "
        f"F1-max  : {metrics['f1_max']:.4f}    "
        f"Accuracy (准确率) : {metrics['accuracy']:.4f}"
    )
    print(
        f"  TP (真阳) : {metrics['tp']:>4}    TN (真阴) : {metrics['tn']:>4}    "
        f"FP (假阳) : {metrics['fp']:>4}    FN (假阴) : {metrics['fn']:>4}    "
        f"Total (总数) : {metrics['total']:>4}"
    )
    print(display_sep("="))


# ── Per-class runner ──────────────────────────────────────────────────────────

def test_class(vlm, class_name, main_questions):
    print()
    print(display_sep("#"))
    print(f"  类别 (CLASS) : {class_name}")
    print(display_sep("#"))

    test_images = load_test_images(class_name)
    if not test_images:
        print("  [警告] 该类别没有测试图像")
        return None

    n_normal = sum(1 for t in test_images if t[1] == 0)
    n_anomaly = sum(1 for t in test_images if t[1] == 1)
    print(
        f"  测试图像总数: {len(test_images)}  |  "
        f"正常 (Normal): {n_normal}  |  异常 (Anomaly): {n_anomaly}  |  "
        f"主问题数: {len(main_questions)}"
    )

    sub_questions = generate_sub_questions(vlm, class_name, main_questions)

    partial_path = Path(CONFIG["results_dir"]) / f"{class_name}_partial.jsonl"
    finished_map, finished_order = load_partial_results(partial_path)

    if finished_map:
        print()
        print(display_sep("-"))
        print(f"  [断点续跑] 已检测到部分结果文件: {partial_path}")
        print(f"  [断点续跑] 已完成图片数: {len(finished_map)}")
        print(display_sep("-"))

    results = list(finished_order)

    for idx, item in enumerate(test_images, 1):
        seg_paths, label, subfolder, display_name = item
        unique_key = f"{subfolder}/{display_name}"
        true_str = "Normal" if label == 0 else "Anomaly"

        if unique_key in finished_map:
            print()
            print(display_sep("-"))
            print(
                f"  [图像 {idx:>3}/{len(test_images)}] {display_name}"
                f"  (标签={true_str}, 子集={subfolder})"
            )
            print("  [SKIP] 已在 partial.jsonl 中完成，跳过")
            print(display_sep("-"))
            continue

        print()
        print(display_sep("-"))
        print(
            f"  [图像 {idx:>3}/{len(test_images)}] {display_name}"
            f"  (标签={true_str}, 子集={subfolder})"
        )
        print(display_sep("-"))

        try:
            result = test_single_image_multi_seg(
                vlm, seg_paths, class_name,
                main_questions, sub_questions,
                image_display_name=display_name,
            )
        except APIQuotaExceededError as e:
            print(f"  [致命] API 配额耗尽，终止运行 (quota exceeded): {e}")
            save_class_progress(class_name, results, CONFIG["results_dir"],
                                status="quota_exceeded", fatal_error=str(e))
            raise
        except Exception as e:
            print(f"  [错误] 图像处理失败，默认标记为异常 (fallback Anomaly): {e}")
            result = {
                "prediction": "Anomaly",
                "anomaly_score": 1.0,
                "score_median": 1.0,
                "score_components": [],
                "main_q_results": [],
                "main_q_logprobs": [],
                "triggered_questions": [],
                "error": str(e),
            }

        result["image_path"] = seg_paths[0]
        result["display_name"] = display_name
        result["true_label"] = label
        result["subfolder"] = subfolder
        result["unique_key"] = unique_key

        if "num_segments" not in result:
            result["num_segments"] = len(seg_paths)
        if "segment_paths" not in result:
            result["segment_paths"] = seg_paths

        results.append(result)
        finished_map[unique_key] = result

        _print_image_result(result, label)

        append_result_jsonl(
            partial_path,
            {
                "class_name": class_name,
                "image_path": result.get("image_path"),
                "display_name": result.get("display_name"),
                "unique_key": result.get("unique_key"),
                "true_label": result.get("true_label"),
                "subfolder": result.get("subfolder"),
                "segment_paths": result.get("segment_paths", seg_paths),
                "prediction": result.get("prediction"),
                "anomaly_score": result.get("anomaly_score"),
                "score_median": result.get("score_median"),
                "score_components": result.get("score_components", []),
                "triggered_questions": result.get("triggered_questions", []),
                "main_q_results": result.get("main_q_results", []),
                "main_q_logprobs": result.get("main_q_logprobs", []),
                "num_segments": result.get("num_segments"),
                "per_segment_results": result.get("per_segment_results"),
            }
        )

        save_class_progress(class_name, results, CONFIG["results_dir"],
                            status="running", fatal_error=None)

    metrics = calculate_metrics(results, test_images)
    _print_class_metrics(class_name, metrics)

    save_class_progress(class_name, results, CONFIG["results_dir"],
                        status="completed", fatal_error=None)
    return {"class_name": class_name, "metrics": metrics, "results": results}

# ── Result persistence ────────────────────────────────────────────────────────

def save_results(all_results, save_path):
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for class_result in all_results:
        if not class_result:
            continue
        class_name = class_result["class_name"]
        class_file = out_dir / f"{class_name}_results.json"
        simple = {
            "class_name": class_name,
            "metrics": class_result["metrics"],
            "results": [
                {
                    "image_path": r["image_path"],
                    "display_name": r.get("display_name"),
                    "true_label": r["true_label"],
                    "prediction": r["prediction"],
                    "anomaly_score": r["anomaly_score"],
                    "score_median": r.get("score_median"),
                    "score_components": r.get("score_components", []),
                    "triggered_questions": r["triggered_questions"],
                }
                for r in class_result["results"]
            ],
        }
        with class_file.open("w", encoding="utf-8") as f:
            json.dump(simple, f, indent=2, ensure_ascii=False)
        print(f"  [已保存] {class_file}")

    valid = [r for r in all_results if r]
    summary = {"classes": [], "average": {}}
    if valid:
        auroc_list = [r["metrics"]["auroc"] for r in valid]
        f1_list = [r["metrics"]["f1_max"] for r in valid]
        acc_list = [r["metrics"]["accuracy"] for r in valid]
        for r in valid:
            summary["classes"].append({
                "class_name": r["class_name"],
                "auroc": r["metrics"]["auroc"],
                "f1_max": r["metrics"]["f1_max"],
                "accuracy": r["metrics"]["accuracy"],
            })
        summary["average"] = {
            "auroc": sum(auroc_list) / len(auroc_list),
            "f1_max": sum(f1_list) / len(f1_list),
            "accuracy": sum(acc_list) / len(acc_list),
        }

    summary_file = out_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [已保存] {summary_file}")

def save_class_progress(class_name, results, save_path,
                        status="running", fatal_error=None):
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / f"{class_name}_progress.json"
    payload = {
        "class_name":       class_name,
        "status":           status,
        "updated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed_images": len(results),
        "fatal_error":      fatal_error,
        "results": [
            {
                "image_path":          r.get("image_path"),
                "display_name":        r.get("display_name"),
                "true_label":          r.get("true_label"),
                "prediction":          r.get("prediction"),
                "anomaly_score":       r.get("anomaly_score"),
                "triggered_questions": r.get("triggered_questions", []),
            }
            for r in results
        ],
    }
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    setup_runtime_logger(log_dir="./runtime_logs", file_prefix=_LOG_PREFIX,
                         redirect_stdout=True)
    try:
        print(display_sep("="))
        print(f"  LogicQA 测试流程 (Testing Pipeline)  —  {_VLM_BACKEND.upper()}")
        print(display_sep("="))
        print(f"  测试类别列表 (class_list)              : {CONFIG['class_list']}")
        print(f"  每类最大正常图数 (max_good_images)     : {CONFIG['max_good_images_per_class']}")
        print(f"  每类最大图像总数 (max_images)          : {CONFIG['max_images_per_class']}  (0=不限)")
        print(f"  每次最大分片数  (max_segments)         : {CONFIG['max_segments_per_query']}  (0=全部)")
        print(display_sep("="))

        vlm         = VLM(CONFIG)
        class_list  = CONFIG["class_list"]
        all_results = []

        for idx, class_name in enumerate(class_list, 1):
            print()
            print(display_sep("#"))
            print(f"  [{idx}/{len(class_list)}]  类别 (class): {class_name}")
            print(display_sep("#"))
            try:
                main_questions = load_main_questions(class_name, vlm_name=_VLM_BACKEND)
            except Exception as e:
                print(f"  [跳过] 加载主问题失败: {e}")
                continue
            if not main_questions:
                print("  [跳过] 主问题列表为空")
                continue
            try:
                class_result = test_class(vlm, class_name, main_questions)
            except APIQuotaExceededError as e:
                print(f"  [停止] 类别 '{class_name}' 处理中 API 配额耗尽: {e}")
                save_results(all_results, CONFIG["results_dir"])
                raise SystemExit(2)
            all_results.append(class_result)
            save_results(all_results, CONFIG["results_dir"])

        valid = [r for r in all_results if r]
        if not valid:
            return

        print()
        print(display_sep("="))
        print("  整体测试结果 (OVERALL RESULTS)")
        print(display_sep("="))
        print(f"  {'类别 (Class)':<28} {'AUROC':>10} {'F1-max':>10} {'Accuracy':>10}")
        print(display_sep("-"))
        for r in valid:
            m = r["metrics"]
            print(
                f"  {r['class_name']:<28} "
                f"{m['auroc']*100:>9.1f}%  "
                f"{m['f1_max']*100:>9.1f}%  "
                f"{m['accuracy']*100:>9.1f}%"
            )
        avg_auroc = sum(r["metrics"]["auroc"]    for r in valid) / len(valid)
        avg_f1    = sum(r["metrics"]["f1_max"]   for r in valid) / len(valid)
        avg_acc   = sum(r["metrics"]["accuracy"] for r in valid) / len(valid)
        print(display_sep("-"))
        print(
            f"  {'平均值 (Average)':<28} "
            f"{avg_auroc*100:>9.1f}%  "
            f"{avg_f1*100:>9.1f}%  "
            f"{avg_acc*100:>9.1f}%"
        )
        print(display_sep("="))

        save_results(all_results, CONFIG["results_dir"])
    finally:
        shutdown_runtime_logger()


if __name__ == "__main__":
    main()
