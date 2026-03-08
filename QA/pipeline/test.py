import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import median

from PIL import Image

from api.gpt_api import APIQuotaExceededError, VLM
from utils.logger import setup_runtime_logger, shutdown_runtime_logger
from utils.utils import (
    extract_result, load_main_questions,
    load_sub_questions, save_sub_questions, parse_sub_questions,
)
from api.vlm_prompts import p_gen_subq, p_test


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
    "subq_num": 5,                # 论文 = 5（Section 3.5）；不建议修改
    "results_dir": os.environ.get("LOGICQA_RESULTS_DIR", "./test_results"),
    # ── 成本简化开关（论文值见注释，按需恢复）──
    "max_good_images_per_class": 10,   # 论文 = 0（全量）；0 = 不限制，成本高
    "max_images_per_class": 0,         # 0 = 不限制（anomaly 侧已是全量）
    # segment 策略（0 = 全部片段逐一轮询，N = 最多 N 片）
    # 论文正确行为 = 0（全部）；成本缩减时可设为 1
    "max_segments_per_query": 0,       # 论文 = 全部（Section 4 & Appendix F）
    "class_list": _env_list(
        "LOGICQA_CLASSES",
        ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
    ),
}


def _parse_segment_name(path_obj):
    m = re.match(r"^(.*)_(\d+)$", path_obj.stem)
    if m:
        return m.group(1), int(m.group(2))
    return path_obj.stem, None


def load_test_images(class_name):
    root = Path(CONFIG["mvtec_root"]) / class_name / "test"
    if not root.exists():
        print(f"  missing test folder: {root}")
        return []

    good_images = []
    anomaly_images = []
    allowed = {"good", "logical_anomalies"}  # logical AD only

    for sub in root.iterdir():
        if not sub.is_dir() or sub.name not in allowed:
            continue
        label = 0 if sub.name == "good" else 1
        bucket = good_images if label == 0 else anomaly_images

        groups = {}
        for img in sorted(list(sub.glob("*.png")) + list(sub.glob("*.jpg"))):
            base, seg_idx = _parse_segment_name(img)
            rec = groups.setdefault(base, {"seg": [], "raw": []})
            if seg_idx is None:
                rec["raw"].append(img)
            else:
                rec["seg"].append((seg_idx, img))

        for base, rec in groups.items():
            if rec["seg"]:
                seg_paths = [str(p) for _, p in sorted(rec["seg"], key=lambda x: x[0])]
                # 按 max_segments_per_query 限制数量（0 = 全部）
                max_seg = CONFIG["max_segments_per_query"]
                if max_seg > 0:
                    seg_paths = seg_paths[:max_seg]
                display_name = f"{base}.png ({len(seg_paths)} segs)"
            else:
                seg_paths = [str(sorted(rec["raw"])[0])]
                display_name = Path(seg_paths[0]).name
            bucket.append((seg_paths, label, sub.name, display_name))

    if CONFIG["max_good_images_per_class"] > 0:
        good_images = good_images[: CONFIG["max_good_images_per_class"]]

    images = good_images + anomaly_images
    if CONFIG["max_images_per_class"] > 0:
        images = images[: CONFIG["max_images_per_class"]]
    return images


def generate_sub_questions(vlm, class_name, main_questions):
    """
    先尝试加载缓存；若缓存存在且 Main-Q 数量匹配，直接复用。
    否则实时生成并立即保存缓存，保证实验可复现。
    """
    cached = load_sub_questions(class_name)
    if cached is not None and len(cached) == len(main_questions):
        print(f"\n[Sub-Q] loaded from cache ({len(cached)} main-Qs)")
        return cached

    print(f"\n[Sub-Q] cache miss, generating for {class_name} ...")
    all_sub_q = {}
    for i, main_q in enumerate(main_questions):
        prompt = p_gen_subq(main_q)
        response, _ = vlm.ask("", prompt)
        sub_qs = parse_sub_questions(response, fallback_q=main_q, subq_num=CONFIG["subq_num"])
        all_sub_q[i] = sub_qs
        print(f"  Main-Q{i + 1}: {len(sub_qs)} Sub-Qs generated")
    save_sub_questions(class_name, all_sub_q)
    return all_sub_q


def _test_one_segment(vlm, image_path, class_name, main_questions, sub_questions, image_display_name=None):
    """对单个 segment 图像执行所有 Main-Q 的 Sub-Q 投票，返回该 segment 的结果字典。"""
    main_q_results = []
    main_q_logprobs = []  # 元素可为 float 或 None（logprob 不可用时）
    display_name = image_display_name or os.path.basename(image_path)
    segment_name = os.path.basename(image_path)

    for i, main_q in enumerate(main_questions):
        sub_qs = sub_questions.get(i, [main_q] * CONFIG["subq_num"])
        sub_answers = []
        sub_logprobs = []  # 元素可为 float 或 None
        print(f"\n    MainQ {i + 1}/{len(main_questions)}: {main_q}")

        for j, sub_q in enumerate(sub_qs, 1):
            prompt = p_test(sub_q, class_name)
            try:
                response, logprob = vlm.ask1(image_path, prompt)
                result_text = extract_result(response)
                q_ij = 0 if result_text == "Yes" else 1
                sub_answers.append(q_ij)
                sub_logprobs.append(logprob)  # None 时保持 None，不用 -2.0 回退
                lp_display = f"{logprob:.4f}" if logprob is not None else "N/A"
                print(
                    f"      M{i + 1}-S{j} [{display_name} | seg={segment_name}] "
                    f"Result={result_text} logprob={lp_display}"
                )
            except APIQuotaExceededError:
                raise
            except Exception as e:
                print(f"    warn: ask1 failed on sub-question, sub-q skipped ({e})")
                sub_answers.append(1)    # 失败时保守判 No
                sub_logprobs.append(None)  # logprob 未知，标为 None
                print(
                    f"      M{i + 1}-S{j} [{display_name} | seg={segment_name}] "
                    f"Result=No logprob=N/A"
                )

        Q_i = 0 if sub_answers.count(0) > sub_answers.count(1) else 1
        main_q_results.append(Q_i)
        # 仅收集与投票结果一致且 logprob 不为 None 的有效值
        matching_lps = [
            sub_logprobs[j]
            for j in range(len(sub_answers))
            if sub_answers[j] == Q_i and sub_logprobs[j] is not None
        ]
        main_q_logprobs.append(max(matching_lps) if matching_lps else None)

    prediction = "Normal" if sum(main_q_results) == 0 else "Anomaly"

    # Section 5.2：过滤掉 None 后再计算 S 和中位数
    valid_lps = [lp for lp in main_q_logprobs if lp is not None]
    if valid_lps:
        S = [math.exp(lp) for lp in valid_lps]
        med = median(S)
    else:
        # 极端情况：全部 logprob 缺失，回退到 binary score
        med = 0.5
    anomaly_score = (1.0 - med) if prediction == "Normal" else med

    triggered = [main_questions[i] for i, q in enumerate(main_q_results) if q == 1]
    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "main_q_results": main_q_results,
        "main_q_logprobs": main_q_logprobs,
        "triggered_questions": triggered,
    }


def test_single_image_multi_seg(vlm, segment_paths, class_name, main_questions, sub_questions, image_display_name=None):
    """
    逐片轮询策略 — 论文 Section 4 & Appendix F 正确实现。
    segment_paths: list[str]，该图像对应的所有 segment 路径（无分割则长度为 1）。
    任一 segment 的任一 Main-Q 为 No → 整图判 Anomaly。
    Anomaly Score 取所有 segment 的最大值（最可疑片段代表整图）。
    """
    if len(segment_paths) == 1:
        return _test_one_segment(
            vlm, segment_paths[0], class_name,
            main_questions, sub_questions, image_display_name
        )

    # ── 多 segment 逐一推理 ──
    all_seg_results = []
    for seg_idx, seg_path in enumerate(segment_paths):
        seg_display = f"{image_display_name}_seg{seg_idx + 1}"
        print(f"\n    [segment {seg_idx + 1}/{len(segment_paths)}] {os.path.basename(seg_path)}")
        seg_result = _test_one_segment(
            vlm, seg_path, class_name,
            main_questions, sub_questions, seg_display
        )
        all_seg_results.append(seg_result)

    # 任一 segment 触发 anomaly → 整图 Anomaly
    any_anomaly = any(r["prediction"] == "Anomaly" for r in all_seg_results)
    prediction = "Anomaly" if any_anomaly else "Normal"

    # 取所有 segment 的最大 anomaly_score
    anomaly_score = max(r["anomaly_score"] for r in all_seg_results)

    # 合并所有 segment 触发的 Main-Q（去重）
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
        "main_q_results": all_seg_results[0]["main_q_results"],  # 首片供日志参考
        "main_q_logprobs": all_seg_results[0]["main_q_logprobs"],
        "triggered_questions": triggered,
        "num_segments": len(segment_paths),
        "per_segment_results": all_seg_results,
    }


def calculate_metrics(results, test_images):
    y_true = [item[1] for item in test_images]
    y_scores = [r["anomaly_score"] for r in results]
    y_pred = [1 if r["prediction"] == "Anomaly" else 0 for r in results]

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
            f1_max = float(max(f1_values))
        except Exception:
            f1_max = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        auroc = accuracy
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_max = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "auroc": auroc,
        "f1_max": f1_max,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": len(y_true),
    }


def test_class(vlm, class_name, main_questions):
    print("\n" + "=" * 70)
    print(f"Class: {class_name}")
    print("=" * 70)

    test_images = load_test_images(class_name)
    if not test_images:
        print("  no test images")
        return None

    print(f"  test images: {len(test_images)}")
    print(f"  normal: {sum(1 for t in test_images if t[1] == 0)}")
    print(f"  anomaly: {sum(1 for t in test_images if t[1] == 1)}")
    print(f"  main questions: {len(main_questions)}")

    sub_questions = generate_sub_questions(vlm, class_name, main_questions)

    results = []
    for idx, item in enumerate(test_images, 1):
        seg_paths, label, subfolder, display_name = item

        print(f"\n  [{idx}/{len(test_images)}] {display_name}")
        try:
            result = test_single_image_multi_seg(
                vlm,
                seg_paths,
                class_name,
                main_questions,
                sub_questions,
                image_display_name=display_name,
            )
        except APIQuotaExceededError as e:
            print(f"    fatal: quota exceeded, stop testing now ({e})")
            save_class_progress(
                class_name=class_name,
                results=results,
                save_path=CONFIG["results_dir"],
                status="quota_exceeded",
                fatal_error=str(e),
            )
            raise
        except Exception as e:
            print(f"    error: failed on image, fallback prediction used ({e})")
            result = {
                "prediction": "Anomaly",
                "anomaly_score": 1.0,
                "main_q_results": [],
                "main_q_logprobs": [],
                "triggered_questions": [],
                "error": str(e),
            }

        result["image_path"] = seg_paths[0]
        result["display_name"] = display_name
        result["true_label"] = label
        result["subfolder"] = subfolder
        results.append(result)

        ok = (result["prediction"] == "Anomaly") == (label == 1)
        print(f"    pred: {result['prediction']} score={result['anomaly_score']:.4f} {'OK' if ok else 'ERR'}")
        if result["triggered_questions"]:
            print("    triggered:")
            for q in result["triggered_questions"][:3]:
                print(f"      - {q}")
        print("")

        save_class_progress(
            class_name=class_name,
            results=results,
            save_path=CONFIG["results_dir"],
            status="running",
            fatal_error=None,
        )

    metrics = calculate_metrics(results, test_images)
    print(f"\n  AUROC: {metrics['auroc']:.4f}")
    print(f"  F1-max: {metrics['f1_max']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    save_class_progress(
        class_name=class_name,
        results=results,
        save_path=CONFIG["results_dir"],
        status="completed",
        fatal_error=None,
    )
    return {"class_name": class_name, "metrics": metrics, "results": results}


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
                    "true_label": r["true_label"],
                    "prediction": r["prediction"],
                    "anomaly_score": r["anomaly_score"],
                    "triggered_questions": r["triggered_questions"],
                }
                for r in class_result["results"]
            ],
        }
        with class_file.open("w", encoding="utf-8") as f:
            json.dump(simple, f, indent=2, ensure_ascii=False)
        print(f"  saved: {class_file}")

    valid = [r for r in all_results if r]
    summary = {"classes": [], "average": {}}
    if valid:
        auroc_list = [r["metrics"]["auroc"] for r in valid]
        f1_list = [r["metrics"]["f1_max"] for r in valid]
        for r in valid:
            summary["classes"].append(
                {
                    "class_name": r["class_name"],
                    "auroc": r["metrics"]["auroc"],
                    "f1_max": r["metrics"]["f1_max"],
                    "accuracy": r["metrics"]["accuracy"],
                }
            )
        summary["average"] = {
            "auroc": sum(auroc_list) / len(auroc_list),
            "f1_max": sum(f1_list) / len(f1_list),
        }

    summary_file = out_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {summary_file}")


def save_class_progress(class_name, results, save_path, status="running", fatal_error=None):
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / f"{class_name}_progress.json"
    payload = {
        "class_name": class_name,
        "status": status,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed_images": len(results),
        "fatal_error": fatal_error,
        "results": [
            {
                "image_path": r.get("image_path"),
                "display_name": r.get("display_name"),
                "true_label": r.get("true_label"),
                "prediction": r.get("prediction"),
                "anomaly_score": r.get("anomaly_score"),
                "triggered_questions": r.get("triggered_questions", []),
            }
            for r in results
        ],
    }
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    setup_runtime_logger(log_dir="./runtime_logs", file_prefix="GPT", redirect_stdout=True)
    try:
        print("=" * 80)
        print("LogicQA Testing")
        print("=" * 80)
        print(f"config.class_list: {CONFIG['class_list']}")
        print(f"config.max_good_images_per_class: {CONFIG['max_good_images_per_class']}")
        print(f"config.max_images_per_class: {CONFIG['max_images_per_class']}")
        print(f"config.max_segments_per_query: {CONFIG['max_segments_per_query']} (0=all)")

        vlm = VLM(CONFIG)
        class_list = CONFIG["class_list"]
        all_results = []

        for idx, class_name in enumerate(class_list, 1):
            print("\n" + "#" * 80)
            print(f"[{idx}/{len(class_list)}] {class_name}")
            print("#" * 80)
            try:
                main_questions = load_main_questions(class_name)
            except Exception as e:
                print(f"  skip: {e}")
                continue
            if not main_questions:
                print("  skip: empty main questions")
                continue
            try:
                class_result = test_class(vlm, class_name, main_questions)
            except APIQuotaExceededError as e:
                print(f"  stop: quota exceeded during class '{class_name}': {e}")
                save_results(all_results, CONFIG["results_dir"])
                raise SystemExit(2)
            all_results.append(class_result)
            save_results(all_results, CONFIG["results_dir"])

        valid = [r for r in all_results if r]
        if not valid:
            return

        print("\n" + "=" * 80)
        print(f"{'Class':<25} {'AUROC':>10} {'F1-max':>10}")
        print("-" * 50)
        for r in valid:
            print(f"{r['class_name']:<25} {r['metrics']['auroc']*100:>9.1f}% {r['metrics']['f1_max']*100:>9.1f}%")
        avg_auroc = sum(r["metrics"]["auroc"] for r in valid) / len(valid)
        avg_f1 = sum(r["metrics"]["f1_max"] for r in valid) / len(valid)
        print("-" * 50)
        print(f"{'Average':<25} {avg_auroc*100:>9.1f}% {avg_f1*100:>9.1f}%")

        save_results(all_results, CONFIG["results_dir"])
    finally:
        shutdown_runtime_logger()


if __name__ == "__main__":
    main()
