import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import median

from PIL import Image, ImageOps

from api.gpt_api import APIQuotaExceededError, VLM
from utils.logger import setup_runtime_logger, shutdown_runtime_logger
from utils.utils import extract_result, load_main_questions
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
    "subq_num": 5,  # Section 3.5
    "results_dir": os.environ.get("LOGICQA_RESULTS_DIR", "./test_results"),
    "max_good_images_per_class": 10,  # 10
    "max_images_per_class": 0,  # 0 = no limit
    "segment_policy": "largest",  # largest
    "max_segments_per_query": 1,  # 1
    "segment_tile_size": 320,  # 320
    "segment_cache_dir": "./runtime_logs/segment_montage_cache",
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


def _pick_largest(paths):
    best = None
    best_area = -1
    for p in paths:
        try:
            with Image.open(p) as im:
                area = im.size[0] * im.size[1]
            if area > best_area:
                best_area = area
                best = p
        except Exception:
            continue
    return best or paths[0]


def _build_segment_query_image(paths, class_name, subset_name, base_name):
    if len(paths) <= 1:
        return str(paths[0])

    policy = CONFIG["segment_policy"].lower()
    if policy == "first":
        return str(paths[0])
    if policy == "largest":
        return str(_pick_largest(paths))

    max_seg = max(1, CONFIG["max_segments_per_query"])
    tile = max(64, CONFIG["segment_tile_size"])
    selected = paths[:max_seg]
    out_dir = Path(CONFIG["segment_cache_dir"]) / class_name / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}_montage.jpg"

    if out_path.exists():
        return str(out_path)

    images = []
    for p in selected:
        try:
            with Image.open(p) as im:
                images.append(im.convert("RGB").copy())
        except Exception:
            continue
    if not images:
        return str(paths[0])

    cols = 1 if len(images) == 1 else 2
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * tile), (0, 0, 0))
    for idx, im in enumerate(images):
        tile_img = ImageOps.pad(im, (tile, tile), color=(0, 0, 0))
        x = (idx % cols) * tile
        y = (idx // cols) * tile
        canvas.paste(tile_img, (x, y))

    canvas.save(out_path, quality=95)
    return str(out_path)


def _select_segment_query(seg_paths, class_name, subset_name, base_name):
    policy = CONFIG["segment_policy"].lower()
    if policy == "first":
        chosen = seg_paths[0]
        display = f"{base_name}.png (seg=first -> {chosen.name})"
        return str(chosen), display
    if policy == "largest":
        chosen = _pick_largest(seg_paths)
        display = f"{base_name}.png (seg=largest -> {Path(chosen).name})"
        return str(chosen), display

    montage = _build_segment_query_image(seg_paths, class_name, subset_name, base_name)
    display = f"{base_name}.png (seg=montage {min(len(seg_paths), max(1, CONFIG['max_segments_per_query']))}/{len(seg_paths)})"
    return montage, display


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
                seg_paths = [p for _, p in sorted(rec["seg"], key=lambda x: x[0])]
                query_path, display_name = _select_segment_query(seg_paths, class_name, sub.name, base)
            else:
                raw_paths = sorted(rec["raw"])
                query_path = str(raw_paths[0])
                display_name = Path(query_path).name
            bucket.append((query_path, label, sub.name, display_name))

    if CONFIG["max_good_images_per_class"] > 0:
        good_images = good_images[: CONFIG["max_good_images_per_class"]]

    images = good_images + anomaly_images
    if CONFIG["max_images_per_class"] > 0:
        images = images[: CONFIG["max_images_per_class"]]
    return images


def parse_sub_questions(response, fallback_q):
    if not response.strip():
        return [fallback_q] * CONFIG["subq_num"]

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

    deduped = []
    seen = set()
    for q in out:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    if not deduped:
        deduped = [fallback_q]
    while len(deduped) < CONFIG["subq_num"]:
        deduped.append(fallback_q)
    return deduped[: CONFIG["subq_num"]]


def generate_sub_questions(vlm, main_questions):
    all_sub_q = {}
    print("\n[Sub-Q generation]")
    for i, main_q in enumerate(main_questions):
        prompt = p_gen_subq(main_q)
        response, _ = vlm.ask("", prompt)
        sub_qs = parse_sub_questions(response, main_q)
        all_sub_q[i] = sub_qs
        print(f"  Main-Q{i + 1}: {len(sub_qs)} Sub-Q generated")
    return all_sub_q


def test_single_image(vlm, image_path, class_name, main_questions, sub_questions, image_display_name=None):
    main_q_results = []
    main_q_logprobs = []
    display_name = image_display_name or os.path.basename(image_path)
    segment_name = os.path.basename(image_path)

    for i, main_q in enumerate(main_questions):
        sub_qs = sub_questions.get(i, [main_q] * CONFIG["subq_num"])
        sub_answers = []
        sub_logprobs = []
        print(f"\n    MainQ {i + 1}/{len(main_questions)}: {main_q}")

        for j, sub_q in enumerate(sub_qs, 1):
            prompt = p_test(sub_q, class_name)
            try:
                response, logprob = vlm.ask1(image_path, prompt)
                result_text = extract_result(response)
                q_ij = 0 if result_text == "Yes" else 1
                sub_answers.append(q_ij)
                lp = logprob if logprob is not None else -2.0
                sub_logprobs.append(lp)
                print(
                    f"      M{i + 1}-S{j} [{display_name} | seg={segment_name}] "
                    f"Result={result_text} logprob={lp:.4f}"
                )
            except APIQuotaExceededError:
                raise
            except Exception as e:
                print(f"    warn: ask1 failed on sub-question, fallback to 'No' ({e})")
                sub_answers.append(1)
                sub_logprobs.append(-2.0)
                print(
                    f"      M{i + 1}-S{j} [{display_name} | seg={segment_name}] "
                    f"Result=No logprob=-2.0000"
                )

        Q_i = 0 if sub_answers.count(0) > sub_answers.count(1) else 1
        main_q_results.append(Q_i)
        matching = [sub_logprobs[j] for j in range(len(sub_answers)) if sub_answers[j] == Q_i]
        main_q_logprobs.append(max(matching) if matching else -2.0)

    prediction = "Normal" if sum(main_q_results) == 0 else "Anomaly"

    S = [math.exp(s_i) for s_i in main_q_logprobs]
    med = median(S) if S else 0.5
    anomaly_score = (1.0 - med) if prediction == "Normal" else med

    triggered = [main_questions[i] for i, q in enumerate(main_q_results) if q == 1]
    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "main_q_results": main_q_results,
        "main_q_logprobs": main_q_logprobs,
        "triggered_questions": triggered,
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

    sub_questions = generate_sub_questions(vlm, main_questions)

    results = []
    for idx, item in enumerate(test_images, 1):
        if len(item) == 4:
            img_path, label, subfolder, display_name = item
        else:
            img_path, label, subfolder = item
            display_name = os.path.basename(img_path)

        print(f"\n  [{idx}/{len(test_images)}] {display_name}")
        try:
            result = test_single_image(
                vlm,
                img_path,
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

        result["image_path"] = img_path
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
        print(f"config.segment_policy: {CONFIG['segment_policy']}")
        print(f"config.max_segments_per_query: {CONFIG['max_segments_per_query']}")

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
