import time
from datetime import datetime
from pathlib import Path

import train as train_mod
import test as test_mod


ORIG_ROOT = r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection"
BPM_ROOT = r"E:\dataSet\MVtec LOCO AD\preprocessed\bpm"
GSAM_ROOT = r"E:\dataSet\MVtec LOCO AD\preprocessed\grounded_sam"


CLASS_PLAN = [
    {"class_name": "breakfast_box", "mvtec_root": ORIG_ROOT},
    {"class_name": "juice_bottle", "mvtec_root": ORIG_ROOT},
    {"class_name": "pushpins", "mvtec_root": GSAM_ROOT},
    {"class_name": "screw_bag", "mvtec_root": BPM_ROOT},
]


def _is_quota_error(err):
    if not err:
        return False
    msg = str(err).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


def _ensure_root(root, class_name):
    p = Path(root) / class_name
    need = [p / "train" / "good", p / "test" / "good", p / "test" / "logical_anomalies"]
    return all(x.exists() for x in need), p


def _configure_train(class_name, mvtec_root):
    train_mod.CONFIG["mvtec_root"] = mvtec_root
    train_mod.CONFIG["class_list"] = [class_name]
    train_mod.CONFIG["normal_shot"] = 3  # 3
    train_mod.CONFIG["filter_shot"] = 5  # 5
    train_mod.CONFIG["mainq_acc_threshold"] = 0.8  # 0.8
    train_mod.CONFIG["seed"] = 42  # 42


def _configure_test(class_name, mvtec_root):
    test_mod.CONFIG["mvtec_root"] = mvtec_root
    test_mod.CONFIG["class_list"] = [class_name]
    test_mod.CONFIG["max_good_images_per_class"] = 10  # 10
    test_mod.CONFIG["max_images_per_class"] = 0  # 0
    test_mod.CONFIG["segment_policy"] = "largest"  # largest
    test_mod.CONFIG["max_segments_per_query"] = 1  # 1
    test_mod.CONFIG["results_dir"] = f"./test_results/runruntest1/{class_name}"


def _run_train():
    train_mod.main()


def _run_test():
    test_mod.main()


def _run_step(fn):
    t0 = time.time()
    try:
        fn()
        return True, None, time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def main():
    total_start = time.time()
    print("=" * 90)
    print("Run Other 4 Classes: train + test")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    all_results = []
    stop_due_quota = False

    for idx, item in enumerate(CLASS_PLAN, 1):
        if stop_due_quota:
            break

        class_name = item["class_name"]
        mvtec_root = item["mvtec_root"]
        exists, class_path = _ensure_root(mvtec_root, class_name)

        print("\n" + "#" * 90)
        print(f"[{idx}/{len(CLASS_PLAN)}] {class_name}")
        print(f"root: {mvtec_root}")
        print("#" * 90)

        if not exists:
            print(f"skip: class path not found -> {class_path}")
            all_results.append(
                {
                    "class_name": class_name,
                    "root": mvtec_root,
                    "train_ok": False,
                    "test_ok": False,
                    "train_time": 0.0,
                    "test_time": 0.0,
                    "error": f"missing class path: {class_path}",
                }
            )
            continue

        _configure_train(class_name, mvtec_root)
        _configure_test(class_name, mvtec_root)

        train_ok, train_err, train_dt = _run_step(_run_train)
        print(f"train.py: {'OK' if train_ok else 'FAIL'} ({train_dt:.1f}s)")

        test_ok = False
        test_err = None
        test_dt = 0.0
        if train_ok:
            test_ok, test_err, test_dt = _run_step(_run_test)
            print(f"test.py: {'OK' if test_ok else 'FAIL'} ({test_dt:.1f}s)")
        else:
            print("test.py: SKIP (train failed)")

        if (not train_ok and train_err) or (train_ok and not test_ok and test_err):
            print("error detail:")
            print(train_err if not train_ok else test_err)
            if _is_quota_error(train_err if not train_ok else test_err):
                print("quota exhausted, stop remaining classes")
                stop_due_quota = True

        all_results.append(
            {
                "class_name": class_name,
                "root": mvtec_root,
                "train_ok": train_ok,
                "test_ok": test_ok,
                "train_time": train_dt,
                "test_time": test_dt,
                "error": train_err if not train_ok else test_err,
            }
        )

    total_dt = time.time() - total_start
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    for r in all_results:
        print(
            f"{r['class_name']:<18} "
            f"train={'OK' if r['train_ok'] else 'FAIL':<5} "
            f"test={'OK' if r['test_ok'] else 'FAIL':<5} "
            f"t_train={r['train_time']:.1f}s "
            f"t_test={r['test_time']:.1f}s"
        )
    print("-" * 90)
    print(f"Total Elapsed: {total_dt:.1f}s")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    all_ok = all(r["train_ok"] and r["test_ok"] for r in all_results)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
