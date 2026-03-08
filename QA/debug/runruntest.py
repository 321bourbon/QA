import os
import time
from datetime import datetime

import test as test_mod
import train as train_mod


DEFAULT_SPLICE_ROOT = r"E:\dataSet\MVtec LOCO AD\preprocessed\full\selected_largest_bpm"
SPLICE_CLASS = "splicing_connectors"


def _configure_train(mvtec_root):
    train_mod.CONFIG["mvtec_root"] = mvtec_root
    train_mod.CONFIG["class_list"] = [SPLICE_CLASS]
    train_mod.CONFIG["normal_shot"] = 3  # 3
    train_mod.CONFIG["filter_shot"] = 5  # 5
    train_mod.CONFIG["mainq_acc_threshold"] = 0.8  # 0.8
    train_mod.CONFIG["seed"] = 42  # 42


def _configure_test(mvtec_root):
    test_mod.CONFIG["mvtec_root"] = mvtec_root
    test_mod.CONFIG["class_list"] = [SPLICE_CLASS]
    test_mod.CONFIG["max_good_images_per_class"] = 10  # 10
    test_mod.CONFIG["max_images_per_class"] = 0  # 0
    test_mod.CONFIG["segment_policy"] = "largest"  # largest
    test_mod.CONFIG["max_segments_per_query"] = 1  # 1
    test_mod.CONFIG["results_dir"] = "./test_results/runruntest/splicing_connectors"


def _run_step(fn):
    t0 = time.time()
    try:
        fn()
        return True, None, time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


def main():
    total_start = time.time()
    splice_root = os.environ.get("LOGICQA_SPLICE_ROOT", DEFAULT_SPLICE_ROOT)
    class_root = os.path.join(splice_root, SPLICE_CLASS)

    print("=" * 80)
    print("Run Train + Test Once (splicing_connectors)")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"mvtec_root: {splice_root}")
    print("=" * 80)

    need_dirs = [
        os.path.join(class_root, "train", "good"),
        os.path.join(class_root, "test", "good"),
        os.path.join(class_root, "test", "logical_anomalies"),
    ]
    missing = [p for p in need_dirs if not os.path.exists(p)]
    if missing:
        print("missing required directories:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(1)

    _configure_train(splice_root)
    _configure_test(splice_root)

    train_ok, train_err, train_dt = _run_step(train_mod.main)
    print(f"\ntrain.py: {'OK' if train_ok else 'FAIL'} ({train_dt:.1f}s)")
    if train_err:
        print(f"train error: {train_err}")

    test_ok, test_err, test_dt = (False, "train failed", 0.0)
    if train_ok:
        test_ok, test_err, test_dt = _run_step(test_mod.main)
        print(f"test.py:  {'OK' if test_ok else 'FAIL'} ({test_dt:.1f}s)")
        if test_err:
            print(f"test error: {test_err}")
    else:
        print("test.py:  SKIP (train failed)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"train.py: {'OK' if train_ok else 'FAIL'}")
    print(f"test.py:  {'OK' if test_ok else 'FAIL'}")
    print(f"Total Elapsed: {total_elapsed:.1f}s")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    if not (train_ok and test_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
