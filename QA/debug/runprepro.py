import subprocess
import sys
import time


ORIG_ROOT = r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection"
PREP_ROOT = r"E:\dataSet\MVtec LOCO AD\preprocessed"


def run_one(method, class_name, input_root=None, output_root=None):
    cmd = [sys.executable, "preprocess.py", "--method", method, "--class", class_name]
    if input_root:
        cmd += ["--input-root", input_root]
    if output_root:
        cmd += ["--output-root", output_root]

    print(f"\n{'=' * 70}\nrun: {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    dt = time.time() - t0
    print(f"{'OK' if rc == 0 else 'FAIL'} ({dt:.1f}s)")
    return rc == 0


def main():
    """
    Full preprocessing pipeline:
    1) BPM on screw_bag, splicing_connectors
    2) Lang-SAM on pushpins (original images)
    3) Lang-SAM on splicing_connectors after BPM output (chained)
    """
    print("=" * 70)
    print("LogicQA FULL preprocessing runner")
    print("=" * 70)

    jobs = [
        # BPM stage
        ("bpm", "screw_bag", ORIG_ROOT, PREP_ROOT),
        ("bpm", "splicing_connectors", ORIG_ROOT, PREP_ROOT),
        # Lang-SAM stage
        ("grounded_sam", "pushpins", ORIG_ROOT, PREP_ROOT),
        # Chain BPM -> Lang-SAM for splicing connectors
        ("grounded_sam", "splicing_connectors", PREP_ROOT + r"\bpm", PREP_ROOT + r"\full"),
    ]

    ok = 0
    start = time.time()
    for method, cls, in_root, out_root in jobs:
        if run_one(method, cls, in_root, out_root):
            ok += 1

    print("\n" + "=" * 70)
    print(f"done: {ok}/{len(jobs)} | elapsed: {(time.time()-start)/60:.1f} min")
    print("=" * 70)
    print("outputs:")
    print(f"  BPM: {PREP_ROOT}\\bpm\\")
    print(f"  Lang-SAM: {PREP_ROOT}\\grounded_sam\\")
    print(f"  Chained full (splicing): {PREP_ROOT}\\full\\grounded_sam\\splicing_connectors\\")


if __name__ == "__main__":
    main()
