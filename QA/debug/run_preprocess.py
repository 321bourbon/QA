"""
debug/run_preprocess.py — Preprocessing pipeline runner.

Usage:
    python debug/run_preprocess.py                  # run full preprocessing (all classes)
    python debug/run_preprocess.py --method bpm     # BPM only
    python debug/run_preprocess.py --class screw_bag

Preprocessing steps (Appendix F):
    1. BPM  → screw_bag, splicing_connectors
    2. Grounded-SAM → pushpins, splicing_connectors (chained from BPM output)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ORIG_ROOT = r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection"
PREP_ROOT = r"E:\dataSet\MVtec LOCO AD\preprocessed"

# Full ordered job list: (method, class_name, input_root, output_root)
ALL_JOBS = [
    ("bpm",          "screw_bag",            ORIG_ROOT,            PREP_ROOT),
    ("bpm",          "splicing_connectors",  ORIG_ROOT,            PREP_ROOT),
    ("grounded_sam", "pushpins",             ORIG_ROOT,            PREP_ROOT),
    ("grounded_sam", "splicing_connectors",  PREP_ROOT + r"\bpm",  PREP_ROOT),
]

# Map class name → its job(s)
_CLASS_JOBS = {}
for _job in ALL_JOBS:
    _CLASS_JOBS.setdefault(_job[1], []).append(_job)


_PROJECT_ROOT = str(Path(__file__).parent.parent)


def _run_one(method, class_name, input_root, output_root):
    pipeline_dir = Path(__file__).parent.parent / "pipeline"
    cmd = [
        sys.executable, str(pipeline_dir / "preprocess.py"),
        "--method", method,
        "--class",  class_name,
        "--input-root",  input_root,
        "--output-root", output_root,
    ]
    print(f"\n{'='*70}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*70}")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _PROJECT_ROOT + (os.pathsep + existing if existing else "")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False, env=env).returncode
    dt = time.time() - t0
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"  {status}  ({dt:.1f}s)")
    return rc == 0


def main():
    parser = argparse.ArgumentParser(description="LogicQA preprocessing runner")
    parser.add_argument("--method", choices=["bpm", "grounded_sam", "all"], default="all",
                        help="Preprocessing method to run (default: all)")
    parser.add_argument("--class", dest="class_name", default=None,
                        help="Limit to a single class name")
    args = parser.parse_args()

    jobs = ALL_JOBS
    if args.class_name:
        jobs = _CLASS_JOBS.get(args.class_name, [])
        if not jobs:
            print(f"[WARN] no preprocessing jobs defined for class '{args.class_name}'")
            return
    if args.method != "all":
        jobs = [j for j in jobs if j[0] == args.method]

    print("=" * 70)
    print(f"LogicQA Preprocessing  —  {len(jobs)} job(s)")
    print("=" * 70)

    ok    = 0
    start = time.time()
    for method, cls, in_root, out_root in jobs:
        if _run_one(method, cls, in_root, out_root):
            ok += 1

    total = time.time() - start
    print("\n" + "=" * 70)
    print(f"  Done: {ok}/{len(jobs)} succeeded  |  elapsed: {total/60:.1f} min")
    print("=" * 70)
    if ok < len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
