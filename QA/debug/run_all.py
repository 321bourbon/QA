"""
debug/run_all.py — Full end-to-end pipeline: preprocess → train → test.

Chains run_preprocess.py and run_pipeline.py.

Usage:
    python debug/run_all.py                          # preprocess all + GPT train+test
    python debug/run_all.py --vlm gemini             # use Gemini for train+test
    python debug/run_all.py --skip-preprocess        # skip preprocessing step
    python debug/run_all.py --classes splicing_connectors
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


_DEBUG_DIR = Path(__file__).parent


def _run_script(script_path, extra_args):
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n{'='*80}")
    print(f"  Running: {' '.join(str(a) for a in cmd)}")
    print(f"{'='*80}")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    return rc == 0, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="LogicQA full pipeline: preprocess + train + test")
    parser.add_argument("--vlm", choices=["gpt", "gemini", "internvl"], default="internvl",
                        help="VLM backend for train+test (default: gpt)")
    parser.add_argument("--mode", choices=["api", "mock", "model"], default="api",
                        help="api = real calls, mock/model = offline debug (default: api)")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class list (default: all 5)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing; run train+test only")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip train.py; run test.py only")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of train+test runs (paper uses 3, default: 1)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d%H%M")
    print("=" * 80)
    print(f"LogicQA End-to-End Pipeline  —  {stamp}")
    print(f"  vlm    : {args.vlm}")
    print(f"  mode   : {args.mode}")
    print(f"  classes: {args.classes or 'all'}")
    print("=" * 80)

    total_start = time.time()
    steps       = []

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    if not args.skip_preprocess:
        prepro_args = []
        if args.classes:
            # Run preprocess only for specified classes (one at a time)
            for cls in args.classes.split(","):
                cls = cls.strip()
                if cls:
                    ok, dt = _run_script(_DEBUG_DIR / "run_preprocess.py", ["--class", cls])
                    steps.append(("preprocess", cls, ok, dt))
                    if not ok:
                        print(f"\n[WARN] Preprocessing failed for {cls}, continuing ...")
        else:
            ok, dt = _run_script(_DEBUG_DIR / "run_preprocess.py", [])
            steps.append(("preprocess", "all", ok, dt))
            if not ok:
                print("\n[WARN] Preprocessing step had failures, continuing to train+test ...")

    # ── Step 2: Train + Test ──────────────────────────────────────────────────
    pipeline_args = [
        "--vlm",  args.vlm,
        "--mode", args.mode,
        "--num-runs", str(args.num_runs),
    ]
    if args.classes:
        pipeline_args += ["--classes", args.classes]
    if args.skip_train:
        pipeline_args.append("--skip-train")

    ok, dt = _run_script(_DEBUG_DIR / "run_pipeline.py", pipeline_args)
    steps.append(("train+test", args.vlm, ok, dt))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - total_start
    print("\n" + "=" * 80)
    print("  PIPELINE SUMMARY")
    print("=" * 80)
    for step_name, target, ok, dt in steps:
        status = "OK" if ok else "FAIL"
        print(f"  {step_name:<15} [{target}]  {status}  ({dt:.1f}s)")
    print(f"  {'─'*50}")
    print(f"  Total elapsed: {total/60:.1f} min")
    print("=" * 80)

    if not all(ok for _, _, ok, _ in steps):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
