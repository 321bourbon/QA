"""
debug/run_pipeline.py — Train + Test pipeline runner with VLM selection.

Usage:
    python debug/run_pipeline.py                              # GPT, all 5 classes
    python debug/run_pipeline.py --vlm gemini                 # Gemini
    python debug/run_pipeline.py --vlm gpt --mode mock        # GPT mock (no real API)
    python debug/run_pipeline.py --classes breakfast_box,juice_bottle
    python debug/run_pipeline.py --skip-train                 # test only (reuse saved questions)

VLM choices: gpt (default), gemini, internvl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


_PROJECT_ROOT = str(Path(__file__).parent.parent)


def _run(script, env):
    """Run a pipeline script via subprocess and return (ok, elapsed_seconds)."""
    pipeline_dir = Path(__file__).parent.parent / "pipeline"
    cmd = [sys.executable, str(pipeline_dir / script)]
    # Ensure the project root is on PYTHONPATH so 'utils', 'api', etc. are importable.
    run_env = dict(env)
    existing = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = _PROJECT_ROOT + (os.pathsep + existing if existing else "")
    t0  = time.time()
    rc  = subprocess.run(cmd, env=run_env, check=False).returncode
    return rc == 0, time.time() - t0


def _read_summary(results_dir):
    p = Path(results_dir) / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="LogicQA train+test runner")
    parser.add_argument("--vlm", choices=["gpt", "gemini", "internvl"], default="gpt",
                        help="VLM backend (default: gpt)")
    parser.add_argument("--mode", choices=["api", "mock", "model"], default="api",
                        help="api = real API calls, mock/model = offline debug (default: api)")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class list (default: all 5)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip train.py and run test.py only")
    parser.add_argument("--results-dir", default=None,
                        help="Override results output directory")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of repeated runs (paper uses 3, default: 1)")
    args = parser.parse_args()

    vlm_upper  = args.vlm.upper()
    stamp      = datetime.now().strftime("%Y%m%d%H%M")
    base_dir   = args.results_dir or f"./test_results/{args.vlm}_{stamp}"

    class_list = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
    )

    # Determine model name defaults per VLM
    model_defaults = {"gpt": "gpt-4o", "gemini": "gemini-2.5-flash", "internvl": "internvl2-8b"}
    model_name     = model_defaults[args.vlm]
    if args.mode in {"mock", "model"}:
        model_name = f"mock-{model_name}"

    print("=" * 80)
    print(f"LogicQA Pipeline  —  VLM={vlm_upper}  mode={args.mode}")
    print(f"classes : {class_list}")
    print(f"runs    : {args.num_runs}")
    print(f"base_dir: {base_dir}")
    print("=" * 80)

    all_run_summaries = []
    total_start = time.time()

    for run_idx in range(1, args.num_runs + 1):
        seed        = 41 + run_idx
        results_dir = base_dir if args.num_runs == 1 else f"{base_dir}/run_{run_idx}"

        print(f"\n{'#'*80}")
        print(f"  Run {run_idx}/{args.num_runs}  |  seed={seed}")
        print(f"{'#'*80}")

        env = os.environ.copy()
        env.update({
            "LOGICQA_VLM":          args.vlm,
            "LOGICQA_MODE":         args.mode,
            "LOGICQA_MODEL":        model_name,
            "LOGICQA_CLASSES":      ",".join(class_list),
            "LOGICQA_RESULTS_DIR":  results_dir,
            "LOGICQA_SEED":         str(seed),
        })

        run_ok = True

        if not args.skip_train:
            ok_train, t_train = _run("train.py", env)
            print(f"  train.py : {'OK' if ok_train else 'FAIL'}  ({t_train:.1f}s)")
            if not ok_train:
                run_ok = False
                print("  [skip test — train failed]")
                continue

        ok_test, t_test = _run("test.py", env)
        print(f"  test.py  : {'OK' if ok_test else 'FAIL'}  ({t_test:.1f}s)")
        if not ok_test:
            run_ok = False

        summary = _read_summary(results_dir)
        if summary and summary.get("average"):
            avg = summary["average"]
            print(f"  AUROC={avg.get('auroc', 0)*100:.2f}%  F1-max={avg.get('f1_max', 0)*100:.2f}%")
            all_run_summaries.append(summary)

    elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print(f"  Finished  |  elapsed: {elapsed/60:.1f} min")
    if all_run_summaries:
        valid_avgs = [s["average"] for s in all_run_summaries if s.get("average")]
        if valid_avgs:
            mean_auroc = sum(a.get("auroc", 0) for a in valid_avgs) / len(valid_avgs)
            mean_f1    = sum(a.get("f1_max", 0) for a in valid_avgs) / len(valid_avgs)
            print(f"  Mean over {len(valid_avgs)} run(s):  AUROC={mean_auroc*100:.2f}%  F1-max={mean_f1*100:.2f}%")

            if args.num_runs > 1:
                agg = {
                    "vlm": args.vlm, "mode": args.mode,
                    "num_runs": args.num_runs,
                    "mean_auroc": mean_auroc, "mean_f1_max": mean_f1,
                }
                agg_path = Path(base_dir) / "aggregate.json"
                agg_path.parent.mkdir(parents=True, exist_ok=True)
                agg_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  aggregate saved: {agg_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
