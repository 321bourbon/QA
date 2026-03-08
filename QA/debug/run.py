import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


NUM_RUNS = 3  # Paper Section 5.1: mean over 3 runs
BASE_SEED = 42


def run_command(cmd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    start = time.time()
    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode == 0, time.time() - start


def read_summary(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    print("=" * 80)
    print("LogicQA end-to-end runner")
    print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"runs: {NUM_RUNS}")
    print("=" * 80)

    run_summaries = []
    total_start = time.time()

    for run_idx in range(1, NUM_RUNS + 1):
        seed = BASE_SEED + run_idx - 1
        results_dir = f"./test_results/run_{run_idx}"
        env = {
            "LOGICQA_SEED": str(seed),
            "LOGICQA_RESULTS_DIR": results_dir,
        }

        print("\n" + "#" * 80)
        print(f"Run {run_idx}/{NUM_RUNS} | seed={seed}")
        print("#" * 80)

        ok_train, t_train = run_command([sys.executable, "train.py"], env)
        print(f"train.py: {'OK' if ok_train else 'FAIL'} ({t_train:.1f}s)")
        if not ok_train:
            continue

        ok_test, t_test = run_command([sys.executable, "test.py"], env)
        print(f"test.py: {'OK' if ok_test else 'FAIL'} ({t_test:.1f}s)")
        if not ok_test:
            continue

        summary = read_summary(Path(results_dir) / "summary.json")
        if summary:
            run_summaries.append(summary)

    elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"elapsed: {elapsed/60:.1f} min")

    if not run_summaries:
        print("no valid run summary generated")
        print("=" * 80)
        return

    avg_auroc = sum(s["average"]["auroc"] for s in run_summaries if s.get("average")) / len(run_summaries)
    avg_f1 = sum(s["average"]["f1_max"] for s in run_summaries if s.get("average")) / len(run_summaries)
    print(f"mean AUROC over {len(run_summaries)} runs: {avg_auroc*100:.2f}%")
    print(f"mean F1-max over {len(run_summaries)} runs: {avg_f1*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
