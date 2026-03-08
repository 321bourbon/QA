import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image


GSAM_SPLICE_ROOT = Path(r"E:\dataSet\MVtec LOCO AD\preprocessed\grounded_sam\splicing_connectors")
OUTPUT_ROOT = Path(r"E:\dataSet\MVtec LOCO AD\preprocessed\full\selected_largest_bpm")
SPLICE_CLASS = "splicing_connectors"
SUBSETS = ["train/good", "test/good", "test/logical_anomalies", "test/structural_anomalies"]
BPM_CROP = (0.10, 0.10, 0.90, 0.90)


def _parse_name(path_obj):
    m = re.match(r"^(.*)_(\d+)$", path_obj.stem)
    if m:
        return m.group(1), int(m.group(2))
    return path_obj.stem, None


def _iter_images(folder):
    return sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))


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
    return best if best is not None else paths[0]


def _bpm_crop(in_path, out_path):
    with Image.open(in_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        x1 = int(w * BPM_CROP[0])
        y1 = int(h * BPM_CROP[1])
        x2 = int(w * BPM_CROP[2])
        y2 = int(h * BPM_CROP[3])
        cropped = rgb.crop((x1, y1, x2, y2))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out_path)


def build_splicing_selected_bpm():
    in_class_root = GSAM_SPLICE_ROOT
    effective_output_root = OUTPUT_ROOT
    out_class_root = effective_output_root / SPLICE_CLASS
    if out_class_root.exists():
        try:
            shutil.rmtree(out_class_root)
        except Exception as e:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            effective_output_root = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}_{ts}"
            out_class_root = effective_output_root / SPLICE_CLASS
            print(f"  warn: cannot clean old output, switch to new path: {effective_output_root} ({e})")
    out_class_root.mkdir(parents=True, exist_ok=True)

    total = 0
    report_rows = []
    for subset in SUBSETS:
        in_dir = in_class_root / subset
        out_dir = out_class_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.exists():
            print(f"  skip missing subset: {in_dir}")
            continue

        groups = {}
        for p in _iter_images(in_dir):
            base, seg_idx = _parse_name(p)
            rec = groups.setdefault(base, {"seg": [], "raw": []})
            if seg_idx is None:
                rec["raw"].append(p)
            else:
                rec["seg"].append((seg_idx, p))

        print(f"\n[{subset}] groups={len(groups)}")
        done = 0
        for base in sorted(groups.keys()):
            rec = groups[base]
            if rec["seg"]:
                seg_paths = [p for _, p in sorted(rec["seg"], key=lambda x: x[0])]
                chosen = _pick_largest(seg_paths)
                select_type = "seg_largest"
            elif rec["raw"]:
                chosen = rec["raw"][0]
                select_type = "raw"
            else:
                continue

            out_path = out_dir / f"{base}.png"
            _bpm_crop(chosen, out_path)
            report_rows.append(
                {
                    "subset": subset,
                    "base": base,
                    "selected": chosen.name,
                    "select_type": select_type,
                    "saved": str(out_path),
                }
            )
            done += 1

        total += done
        print(f"  saved: {done}")

    report_path = effective_output_root / SPLICE_CLASS / "selection_report.csv"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("subset,base,selected,select_type,saved\n")
        for r in report_rows:
            f.write(f"{r['subset']},{r['base']},{r['selected']},{r['select_type']},{r['saved']}\n")
    print(f"\nselection report: {report_path}")

    return total, effective_output_root


def run_cmd(cmd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    dt = time.time() - t0
    return rc == 0, rc, dt


def main():
    all_start = time.time()
    print("=" * 100)
    print("runruntest_all")
    print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n[1/3] Build splicing_connectors (largest-segment + BPM)")
    num_saved, effective_output_root = build_splicing_selected_bpm()
    print(f"saved total images: {num_saved}")
    print(f"splicing dual-preprocess root: {effective_output_root}")

    print("\n[2/3] Run runruntest.py (splicing_connectors)")
    env = {"LOGICQA_SPLICE_ROOT": str(effective_output_root)}
    ok1, rc1, dt1 = run_cmd([sys.executable, "runruntest.py"], env)
    print(f"runruntest.py: {'OK' if ok1 else f'FAIL(code={rc1})'} ({dt1:.1f}s)")

    print("\n[3/3] Run runruntest1.py (other 4 classes)")
    ok2, rc2, dt2 = run_cmd([sys.executable, "runruntest1.py"])
    print(f"runruntest1.py: {'OK' if ok2 else f'FAIL(code={rc2})'} ({dt2:.1f}s)")

    total_dt = time.time() - all_start
    print("\n" + "=" * 100)
    print(f"end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"elapsed: {total_dt:.1f}s")
    print("=" * 100)

    if not (ok1 and ok2):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
