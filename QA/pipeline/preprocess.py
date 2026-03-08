import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


CONFIG = {
    "mvtec_root": r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection",
    "output_root": r"E:\dataSet\MVtec LOCO AD\preprocessed",
    "box_threshold": 0.2,
    "text_threshold": 0.2,
    "threshold_schedule": [0.2, 0.15, 0.1, 0.05],
    "max_detections": 50,
    "grounding_model_candidates": [
        "IDEA-Research/grounding-dino-base",
        "IDEA-Research/grounding-dino-tiny",
    ],
    # Paper Appendix F:
    "bpm_classes": {"screw_bag", "splicing_connectors"},
    # Paper Appendix B.3:
    "lang_sam_prompts": {
        "splicing_connectors": "Connector Block",
        "pushpins": "The individual black compartments within the transparent plastic storage box",
    },
}


def iter_images(folder):
    return sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))


class BPMProcessor:
    """Lightweight object-centered crop for BPM-like preprocessing."""

    # Class-specific rough crop (x1, y1, x2, y2) ratio
    CROP = {
        "screw_bag": (0.10, 0.10, 0.90, 0.90),
        "splicing_connectors": (0.10, 0.10, 0.90, 0.90),
    }

    def process(self, image_path, class_name, output_path):
        if class_name not in CONFIG["bpm_classes"]:
            return False, f"skip (paper: BPM not used for {class_name})"

        img = cv2.imread(image_path)
        if img is None:
            return False, "read failed"

        h, w = img.shape[:2]
        x1r, y1r, x2r, y2r = self.CROP[class_name]
        x1, y1 = int(w * x1r), int(h * y1r)
        x2, y2 = int(w * x2r), int(h * y2r)
        crop = img[y1:y2, x1:x2]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, crop)
        return True, "ok"


class GroundedSAMProcessor:
    """GroundingDINO + SAM-based object extraction for Lang-SAM-like preprocessing."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_ready = False
        self.sam_ready = False
        self._load_detector()
        self._load_sam()

    def _load_detector(self):
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            for model_id in CONFIG["grounding_model_candidates"]:
                try:
                    print(f"[Grounded-SAM] loading detector: {model_id}")
                    self.dino_processor = AutoProcessor.from_pretrained(model_id)
                    self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
                    self.dino_model.eval()
                    self.detector_ready = True
                    print(f"[Grounded-SAM] detector ready: {model_id}")
                    break
                except Exception as e:
                    print(f"[Grounded-SAM] detector load failed: {model_id} ({e})")
        except Exception as e:
            print(f"[Grounded-SAM] detector load failed: {e}")

    def _load_sam(self):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as e:
            print(f"[Grounded-SAM] segment-anything not available: {e}")
            return

        ckpt, model_type = self._find_sam_checkpoint()
        if not ckpt:
            print("[Grounded-SAM] SAM checkpoint not found, fallback to box crop.")
            return

        try:
            sam = sam_model_registry[model_type](checkpoint=ckpt).to(self.device)
            self.sam_predictor = SamPredictor(sam)
            self.sam_ready = True
        except Exception as e:
            print(f"[Grounded-SAM] SAM load failed: {e}")

    @staticmethod
    def _find_sam_checkpoint():
        candidates = [
            ("sam_vit_h", "vit_h"),
            ("sam_vit_l", "vit_l"),
            ("sam_vit_b", "vit_b"),
        ]
        search_dirs = [
            r"C:\Users\pc\Desktop\LAD_QA\pt",
            r"C:\Users\pc\Desktop\LAD_QA\weights",
            r".\pt",
            r".\weights",
            ".",
        ]
        for pattern, model_type in candidates:
            for d in search_dirs:
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if pattern in fn and fn.endswith(".pth"):
                        return os.path.join(d, fn), model_type
        return None, None

    def _detect(self, image_pil, prompt, box_threshold=None, text_threshold=None):
        if not self.detector_ready:
            return None, None
        if box_threshold is None:
            box_threshold = CONFIG["box_threshold"]
        if text_threshold is None:
            text_threshold = CONFIG["text_threshold"]

        # GroundingDINO is generally more stable with a trailing period.
        if not prompt.endswith("."):
            prompt = prompt + "."
        inputs = self.dino_processor(images=image_pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)
        try:
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = self.dino_processor.post_process_object_detection(
                outputs, threshold=box_threshold, target_sizes=target_sizes
            )

        if results and len(results[0]["boxes"]) > 0:
            return results[0]["boxes"].cpu().numpy(), results[0]["scores"].cpu().numpy()
        return None, None

    def _segment_masks(self, image_np, boxes):
        self.sam_predictor.set_image(image_np)
        box_tensor = torch.tensor(boxes, device=self.device)
        trans_boxes = self.sam_predictor.transform.apply_boxes_torch(box_tensor, image_np.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=trans_boxes,
            multimask_output=False,
        )
        return masks.squeeze(1).cpu().numpy()

    @staticmethod
    def _candidate_prompts(class_name):
        base = CONFIG["lang_sam_prompts"].get(class_name)
        if class_name == "splicing_connectors":
            return [base, "splicing connector", "connector block", "electrical connector"]
        if class_name == "pushpins":
            return [base, "pushpin compartment", "black compartment", "plastic compartment"]
        return [base] if base else []

    @staticmethod
    def _fallback_center_crop(image_np, ratio=0.9):
        h, w = image_np.shape[:2]
        nw, nh = int(w * ratio), int(h * ratio)
        x1 = (w - nw) // 2
        y1 = (h - nh) // 2
        return image_np[y1:y1 + nh, x1:x1 + nw]

    def process(self, image_path, class_name, output_path):
        prompt = CONFIG["lang_sam_prompts"].get(class_name)
        if prompt is None:
            return False, f"skip (paper: Lang-SAM not used for {class_name})"
        if not self.detector_ready:
            return False, "detector unavailable"

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        boxes, scores = None, None
        detect_meta = None

        # Retry with multiple prompts + threshold schedule to avoid all-0 detections.
        for p in self._candidate_prompts(class_name):
            if not p:
                continue
            for th in CONFIG["threshold_schedule"]:
                b, s = self._detect(image_pil, p, box_threshold=th, text_threshold=min(th, CONFIG["text_threshold"]))
                if b is not None and len(b) > 0:
                    boxes, scores = b, s
                    detect_meta = f"prompt='{p}', th={th}"
                    break
            if boxes is not None and len(boxes) > 0:
                break

        if boxes is None or len(boxes) == 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Last-resort fallback: keep object-centered crop instead of full original.
            fallback = self._fallback_center_crop(image_np, ratio=0.9)
            Image.fromarray(fallback.astype(np.uint8)).save(output_path)
            return True, "0 objects -> fallback center crop saved"

        if len(boxes) > CONFIG["max_detections"]:
            top_ids = np.argsort(scores)[::-1][: CONFIG["max_detections"]]
            boxes = boxes[top_ids]

        out_dir = os.path.dirname(output_path)
        base = os.path.splitext(os.path.basename(output_path))[0]
        os.makedirs(out_dir, exist_ok=True)

        masks = self._segment_masks(image_np, boxes) if self.sam_ready else None
        h, w = image_np.shape[:2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(w, x2 + 10), min(h, y2 + 10)

            if masks is not None:
                mask = masks[i].astype(bool)
                masked = image_np.copy()
                masked[~mask] = 0
                crop = masked[y1:y2, x1:x2]
            else:
                crop = image_np[y1:y2, x1:x2]

            save_path = os.path.join(out_dir, f"{base}_{i+1}.png")
            Image.fromarray(crop.astype(np.uint8)).save(save_path)

        return True, f"{len(boxes)} objects saved ({detect_meta})"


def get_output_path(method, class_name, subset, filename):
    return str(Path(CONFIG["output_root"]) / method / class_name / subset / filename)


def process_class(processor, method_name, class_name):
    subsets = ["train/good", "test/good", "test/logical_anomalies", "test/structural_anomalies"]
    base = Path(CONFIG["mvtec_root"]) / class_name

    total, success = 0, 0
    start = time.time()
    for subset in subsets:
        folder = base / subset
        if not folder.exists():
            continue
        imgs = iter_images(folder)
        if not imgs:
            continue
        print(f"\n  [{subset}] {len(imgs)} images")
        for i, img in enumerate(imgs, 1):
            out = get_output_path(method_name, class_name, subset, img.name)
            ok, msg = processor.process(str(img), class_name, out)
            total += 1
            if ok:
                success += 1
            print(f"    [{i}/{len(imgs)}] {img.name} -> {msg}")
    print(f"\n  done: {success}/{total} ({time.time()-start:.1f}s)")
    return success, total


def main():
    parser = argparse.ArgumentParser(description="LogicQA preprocessing.")
    parser.add_argument("--method", choices=["bpm", "grounded_sam"], default="bpm")
    parser.add_argument("--class", dest="class_name", default="all")
    parser.add_argument("--input-root", dest="input_root", default=None,
                        help="Override input dataset root")
    parser.add_argument("--output-root", dest="output_root", default=None,
                        help="Override output root")
    args = parser.parse_args()

    if args.input_root:
        CONFIG["mvtec_root"] = args.input_root
    if args.output_root:
        CONFIG["output_root"] = args.output_root

    all_classes = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
    class_list = all_classes if args.class_name == "all" else [args.class_name]

    print("=" * 70)
    print("Preprocessing")
    print("=" * 70)
    print(f"method: {args.method}")
    print(f"classes: {', '.join(class_list)}")
    print(f"input_root: {CONFIG['mvtec_root']}")
    print(f"output_root: {CONFIG['output_root']}")

    processor = BPMProcessor() if args.method == "bpm" else GroundedSAMProcessor()
    for cls in class_list:
        print("\n" + "-" * 50)
        print(f"class: {cls}")
        print("-" * 50)
        process_class(processor, args.method, cls)

    print("\nDone.")


if __name__ == "__main__":
    main()
