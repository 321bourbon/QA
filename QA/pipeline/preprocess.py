import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


CONFIG = {
    "mvtec_root": "/home/liuxinyi/datasets/MVtec_LOCO_AD/mvtec_loco_anomaly_detection",
    "output_root": "/home/liuxinyi/datasets/MVtec_preprocessed",
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
    # BPM ViT 参数（论文使用 DINO ViT-S/16，Figure 3 & Appendix F）
    "bpm_vit_model":      "dino_vits16",
    "bpm_input_size":     480,
    "bpm_patch_size":     16,
    "bpm_attn_threshold": 60,   # 百分位，越高前景越保守
    # Paper Appendix B.3:
    "lang_sam_prompts": {
        "splicing_connectors": "Connector Block",
        "pushpins": "The individual black compartments within the transparent plastic storage box",
    },
    "sam_version": "sam3",   # "sam2" | "sam3"，手动切换
}


def iter_images(folder):
    return sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))


class BPMProcessor:
    """
    Back Patch Masking (BPM) — 论文 Figure 3 & Appendix F 实现。
    使用预训练 DINO ViT 的 CLS token 注意力图，将低注意力区域（背景）置零，
    保留前景物体像素。
    步骤：
      1. 将原图 resize 到 ViT 输入尺寸 (bpm_input_size)
      2. 用 DINO ViT 获取所有 head 的 CLS 注意力图
      3. 对多 head 取平均 → [H_patch, W_patch]
      4. Resize 注意力图到原图尺寸
      5. 二值化：高于第 bpm_attn_threshold 百分位的像素为前景
      6. 用掩码置零背景像素，保存结果
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._transform = None
        self._load_vit()

    def _load_vit(self):
        vit_model = CONFIG["bpm_vit_model"]
        input_size = CONFIG["bpm_input_size"]
        try:
            print(f"[BPM] loading DINO ViT: {vit_model}")
            self.model = torch.hub.load(
                "/home/liuxinyi/model/dino",   # 本地仓库，不联网
                vit_model,
                pretrained=False,              # 不自动下载权重
                source="local",
            )
            # 手动加载本地权重
            ckpt = torch.load(
                "/home/liuxinyi/model/dino/dino_deitsmall16_pretrain.pth",
                map_location="cpu",
            )
            self.model.load_state_dict(ckpt, strict=True)
            self.model.eval().to(self.device)
            self._transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            print(f"[BPM] ViT ready on {self.device}")
        except Exception as e:
            print(f"[BPM] ViT load failed: {e}  -> will use fallback center crop")
            self.model = None

    def _get_attention_mask(self, pil_image) -> np.ndarray:
        w_orig, h_orig = pil_image.size
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            attn = self.model.get_last_selfattention(tensor)

        attn_cls = attn[0, :, 0, 1:]

        attn_map = attn_cls.max(dim=0).values

        patch_size = CONFIG["bpm_patch_size"]
        input_size = CONFIG["bpm_input_size"]
        h_p = w_p = input_size // patch_size

        attn_map = attn_map.reshape(h_p, w_p).cpu().numpy()

        attn_full = cv2.resize(
            attn_map,
            (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR
        )

        # normalize 到 0~1
        attn_full = (attn_full - attn_full.min()) / (attn_full.max() - attn_full.min() + 1e-6)

        return attn_full

    @staticmethod
    def _attention_bbox(mask: np.ndarray, padding: float = 0.05):
        """
        根据 float32 attention mask 计算前景 bounding box。
        padding: 相对于图像尺寸的额外边距比例（防止裁太紧）。
        返回 (x1, y1, x2, y2)，若 mask 全零或 bbox 过小则返回 None。
        """
        h, w = mask.shape
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0 or len(ys) == 0:
            return None

        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0,     int(xs.min()) - pad_x)
        y1 = max(0,     int(ys.min()) - pad_y)
        x2 = min(w - 1, int(xs.max()) + pad_x)
        y2 = min(h - 1, int(ys.max()) + pad_y)

        # 防止裁切后面积过小（宽或高 < 原图 10%）
        if (x2 - x1) < w * 0.1 or (y2 - y1) < h * 0.1:
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _fallback_center_crop(image_np, ratio=0.9):
        h, w = image_np.shape[:2]
        nw, nh = int(w * ratio), int(h * ratio)
        x1 = (w - nw) // 2
        y1 = (h - nh) // 2
        return image_np[y1:y1 + nh, x1:x1 + nw]

    def process(self, image_path, class_name, output_path):
        if class_name not in CONFIG["bpm_classes"]:
            return False, f"skip (paper: BPM not used for {class_name})"

        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img, dtype=np.uint8)
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        if self.model is None:
            # ViT 未加载时使用 center crop 作为 fallback
            fallback = self._fallback_center_crop(img_np)
            Image.fromarray(fallback).save(output_path)
            return True, "fallback center crop (ViT unavailable)"

        try:
            mask = self._get_attention_mask(pil_img)
            threshold = np.percentile(mask, CONFIG["bpm_attn_threshold"])
            binary_mask = (mask > threshold).astype(np.uint8)
            kernel = np.ones((15, 15), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_DILATE, kernel)
            masked_img_np = img_np * binary_mask[:, :, np.newaxis]
            Image.fromarray(masked_img_np.astype(np.uint8)).save(output_path)
            return True, "bpm hard mask applied"

        except Exception as e:
            # ViT 推理失败时降级到 center crop（无遮蔽）
            fallback = self._fallback_center_crop(img_np)
            Image.fromarray(fallback).save(output_path)
            return True, f"fallback center crop ({e})"


class GroundedSAMProcessor:
    """GroundingDINO + SAM-based object extraction for Lang-SAM-like preprocessing."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_ready = False
        self.sam_ready = False
        self._load_detector()
        self._load_sam()

    def _load_sam(self):
        self.sam_version = None
        sam_ver = CONFIG.get("sam_version", "sam2")
        if sam_ver == "sam2":
            self._load_sam2()
        else:
            self._load_sam3()

    def _load_sam2(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_cfg  = os.environ.get("SAM2_CONFIG",     "/home/liuxinyi/model/sam/sam2_hiera_large.yaml")
            sam2_ckpt = os.environ.get("SAM2_CHECKPOINT", "/home/liuxinyi/model/sam/sam2_hiera_large.pt")

            sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            self.sam_ready = True
            self.sam_version = "sam2"
            print(f"[Grounded-SAM] SAM2 loaded: {sam2_ckpt}")
        except Exception as e:
            print(f"[Grounded-SAM] SAM2 load failed: {e}")

    def _load_sam3(self):
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            ckpt_path = "/home/liuxinyi/model/sam/sam3.pt"
            model = build_sam3_image_model(checkpoint=ckpt_path)
            self.sam_predictor = Sam3Processor(model)
            self.sam_ready = True
            self.sam_version = "sam3"
            print(f"[Grounded-SAM] SAM3 loaded: {ckpt_path}")
        except Exception as e:
            print(f"[Grounded-SAM] SAM3 load failed: {e}")

    @staticmethod
    def _find_sam_checkpoint():
        ckpt_path = "/home/liuxinyi/model/sam/sam3.pt"
        if os.path.exists(ckpt_path):
            return ckpt_path, "vit_h"
        print(f"[错误] 找不到 SAM 权重文件: {ckpt_path}")
        return None, None

    def _load_detector(self):
        try:
            from groundingdino.util.inference import Model

            config_path = "/home/liuxinyi/model/sam/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
            weight_path = "/home/liuxinyi/model/sam/GroundingDINO/groundingdino_swinb_cogcoor.pth"

            self.dino_model = Model(model_config_path=config_path, model_checkpoint_path=weight_path, device=self.device)
            self.detector_ready = True
            print("[Grounded-SAM] Native GroundingDINO detector loaded")
        except Exception as e:
            print(f"[Grounded-SAM] Native detector load failed: {e}")
            self.detector_ready = False

    def _detect(self, image_pil, prompt, box_threshold=None, text_threshold=None):
        if not self.detector_ready:
            return None, None

        box_th = box_threshold if box_threshold is not None else CONFIG["box_threshold"]
        text_th = text_threshold if text_threshold is not None else CONFIG["text_threshold"]

        image_np = np.array(image_pil)

        boxes, logits, phrases = self.dino_model.predict_with_caption(
            image=image_np,
            caption=prompt,
            box_threshold=box_th,
            text_threshold=text_th
        )

        if len(boxes) == 0:
            return None, None

        h, w = image_np.shape[:2]
        boxes = boxes.numpy()
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2

        return boxes_xyxy, logits.numpy()

    def _segment_masks(self, image_np, boxes, prompt):
        if self.sam_version == "sam2":
            self.sam_predictor.set_image(image_np)
            all_masks = []
            for box in boxes:
                x1, y1, x2, y2 = map(float, box)
                masks_i, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False,
                )
                all_masks.append(masks_i[0].astype(np.uint8))
            return np.array(all_masks)

        if self.sam_version == "sam3":
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image_np)
            all_masks = []
            h, w = image_np.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop_pil = pil_image.crop((x1, y1, x2, y2))
                state = self.sam_predictor.set_image(crop_pil)
                out = self.sam_predictor.set_text_prompt(state=state, prompt=prompt)
                full_mask = np.zeros((h, w), dtype=np.uint8)
                if out["masks"] is not None and len(out["masks"]) > 0:
                    crop_mask = out["masks"][0].astype(np.uint8)
                    ch, cw = crop_mask.shape[:2]
                    full_mask[y1:y1+ch, x1:x1+cw] = crop_mask
                else:
                    full_mask[y1:y2, x1:x2] = 1
                all_masks.append(full_mask)
            return np.array(all_masks)

        return None

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

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        viz_dir = os.path.join(out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(output_path))[0]

        def save_viz(boxes_list):
            viz = image_np.copy()
            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            viz_path = os.path.join(viz_dir, f"{base}_boxes.png")
            Image.fromarray(viz.astype(np.uint8)).save(viz_path)

        if self.sam_version == "sam3":
            if not self.sam_ready:
                return False, "SAM3 unavailable"

            boxes, masks, scores, detect_meta = None, None, None, None
            for p in self._candidate_prompts(class_name):
                if not p:
                    continue
                try:
                    state = self.sam_predictor.set_image(image_pil)
                    output = self.sam_predictor.set_text_prompt(state=state, prompt=p)
                    if output["masks"] is not None and len(output["masks"]) > 0:
                        masks  = output["masks"]
                        boxes  = output["boxes"]
                        scores = output["scores"]
                        detect_meta = f"prompt='{p}'"
                        break
                except Exception as e:
                    print(f"    [SAM3] prompt='{p}' failed: {e}")

            if masks is None or len(masks) == 0:
                fallback = self._fallback_center_crop(image_np, ratio=0.9)
                Image.fromarray(fallback.astype(np.uint8)).save(output_path)
                return True, "0 objects -> fallback center crop saved"

            if len(boxes) > CONFIG["max_detections"]:
                top_ids = np.argsort(scores)[::-1][: CONFIG["max_detections"]]
                masks = masks[top_ids]
                boxes = boxes[top_ids]

            save_viz(boxes)

            h, w = image_np.shape[:2]
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
                x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
                crop = image_np[y1:y2, x1:x2]
                Image.fromarray(crop.astype(np.uint8)).save(
                    os.path.join(out_dir, f"{base}_{i+1}.png"))
            return True, f"{len(masks)} objects saved (bbox crop + viz, {detect_meta})"

        else:
            if not self.detector_ready:
                return False, "detector unavailable"
            if not self.sam_ready:
                return False, "SAM2 unavailable"

            boxes, scores, detect_meta = None, None, None
            for p in self._candidate_prompts(class_name):
                if not p:
                    continue
                for th in CONFIG["threshold_schedule"]:
                    b, s = self._detect(image_pil, p, box_threshold=th,
                                        text_threshold=min(th, CONFIG["text_threshold"]))
                    if b is not None and len(b) > 0:
                        boxes, scores, detect_meta = b, s, f"prompt='{p}', th={th}"
                        break
                if boxes is not None:
                    break

            if boxes is None or len(boxes) == 0:
                fallback = self._fallback_center_crop(image_np, ratio=0.9)
                Image.fromarray(fallback.astype(np.uint8)).save(output_path)
                return True, "0 objects -> fallback center crop saved"

            if len(boxes) > CONFIG["max_detections"]:
                top_ids = np.argsort(scores)[::-1][: CONFIG["max_detections"]]
                boxes = boxes[top_ids]

            save_viz(boxes)

            h, w = image_np.shape[:2]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
                x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
                crop = image_np[y1:y2, x1:x2]
                Image.fromarray(crop.astype(np.uint8)).save(
                    os.path.join(out_dir, f"{base}_{i+1}.png"))
            return True, f"{len(boxes)} objects saved (bbox crop + viz, {detect_meta})"


def get_output_path(method, class_name, subset, filename):
    return str(Path(CONFIG["output_root"]) / method / class_name / subset / filename)


def process_class(processor, method_name, class_name):
    subsets = ["train/good", "test/good", "test/logical_anomalies"]
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
