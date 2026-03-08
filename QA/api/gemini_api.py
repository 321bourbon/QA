import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.logger import get_runtime_logger, runtime_logger_active

try:
    from google import genai
    from google.genai import types
except Exception as _import_err:
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = _import_err
else:
    _GENAI_IMPORT_ERROR = None


GEMINI_API_KEY = ""


class APIQuotaExceededError(RuntimeError):
    pass


class VLM:
    """Gemini client for LogicQA, with mock/model mode for debugging."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.mode = str(self.cfg.get("mode", "api")).lower()
        self.mock_mode = self.mode in {"mock", "model", "fake", "debug"}

        if not self.mock_mode:
            if genai is None or types is None:
                raise ImportError(
                    "Gemini SDK not available in current Python env. "
                    "Install with: pip install google-genai. "
                    f"Original import error: {_GENAI_IMPORT_ERROR}"
                )
            api_key = self.cfg.get("api_key") or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
            if not api_key:
                raise ValueError("Gemini API key missing. Set cfg['api_key'] or GEMINI_API_KEY env var.")
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

        self.model = self.cfg.get("model", "mock-gemini-2.5-flash" if self.mock_mode else "gemini-2.5-flash")
        self.temperature = self.cfg.get("temperature", 1.0)
        self.top_p = self.cfg.get("top_p")
        self.max_tokens = self.cfg.get("max_new_tokens")

        shared_log = os.environ.get("LOGICQA_LOG_PATH", "").strip()
        if shared_log:
            self.log_path = shared_log
            self.log_dir = os.path.dirname(shared_log) or "./runtime_logs"
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = self.cfg.get("log_dir", "./runtime_logs")
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(self.log_dir, f"GEMINI+{ts}+results.log")

        self._log("=" * 78)
        self._log("[Gemini] initialized")
        self._log(f"  mode: {'mock(model)' if self.mock_mode else 'api'}")
        self._log(f"  model: {self.model}")
        self._log(f"  temperature: {self.temperature}")
        self._log(f"  top_p: {self.top_p if self.top_p is not None else 'default'}")
        self._log(f"  max_tokens: {self.max_tokens if self.max_tokens is not None else 'default'}")
        self._log(f"  log_file: {self.log_path}")
        self._log("=" * 78)

    def _log(self, message):
        if runtime_logger_active():
            lg = get_runtime_logger()
            if lg is not None:
                lg.info(message)
                return
        print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    @staticmethod
    def _get_image_media_type(image_path):
        ext = os.path.splitext(image_path)[1].lower()
        mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mapping.get(ext, "image/jpeg")

    def _build_config(self, with_logprobs=False):
        kwargs = {"temperature": self.temperature}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        if with_logprobs:
            kwargs["response_logprobs"] = True
            kwargs["logprobs"] = 5
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _is_quota_error(exc):
        msg = str(exc).lower()
        keys = ["resource_exhausted", "quota", "rate limit", "429", "billing"]
        return any(k in msg for k in keys)

    @staticmethod
    def _stable_seed(prompt, image_path):
        key = f"{prompt}|{os.path.basename(image_path) if image_path else 'no_image'}"
        return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    def _mock_yes_no(self, prompt, image_path):
        seed = self._stable_seed(prompt, image_path)
        answer = "Yes" if (seed % 3) != 0 else "No"
        if answer == "Yes":
            logprob = -0.0001 - ((seed % 9) * 0.0001)
        else:
            logprob = -0.0005 - ((seed % 11) * 0.0002)
        return answer, logprob

    def _mock_response_text(self, prompt, image_path):
        low = prompt.lower()
        if "generate five variations" in low and "input" in low:
            m = re.search(r"input\s*:\s*(.+)", prompt, flags=re.IGNORECASE | re.DOTALL)
            base_q = (m.group(1).strip() if m else "Is the object normal?").strip()
            if not base_q.endswith("?"):
                base_q += "?"
            return (
                f"Output1: {base_q}\n"
                f"Output2: Could you confirm: {base_q}\n"
                f"Output3: Please verify whether {base_q[0].lower() + base_q[1:]}\n"
                f"Output4: Is it correct that {base_q[0].lower() + base_q[1:]}\n"
                f"Output5: Based on the image, {base_q[0].lower() + base_q[1:]}"
            )

        if "analyze the image and describe" in low:
            return (
                "Type: Industrial object under inspection.\n"
                "Color: Mixed metallic/plastic tones.\n"
                "Size: Compact object occupying center area.\n"
                "Material: Plastic and metal-like parts.\n"
                "Composition: Core object with repeated components.\n"
                "Quantity: Appears consistent with a normal sample.\n"
                "Relative location: Main object is centered with stable arrangement."
            )

        if "combine the three descriptions" in low:
            return (
                "Common features: centered target object, stable component count, "
                "consistent arrangement, and no obvious misplaced parts."
            )

        if "create several" in low and "normal or abnormal" in low:
            return (
                "Q1: Is the main object count visually correct?\n"
                "Q2: Are key components arranged in expected positions?\n"
                "Q3: Is there any missing required component?\n"
                "Q4: Is there any extra unexpected component?\n"
                "Q5: Does the global layout look consistent with normal samples?"
            )

        if "your response must end with" in low and "result" in low:
            ans, _ = self._mock_yes_no(prompt, image_path)
            return (
                "Mock reasoning: inspected overall structure and component consistency.\n"
                f"- Result: {ans}"
            )

        ans, _ = self._mock_yes_no(prompt, image_path)
        return f"Mock response generated for debugging.\n- Result: {ans}"

    def _build_contents(self, image_path, prompt):
        parts = []
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            parts.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=self._get_image_media_type(image_path),
                )
            )
        parts.append(prompt)
        return parts

    def ask(self, image_path, prompt):
        try:
            prompt_preview = " ".join(prompt.strip().split())[:120]
            self._log(
                f"  [ask] image={'yes' if image_path and os.path.exists(image_path) else 'no'} "
                f"prompt_len={len(prompt)} prompt='{prompt_preview}'"
            )

            if self.mock_mode:
                text = self._mock_response_text(prompt, image_path)
                self._log("  [ask] response.model: mock-gemini-2.5-flash")
                return text.strip(), None

            response = self.client.models.generate_content(
                model=self.model,
                contents=self._build_contents(image_path, prompt),
                config=self._build_config(with_logprobs=False),
            )
            text = (getattr(response, "text", "") or "").strip() or self._safe_extract_text(response)
            self._log(f"  [ask] response.model: {self.model}")
            return text, None
        except Exception as e:
            self._log(f"  [ask] error: {e}")
            if self._is_quota_error(e):
                raise APIQuotaExceededError(str(e))
            return f"API error: {e}", None

    def ask1(self, image_path, prompt):
        try:
            if not self.mock_mode and image_path and not os.path.exists(image_path):
                return f"Image not found: {image_path}", None

            if self.mock_mode:
                answer, answer_logprob = self._mock_yes_no(prompt, image_path)
                text = (
                    "Mock reasoning: simulated visual inspection for debugging only.\n"
                    f"- Result: {answer}"
                )
                return text, answer_logprob

            response = self.client.models.generate_content(
                model=self.model,
                contents=self._build_contents(image_path, prompt),
                config=self._build_config(with_logprobs=True),
            )
            text = (getattr(response, "text", "") or "").strip() or self._safe_extract_text(response)
            answer_logprob = self._extract_answer_logprob(response, text)
            return text, answer_logprob
        except Exception as e:
            self._log(f"  [qa] error: {e}")
            if self._is_quota_error(e):
                raise APIQuotaExceededError(str(e))
            return f"API error: {e}", None

    @staticmethod
    def _safe_extract_text(response):
        try:
            cands = getattr(response, "candidates", None) or []
            if not cands:
                return ""
            cand0 = cands[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) or []
            chunks = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            return "\n".join(chunks).strip()
        except Exception:
            return ""

    @staticmethod
    def _find_answer_in_response(response_text):
        low = response_text.lower()
        if re.search(r"result\s*:\s*yes", low):
            return "Yes"
        if re.search(r"result\s*:\s*no", low):
            return "No"
        return "Unknown"

    def _extract_answer_logprob(self, response, response_text):
        ans = None
        low = (response_text or "").lower()
        if "result: yes" in low:
            ans = "yes"
        elif "result: no" in low:
            ans = "no"
        if ans is None:
            return None

        try:
            cands = getattr(response, "candidates", None) or []
            if not cands:
                return None

            cand = cands[0]
            lpr = getattr(cand, "logprobs_result", None)
            if not lpr:
                return None

            chosen = getattr(lpr, "chosen_candidates", None) or []
            for tok in reversed(chosen):
                token = (getattr(tok, "token", "") or "").lower().strip()
                lp = getattr(tok, "log_probability", None)
                if lp is None:
                    lp = getattr(tok, "logprob", None)
                if token in {"yes", "yes.", "no", "no."} and token.startswith(ans):
                    return lp

            top_steps = getattr(lpr, "top_candidates", None) or []
            for step in reversed(top_steps):
                candidates = getattr(step, "candidates", None) or []
                for tok in candidates:
                    token = (getattr(tok, "token", "") or "").lower().strip()
                    lp = getattr(tok, "log_probability", None)
                    if lp is None:
                        lp = getattr(tok, "logprob", None)
                    if token in {"yes", "yes.", "no", "no."} and token.startswith(ans):
                        return lp
            return None
        except Exception as e:
            self._log(f"    [qa-logprob] extract failed: {e}")
            return None


def _run_train_with_gemini(env):
    code = "import gemini_api, train; train.VLM = gemini_api.VLM; train.main()"
    return subprocess.call([sys.executable, "-c", code], env=env)


def _run_test_with_gemini(env):
    code = (
        "import gemini_api, test; "
        "test.VLM = gemini_api.VLM; "
        "test.APIQuotaExceededError = gemini_api.APIQuotaExceededError; "
        "test.main()"
    )
    return subprocess.call([sys.executable, "-c", code], env=env)


if __name__ == "__main__":
    import argparse
    from logger import setup_runtime_logger, shutdown_runtime_logger

    parser = argparse.ArgumentParser(description="Gemini API wrapper with mock(full-pipeline) mode.")
    parser.add_argument("--demo", action="store_true", help="Run quick API/mock demo only.")
    parser.add_argument("--root", default=os.environ.get("LOGICQA_MVTEC_ROOT", ""), help="Dataset root for train/test.")
    parser.add_argument(
        "--classes",
        default=os.environ.get("LOGICQA_CLASSES", "breakfast_box,juice_bottle,pushpins,screw_bag,splicing_connectors"),
        help="Comma-separated class list.",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("LOGICQA_MODE", "model"),
        choices=["model", "mock", "api"],
        help="model/mock for debug, api for real Gemini calls.",
    )
    args = parser.parse_args()

    setup_runtime_logger(log_dir="./runtime_logs", file_prefix="GEMINI", redirect_stdout=True)
    try:
        if args.demo:
            is_mock = args.mode in {"model", "mock"}
            demo_cfg = {
                "mode": "model" if is_mock else "api",
                "model": "mock-gemini-2.5-flash" if is_mock else "gemini-2.5-flash",
                "temperature": 1.0,
            }
            vlm = VLM(demo_cfg)

            print("\n[demo] ask() - generate sub-questions")
            demo_subq_prompt = (
                "Generate five variations of the following question while keeping the semantic meaning.\n"
                "Input: Is there exactly one cable connecting the two splicing connectors?"
            )
            text, _ = vlm.ask("", demo_subq_prompt)
            print(text)

            print("\n[demo] ask1() - test question with simulated yes/no + logprob")
            demo_test_prompt = (
                "Question: Is there exactly one cable connecting the two splicing connectors?\n"
                "At first, describe splicing_connectors image and then answer the question.\n"
                "Your response must end with '- Result: Yes' or '- Result: No'.\n"
                "Let's think step by step."
            )
            text1, lp = vlm.ask1("", demo_test_prompt)
            print(text1)
            print(f"[demo] extracted logprob: {lp}")
        else:
            is_mock = args.mode in {"model", "mock"}
            print("=" * 90)
            print("Mock Full Pipeline (Paper flow except preprocessing)" if is_mock else "Full Pipeline (Paper flow except preprocessing)")
            print("=" * 90)
            print("Mode: model/mock (no real API call)" if is_mock else "Mode: api (real Gemini API call)")
            class_list = [x.strip() for x in args.classes.split(",") if x.strip()]
            print(f"Classes: {','.join(class_list)}")

            env = os.environ.copy()
            env["LOGICQA_MODE"] = "model" if is_mock else "api"
            env["LOGICQA_MODEL"] = "mock-gemini-2.5-flash" if is_mock else "gemini-2.5-flash"

            if args.root:
                class_root_map = {c: args.root for c in class_list}
            else:
                class_root_map = {
                    "breakfast_box": r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection",
                    "juice_bottle": r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection",
                    "pushpins": r"E:\dataSet\MVtec LOCO AD\preprocessed\bpm",
                    "screw_bag": r"E:\dataSet\MVtec LOCO AD\preprocessed\bpm",
                    "splicing_connectors": r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection",
                }

            overall_ok = True
            per_class_metrics = []
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, class_name in enumerate(class_list, 1):
                root = class_root_map.get(class_name, args.root or "")
                if not root:
                    print(f"\n[{i}/{len(class_list)}] {class_name}: no root configured, skip")
                    overall_ok = False
                    continue

                env_one = env.copy()
                env_one["LOGICQA_CLASSES"] = class_name
                env_one["LOGICQA_MVTEC_ROOT"] = root
                class_results_dir = str(Path("./test_results") / f"full_run_{run_stamp}" / class_name)
                env_one["LOGICQA_RESULTS_DIR"] = class_results_dir

                print("\n" + "#" * 90)
                print(f"[{i}/{len(class_list)}] class={class_name}")
                print(f"root={root}")
                print(f"results_dir={class_results_dir}")
                print("#" * 90)

                print("[1/2] Running train.py ...")
                rc_train = _run_train_with_gemini(env_one)
                print(f"train.py exit_code={rc_train}")
                if rc_train != 0:
                    overall_ok = False
                    continue

                print("[2/2] Running test.py ...")
                rc_test = _run_test_with_gemini(env_one)
                print(f"test.py exit_code={rc_test}")
                if rc_test != 0:
                    overall_ok = False
                    continue

                summary_path = Path(class_results_dir) / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        avg = summary.get("average", {})
                        auroc = float(avg.get("auroc", 0.0))
                        f1_max = float(avg.get("f1_max", 0.0))
                        per_class_metrics.append(
                            {
                                "class_name": class_name,
                                "root": root,
                                "results_dir": class_results_dir,
                                "auroc": auroc,
                                "f1_max": f1_max,
                            }
                        )
                        print(
                            f"[class-metrics] {class_name}: "
                            f"AUROC={auroc:.4f} F1-max={f1_max:.4f}"
                        )
                    except Exception as e:
                        overall_ok = False
                        print(f"[class-metrics] parse failed for {class_name}: {e}")
                else:
                    overall_ok = False
                    print(f"[class-metrics] summary missing: {summary_path}")

            print("\nFull pipeline finished.")
            if per_class_metrics:
                overall_auroc = sum(x["auroc"] for x in per_class_metrics) / len(per_class_metrics)
                overall_f1 = sum(x["f1_max"] for x in per_class_metrics) / len(per_class_metrics)
                print("\n" + "=" * 90)
                print("Final Metrics (Per-Class + Overall)")
                print("=" * 90)
                for row in per_class_metrics:
                    print(f"{row['class_name']:<22} AUROC={row['auroc']:.4f} F1-max={row['f1_max']:.4f}")
                print("-" * 90)
                print(f"{'OVERALL-AVG':<22} AUROC={overall_auroc:.4f} F1-max={overall_f1:.4f}")

                final_summary = {
                    "mode": env["LOGICQA_MODE"],
                    "timestamp": run_stamp,
                    "class_metrics": per_class_metrics,
                    "overall_average": {"auroc": overall_auroc, "f1_max": overall_f1},
                }
                final_summary_path = Path("./test_results") / f"full_run_{run_stamp}" / "final_summary.json"
                final_summary_path.parent.mkdir(parents=True, exist_ok=True)
                final_summary_path.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"final summary saved: {final_summary_path}")

            if not overall_ok:
                raise SystemExit(1)
    finally:
        shutdown_runtime_logger()
