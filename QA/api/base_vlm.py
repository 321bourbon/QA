"""
api/base_vlm.py — Abstract base class for all VLM API wrappers.

Shared across GeminiVLM, GPTVLM, InternVLM:
  - APIQuotaExceededError
  - common __init__ (mode, mock_mode, log setup)
  - _log / _is_quota_error / _stable_seed / _get_image_media_type
  - mock responses: _mock_yes_no, _mock_response_text
"""

import hashlib
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime

from utils.logger import get_runtime_logger, runtime_logger_active


class APIQuotaExceededError(RuntimeError):
    """Raised when the API quota / rate limit is exhausted."""


class BaseVLM(ABC):
    """Abstract base for Gemini, GPT-4o, and InternVL wrappers."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.mode      = str(self.cfg.get("mode", "api")).lower()
        self.mock_mode = self.mode in {"mock", "model", "fake", "debug"}

        self.model       = self.cfg.get("model", "")
        self.temperature = self.cfg.get("temperature", 1.0)
        self.max_tokens  = self.cfg.get("max_new_tokens")

        # ── log path ────────────────────────────────────────────────────────
        shared_log = os.environ.get("LOGICQA_LOG_PATH", "").strip()
        if shared_log:
            self.log_path = shared_log
            self.log_dir  = os.path.dirname(shared_log) or "./runtime_logs"
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir  = self.cfg.get("log_dir", "./runtime_logs")
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d%H%M")
            name = type(self).__name__
            self.log_path = os.path.join(self.log_dir, f"{name}+{ts}.log")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, message):
        if runtime_logger_active():
            lg = get_runtime_logger()
            if lg is not None:
                lg.info(message)
                return
        print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_image_media_type(image_path):
        ext = os.path.splitext(image_path)[1].lower()
        return {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    @staticmethod
    def _is_quota_error(exc):
        msg = str(exc).lower()
        keys = [
            "resource_exhausted", "quota", "rate limit", "429", "billing",
            "insufficient_quota", "exceeded your current quota",
        ]
        return any(k in msg for k in keys)

    @staticmethod
    def _stable_seed(prompt, image_path):
        key = f"{prompt}|{os.path.basename(image_path) if image_path else 'no_image'}"
        return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    # ── Mock helpers ──────────────────────────────────────────────────────────

    def _mock_yes_no(self, prompt, image_path):
        """Stable (deterministic) Yes/No + logprob for mock mode."""
        seed   = self._stable_seed(prompt, image_path)
        answer = "Yes" if (seed % 3) != 0 else "No"
        logprob = (
            -0.0001 - (seed % 9)  * 0.0001 if answer == "Yes"
            else -0.0005 - (seed % 11) * 0.0002
        )
        return answer, logprob

    def _mock_response_text(self, prompt, image_path):
        """Generate a plausible mock response for different prompt types."""
        low = prompt.lower()

        if "generate five variations" in low and "input" in low:
            m     = re.search(r"input\s*:\s*(.+)", prompt, flags=re.IGNORECASE | re.DOTALL)
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

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def ask(self, image_path, prompt):
        """Send prompt (with optional image); return (text, None)."""

    @abstractmethod
    def ask1(self, image_path, prompt):
        """Send yes/no prompt; return (text, logprob_or_None)."""
