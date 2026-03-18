"""
api/gemini_api.py — Google Gemini VLM wrapper (inherits BaseVLM).

Gemini-specific: google-genai SDK, top_p/top_k, response_logprobs extraction.
APIQuotaExceededError is re-exported here for backward compatibility.
"""

import os

from api.base_vlm import APIQuotaExceededError, BaseVLM  # noqa: F401  (re-export)

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


class VLM(BaseVLM):
    """Gemini client for LogicQA, with mock mode for offline debugging."""

    def __init__(self, cfg=None):
        super().__init__(cfg)

        # Gemini-specific generation params (Appendix B.1)
        self.top_p = self.cfg.get("top_p", 0.95)   # Appendix B.1 = 0.95
        self.top_k = self.cfg.get("top_k", 40)     # Appendix B.1 = 40

        if not self.mock_mode:
            if genai is None or types is None:
                raise ImportError(
                    "Gemini SDK not available. Install: pip install google-genai. "
                    f"Original error: {_GENAI_IMPORT_ERROR}"
                )
            api_key = self.cfg.get("api_key") or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
            if not api_key:
                raise ValueError("Gemini API key missing. Set cfg['api_key'] or GEMINI_API_KEY env var.")
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

        if not self.model:
            self.model = "mock-gemini-2.5-flash" if self.mock_mode else "gemini-2.5-flash"

        self._log("=" * 78)
        self._log("[Gemini] 初始化完成 (initialized)")
        self._log(f"  模式 (mode)        : {'mock 离线' if self.mock_mode else 'api 在线'}")
        self._log(f"  模型 (model)       : {self.model}")
        self._log(f"  温度 (temperature) : {self.temperature}")
        self._log(f"  top_p              : {self.top_p  if self.top_p  is not None else '默认 (default)'}")
        self._log(f"  top_k              : {self.top_k  if self.top_k  is not None else '默认 (default)'}")
        self._log(f"  最大token (max_tokens): {self.max_tokens if self.max_tokens is not None else '默认 (default)'}")
        self._log(f"  日志文件 (log_file): {self.log_path}")
        self._log("=" * 78)

    # ── Gemini-specific builders ──────────────────────────────────────────────

    def _build_config(self, with_logprobs=False):
        kwargs = {"temperature": self.temperature}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        if with_logprobs:
            kwargs["response_logprobs"] = True
            kwargs["logprobs"] = 5
        return types.GenerateContentConfig(**kwargs)

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

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, image_path, prompt):
        try:
            preview = " ".join(prompt.strip().split())[:120]
            self._log(
                f"  [ask] image={'yes' if image_path and os.path.exists(image_path) else 'no'} "
                f"prompt_len={len(prompt)} prompt='{preview}'"
            )
            if self.mock_mode:
                return self._mock_response_text(prompt, image_path).strip(), None

            response = self.client.models.generate_content(
                model=self.model,
                contents=self._build_contents(image_path, prompt),
                config=self._build_config(with_logprobs=False),
            )
            text = (getattr(response, "text", "") or "").strip() or self._safe_extract_text(response)
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
                answer, logprob = self._mock_yes_no(prompt, image_path)
                return f"Mock reasoning: simulated visual inspection.\n- Result: {answer}", logprob

            response = self.client.models.generate_content(
                model=self.model,
                contents=self._build_contents(image_path, prompt),
                config=self._build_config(with_logprobs=True),
            )
            text = (getattr(response, "text", "") or "").strip() or self._safe_extract_text(response)
            return text, self._extract_answer_logprob(response, text)
        except Exception as e:
            self._log(f"  [qa] error: {e}")
            if self._is_quota_error(e):
                raise APIQuotaExceededError(str(e))
            return f"API error: {e}", None

    # ── Gemini-specific helpers ───────────────────────────────────────────────

    @staticmethod
    def _safe_extract_text(response):
        try:
            cands = getattr(response, "candidates", None) or []
            if not cands:
                return ""
            chunks = []
            content = getattr(cands[0], "content", None)
            for p in (getattr(content, "parts", None) or []):
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            return "\n".join(chunks).strip()
        except Exception:
            return ""

    def _extract_answer_logprob(self, response, response_text):
        low = (response_text or "").lower()
        if "result: yes" in low:
            ans = "yes"
        elif "result: no" in low:
            ans = "no"
        else:
            return None
        try:
            cands = getattr(response, "candidates", None) or []
            if not cands:
                return None
            lpr = getattr(cands[0], "logprobs_result", None)
            if not lpr:
                return None
            for tok in reversed(getattr(lpr, "chosen_candidates", None) or []):
                token = (getattr(tok, "token", "") or "").lower().strip()
                lp    = getattr(tok, "log_probability", None) or getattr(tok, "logprob", None)
                if token in {"yes", "yes.", "no", "no."} and token.startswith(ans):
                    return lp
            for step in reversed(getattr(lpr, "top_candidates", None) or []):
                for tok in (getattr(step, "candidates", None) or []):
                    token = (getattr(tok, "token", "") or "").lower().strip()
                    lp    = getattr(tok, "log_probability", None) or getattr(tok, "logprob", None)
                    if token in {"yes", "yes.", "no", "no."} and token.startswith(ans):
                        return lp
            return None
        except Exception as e:
            self._log(f"    [qa-logprob] extract failed: {e}")
            return None


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.logger import setup_runtime_logger, shutdown_runtime_logger

    setup_runtime_logger(log_dir="./runtime_logs", file_prefix="Gemini+demo", redirect_stdout=True)
    try:
        cfg = {"mode": "model", "model": "mock-gemini-2.5-flash", "temperature": 1.0}
        vlm = VLM(cfg)

        print("\n[demo] ask() — sub-question generation")
        text, _ = vlm.ask("", "Generate five variations of: Is there exactly one cable?")
        print(text)

        print("\n[demo] ask1() — yes/no + logprob")
        text, lp = vlm.ask1("", "Is the image normal?\nYour response must end with '- Result: Yes' or '- Result: No'.")
        print(text)
        print(f"logprob: {lp}")
    finally:
        shutdown_runtime_logger()
