"""
api/gpt_api.py — OpenAI GPT VLM wrapper (inherits BaseVLM).

GPT-specific: openai SDK, base64 image encoding, top_logprobs extraction.
APIQuotaExceededError is re-exported here for backward compatibility.
"""

import base64
import os

from openai import OpenAI

from api.base_vlm import APIQuotaExceededError, BaseVLM  # noqa: F401  (re-export)


class VLM(BaseVLM):
    """GPT client for LogicQA, with mock mode for offline debugging."""

    def __init__(self, cfg=None):
        super().__init__(cfg)

        self.top_p             = self.cfg.get("top_p")
        self.frequency_penalty = self.cfg.get("frequency_penalty")

        if not self.mock_mode:
            api_key = self.cfg.get("api_key")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        if not self.model:
            self.model = "mock-gpt" if self.mock_mode else "gpt"

        self._log("=" * 78)
        self._log("[GPT] 初始化完成 (initialized)")
        self._log(f"  模式 (mode)              : {'mock 离线' if self.mock_mode else 'api 在线'}")
        self._log(f"  模型 (model)             : {self.model}")
        self._log(f"  温度 (temperature)       : {self.temperature}")
        self._log(f"  top_p                    : {self.top_p             if self.top_p             is not None else '默认 (default)'}")
        self._log(f"  最大token (max_tokens)   : {self.max_tokens        if self.max_tokens        is not None else '默认 (default)'}")
        self._log(f"  frequency_penalty        : {self.frequency_penalty if self.frequency_penalty is not None else '默认 (default)'}")
        self._log(f"  日志文件 (log_file)      : {self.log_path}")
        self._log("=" * 78)

    # ── GPT-specific builders ─────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _build_completion_kwargs(self, content, with_logprobs=False):
        kwargs = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if with_logprobs:
            kwargs["logprobs"]     = True
            kwargs["top_logprobs"] = 5
        return kwargs

    def _build_content(self, image_path, prompt):
        content = []
        if image_path and os.path.exists(image_path):
            media_type = self._get_image_media_type(image_path)
            content.append({
                "type":      "image_url",
                "image_url": {"url": f"data:{media_type};base64,{self._encode_image(image_path)}"},
            })
        content.append({"type": "text", "text": prompt})
        return content

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

            resp = self.client.chat.completions.create(
                **self._build_completion_kwargs(self._build_content(image_path, prompt), False)
            )
            return (resp.choices[0].message.content or "").strip(), None
        except Exception as e:
            self._log(f"  [ask] error: {e}")
            if self._is_quota_error(e):
                raise APIQuotaExceededError(str(e))
            return f"API error: {e}", None

    def ask1(self, image_path, prompt):
        try:
            if not self.mock_mode and not os.path.exists(image_path):
                return f"Image not found: {image_path}", None
            if self.mock_mode:
                answer, logprob = self._mock_yes_no(prompt, image_path)
                return f"Mock reasoning: simulated visual inspection.\n- Result: {answer}", logprob

            resp = self.client.chat.completions.create(
                **self._build_completion_kwargs(self._build_content(image_path, prompt), True)
            )
            text = (resp.choices[0].message.content or "").strip()
            return text, self._extract_answer_logprob(resp)
        except Exception as e:
            self._log(f"  [qa] error: {e}")
            if self._is_quota_error(e):
                raise APIQuotaExceededError(str(e))
            return f"API error: {e}", None

    # ── GPT-specific logprob extraction ──────────────────────────────────────

    def _extract_answer_logprob(self, response):
        try:
            lp = response.choices[0].logprobs
            if lp is None or not lp.content:
                return None
            response_text = (response.choices[0].message.content or "").lower()
            if "result: yes" in response_text:
                answer = "yes"
            elif "result: no" in response_text:
                answer = "no"
            else:
                return None
            for token_info in reversed(lp.content):
                t = token_info.token.lower().strip()
                if t in {"yes", "yes.", "no", "no."} and t.startswith(answer):
                    return token_info.logprob
            for token_info in reversed(lp.content):
                for cand in (getattr(token_info, "top_logprobs", None) or []):
                    t = cand.token.lower().strip()
                    if t in {"yes", "yes.", "no", "no."} and t.startswith(answer):
                        return cand.logprob
            return None
        except Exception as e:
            self._log(f"    [qa-logprob] extract failed: {e}")
            return None


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.logger import setup_runtime_logger, shutdown_runtime_logger

    setup_runtime_logger(log_dir="./runtime_logs", file_prefix="GPT+demo", redirect_stdout=True)
    try:
        cfg = {"mode": "model", "model": "mock-gpt", "temperature": 1.0}
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
