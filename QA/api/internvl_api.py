import gc
import os
import re
import traceback

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from api.base_vlm import APIQuotaExceededError, BaseVLM  # noqa: F401  (re-export)


class VLM(BaseVLM):
    """
    InternVL2-8B client for LogicQA.

    mock_mode (cfg["use_mock"] = True  OR  cfg["mode"] in {"mock","model","fake","debug"}):
        Uses base class stable-seed mock — no GPU required.
    real mode:
        Loads InternVL2-8B with 4-bit quantization; requires CUDA.
    """

    # Default model location; override via cfg["model_dir"] or LOGICQA_MODEL_DIR env var
    _DEFAULT_MODEL_DIR ="/home/liuxinyi/model/InternVL2_5-38B"

    def __init__(self, cfg=None):
        super().__init__(cfg)

        # Support legacy cfg["use_mock"] flag as well as inherited mock_mode
        if self.cfg.get("use_mock", False):
            self.mock_mode = True

        if self.mock_mode:
            self._log("[InternVL] Mock 模式 — 不加载模型 (mock mode, no model loaded)")
            return

        self.model_dir = (
            self.cfg.get("model_dir")
            or os.environ.get("LOGICQA_MODEL_DIR", "")
            or self._DEFAULT_MODEL_DIR
        )

        # Paper params (Appendix B.1) — InternVL-specific defaults，不继承 GPT/Gemini 的 cfg 值
        # cfg["temperature"] 通常是 GPT 用的 1.0，InternVL 论文要求 0.2，此处独立设置
        self.temperature        = self.cfg.get("internvl_temperature", 0.2)
        self.top_p              = self.cfg.get("internvl_top_p", 0.7)
        self.repetition_penalty = self.cfg.get("repetition_penalty", 1.1)
        self.do_sample          = self.cfg.get("do_sample", True)
        self.max_new_tokens     = self.cfg.get("max_new_tokens", 512)

        self._log(f"[InternVL] 加载模型 (loading model from): {self.model_dir}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self._log("[1/2] 加载分词器 (Loading tokenizer) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True
        )

        #self._log("[2/3] 跳过量化，准备全精度环境 (Skipping quantization) ...")
        # 38B 模型加载需要大量系统内存 (RAM)，确保服务器有 >80GB 物理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空缓存
            gc.collect()  # 强制垃圾回收
        self._log("[2/2] 加载模型权重 (Loading model weights in BF16) ...")
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动分配显存到多个GPU
            max_memory={
                0: "20GiB",
                1: "20GiB",
                2: "20GiB",
                3: "20GiB",
            },
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="eager",
        ).eval()

        self._log("[InternVL] 模型加载成功 (model loaded successfully)")
        if torch.cuda.is_available():
            # 遍历所有可见卡统计显存
            gpu_mem = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024 ** 3
            self._log(f"  GPU 总显存占用 (total memory used): {gpu_mem:.2f} GB")

        # Image preprocessing pipeline
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)
        self.image_processor = transforms.Compose([
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Cache Yes/No token IDs for fast logprob lookup
        self.yes_token_ids: set = set()
        self.no_token_ids:  set = set()
        for v in ["Yes", "yes", "YES", " Yes", " yes", "Yes.", "yes."]:
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if toks:
                self.yes_token_ids.add(toks[0])
        for v in ["No", "no", "NO", " No", " no", "No.", "no."]:
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if toks:
                self.no_token_ids.add(toks[0])
        self._log(f"  Yes 词元 ID (token IDs): {self.yes_token_ids}")
        self._log(f"  No  词元 ID (token IDs): {self.no_token_ids}")

        # IMG_CONTEXT token and language model reference
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.language_model       = self.model.language_model
        self._log(f"  IMG_CONTEXT 词元 ID               : {self.img_context_token_id}")
        self._log(f"  语言模型类型 (language_model type): {type(self.language_model).__name__}")

    # ── Internal image helpers ────────────────────────────────────────────────

    def _load_pixel_values(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pv = self.image_processor(image).unsqueeze(0)  # [1, C, H, W]
        pv = pv.to(torch.bfloat16)
        if torch.cuda.is_available():
            pv = pv.to(self.model.device)  # 使用模型所在的设备
        return pv
        
    def _generation_config(self):
        return {
            "max_new_tokens":    self.max_new_tokens,
            "do_sample":         self.do_sample,
            "temperature":       self.temperature,
            "top_p":             self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "num_beams":         1,
        }

    def _free_gpu(self, *tensors):
        for t in tensors:
            del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, image_path, prompt):
        if self.mock_mode:
            return self._mock_response_text(prompt, image_path).strip(), None
        try:
            if not image_path or not os.path.exists(image_path):
                return self._ask_text_only(prompt)

            pv = self._load_pixel_values(image_path)
            response = self._ask_with_image(pv, prompt)
            self._log(f"  [回复]\n{response}")
            self._free_gpu(pv)
            return response.strip(), None
        except Exception as e:
            self._free_gpu()
            return f"Inference error: {e}", None

    def ask1(self, image_path, prompt):
        if self.mock_mode:
            answer, logprob = self._mock_yes_no(prompt, image_path)
            return f"Mock reasoning: simulated visual inspection.\n- Result: {answer}", logprob
        try:
            if not os.path.exists(image_path):
                return f"Image not found: {image_path}", None

            pv = self._load_pixel_values(image_path)
            response = self._ask_with_image(pv, prompt)
            self._log(f"  [回复]\n{response}") 
            logprob = self._compute_logprob(pv, prompt, response)
            self._free_gpu(pv)
            return response.strip(), logprob
        except Exception as e:
            self._free_gpu()
            self._log(f"  ask1 error: {e}")
            traceback.print_exc()
            return f"Inference error: {e}", None

    # ── InternVL-specific: image+text, text-only and logprob computation ──────

    def _ask_with_image(self, pixel_values, prompt):
        """
        Image+text inference without model.chat() / language_model.generate().

        model.chat() internally calls language_model.generate(), which is unavailable
        on 4-bit quantized InternLM2ForCausalLM. Instead:
          1. Extract image embeddings via model.extract_feature().
          2. Build the full prompt with <IMG_CONTEXT> placeholders.
          3. Replace placeholder positions with image embeddings.
          4. Decode autoregressively using raw LM forward passes (same as _ask_text_only).
        """
        with torch.no_grad():
            image_embeds = self.model.extract_feature(pixel_values)
        num_image_tokens = image_embeds.shape[1]

        image_placeholder = " ".join(["<IMG_CONTEXT>"] * num_image_tokens)
        full_prompt = (
            f"<|im_start|>user\n{image_placeholder}\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = self.tokenizer(
            full_prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)

        eos_ids = set()
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid > 0:
                eos_ids.add(tid)
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)

        embed_fn = self.language_model.get_input_embeddings()

        with torch.no_grad():
            input_embeds = embed_fn(input_ids)
            img_positions = (input_ids[0] == self.img_context_token_id).nonzero(as_tuple=True)[0]
            if len(img_positions) == num_image_tokens:
                for i, pos in enumerate(img_positions):
                    input_embeds[0, pos] = image_embeds[0, i]

        gen_ids   = []
        past_kv   = None
        cur_embeds = input_embeds
        max_tokens = self.max_new_tokens or 512

        with torch.no_grad():
            for _ in range(max_tokens):
                out     = self.language_model(inputs_embeds=cur_embeds, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                logits  = out.logits[0, -1, :]

                logits = logits / max(self.temperature, 1e-6)
                if self.top_p is not None and self.top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > self.top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(0, sorted_idx, sorted_logits)
                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                if next_id in eos_ids:
                    break
                gen_ids.append(next_id)

                # Feed only the new token embedding for the next step (KV-cache)
                next_id_tensor = torch.tensor([[next_id]], device=input_ids.device)
                cur_embeds = embed_fn(next_id_tensor)

        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    

    def _ask_text_only(self, prompt):
        """
        Text-only inference via manual autoregressive loop (KV-cache optimized).

        保持原有 sampling / temperature / top-p / eos 逻辑完全一致，
        但使用 KV cache 避免每一步重新计算整个序列。
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            im_start = "<|im_start|>"
            im_end   = "<|im_end|>"

            full_prompt = f"{im_start}user\n{prompt}{im_end}\n{im_start}assistant\n"

            input_ids = self.tokenizer(
                full_prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.to(self.model.device)

            eos_ids = set()
            for tok in [im_end, "<|endoftext|>"]:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid > 0:
                    eos_ids.add(tid)

            if self.tokenizer.eos_token_id is not None:
                eos_ids.add(self.tokenizer.eos_token_id)

            embed_fn = self.language_model.get_input_embeddings()

            gen_ids = []
            max_tokens = self.max_new_tokens or 512

            with torch.no_grad():

                # ---------- Step1: 处理完整 prompt ----------
                embeds = embed_fn(input_ids)

                out = self.language_model(
                    inputs_embeds=embeds,
                    use_cache=True
                )

                past_kv = out.past_key_values
                logits  = out.logits[0, -1, :]

                # ---------- Step2: autoregressive decode ----------
                for _ in range(max_tokens):

                    logits = logits / max(self.temperature, 1e-6)

                    if self.top_p is not None and self.top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True)

                        cum_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )

                        remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > self.top_p
                        sorted_logits[remove_mask] = float("-inf")

                        logits_filtered = torch.full_like(logits, float("-inf"))
                        logits_filtered.scatter_(0, sorted_idx, sorted_logits)
                        logits = logits_filtered

                    probs   = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()

                    if next_id in eos_ids:
                        break

                    gen_ids.append(next_id)

                    # ---------- Step3: 只输入新 token ----------
                    next_id_tensor = torch.tensor([[next_id]], device=input_ids.device)
                    next_embed = embed_fn(next_id_tensor)

                    out = self.language_model(
                        inputs_embeds=next_embed,
                        past_key_values=past_kv,
                        use_cache=True
                    )

                    past_kv = out.past_key_values
                    logits  = out.logits[0, -1, :]

            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            return response.strip(), None

        except Exception as e:
            self._log(f"  text-only (kv-cache) failed: {e}")
            traceback.print_exc()
            return "", None

    def _compute_logprob(self, pixel_values, prompt, response):
        """
        Compute log P("Yes") or log P("No") using full token sequence probability.
        """
        try:
            answer, answer_pos = self._find_answer_position(response)
            if answer is None:
                self._log("    [logprob] Yes/No answer not found in response")
                return None

            partial_response = response[:answer_pos]

            with torch.no_grad():
                image_embeds = self.model.extract_feature(pixel_values)

            num_image_tokens = image_embeds.shape[1]

            image_placeholder = " ".join(["<IMG_CONTEXT>"] * num_image_tokens)

            full_text = (
                f"<|im_start|>user\n{image_placeholder}\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{partial_response}"
            )

            inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.to(self.model.device)

            embed_tokens = self.language_model.get_input_embeddings()

            with torch.no_grad():

                input_embeds = embed_tokens(input_ids)

                img_positions = (input_ids[0] == self.img_context_token_id).nonzero(as_tuple=True)[0]

                if len(img_positions) == num_image_tokens:
                    for i, pos in enumerate(img_positions):
                        input_embeds[0, pos] = image_embeds[0, i]

                # forward once to get cache
                outputs = self.language_model(inputs_embeds=input_embeds, use_cache=True)

                past_kv = outputs.past_key_values
                logits = outputs.logits[0, -1, :]

                log_probs = F.log_softmax(logits.float(), dim=-1)

                # candidate answer tokens
                candidates = ["Yes", " yes", "Yes.", "yes."] if answer == "Yes" else ["No", " no", "No.", "no."]

                best_lp = None

                for cand in candidates:

                    token_seq = self.tokenizer.encode(cand, add_special_tokens=False)

                    cur_lp = 0.0
                    cur_past = past_kv
                    cur_logits = logits

                    for tid in token_seq:

                        log_probs = F.log_softmax(cur_logits.float(), dim=-1)

                        if tid >= len(log_probs):
                            cur_lp = None
                            break

                        cur_lp += log_probs[tid].item()

                        next_token = torch.tensor([[tid]], device=input_ids.device)
                        next_embed = embed_tokens(next_token)

                        out = self.language_model(
                            inputs_embeds=next_embed,
                            past_key_values=cur_past,
                            use_cache=True
                        )

                        cur_past = out.past_key_values
                        cur_logits = out.logits[0, -1, :]

                    if cur_lp is not None:
                        if best_lp is None or cur_lp > best_lp:
                            best_lp = cur_lp

            if best_lp is not None:
                self._log(f"    [logprob] {answer}: {best_lp:.4f}")
                return best_lp

            self._log(f"    [logprob] failed for '{answer}'")
            return None

        except Exception as e:
            self._log(f"    [logprob] computation failed: {e}")
            traceback.print_exc()
            return None

    def _find_answer_position(self, response):
        """Locate the character position of Yes/No in the response."""
        low = response.lower()
        patterns = [
            (r"[-\u2013]\s*result\s*:\s*yes", "Yes"),
            (r"[-\u2013]\s*result\s*:\s*no",  "No"),
            (r"result\s*:\s*yes",              "Yes"),
            (r"result\s*:\s*no",               "No"),
        ]
        for pattern, answer in patterns:
            m = re.search(pattern, low)
            if m:
                keyword = "yes" if answer == "Yes" else "no"
                sub = re.search(keyword, low[m.start():], re.IGNORECASE)
                if sub:
                    return answer, m.start() + sub.start()

        # Fallback: last occurrence in final 100 chars
        tail     = low[-100:] if len(low) > 100 else low
        offset   = len(low) - len(tail)
        yes_pos  = tail.rfind("yes")
        no_pos   = tail.rfind("no")
        if yes_pos > no_pos and yes_pos != -1:
            return "Yes", offset + yes_pos
        if no_pos != -1:
            return "No", offset + no_pos
        return None, -1
