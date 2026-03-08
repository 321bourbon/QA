# vlm_api.py - VLM接口封装
#
# InternVL2-8B的logprob计算方案：
# 使用模型的extract_feature方法获取图像embedding，
# 然后手动构建输入给language_model计算logits

import os
import re
import traceback
import torch
import gc
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F


class MockVLM:
    """模拟VLM，用于测试"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.call_count = 0
        print("[MockVLM] 使用模拟模型")

    def ask(self, image_path, prompt):
        self.call_count += 1
        response = f"模拟输出 #{self.call_count}: - Result: Yes"
        return response, None

    def ask1(self, image_path, prompt):
        self.call_count += 1
        import random
        if random.random() > 0.3:
            response = f"The image appears normal. - Result: Yes"
            logprob = random.uniform(-0.5, -0.1)
        else:
            response = f"The image shows anomaly. - Result: No"
            logprob = random.uniform(-1.5, -0.5)
        return response, logprob


class RealVLM:
    """真实VLM实现 - InternVL2-8B"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_dir = cfg.get("model_dir", r"C:\Users\pc\Desktop\LAD_QA\models\InternVL2-8B")

        # 论文参数 (Appendix B.1)
        self.temperature = cfg.get("temperature", 0.2)
        self.top_p = cfg.get("top_p", 0.7)
        self.repetition_penalty = cfg.get("repetition_penalty", 1.1)
        self.do_sample = cfg.get("do_sample", True)
        self.max_new_tokens = cfg.get("max_new_tokens", 512)

        print(f"[RealVLM] 开始加载 InternVL2-8B")
        print(f"  模型路径: {self.model_dir}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 加载Tokenizer
        print("\n[1/3] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True
        )

        # 配置4bit量化
        print("\n[2/3] 配置4bit量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # 加载模型
        print("\n[3/3] 加载模型...")
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()

        print("\n✓ 模型加载成功！")

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
            print(f"  GPU显存: {gpu_mem:.2f} GB")

        # 图像预处理
        self.image_processor = self._build_image_transform()

        # 缓存Yes/No的token IDs
        self._cache_answer_token_ids()

        # 获取模型配置
        self._setup_model_config()

    def _build_image_transform(self):
        """构建图像预处理管道"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        return transforms.Compose([
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _cache_answer_token_ids(self):
        """缓存Yes/No的token IDs"""
        self.yes_token_ids = set()
        self.no_token_ids = set()

        yes_variants = ["Yes", "yes", "YES", " Yes", " yes", "Yes.", "yes."]
        no_variants = ["No", "no", "NO", " No", " no", "No.", "no."]

        for variant in yes_variants:
            tokens = self.tokenizer.encode(variant, add_special_tokens=False)
            if tokens:
                self.yes_token_ids.add(tokens[0])

        for variant in no_variants:
            tokens = self.tokenizer.encode(variant, add_special_tokens=False)
            if tokens:
                self.no_token_ids.add(tokens[0])

        print(f"  Yes token IDs: {self.yes_token_ids}")
        print(f"  No token IDs: {self.no_token_ids}")

    def _setup_model_config(self):
        """获取模型配置"""
        # 获取IMG_CONTEXT token ID
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        print(f"  IMG_CONTEXT token ID: {self.img_context_token_id}")

        # 获取language_model
        self.language_model = self.model.language_model
        print(f"  language_model: {type(self.language_model).__name__}")

    def ask(self, image_path, prompt):
        """基本问答方法"""
        try:
            if not image_path or not os.path.exists(image_path):
                return self._ask_text_only(prompt)

            image = Image.open(image_path).convert('RGB')
            pixel_values = self.image_processor(image)
            pixel_values = pixel_values.unsqueeze(0).to(dtype=torch.float16)
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()

            generation_config = {
                'max_new_tokens': self.max_new_tokens,
                'do_sample': self.do_sample,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'repetition_penalty': self.repetition_penalty,
                'num_beams': 1,
            }

            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=generation_config,
                )

            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return response.strip(), None

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"推理错误: {e}", None

    def _ask_text_only(self, prompt):
        """纯文本问答"""
        try:
            generation_config = {
                'max_new_tokens': self.max_new_tokens,
                'do_sample': self.do_sample,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'repetition_penalty': self.repetition_penalty,
                'num_beams': 1,
            }

            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=None,
                    question=prompt,
                    generation_config=generation_config,
                )

            return response.strip(), None

        except Exception as e:
            print(f"  纯文本调用失败: {e}")
            return "", None

    def ask1(self, image_path, prompt):
        """
        问答 + 获取Yes/No答案的log probability
        """
        try:
            if not os.path.exists(image_path):
                return f"图像不存在: {image_path}", None

            # 加载图像
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.image_processor(image)
            pixel_values = pixel_values.unsqueeze(0).to(dtype=torch.float16)
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()

            generation_config = {
                'max_new_tokens': self.max_new_tokens,
                'do_sample': self.do_sample,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'repetition_penalty': self.repetition_penalty,
                'num_beams': 1,
            }

            # 步骤1: 使用chat生成response
            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=generation_config,
                )

            # 步骤2: 计算Yes/No的logprob
            answer_logprob = self._compute_logprob_with_image(
                pixel_values, prompt, response
            )

            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return response.strip(), answer_logprob

        except Exception as e:
            print(f"  ✗ ask1错误: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"推理错误: {e}", None

    def _compute_logprob_with_image(self, pixel_values, prompt, response):
        """
        使用InternVL2的内部方法计算带图像的logprob

        关键：使用model.extract_feature()获取图像embedding，
        这是InternVL2官方提供的方法，已经处理好了所有细节
        """
        try:
            # 找到答案
            answer, answer_pos = self._find_answer_position(response)
            if answer is None:
                print(f"    [logprob] 未找到Yes/No答案")
                return None

            # 构建到答案之前的部分response
            partial_response = response[:answer_pos]

            # ===== 使用InternVL2的extract_feature方法 =====
            with torch.no_grad():
                # extract_feature是InternVL2官方提供的方法
                # 它会正确处理vision_model的输出并通过mlp1投影
                image_embeds = self.model.extract_feature(pixel_values)
                # image_embeds: [1, num_patches, hidden_size]

            num_image_tokens = image_embeds.shape[1]

            # ===== 构建文本输入 =====
            # 创建图像占位符
            image_placeholder = '<IMG_CONTEXT>' * num_image_tokens

            # InternVL2的对话模板
            full_text = f"<|im_start|>user\n{image_placeholder}\n{prompt}<|im_end|>\n<|im_start|>assistant\n{partial_response}"

            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs['input_ids']

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # ===== 获取文本embedding并替换图像位置 =====
            with torch.no_grad():
                # 获取embedding层
                embed_tokens = self.language_model.get_input_embeddings()
                input_embeds = embed_tokens(input_ids)  # [1, seq_len, hidden_size]

                # 找到IMG_CONTEXT token的位置
                img_positions = (input_ids[0] == self.img_context_token_id).nonzero(as_tuple=True)[0]

                # 替换图像位置的embedding
                if len(img_positions) == num_image_tokens:
                    for i, pos in enumerate(img_positions):
                        input_embeds[0, pos] = image_embeds[0, i]
                else:
                    print(f"    [logprob] 警告: 图像token数量不匹配 ({len(img_positions)} vs {num_image_tokens})")

                # ===== 用language_model计算logits =====
                outputs = self.language_model(
                    inputs_embeds=input_embeds,
                )

                # 获取最后一个位置的logits（预测下一个token）
                last_logits = outputs.logits[0, -1, :]

                # 计算log softmax
                log_probs = F.log_softmax(last_logits, dim=-1)

                # 提取Yes/No的logprob
                if answer == "Yes":
                    logprobs_list = [log_probs[tid].item() for tid in self.yes_token_ids if tid < len(log_probs)]
                else:
                    logprobs_list = [log_probs[tid].item() for tid in self.no_token_ids if tid < len(log_probs)]

                if logprobs_list:
                    answer_logprob = max(logprobs_list)
                    print(f"    [logprob] {answer}: {answer_logprob:.4f}")
                    return answer_logprob
                else:
                    print(f"    [logprob] 未找到{answer}的token ID在vocab中")
                    return None

        except Exception as e:
            print(f"    [logprob] 计算失败: {e}")
            traceback.print_exc()
            return None

    def _find_answer_position(self, response):
        """在response中找到Yes/No答案的位置"""
        response_lower = response.lower()

        # 查找 "Result: Yes" 或 "Result: No" 的模式
        patterns = [
            (r'[-–]\s*result\s*:\s*yes', "Yes"),
            (r'[-–]\s*result\s*:\s*no', "No"),
            (r'result\s*:\s*yes', "Yes"),
            (r'result\s*:\s*no', "No"),
        ]

        for pattern, answer in patterns:
            match = re.search(pattern, response_lower)
            if match:
                # 找到"Yes"或"No"在match区域内的实际位置
                if answer == "Yes":
                    sub_match = re.search(r'yes', response_lower[match.start():], re.IGNORECASE)
                    if sub_match:
                        return "Yes", match.start() + sub_match.start()
                else:
                    sub_match = re.search(r'no', response_lower[match.start():], re.IGNORECASE)
                    if sub_match:
                        return "No", match.start() + sub_match.start()

        # 备选：在最后100字符中找
        last_part = response_lower[-100:] if len(response_lower) > 100 else response_lower

        yes_pos = last_part.rfind('yes')
        no_pos = last_part.rfind('no')

        if yes_pos > no_pos and yes_pos != -1:
            actual_pos = len(response_lower) - len(last_part) + yes_pos
            return "Yes", actual_pos
        elif no_pos != -1:
            actual_pos = len(response_lower) - len(last_part) + no_pos
            return "No", actual_pos

        return None, -1


class VLM:
    """VLM统一接口"""

    def __init__(self, cfg):
        use_mock = cfg.get("use_mock", False)
        if use_mock:
            self.client = MockVLM(cfg)
        else:
            self.client = RealVLM(cfg)

    def ask(self, image_path, prompt):
        """基本问答"""
        return self.client.ask(image_path, prompt)

    def ask1(self, image_path, prompt):
        """问答 + Yes/No的logprob"""
        return self.client.ask1(image_path, prompt)


if __name__ == "__main__":
    test_cfg = {
        "use_mock": False,
        "temperature": 0.2,
        "max_new_tokens": 256
    }

    try:
        print("\n加载模型...")
        vlm = VLM(test_cfg)

        test_image = None
        mvtec_root = r"E:\dataSet\MVtec LOCO AD\mvtec_loco_anomaly_detection"

        if os.path.exists(mvtec_root):
            from pathlib import Path

            samples = list(Path(mvtec_root).rglob("*.png"))
            if samples:
                test_image = str(samples[0])

        if test_image and os.path.exists(test_image):
            print(f"\n测试图像: {os.path.basename(test_image)}")

            test_prompt = """Question: Is the juice bottle filled with at least 90% of its capacity?

At first, describe juice_bottle image and then answer the question.
Your response must end with '- Result: Yes' or '- Result: No'.
Let's think step by step."""

            print(f"\n{'=' * 60}")
            print(f"测试 ask1 (使用extract_feature)")
            print(f"{'=' * 60}")

            response, logprob = vlm.ask1(test_image, test_prompt)

            print(f"\n响应:\n{response}")
            print(f"\n答案logprob: {logprob}")

        else:
            print("\n⚠️ 未找到测试图像")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        traceback.print_exc()