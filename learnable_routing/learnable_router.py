# -*- encoding: utf-8 -*-
"""
learnable_router.py
===================

加载 SFT 后的 Qwen3-0.6B LoRA checkpoint，对外提供：

    LearnableRouter(model_path).predict_route(question, table_meta) -> "full"|"partial"|"filter"

agent.py 在 route_mode="learnable" 时调用这个对象。
"""
from __future__ import annotations
import os
import re
from threading import Lock
from typing import Any, Dict, Optional

# 重型依赖（torch / peft / transformers）延迟到 LearnableRouter.__init__ 内导入，
# 这样无 GPU / 未安装 peft 的环境仍可使用 StubLearnableRouter 跑通管线。

# system prompt 与 merge_labels.py 严格保持一致
SYSTEM_PROMPT = (
    "你是表格路由分类器。给定用户的查询和表格元信息，预测最适合的路由强度：\n"
    "- filter  : 强筛选，只保留命中的实体（适合稀疏证据检索）\n"
    "- partial : 保守过滤 + 覆盖率熔断（适合通用问答）\n"
    "- full    : 全量表格喂给模型（适合全局聚合 / 开放式分析 / 代码生成）\n"
    "只输出 filter / partial / full 三个标签之一，不输出任何解释。"
)

_LABELS = ("full", "partial", "filter")
_LABEL_RE = re.compile(r"\b(full|partial|filter)\b", re.IGNORECASE)


def _build_user(question: str, meta: Dict[str, Any]) -> str:
    return (
        f"benchmark: {meta.get('benchmark','?')}\n"
        f"sub_task : {meta.get('sub_task_name','?')}\n"
        f"language : {meta.get('lang','?')}\n"
        f"table    : {meta.get('rows',0)} rows x {meta.get('cols',0)} cols ({meta.get('cells',0)} cells)\n"
        f"query    : {question}"
    )


class LearnableRouter:
    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        default_label: str = "full",
        dtype: Optional[Any] = None,
    ):
        """
        model_path: LoRA checkpoint 目录 (sft_train.py 的 --output_dir)
        base_model: 若 model_path 是纯 LoRA adapter 需提供 base；
                    若 model_path 是 trainer.save_model 后已合并的 full ckpt 可不传。
        """
        import torch  # noqa: WPS433
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

        self.default_label = default_label if default_label in _LABELS else "full"
        self.lock = Lock()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            if base_model is None:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if is_lora:
            if base_model is None:
                raise ValueError(
                    f"{model_path} 看起来是 LoRA adapter，请同时提供 base_model="
                )
            from peft import PeftModel  # noqa: WPS433
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=dtype, trust_remote_code=True,
            ).to(device)
            self.model = PeftModel.from_pretrained(base, model_path).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True,
            ).to(device)

        self.model.eval()
        self.device = device
        self._torch = torch  # 留作 inference_mode 用

    def predict_route(self, question: str, table_meta: Dict[str, Any]) -> str:
        torch = self._torch
        with torch.inference_mode():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user(question, table_meta)},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with self.lock:
                out = self.model.generate(
                    **inputs,
                    # Qwen3 默认开启 thinking mode：assistant 段先输出 <think>\n\n</think>\n\nLABEL，
                    # 所以 max_new_tokens 必须 ≥ 8 才能保证看到 LABEL；这里给 32 留余量。
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).lower()
            # 优先从 </think> 之后匹配，避免误命中 thinking 段里偶然出现的 'full/partial/filter'
            after_think = text.split("</think>", 1)[-1] if "</think>" in text else text
            m = _LABEL_RE.search(after_think) or _LABEL_RE.search(text)
            if m:
                return m.group(1).lower()
            return self.default_label


class StubLearnableRouter:
    """无 GPU 跑通管线时用：按 cells 阈值给出确定性的 full/partial/filter。"""

    def __init__(self, *, cells_partial: int = 200, cells_filter: int = 600):
        self.cells_partial = cells_partial
        self.cells_filter = cells_filter

    def predict_route(self, question: str, table_meta: Dict[str, Any]) -> str:
        c = int(table_meta.get("cells", 0))
        if c >= self.cells_filter:
            return "filter"
        if c >= self.cells_partial:
            return "partial"
        return "full"
