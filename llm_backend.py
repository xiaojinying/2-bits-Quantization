#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版本地量化 LLM 后端

功能：
- 加载你自己的量化 LLaMA / Qwen 模型（来自 utils.py）
- 提供 HTTP API: POST /api/chat
- 提供一个简单的 DSPy 适配类，方便在 DSPy 里直接调用本地模型

不包含：
- Chainlit 前端 UI
- 本地会话持久化（SQLite / JSONL 日志）
"""

import os
import typing as t

import torch
from transformers import AutoTokenizer

from fastapi import FastAPI, Body
from pydantic import BaseModel

# 你的自定义工具
from utils import (
    load_llama_or_qwen_model,
    load_non_linear_params_from_true_quant,
    replace_llama_with_packed_w2_layers,
)

# ==========================
# 0. 环境设置
# ==========================

# 如果你想指定 GPU，可以在启动前导出 CUDA_VISIBLE_DEVICES；
# 这里给一个默认值（如果外部没设，就用 "1"）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# ==========================
# 1. 配置
# ==========================

class Args:
    # 按需改成你的真实路径（也可以用环境变量覆盖）
    quant_pt: str = os.getenv(
        "QUANT_PT",
        "/data2/xx_llms/output/l2-13b-wiki-gs64/true_quant.pth",
    )
    model_path: str = os.getenv(
        "MODEL_PATH",
        "/data2/llms/Qwen2.5-14B-Instruct",
    )
    group_size: int = int(os.getenv("GROUP_SIZE", "64"))


# ==========================
# 2. 模型加载（后端核心）
# ==========================

MODEL = None          # type: ignore
TOKENIZER = None      # type: ignore


def init_model():
    """懒加载模型：首次调用时加载，后面直接复用"""
    global MODEL, TOKENIZER
    if MODEL is not None:
        return

    print(">>> [System] 正在加载本地量化模型...")

    # 1) Tokenizer
    print(f">>> [System] 加载 Tokenizer: {Args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(Args.model_path, use_fast=True)

    # 2) 基础结构（先在 CPU 上）
    print(">>> [System] 加载基础模型结构 (CPU)...")
    model = load_llama_or_qwen_model(Args.model_path)

    # 3) 量化权重
    print(f">>> [System] 加载量化权重: {Args.quant_pt}")
    state_dict = torch.load(Args.quant_pt, map_location="cpu")

    print(">>> [System] 应用自定义量化层 (CPU)...")
    # 如果你已经在 utils 里集成好了 true_quant 恢复，这里可以按需打开：
    replace_llama_with_packed_w2_layers(model, state_dict, Args.group_size)

    # 4) 移到 GPU
    print(">>> [System] 将模型移动到 GPU...")
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)

    MODEL = model
    TOKENIZER = tokenizer

    print(">>> [System] 模型加载完毕")


def _build_full_prompt(history: t.List[t.Tuple[str, str]], user_input: str) -> str:
    """
    把多轮对话变成一个 prompt。
    - history 由外部调用方管理（服务端不做持久化）
    """
    messages = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    try:
        full_prompt = TOKENIZER.apply_chat_template(  # type: ignore
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 简单 fallback
        full_prompt = ""
        for u, a in history:
            full_prompt += f"User: {u}\nAssistant: {a}\n"
        full_prompt += f"User: {user_input}\nAssistant:"

    return full_prompt


def generate_once(
    user_input: str,
    history: t.Optional[t.List[t.Tuple[str, str]]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> t.Tuple[str, t.List[t.Tuple[str, str]]]:
    """
    核心同步推理函数：
    - HTTP API / DSPy 都建议走这个函数
    - history 仅由调用方在请求体中携带，本地不做存盘
    """
    init_model()

    if history is None:
        history = []

    full_prompt = _build_full_prompt(history, user_input)

    device = next(MODEL.parameters()).device  # type: ignore
    inputs = TOKENIZER(full_prompt, return_tensors="pt").to(device)  # type: ignore

    with torch.inference_mode():
        output_ids = MODEL.generate(  # type: ignore
            input_ids=inputs["input_ids"],
            max_new_tokens=int(max_new_tokens),
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id,
            do_sample=True,
            temperature=float(temperature),
            use_cache=True,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    reply = TOKENIZER.decode(new_tokens, skip_special_tokens=True)  # type: ignore

    new_history = history + [(user_input, reply)]
    return reply, new_history


# ==========================
# 3. FastAPI HTTP API
# ==========================

app = FastAPI(title="Local Quantized LLM API")


class ChatRequest(BaseModel):
    message: str
    history: t.Optional[t.List[t.List[str]]] = None  # [[user, assistant], ...]
    max_new_tokens: int = 512
    temperature: float = 0.7


class ChatResponse(BaseModel):
    reply: str
    history: t.List[t.List[str]]


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest = Body(...)):
    """
    HTTP 调用示例：

    POST http://localhost:8000/api/chat
    JSON:
    {
        "message": "你好，你是谁？",
        "history": [["hi", "hello"]],
        "max_new_tokens": 128,
        "temperature": 0.7
    }
    """
    # 把 [[u, a], ...] 转成 [(u, a), ...]
    history_pairs: t.List[t.Tuple[str, str]] = []
    if req.history:
        for item in req.history:
            if len(item) == 2:
                history_pairs.append((item[0], item[1]))

    reply, new_history = generate_once(
        user_input=req.message,
        history=history_pairs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    history_out = [[u, a] for (u, a) in new_history]
    return ChatResponse(reply=reply, history=history_out)


# ==========================
# 4. DSPy 适配（可选）
# ==========================

try:
    import dspy

    class LocalLlamaModule(dspy.Module):
        """
        最简单的 DSPy 模块适配：
        - 不走 dspy.LM / LiteLLM
        - 直接调用本地 generate_once
        """

        class LocalSignature(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()

        def forward(self, question: str):
            # 如需多轮对话，可以在这里自己维护 history
            answer, _ = generate_once(question, history=[])
            return dspy.Prediction(answer=answer)

except ImportError:
    # dspy 没装也没关系，HTTP API 仍然可用
    LocalLlamaModule = None  # type: ignore


# ==========================
# 5. 直接运行（可选）
# ==========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "llm_backend:app",  # 文件名如果不是 llm_backend.py，改成对应模块路径
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )
