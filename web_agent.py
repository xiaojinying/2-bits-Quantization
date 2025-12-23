#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NUDT LLM Lite Web Agent (English QA)
- Designed for relatively weak / slow local models
- Simple pipeline: (optional) web search with Exa + single-turn summarization
- All model prompts and answers are in English
"""

import os
import argparse
from typing import List, Dict, Tuple

import torch
import gradio as gr
from transformers import AutoTokenizer
from exa_py import Exa

# ============================================================
# 0. Environment & default config
# ============================================================


EXA_API_KEY = os.getenv("EXA_API_KEY", "")
exa = Exa(api_key=EXA_API_KEY) if EXA_API_KEY else None


class Args:
    quant_pt = os.getenv("QUANT_PT", "")
    model_path = os.getenv("MODEL_PATH", "")
    group_size = int(os.getenv("GROUP_SIZE", "64"))
    max_new_tokens = 256
    temperature = 0.5
    max_context_chars = 2000  # total characters of retrieved context given to the model


def parse_cli_args():
    parser = argparse.ArgumentParser(description="NUDT Lite Web Agent (English)")
    parser.add_argument("--model-path", type=str, default=Args.model_path)
    parser.add_argument("--quant-pt", type=str, default=Args.quant_pt)
    parser.add_argument("--group-size", type=int, default=Args.group_size)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


# ============================================================
# 1. Your quantized model utilities
# ============================================================

from utils import (
    load_llama_or_qwen_model,
    replace_llama_with_packed_w2_layers,
)

MODEL = None
TOKENIZER = None
WARMED_UP = False  # 标记是否已经做过一次预热


def init_model():
    """Lazy load the quantized model only once."""
    global MODEL, TOKENIZER
    if MODEL is not None:
        return

    if not Args.model_path or not Args.quant_pt:
        raise ValueError(
            "MODEL_PATH and QUANT_PT must be set via env or CLI arguments."
        )

    print(">>> [System] Loading model ...")
    TOKENIZER = AutoTokenizer.from_pretrained(Args.model_path, use_fast=True)

    base_model = load_llama_or_qwen_model(Args.model_path)
    state_dict = torch.load(Args.quant_pt, map_location="cpu")
    replace_llama_with_packed_w2_layers(base_model, state_dict, Args.group_size)

    base_model.cuda().eval()
    MODEL = base_model
    print(">>> [System] Model loaded.")

    # 🔥 模型加载完成后，立刻进行一次轻量预热
    warmup_model()


def warmup_model(num_new_tokens: int = 16):
    """
    Run a tiny dummy generation once to warm up CUDA kernels, caches, etc.
    This happens before real user inference to avoid first-request latency.
    """
    global MODEL, TOKENIZER, WARMED_UP
    if MODEL is None or TOKENIZER is None:
        return

    if WARMED_UP:
        return

    print(">>> [System] Warming up model ...")
    dummy_prompt = "Warmup."

    inputs = TOKENIZER(dummy_prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        _ = MODEL.generate(
            **inputs,
            max_new_tokens=num_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    # 确保所有 CUDA kernel 执行完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    WARMED_UP = True
    print(">>> [System] Warmup done.")


def generate_once(prompt: str) -> str:
    """Single-turn generation; no ReAct loop."""
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out_ids = MODEL.generate(
            **inputs,
            max_new_tokens=Args.max_new_tokens,
            do_sample=True,
            temperature=Args.temperature,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    text = TOKENIZER.decode(gen_ids[0], skip_special_tokens=True)
    return text.strip()


# ============================================================
# 2. Tool layer: simple web search via Exa
# ============================================================

def web_search_exa(query: str, k: int = 3) -> Tuple[str, List[Dict]]:
    """
    Use Exa to search and return:
    - merged context string (fed into the model)
    - list of references (for user display)
    """
    if exa is None:
        print(">>> [Tool] EXA_API_KEY not set, skip web search.")
        return "", []

    print(f">>> [Tool] Exa search: {query}")
    try:
        result = exa.search_and_contents(
            query,
            num_results=k,
            text=True,
        )
    except Exception as e:
        print(f">>> [Tool] Exa error: {e}")
        return "", []

    contexts: List[str] = []
    refs: List[Dict] = []

    for item in result.results:
        title = item.title or "No Title"
        url = item.url or ""
        text = item.text or ""
        if not text:
            continue

        snippet = text[:600]  # limit each snippet
        block = f"[Title] {title}\n[URL] {url}\n[Content] {snippet}\n"
        contexts.append(block)

        refs.append(
            {
                "title": title,
                "url": url,
            }
        )

    if not contexts:
        return "", []

    # limit total context length for the model
    merged = ""
    for c in contexts:
        if len(merged) + len(c) > Args.max_context_chars:
            break
        merged += c + "\n"

    return merged, refs


def format_refs_markdown(refs: List[Dict]) -> str:
    """Format reference list as Markdown."""
    if not refs:
        return ""

    seen = set()
    uniq: List[Dict] = []
    for r in refs:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            uniq.append(r)

    if not uniq:
        return ""

    s = "\n\n---\n### 🔗 References\n"
    for i, r in enumerate(uniq, 1):
        title = r.get("title", "Unknown Page")
        url = r.get("url", "#")
        s += f"{i}. [{title}]({url})\n"
    return s


# ============================================================
# 3. Routing layer: decide whether to use web search
# ============================================================

def need_web_search(user_query: str) -> bool:
    """
    Simple rule-based router for weak models.

    - If query clearly asks for recent / factual / up-to-date info -> use web
    - Very short chit-chat messages -> let the model answer directly
    - Otherwise: default to using web search when possible
    """
    q = user_query.lower()

    # keywords that strongly suggest we need the internet
    hot_keywords = [
        "recent",
        "latest",
        "update",
        "updates",
        "news",
        "trend",
        "trends",
        "today",
        "yesterday",
        "this year",
        "2023",
        "2024",
        "2025",
    ]
    for kw in hot_keywords:
        if kw in q:
            return True

    # obvious math / logic questions might not need web
    math_ops = ["+", "-", "*", "/", "%"]
    if all(ch.isdigit() or ch.isspace() or ch in math_ops for ch in q):
        return False

    # very short questions (e.g. "Who are you?") – just chat
    if len(q) < 15:
        return False

    # default: prefer using web when possible
    return True


# ============================================================
# 4. Agent logic (single-turn)
# ============================================================

SYSTEM_PROMPT = """You are a concise, reliable English assistant.

- If external context (documents or web snippets) is provided, you must base your answer on that context as much as possible.
- If the context is not sufficient to answer confidently, clearly say that you are uncertain instead of hallucinating.
- Always answer in clear, natural English.
- Keep the answer focused and reasonably short unless the question explicitly asks for a long or detailed explanation.
"""


def agent_answer(user_query: str) -> str:
    """Main agent entry: route -> optional web search -> summarize once."""
    init_model()  # 确保模型已加载并预热（若尚未完成）

    use_web = need_web_search(user_query)
    context = ""
    refs: List[Dict] = []

    if use_web:
        context, refs = web_search_exa(user_query)
        print(f">>> [Agent] use_web={use_web}, context_len={len(context)}")
    else:
        print(f">>> [Agent] use_web={use_web}, no search.")

    # Build prompt for the model (single-turn)
    if context:
        prompt = (
            SYSTEM_PROMPT
            + "\n\nUser question:\n"
            + user_query
            + "\n\nI have retrieved some external information:\n"
            + context
            + "\nBased on the information above, answer the user's question in English. "
              "If the information is insufficient, say that you are uncertain."
        )
    else:
        prompt = (
            SYSTEM_PROMPT
            + "\n\nThere is no additional external context available. "
              "You must answer based only on your own knowledge.\n\n"
            + "User question:\n"
            + user_query
        )

    answer = generate_once(prompt).strip()

    # Append reference list for the UI
    answer += format_refs_markdown(refs)
    return answer


# ============================================================
# 5. Gradio UI wrapper – cyber cute style ✨
# ============================================================

def chat_fn(message, history):
    # history is ignored in this simple single-turn agent,
    # but you could condition on it later if needed.
    return agent_answer(message)


# 赛博暗色 + 霓虹边框 + 表情
CUSTOM_CSS = """
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 16px 0 24px 0;
    background: radial-gradient(circle at top, #0f172a 0, #020617 45%, #000000 100%);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #e5e7eb !important;
}

.gradio-container h1 {
    text-align: center;
    font-size: 1.9rem;
    letter-spacing: 0.03em;
}

.gradio-container h3 {
    text-align: center;
}

/* 顶部说明文字更紧凑一点 */
.gradio-container .prose p {
    font-size: 0.95rem;
}

/* 聊天框整体：赛博卡片 */
#chatbot {
    border-radius: 18px !important;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.9)) !important;
    box-shadow:
        0 0 0 1px rgba(56,189,248,0.15),
        0 18px 45px rgba(15,23,42,0.9) !important;
}

/* 用户 & 机器人消息气泡 */
#chatbot .message {
    border-radius: 14px !important;
    margin: 8px 12px !important;
    padding: 9px 12px !important;
    font-size: 0.94rem !important;
    line-height: 1.4;
}

/* 用户 = 绿色侧边条 */
#chatbot .message.user {
    background: rgba(15, 23, 42, 0.95) !important;
    border-left: 3px solid #22c55e !important;
}

/* Bot = 蓝紫色侧边条 */
#chatbot .message.bot {
    background: rgba(15, 23, 42, 0.98) !important;
    border-left: 3px solid #38bdf8 !important;
}

/* 链接颜色 */
#chatbot .message a {
    color: #38bdf8 !important;
    font-weight: 600;
    text-decoration: none;
}
#chatbot .message a:hover {
    text-decoration: underline;
}

/* 输入框：发光圆角条 */
textarea {
    border-radius: 999px !important;
    border: 1px solid #1f2937 !important;
    padding: 10px 16px !important;
    background: rgba(15,23,42,0.95) !important;
    color: #e5e7eb !important;
    font-size: 0.95rem !important;
}
textarea::placeholder {
    color: #6b7280 !important;
}

/* 示例区域：小按钮风格 */
.gradio-container .examples {
    margin-top: 8px !important;
}
.gradio-container .examples button {
    border-radius: 999px !important;
    font-size: 0.8rem !important;
}

/* 隐藏多余的底部 footer（如果有的话） */
footer {visibility: hidden}
"""


def build_demo():
    """构建带自定义样式的 ChatInterface，兼容老版本 gradio。"""
    try:
        demo = gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(
                height=580,
                elem_id="chatbot",
            ),
            title="⚡ NUDT Deep Mind Web Agent · Quantized LLM + Web Search",
            description=(
                "🧠 **Mode:** English-only · single-turn\n"
                "💾 **Engine:** Quantized local LLM with optional Exa web search\n"
            ),
            textbox=gr.Textbox(
                placeholder="Type your question here… (e.g. \"Explain 2-bit quantization like I'm 12 🌱\")",
                show_label=False,
                lines=2,
                autofocus=True,
            ),
            examples=[
                "🌈 Give me a friendly explanation of 2-bit weight-only quantization.",
                "🧪 Compare INT4 and INT2 for LLM inference efficiency.",
                "🏫 Briefly introduce THU to an international student.",
                "⚙️ How does knowledge distillation help compress large language models?",
                "📚 What are some recent trends in LLM evaluation?",
            ],
            cache_examples=False,
            css=CUSTOM_CSS,   
        )
    except TypeError:
        demo = gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(
                height=580,
                elem_id="chatbot",
            ),
            title="NUDT Deep Mind Web Agent",
            description="A web-augmented English assistant on a quantized local LLM.",
            textbox=gr.Textbox(
                placeholder="Type your question here…",
                show_label=False,
                lines=2,
                autofocus=True,
            ),
            examples=[
                "Explain 2-bit weight-only quantization in simple terms.",
                "What are the trade-offs between INT4 and INT2 for LLMs?",
            ],
            cache_examples=False,
        )
    return demo


if __name__ == "__main__":
    args = parse_cli_args()
    Args.model_path = args.model_path
    Args.quant_pt = args.quant_pt
    Args.group_size = args.group_size
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # 💡 启动服务前就加载并预热模型
    init_model()

    demo = build_demo()
    demo.launch(
        server_port=args.port,
        server_name="0.0.0.0",
    )
