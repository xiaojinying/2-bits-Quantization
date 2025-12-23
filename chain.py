import os
import torch
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import json
import datetime

# 引入自定义库
from utils import load_llama_or_qwen_model, replace_llama_with_packed_w2_layers

# ==============================================================================
# 0. 并行与环境设置 (关键修改区域)
# ==============================================================================

# 【核心设置 1】指定你要使用的显卡 ID（支持命令行环境变量覆盖）
# 命令行未设置 TARGET_GPUS 时，默认用 "1"
TARGET_GPUS = os.getenv("TARGET_GPUS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPUS

print(f">>> [System] 已指定 GPU: {TARGET_GPUS}")


# ==============================================================================
# 1. 存储机制配置
# ==============================================================================
current_dir = os.getcwd()
db_path = os.path.join(current_dir, "chat.db")
storage = SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{db_path}")
cl.data._data_layer = storage


def save_log_to_json(session_id, user_input, ai_output):
    """保存日志到 JSONL"""
    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join(current_dir, "chat_logs", date_str)
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{session_id}.jsonl")
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "role_user": user_input,
            "role_assistant": ai_output
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"!!! [Log] 保存日志失败: {e}")


# ==============================================================================
# 2. 模型加载区 + 预热功能
# ==============================================================================
class Args:
    # 支持用环境变量覆盖默认路径
    quant_pt = os.getenv("QUANT_PT", "/data1/xjy/2-bit/true_quant.pth")
    model_path = os.getenv("MODEL_PATH", "/data2/llms/Qwen2.5-14B-Instruct")
    # group_size 也可以顺便做成可配置
    group_size = int(os.getenv("GROUP_SIZE", "64"))


# 全局变量
MODEL = None
TOKENIZER = None
WARMED_UP = False  # 标记是否已经预热，避免重复预热


def warmup_model(num_new_tokens: int = 16):
    """
    进行一次极短的 dummy 推理，用于预热 CUDA kernel / KV cache 等，
    降低用户第一轮对话的延迟。
    """
    global MODEL, TOKENIZER, WARMED_UP
    if MODEL is None or TOKENIZER is None:
        return

    if WARMED_UP:
        return

    print(">>> [System] 正在进行模型预热 (warmup) ...")
    dummy_prompt = "Warmup."

    try:
        inputs = TOKENIZER(dummy_prompt, return_tensors="pt").to(MODEL.device)
        with torch.no_grad():
            _ = MODEL.generate(
                **inputs,
                max_new_tokens=num_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=TOKENIZER.eos_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )

        # 同步一下，确保所有 CUDA kernel 已经执行完毕
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        WARMED_UP = True
        print(">>> [System] 预热完成，模型就绪 ✅")
    except Exception as e:
        # 预热失败不影响主流程，只打印一下
        print(f"!!! [System] 预热过程中出现异常，但不会影响正常推理: {e}")


def init_model():
    """初始化模型，并在加载完成后进行一次预热"""
    global MODEL, TOKENIZER
    if MODEL is not None:
        return

    print(">>> [System] 正在加载 Tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(Args.model_path, use_fast=True)

    print(">>> [System] 正在加载基础模型 (CPU)...")
    # 注意：这里加载到 CPU 内存中，不要急着 .cuda()
    model = load_llama_or_qwen_model(Args.model_path)

    print(f">>> [System] 正在加载量化权重: {Args.quant_pt}...")
    state_dict = torch.load(Args.quant_pt, map_location="cpu")  # 确保加载到 CPU

    print(">>> [System] 正在应用自定义 2-bit 量化层 (CPU)...")
    # 在 CPU 上完成结构替换，避免显存碎片
    replace_llama_with_packed_w2_layers(model, state_dict, Args.group_size)

    print(">>> [System] 正在将模型移动到 GPU...")
    model.cuda()
    model.eval()

    MODEL = model
    print(">>> [System] 模型加载完毕！")

    # 🔥 模型加载完毕后，立刻进行一次轻量预热
    warmup_model()


# 启动时就尝试加载模型 + 预热，失败时保持 UI 可用（只是不执行真实推理）
try:
    init_model()
except Exception as e:
    print(f"!!! 模型加载失败 (仅 UI 模式，可查看报错): {e}")


# ==============================================================================
# 3. Chainlit 交互逻辑
# ==============================================================================

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    if not cl.user_session.get("id"):
        cl.user_session.set("id", str(int(time.time())))

    await cl.Message(
        content=f"👋 你好！我是 NUDT Deep Mind 研发的对话助手 (运行于 GPU {TARGET_GPUS})～"
    ).send()


@cl.on_chat_resume
async def on_resume(thread):
    steps = thread["steps"]
    history = []
    last_user_input = ""
    for step in steps:
        if step["type"] == "user_message":
            last_user_input = step["output"]
        elif step["type"] == "assistant_message":
            if last_user_input:
                history.append((last_user_input, step["output"]))
                last_user_input = ""
    cl.user_session.set("history", history)


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    history = cl.user_session.get("history", [])

    # 1. 构建 Prompt（多轮对话拼接）
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn[0]})
        messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": user_input})

    try:
        full_prompt = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 万一模板不可用，就简单拼接
        full_prompt = ""
        for m in messages:
            full_prompt += f"{m['role']}: {m['content']}\n"
        full_prompt += "assistant:"

    # 2. 推理
    msg = cl.Message(content="")
    await msg.send()
    response_text = ""

    if MODEL is None:
        # 模型没加载成功时，给一个占位回复，避免前端挂死
        time.sleep(1)
        response_text = f"【模拟】收到了你的消息：{user_input}\n当前后台模型尚未成功加载，请检查日志。"
        await msg.stream_token(response_text)
    else:
        # 确保已经预热过（理论上在 init_model 中已经预热一次，这里只是兜底）
        warmup_model()

        # 输入移动到模型所在设备
        inputs = TOKENIZER(full_prompt, return_tensors="pt").to(MODEL.device)

        streamer = TextIteratorStreamer(
            TOKENIZER,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            streamer=streamer,
            max_new_tokens=2048,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id,
            temperature=0.7,
            do_sample=True,
        )

        thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            response_text += new_text
            await msg.stream_token(new_text)

        thread.join()

    history.append((user_input, response_text))
    cl.user_session.set("history", history)
    await msg.update()

    session_id = cl.user_session.get("id")
    save_log_to_json(session_id, user_input, response_text)
