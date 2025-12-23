import sys
sys.path.append(".")

import os
import random
import copy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Any, Union
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

IGNORE_INDEX = -100


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataCollatorForCausalLM(object):
    """
    输入样本格式:
        {
            "input":  "prompt / source 文本",
            "output": "target 文本"
        }
    """
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 拆出 source / target
        sources = [example["input"] for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # 编码 source
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
        )
        # 编码 target（不加 bos 之类的特殊符号）
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = []
        labels = []

        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"],
            tokenized_targets["input_ids"],
        ):
            if not self.predict_with_generate:
                # 训练模式：直接把 source + target 拼在一起
                ids = torch.tensor(tokenized_source + tokenized_target)
                input_ids.append(ids)

                if not self.train_on_source:
                    # source 部分不参与 loss，填 IGNORE_INDEX
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                # 纯生成模式，只喂 source
                input_ids.append(torch.tensor(tokenized_source))

        # padding
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )

        data_dict: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels

        return data_dict


def _print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}
def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = PROMPT_DICT["prompt_input"]
    else:
        prompt_format = PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}
def get_dataset():
    dataset = load_dataset("tatsu-lab/alpaca")
    dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    train_dataset = dataset['train']
    train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    train_dataset=train_dataset.shuffle(seed=42)
    return train_dataset

def _prepare_model_for_training(model: nn.Module) -> nn.Module:
    """
    冻结基座参数 + 把 fp16/bf16 转成 fp32 + 打开 gradient checkpointing。
    只给 LoRA 的 adapter 层梯度。
    """
    # 1. 冻结所有参数
    for _, param in model.named_parameters():
        param.requires_grad = False

    # 2. dtype 统一到 fp32，提高数值稳定性
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    # 3. 让输入需要 grad，方便 LoRA 训练
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 4. gradient checkpointing 节省显存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model


def _load_base_model_and_tokenizer(
    model_path: str,
    ckpt_path: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
):
    """
    根据模型路径加载 base model + tokenizer，
    可选再加载一个额外 ckpt（比如 true_quant 权重）。
    """
    print(f"[INFO] Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype,
    )

    print(f"[INFO] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 保证有 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    # 可选额外 ckpt
    if ckpt_path is not None and ckpt_path != "":
        print(f"[INFO] Loading extra checkpoint weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")

    return model, tokenizer


def lora_finetune_and_merge_model(
    model: nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    train_steps: int = 1000,
    model_path: str = "",   
    seed: int = 42,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    device_map: str = "auto",      
) -> Dict[str, Any]:

    # 固定随机种子
    set_seed(seed)

    # 保证 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))

    # 1) 冻结 base model 等准备工作
    model = _prepare_model_for_training(model)


    # 2) 准备训练数据（这里假设你有一个无参的 get_dataset()）

    train_dataset = get_dataset()
    if 'llama' in model_path.lower():
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    elif 'qwen' in model_path.lower():
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj","o_proj"]

    # 3) 配置 LoRA

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        
    )
    model = get_peft_model(model, peft_config)
    _print_trainable_parameters(model)

    # 4) 训练配置
    num_gpus = max(1, torch.cuda.device_count())
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 2
    warmup_steps = max(1, int(train_steps * 0.05))

    # 简单根据算力判断用 bf16 还是 fp16
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        cc_major, _ = torch.cuda.get_device_capability(0)
        if cc_major >= 8:
            use_bf16 = True
        else:
            use_fp16 = True

    print(
        f"[INFO] Training config: steps={train_steps}, "
        f"bs={per_device_train_batch_size}, grad_acc={gradient_accumulation_steps}, "
        f"gpus={num_gpus}, bf16={use_bf16}, fp16={use_fp16}"
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        max_steps=train_steps  ,

        output_dir='./outputs',  
        optim="adamw_torch",
        remove_unused_columns=False,

        bf16=use_bf16,
        fp16=use_fp16,


        # logging_strategy="no",
        # disable_tqdm=True,
        report_to="none",


        save_strategy="no",
    )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=1024,
        target_max_len=256,
        train_on_source=False,
        predict_with_generate=False,
    )

    # 5) 构建 Trainer

    if hasattr(model, "config"):
        model.config.use_cache = False  # 与 gradient checkpointing 配套（如果你在 _prepare_model_for_training 里开了）

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    # 6) 训练 LoRA

    trainer.train()

    # 7) 合并 LoRA 权重，得到最终完整模型（只在内存中）

    merged_model = trainer.model.merge_and_unload()


    return {
        "merged_model": merged_model,
        "tokenizer": tokenizer,
    }

