"""
The code in this file is built on thr top of OPTQ, please visit:
https://github.com/IST-DASLab/gptq
for their origin contribution

SPDX-License-Identifier: Apache-2.0

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates.
"""
import time
import os
import torch
import torch.nn as nn
from Quantization import LayerQuantizerHelper,finetune_block_scales,collect_quant_layers,move_to_device,WeightQuantizer


import shutil
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
import time



@torch.no_grad()
def run_sequential_quantization(cfg, model, layer_list, calib_dataloader, dev):
    """
    顺序逐层量化入口：
    1. 先用一个 Catcher 抓取每一层的输入 (cache)
    2. 对每一层构建 decoupleQ + MoQQuantizer，使用校准 batch 收集 Hessian 近似
    3. 调用 startquant 完成权重量化
    4. 可选做 blockwise_minimize_layer 进行 block 粒度微调
    """
    # print(cfg)
    print("start quant====")
    input_cache = []

    class ActivationCatcher(nn.Module):
        """
        Hook 住第一层输入，把所有输入 batch 保存到 CPU 上。
        通过抛出异常提前终止前向，从而避免真正跑完整个模型。
        """

        def __init__(self, wrapped_module):
            super().__init__()
            self.wrapped_module = wrapped_module

        def forward(self, *args, **kwargs):
            inputs = [list(args), kwargs]
            # 将 inputs 统一搬到 CPU，避免显存爆掉
            input_cache.append(move_to_device(inputs, "cpu"))
            # 利用异常中断后续 forward
            raise ValueError

    # 用 ActivationCatcher 替代第 0 层
    layer_list[0] = ActivationCatcher(layer_list[0])

    use_cache_flag = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 只把 embedding 和最后的 norm 搬到目标设备上
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)

    model.eval()
    torch.cuda.empty_cache()
    model.requires_grad_(False)

    # 目前 mask 用不到，先占位
    masks = [None] * len(calib_dataloader)

    # 遍历校准集，触发 ActivationCatcher 将输入缓存到 input_cache
    for batch in calib_dataloader:
        batch = move_to_device(batch, dev)
        try:
            model(batch)
        except ValueError:
            # 预期中的中断，不是错误
            pass

    del calib_dataloader, batch
    gc.collect()

    # 恢复原始第 0 层
    layer_list[0] = layer_list[0].wrapped_module
    model = model.cpu()
    inps = input_cache
    torch.cuda.empty_cache()

    print('Ready.')
    layer_index_shift = 0  # 预留 shift，兼容 decoder 层偏移
    quant_state_dict = {}
    layer_outputs = []

    for layer_idx in range(len(layers)):
        t_layer_start = time.time()
        layer_module = layers[layer_idx]

        # 收集当前层里所有 Linear / Conv / Conv1D（通过 collect_target_layers 实现）
        sub_layer_dict = collect_quant_layers(layer_module)

        # true_sequential 的时候按项目中的 grouping 顺序量化
        if cfg.true_sequential:
            sequential_groups = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj'],
            ]
        else:
            # 否则就把这一层所有可量化权重整体作为一组
            sequential_groups = [list(sub_layer_dict.keys())]
        ts = int(time.time()) 
        dir_name = f"tmp_blockwise_{ts}"
        # 对每一个 group 依次量化
        for group_idx, layer_name_group in enumerate(sequential_groups):
            # subset: 当前 group 内需要量化的子模块
            sub_layer_group = {n: sub_layer_dict[n] for n in layer_name_group}
            quant_wrappers = {}

            # 为当前 group 中的每一个子层构造 quant 包装器和 MoQQuantizer
            for sub_name in sub_layer_group:
                quant_wrappers[sub_name] = LayerQuantizerHelper(
                    sub_layer_group[sub_name], layer_name=f"layer.{layer_idx}.{sub_name}"
                )
                quant_wrappers[sub_name].quantizer = WeightQuantizer()
                quant_wrappers[sub_name].quantizer.configure(
                    cfg.qbits,
                    perchannel=True,
                    sym=not cfg.asym,
                )
                # mask 目前只是占位，按项目接口保留
                sub_layer_group[sub_name].mask = [None]

            def build_forward_hook(sub_layer_name):
                """
                构建 forward_hook，用于：
                - 把该子层的输入 / 输出喂给 LayerQuantizerHelper.accumulate_batch_stats
                - 进而累积 Hessian 近似
                """

                def forward_hook(module, inp, out):
                    quant_wrappers[sub_layer_name].accumulate_batch_stats(
                        inp[0].data, out.data, module.mask[0]
                    )

                return forward_hook

            hook_handles = []
            for sub_name in sub_layer_group:
                handle = sub_layer_group[sub_name].register_forward_hook(
                    build_forward_hook(sub_name)
                )
                hook_handles.append(handle)

            # 把当前层搬到 GPU，利用 inps 做一遍前向，收集统计信息
            layer_module = layer_module.to(dev)
            for idx, batch_inputs in enumerate(inps):
                batch_inputs = move_to_device(batch_inputs, dev)
                out = layer_module(*(batch_inputs[0]), **batch_inputs[1])
                if group_idx == 0 and cfg.blockwise_minimize_lr > 0:
                    # 保存中间输出，为 blockwise_minimize 使用
                    os.makedirs(f"./{dir_name}", exist_ok=True)
                    out_cpu = {"out": move_to_device(out, "cpu")}
                    torch.save(out_cpu, f"./{dir_name}/out_{idx}.pth")
                del out
            layer_module = layer_module.cpu()

            # 移除 forward hook，防止后续多次触发
            for h in hook_handles:
                h.remove()

            # 每个子层分别调用 LayerQuantizerHelper.startquant 进行权重量化
            for sub_name in layer_name_group:
                del sub_layer_group[sub_name].mask
                print(layer_idx, sub_name)
                print('Quantizing ...')
                t_quant_start = time.time()
                torch.cuda.empty_cache()

                scale_out, zero_out, w_int, loss = quant_wrappers[sub_name].run_quantization(
                    device=dev,
                    group_size=cfg.group_size,
                    is_symmetric=not cfg.asym,
                    max_iter_num=cfg.max_iter_num,
                    inner_iters_for_round=cfg.inner_iters_for_round,
                    iters_before_round=cfg.iters_before_round,
                    learning_rate=cfg.lr,
                    actorder=cfg.act_order,
                    round_fn=cfg.round_fn,
                )
                t_quant_end = time.time()
                print(
                    f"time cost {t_quant_end - t_quant_start}, "
                    f"model.decoder.layers.{layer_idx + layer_index_shift}.{sub_name}.weight, "
                    f"loss is {loss.mean().item()}"
                )
                print()

                # 把 scale / zero / int 权重保存到 quant_state_dict 中，后续写回 ckpt
                scale_list = [s.cpu() for s in [scale_out, zero_out]]
                quant_state_dict[f"{layer_idx + layer_index_shift}.{sub_name}.weight"] = {
                    "scales": scale_list,
                    "weights": w_int.cpu(),
                    "loss": loss.cpu(),
                }

                # 释放 LayerQuantizerHelper 内部缓存（Hessian 等）
                quant_wrappers[sub_name].free()
                quant_wrappers[sub_name].quantizer.free()
                del quant_wrappers[sub_name], scale_out, zero_out, w_int

        layer_outputs = []

        # blockwise 最小化：对整个层块做一次小规模微调，进一步降低量化误差
        if cfg.blockwise_minimize_lr > 0:
            t_block_start = time.time()
            finetune_block_scales(
                cfg, quant_state_dict, layer_module, inps, dev, layer_idx + layer_index_shift, masks,dir_name
            )
            shutil.rmtree(f"./{dir_name}")
            print("time cost for block minimization:", time.time() - t_block_start)

        # 重新跑一遍当前层，把量化后的输出作为下一层的输入
        layer_module = layer_module.to(dev)
        for batch_inputs in inps:
            batch_inputs = move_to_device(batch_inputs, dev)
            layer_outputs.append(move_to_device(layer_module(*(batch_inputs[0]), **batch_inputs[1]), "cpu"))

        layers[layer_idx] = layer_module.cpu()
        del layer_module
        del quant_wrappers
        torch.cuda.empty_cache()

        # 将当前层的输出写回 inps，用于下一层量化
        for j in range(len(layer_outputs)):
            inps[j][0][0] = layer_outputs[j][0]
        del layer_outputs

        print(f"quant layer {layer_idx} done! time cost {time.time() - t_layer_start}")
        print()

    del inps
    model.config.use_cache = use_cache_flag
    return quant_state_dict



if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--quant_pt', type=str,default=None,
        help='quant model to infer'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.',
        default='c4'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--group-size', type=int, default=64,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Whether to save the fake and true checkpoints'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--quant-method', type=str, choices=['optq', 'moq', 'moq_sequential', ""], default="",
        help='the quant method'
    )
    parser.add_argument(
        '--loss-thr', type=float, default=0.02,
        help='The loss threshold to exit loop'
    )
    parser.add_argument(
        '--max-iter-num', type=int, default=3,
        help='The max iter num for the whole loop'
    )
    parser.add_argument(
        '--inner-iters-for-round', type=int, default=50,
        help='the number of iters for PGD when use first level approximation'
    )
    parser.add_argument(
        '--iters-before-round', type=int, default=0,
        help='the number of iters before entering PGD when use first level approximation'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='the learning rate for PGD'
    )
    parser.add_argument(
        '--round-fn', type=str, choices=["gptq", "train"], default="train",
        help='the quant method'
    )
    parser.add_argument(
        '--blockwise-minimize-lr', type=float, default=-1.0,
        help='the learning rate for block minimization'
    )
    parser.add_argument(
        '--blockwise-minimize-wd', type=float, default=1.0e-6,
        help='the weight decaying rate for block minimization'
    )
    parser.add_argument(
        '--blockwise-minimize-epoch', type=int, default=3,
        help='the number of epoch for training the float point part'
    )
    parser.add_argument(
        '--train-LN', action='store_true',
        help='Whether to train the parameters in norm'
    )
    parser.add_argument(
        '--train-bias', action='store_true',
        help='Whether to train the bias in linear layer'
    )
    parser.add_argument(
        '--inference', action='store_true',
        help="inference trained model"
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
        help='Tasks to evaluate on LM Eval.'
    )
    parser.add_argument(
        '--lm_eval_batch_size', type=int, default=8,
        help='Batch size for evaluation with lm eval harness.'
    )

    parser.add_argument(
        '--ass_model', type=str,default=None
    )
    parser.add_argument(
        '--ass_model_quant_path', type=str,default=None
    )

    args = parser.parse_args()
    print(args)

    # 对称 / 非对称标记转换
    args.asym = not args.sym
    args.qbits = args.wbits


    # 加载原始 HF 模型
    model = load_llama_or_qwen_model(args.model,args.inference)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model.eval()


    

    if args.inference:
        # 推理模式：从 true_quant ckpt 中恢复权重并替换为 W2 线性层
        if args.quant_pt!=None:
            state_dict = torch.load(f"{args.quant_pt}")
            replace_llama_with_packed_w2_layers(model, state_dict, args.group_size)

        prompts = ["who are you?"]

        inputs = tokenizer(prompts, return_tensors="pt").to('cuda:0')
        model.eval()

        repeat = 10
        batch_size=1

        gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        )
        if args.ass_model!=None:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                args.ass_model,
                torch_dtype='auto',
                device_map=None,
            ).eval()
            state_dict=torch.load(f"{args.ass_model_quant_path}")
            replace_llama_with_packed_w2_layers(assistant_model, state_dict, args.group_size)
            assistant_model=assistant_model.to('cuda:0')
            gen_kwargs['assistant_model']=assistant_model
            gen_kwargs['do_sample']=False

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        with torch.inference_mode():
            # 做两次 warmup，避免第一次运行的冷启动干扰计时
            _ = model.generate(**inputs, **gen_kwargs)
            _ = model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()        
            t_start = time.time()
            for _ in range(repeat):
                model_output = model.generate(**inputs,**gen_kwargs)
                torch.cuda.synchronize()
            t_end = time.time()
        peak_allocated = torch.cuda.max_memory_allocated()       
        peak_reserved = torch.cuda.max_memory_reserved()         

        peak_allocated_gb = peak_allocated / (1024 ** 3)
        peak_reserved_gb = peak_reserved / (1024 ** 3)

        print(f"[GPU Memory] peak allocated: {peak_allocated_gb:.2f} GB ({peak_allocated/1024**2:.0f} MB)")
        print(f"[GPU Memory] peak reserved : {peak_reserved_gb:.2f} GB ({peak_reserved/1024**2:.0f} MB)")
        out_text = tokenizer.batch_decode(model_output)
        print(f"out_text: {out_text}")
        infer_time_ms = (t_end - t_start) * 1000
        print(
            f"inference speed: e2e {infer_time_ms / repeat} ms, "
            f"pertoken {infer_time_ms / model_output.shape[-1] / repeat} ms"
        )

        # 下面是 PPL 评测
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        dev = "cuda:0"

        datasets = ['wikitext2', 'c4']
        for dataset in datasets:
            print(dataset)
            dataloader, testloader = get_loaders(
                dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
            )
            evaluate_llama_perplexity(model, testloader, dev)

    else:
        # 训练 / 量化模式：先收集校准数据，再调用顺序量化
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        dev = "cuda"
        layers = model.model.layers

        # dataloader 里每个元素形如 (inputs, labels, ...)，这里只取 inputs
        calib_batches = [b[0] for b in dataloader]
        tick = time.time()

        quant_state_dict = run_sequential_quantization(
            args,
            model,
            layers,
            calib_batches,
            dev=dev,
        )
        from LoRA import lora_finetune_and_merge_model
        result=lora_finetune_and_merge_model(
    model=model,
    tokenizer=tokenizer,
    model_path=args.model,

)       
        model=result["merged_model"]
        layers = model.model.layers
        quant_state_dict = run_sequential_quantization(
            args,
            model,
            layers,
            calib_batches,
            dev=dev,
        )

        if args.save:
            export_quantized_checkpoints(args, model, quant_state_dict, prefix="model.layers.")

        print("The quantization duration is ", (time.time() - tick) / 3600)

        datasets = ['wikitext2', 'c4']

        for dataset in datasets:
            print(dataset)
            dataloader, testloader = get_loaders(
                dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
            )
            evaluate_llama_perplexity(model, testloader, dev)
        # 如需 lm-eval-harness，可以解开：
        # run_lm_eval_harness(model, args, dev)
