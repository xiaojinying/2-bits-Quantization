import torch
from torch import nn 
from .Quant_kernels import preprocess_weights_int2_for_weight_only, asymm_qw2_gemm
from typing import Optional



class PackedW2A16Linear(nn.Module):
    """
    权重 2bit、激活 16bit 的自定义 Linear 层封装。

    使用方式（典型流水）：
        1. 构造：
            layer = PackedW2A16Linear(in_features, out_features, bias=old_linear.bias,
                                 group_size=64, name="model.layers.0.self_attn.q_proj")
        2. 量化加载：
            layer.weight = packed_int2_weight.t().contiguous()   # int8, 低2bit有效
            layer.scale  = scale_tensor                          # [out, num_groups]
            layer.zp     = zp_tensor                             # [out, num_groups]
        3. 前向：
            out = layer(x)  # 内部会在第一次 forward 时做预处理 + GEMM 调用
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor],
        group_size: int
    ) -> None:
        super().__init__()

        # 记录 in/out 维度（方便 debug）
        self.k = in_features
        self.n = out_features

        # 是否原始 Linear 有 bias
        self.with_bias: bool = bias is not None

        # 权重在外部赋值：packed int2（int8 存储）
        #   形状约为 [out_features, in_features] 或对应的打包格式
        self.weight: Optional[torch.Tensor] = None

        # bias 处理：
        #  - 若源 Linear 有 bias，则转为 float16 保存；
        #  - 若没有，则用全 0 bias，保持接口统一。
        if isinstance(bias, torch.Tensor):
            # 不强制放到 GPU，后面 forward 会统一按 input.device 搬迁
            self.bias = bias.detach().to(dtype=torch.float16)
        else:
            self.bias = torch.zeros(out_features, dtype=torch.float16)

        # 量化参数：scale / zero-point
        #   预期由外部在替换线性层时赋值：
        #       self.scale: [out, num_groups]  或  [out, k/group_size]
        #       self.zp   : 同上
        self.scale: Optional[torch.Tensor] = None
        self.zp: Optional[torch.Tensor] = None

        # 每组多少个权重（group-wise 量化）
        self.group_size: int = group_size

        # 标记权重是否已经做过一次 C++ 预处理（Layout + pack）
        self.weight_processed: bool = False


    def _ensure_preprocessed(self, input_device: torch.device) -> None:
        """
        在第一次 forward 时调用：
        - 检查 weight / scale / zp 是否已经被赋值
        - 调用 C++ 的 dQ_preprocess_weights_int2_for_weight_only 进行预处理
        - 把预处理后的权重移动到 input 所在设备
        """
        if self.weight_processed:
            return

        if self.weight is None:
            raise RuntimeError(
                f"[{self.name}] LinearW2A16.forward: 需要先为 `weight` 赋值再调用 forward"
            )

        if self.with_bias and self.bias is None:
            raise RuntimeError(
                f"[{self.name}] LinearW2A16.forward: 配置了 with_bias=True 但 `bias` 为空"
            )

        if self.scale is None:
            raise RuntimeError(
                f"[{self.name}] LinearW2A16.forward: 需要事先赋值 `scale`（量化比例因子）"
            )

        if self.zp is None:
            raise RuntimeError(
                f"[{self.name}] LinearW2A16.forward: 需要事先赋值 `zp`（量化零点）"
            )

        # C++ 预处理接口目前假定权重在 CPU 上，且为连续内存
        weight_cpu = self.weight.detach().to("cpu").contiguous()
        processed_weight_cpu = preprocess_weights_int2_for_weight_only(weight_cpu)

        # 预处理后的权重放回到 Linear 内部，并迁移到与输入相同的设备
        self.weight = processed_weight_cpu.to(input_device)

        # scale / zp / bias 也一并搬到对应设备（防止调用时 device 不一致）
        self.scale = self.scale.to(input_device)
        self.zp = self.zp.to(input_device)
        self.bias = self.bias.to(input_device)

        self.weight_processed = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向计算：
        - 第一次调用时会执行权重预处理（打包 + 布局变换）
        - 调用 C++ 的 dQ_asymm_qw2_gemm 完成 GEMM + dequant
        """
        # 确保所有内部张量已经按正确格式 / 设备预处理完成
        self._ensure_preprocessed(input.device)

        # 调用 C++ kernel：
        #   input : [*, k]
        #   weight: 打包好的 int2 权重（已预处理）
        #   scale / zp: group-wise 量化参数
        #   bias  : float16
        #   group_size: 分组大小
        output = asymm_qw2_gemm(
            input,
            self.weight,
            self.scale,
            self.zp,
            self.bias,
            self.group_size,
        )

        return output


    
