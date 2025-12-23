/*
Copyright (2024) Bytedance Ltd. and/or its affiliates
*/
#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "fpA_intB_gemm/fpA_intB_gemm.h"

#include "cutlass_preprocessors.h"
#include "cutlass/numeric_types.h"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#include <vector>
#endif  // CUDA_VERSION >= 11000

// 给 torch 命名空间起个别名 th
namespace th = ::torch;
// 直接使用 tensorrt_llm 中 cutlass_kernels 里的符号
using namespace ::tensorrt_llm::kernels::cutlass_kernels;

// 从 Tensor 中拿出底层数据指针的工具函数（可写版本）
template <typename T>
inline T* get_ptr(th::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

// 从 Tensor 中拿出底层数据指针的工具函数（只读版本）
template <typename T>
inline const T* get_ptr(const th::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

// GEMM 接口的抽象基类：用于做 runtime 动态多态（不同 T/WeightType/QuantOp 的统一封装）
class ITrtllmFpAIntBGemm {
 public:
  ITrtllmFpAIntBGemm() {}
  virtual ~ITrtllmFpAIntBGemm() {}

  // 纯虚函数：具体的 GEMM 实现需要自己实现 forward
  // A: 激活，B: 量化权重，C: 输出
  // scale/zp: 权重量化用的 scale / zero-point
  // bias: 线性层 bias
  // m, n, k: GEMM 维度 (m x k) * (k x n)
  // group_size: 权重量化的分组大小（如 per-group scale/zp）
  virtual void forward(const th::Tensor& A, const th::Tensor& B, th::Tensor& C,
                       const th::Tensor& scale, const th::Tensor& zp,
                       const th::Tensor& bias, const int64_t m, const int64_t n,
                       const int64_t k, int group_size) = 0;

};

  
// 具体的 GEMM 实现：
// T         : 计算精度类型（如 half / bfloat16）
// WeightType: 权重存储类型（这里是 cutlass::uint2b_t，对应 2bit）
// QuantOp   : cutlass 里定义的权重仅量化算子类型
template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class TrtllmFpAIntBGemm : public ITrtllmFpAIntBGemm {
 public:
  TrtllmFpAIntBGemm() {}

  ~TrtllmFpAIntBGemm() override {}

  void forward(const th::Tensor& A, const th::Tensor& B, th::Tensor& C,
               const th::Tensor& scale, const th::Tensor& zp,
               const th::Tensor& bias, const int64_t m, const int64_t n,
               const int64_t k, int group_size) override {
    // 获取当前 CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // 输入激活、权重、scale、zero-point、bias 的设备指针
    const T* input_act_ptr = get_ptr<const T>(A);
    const WeightType* weight_ptr = get_ptr<const WeightType>(B);
    const T* scales_ptr = get_ptr<const T>(scale);
    const T* zp_ptr = get_ptr<const T>(zp);
    const T* bias_ptr = get_ptr<const T>(bias);
    //const T* res_ptr = nullptr;  // 目前没直接使用，保留以兼容接口

    // 计算 fused GEMM + dequant 需要的 workspace 大小（字节）
    const int64_t ws_bytes = fused_gemm_runner.getWorkspaceSize(m, n, k);

    // 分配 workspace 张量（int8 类型，只是用来占用内存，真正语义由 kernel 决定）
    auto ws_tensor =
        th::empty({ws_bytes},
                  th::dtype(th::kInt8).device(th::kCUDA).requires_grad(false));

    // 输出张量指针
    T* output_tensor_ptr = get_ptr<T>(C);
    // workspace 底层指针
    char* ws_ptr = get_ptr<char>(ws_tensor);

    // 从 runner 拿到 kernel 配置（多种实现/调优参数）
    auto configs = fused_gemm_runner.getConfigs();
    // 默认配置在 L20 上不能跑，我们这里硬编码修改 stage 数
    // TODO: 从根源修复配置，避免硬编码
    configs[0].stages = 3;

    // 调用 fused GEMM + dequant kernel：
    // A: input_act_ptr
    // B: weight_ptr (int2 packed)
    // scales/zp: 每group的量化参数
    // bias_ptr: bias
    // 输出: output_tensor_ptr
    fused_gemm_runner.gemm(input_act_ptr, weight_ptr, scales_ptr, zp_ptr,
                              bias_ptr, output_tensor_ptr, m, n, k,
                              group_size, configs[0], ws_ptr,
                              ws_bytes, stream);
  }

 private:
  // Cutlass 封装的 FpA + IntB GEMM runner，模板参数决定数据类型和量化算子
  CutlassFpAIntBGemmRunner<T, WeightType, QuantOp> fused_gemm_runner;
};

// ==================== w2 接口：int2 权重 + A16 计算 ====================

// w2 接口：异步非对称权重仅量化 GEMM（权重 2bit，激活 FP，bias 也在这算）
// A: [*, k]  输入激活，最后一维是 k
// B: [k/4, n] 打包后的 int2 权重（每字节 4 个 2bit）
// scale/zp: 权重量化参数
// bias: 输出 bias
// group_size: 用于 weight-only 量化的分组大小
th::Tensor asymm_qw2_gemm(const th::Tensor& A, const th::Tensor& B,
                                  const th::Tensor& scale, const th::Tensor& zp,
                                  const th::Tensor& bias, int group_size) {
  // m 是把 A 除了最后一维之外所有维度展平后的 batch size
  int64_t m = 1;
  // 因为 B 是 int2 packed，每个输出通道由 2bit 表示，
  // 这里 n = B.size(1) * 4 表示还原后的输出通道数
  const int64_t n = B.size(1) * 4;
  const int64_t k = A.size(-1);

  // 构建输出张量的 shape：等同于 A 前面的所有维度 + n
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < A.dim() - 1; ++i) {
    m *= A.size(i);
    out_shape.push_back(A.size(i));
  }
  out_shape.push_back(n);

  // 计算 dtype 与 A 保持一致（如 half / bfloat16）
  auto compute_dtype = A.scalar_type();

  // 用基类指针做多态，根据 dtype 决定具体模板实例
  std::unique_ptr<ITrtllmFpAIntBGemm> qgemm;

  if (compute_dtype == at::ScalarType::Half) {
    // 半精度计算：T=half，权重为 cutlass::uint2b_t，量化算子为微粒度 scale+zp
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
                half, cutlass::uint2b_t,
                cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#if CUDA_VERSION >= 11000
  else if (compute_dtype == at::ScalarType::BFloat16) {
    // bfloat16 计算：T=__nv_bfloat16
    qgemm = std::make_unique<TrtllmFpAIntBGemm<
                __nv_bfloat16, cutlass::uint2b_t,
                cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
  }
#endif
  else {
    // 不支持的计算类型，报错
    std::string err_msg =
        "Unsupported compute type " + std::string(at::toString(compute_dtype));
    throw std::runtime_error(err_msg);
  }

  // 按计算 dtype 和输出形状分配输出张量（在 GPU 上）
  auto output_tensor = th::empty(
      out_shape,
      th::dtype(compute_dtype).device(th::kCUDA).requires_grad(false));

  // 调用多态的 GEMM 实现
  qgemm->forward(A, B, output_tensor, scale, zp, bias, m, n, k, group_size);

  return output_tensor;
}

// ==================== 权重预处理：int2 打包 + cutlass 布局转换 ====================

// 预处理 int2 权重，使其满足 cutlass weight-only kernel 的布局要求
th::Tensor preprocess_weights_int2_for_weight_only(const th::Tensor& in) {
  // 输入应该是 int8/uint8，数值范围 0~3
  TORCH_CHECK(
      in.scalar_type() == th::kChar || in.scalar_type() == th::kByte,
      "preprocess_weights_int2_for_weight_only expects int8/uint8 tensor");

  const int8_t* in_ptr = reinterpret_cast<const int8_t*>(in.data_ptr());

  // 记录维度
  const int dims = in.dim();
  std::vector<size_t> shape;
  shape.reserve(dims);
  for (int i = 0; i < dims; ++i) {
    shape.push_back(static_cast<size_t>(in.size(i)));
  }

  auto dtype = QuantType::W2_A16;

  // 支持 [rows, cols] 或 [experts, rows, cols]
  const size_t num_experts = (shape.size() == 2) ? 1 : shape[0];
  const size_t num_rows    = (shape.size() == 2) ? shape[0] : shape[1];
  const size_t num_cols    = (shape.size() == 2) ? shape[1] : shape[2];

  constexpr int bits_in_type      = 2;               // 2bit
  const size_t bytes_per_out_col  = (num_cols * bits_in_type) / 8;  // 等价于 num_cols / 4

  // 原始矩阵的元素个数（按 2D 展开）
  const size_t input_mat_size     = num_rows * num_cols;
  // 打包后的矩阵元素个数（以 byte 为单位）
  const size_t quantized_mat_size = num_rows * bytes_per_out_col;

  // 中间缓冲区：存放打包后的 int2 字节（注意这里仍保持原逻辑：
  //    expert 维度按块连续排列，每个 expert 是 [num_rows, bytes_per_out_col]）
  std::vector<int8_t> weight_buf(num_experts * quantized_mat_size);
  int8_t* unprocessed_quantized_weight = weight_buf.data();

  // 按 expert / row 两层循环，内层按 4 个 2bit -> 1 byte 打包
  for (size_t expert = 0; expert < num_experts; ++expert) {
    const int8_t* expert_in_base =
        in_ptr + expert * input_mat_size;             // [num_rows, num_cols]
    int8_t* expert_out_base =
        unprocessed_quantized_weight + expert * quantized_mat_size; // [num_rows, bytes_per_out_col]

    for (size_t row = 0; row < num_rows; ++row) {
      const int8_t* row_in  = expert_in_base + row * num_cols;
      int8_t* row_out       = expert_out_base + row * bytes_per_out_col;

      // 主循环：每次处理 4 个权重 → 1 byte
      size_t col = 0;
      size_t byte_idx = 0;

      // 这里假设 num_cols 通常是 4 的倍数（大模型里基本如此），
      // 若不是 4 的倍数，尾巴部分交给后面的补丁逻辑处理。
      for (; col + 4 <= num_cols && byte_idx < bytes_per_out_col; col += 4, ++byte_idx) {
        const int8_t w0 = row_in[col + 0];
        const int8_t w1 = row_in[col + 1];
        const int8_t w2 = row_in[col + 2];
        const int8_t w3 = row_in[col + 3];

        // 只取低 2bit，然后按 [w0, w1, w2, w3] 顺序打包成一个 byte
        const int8_t packed =
            (w0 & 0x03) |
            ((w1 & 0x03) << 2) |
            ((w2 & 0x03) << 4) |
            ((w3 & 0x03) << 6);

        row_out[byte_idx] = packed;
      }

      // 万一 num_cols 不是 4 的倍数，处理剩余的 1~3 个元素
      // （这个逻辑和你原来的 if(input_idx < num_cols) 等价）
      if (col < num_cols && byte_idx < bytes_per_out_col) {
        int8_t packed_tail = 0;
        int shift = 0;
        for (; col < num_cols; ++col, shift += 2) {
          packed_tail |= (row_in[col] & 0x03) << shift;
        }
        row_out[byte_idx] = packed_tail;
      }
    }
  }

  // out 的 shape 仍然是 [num_experts, num_rows, num_cols/4]，dtype=int8，放在 CPU 上
  th::Tensor out =
      th::empty({static_cast<long>(num_experts),
                 static_cast<long>(num_rows),
                 static_cast<long>(bytes_per_out_col)},
                th::dtype(th::kInt8).device(th::kCPU));

  int8_t* out_ptr = reinterpret_cast<int8_t*>(out.data_ptr());

  // 交给 cutlass 相关的预处理，做 layout transform / swizzle 等
  preprocess_weights_for_mixed_gemm(
      out_ptr,                      // 目标：cutlass 友好布局
      unprocessed_quantized_weight, // 源：我们刚刚 pack 好的 row-major byte 矩阵
      shape,
      dtype);

  // 如果原始 shape 是 2D，把 expert 维度 squeeze 掉，恢复成 [num_rows, num_cols/4]
  if (shape.size() == 2) {
    out = out.view({static_cast<long>(num_rows),
                    static_cast<long>(bytes_per_out_col)});
  }

  return out;
}


// ==================== PyBind11 导出接口 ====================

PYBIND11_MODULE(Quant_kernels, m) {
  // Python 接口：asymm_qw2_gemm(A, B, scale, zp, bias, group_size)
  // 描述：权重仅量化（2bit）的 GEMM，非对称量化方式
  m.def(
    "asymm_qw2_gemm",
    &asymm_qw2_gemm,
    "weight only int2 gemm for asymm quant"
  );

  // Python 接口：preprocess_weights_int2_for_weight_only(in)
  // 描述：在运行 weight-only int2 GEMM 之前，对 int2 权重进行打包 + 布局预处理
  m.def(
    "preprocess_weights_int2_for_weight_only",
    &preprocess_weights_int2_for_weight_only,
    "preprocess weight before weight only int2 gemm run"
  );
}
