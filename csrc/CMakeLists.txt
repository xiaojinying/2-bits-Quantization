# ===================== trtllm =========================
cmake_minimum_required(VERSION 3.21) #最低cmake版本要求
project(TRTLLM_KERNELS CXX) #定义项目名称和指定语言为c++

enable_language(CUDA) #启用CUDA语言支持

set(CMAKE_CXX_STANDARD 17)  #强制要求c++17标准
set(CMAKE_CUDA_STANDARD 17) #强制要求cuda c++17标准

# 设计变量的值（定义依赖库的路径） set(<variable> <value>...) 通过环境变量传递路径
set(CUDA_PATH ${QUANT_CUDA_HOME})
set(CUDNN_PATH ${QUANT_CUDNN_HOME})
set(TORCH_PATH ${QUANT_TORCH_HOME})
set(CUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/cutlass)
set(TRTLLM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/TensorRT-LLM)
set(TRTLLM_KERNEL_PATH ${TRTLLM_PATH}/cpp/tensorrt_llm/kernels)

# 指定目标GPU架构为Ampere（SM 80）和 Ada Lovelace （Sm 89）
# -gencode: 显式生成对应架构的PTX和二进制代码，确保兼容性。
set(CMAKE_CUDA_ARCHITECTURES 80-real 89-real)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")

# 查找所需的包
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
if(Python3_FOUND)
  message(f"python3 found")
elseif()
  message(f"python3 not found")
endif()
if(Torch_FOUND)
  message(f"torch found")
elseif()
  message(f"torch not found")
endif()
message(STATUS "开始查找cuDNN库，搜索路径: ${CUDNN_PATH}")
find_library(TORCH_LIBS torch_python PATHS ${TORCH_PATH})
find_library(CUDA_LIBS cudart PATHS ${CUDA_PATH}/lib64)
find_library(CUDNN_LIBS libcudnn.so.8 PATHS ${CUDNN_PATH})

#添加头文件搜索路径
include_directories(${CUDA_PATH}/include)
include_directories(${CUDNN_PATH}/include)
include_directories(${CUTLASS_PATH}/include)
include_directories(${CUTLASS_PATH}/tools/util/include)
include_directories(${TRTLLM_PATH}/cpp/include)
include_directories(${TORCH_PATH}/include)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include)
include_directories(${Python3_INCLUDE_DIRS})

#指定链接库的搜索路径
link_directories(${CUDA_PATH}/lib64)
link_directories(${CUDNN_PATH}/lib64)

#启动BF16支持和NDEBUG模式（禁用调试断言）
add_definitions("-DENABLE_BF16")
add_definitions("-DNDEBUG")
# CUDA扩展选项: 允许使用扩展Lambda和宽松的常量表达式。
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(TRTLLM_HEADERS
  ${TRTLLM_KERNEL_PATH}/../..
  ${TRTLLM_KERNEL_PATH}/cutlass_kernels/
  ${TRTLLM_KERNEL_PATH}/../cutlass_extensions/include
  )
include_directories(${TRTLLM_HEADERS})

set(TRTLLM_SRC_CPP
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/cutlass_heuristic.cpp
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/cutlass_preprocessors.cpp
    ${TRTLLM_KERNEL_PATH}/../../tensorrt_llm/common/logger.cpp
    ${TRTLLM_KERNEL_PATH}/../../tensorrt_llm/common/tllmException.cpp
    ${TRTLLM_KERNEL_PATH}/../../tensorrt_llm/common/stringUtils.cpp
   )

list(APPEND TRTLLM_SRC_CU
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/fpA_intB_gemm/fp16_int2_gemm_fg_scalebias.cu
    )
list(APPEND TRTLLM_SRC_CU
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/fpA_intB_gemm/bf16_int2_gemm_fg_scalebias.cu
    )
add_definitions("-DUSE_W2A16")

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

set(QUANT_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/w2a16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_kernel_file_1.generated.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_kernel_file_2.generated.cu
    )

#创建一个库（静态或动态）和其源文件
add_library(Quant_kernels SHARED ${TRTLLM_SRC_CPP} ${TRTLLM_SRC_CU} ${QUANT_SRC_CU})
#链接目标文件和其他库
target_link_libraries(Quant_kernels ${TORCH_LIBS} ${CUDA_LIBS} ${CUDNN_LIBS})
set_property(TARGET Quant_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET Quant_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
