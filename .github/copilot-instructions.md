# Copilot 指令（仓库专用）

目的：帮助 AI 编码代理快速上手本仓库的结构、约定和常见改动模式，以便做出符合项目风格和正确性的修改。

快速启动
- 构建：在仓库根目录执行：

  ```bash
  mkdir -p build && cd build
  cmake ..
  make -j
  ```

  可执行文件通常位于 `build/inference_app`（参见已有 `build/` 输出）。如果 CMake 找不到 CUDA 编译器，请确保系统安装 CUDA 并设置 `CMAKE_CUDA_COMPILER`。

关键文件与架构概览
- `main.cpp`：程序入口，负责初始化并驱动推理流程。
- `InferencePipeline.cpp` / `InferencePipeline.hpp`：整体推理流程的主线，数据从输入->预处理->推理->后处理通过这里串联。
- `preprocess.cu`：所有 CUDA 预处理内核（归一化、裁剪/双线性缩放等）。这是修改输入数据布局或归一化参数的首要位置。[preprocess.cu](preprocess.cu)
- `MemoryState.cpp` / `MemoryState.h`：管理设备/主机缓冲区和生命周期，注意内存同步与流（stream）使用。
- `dbscan.cpp` / `dbscan.h`：后处理（聚类）实现示例，展示 CPU/GPU 混合实现思路。
- `CMakeLists.txt`：工程构建规则，新增源文件时需更新此文件以纳入编译。

项目约定（重要，务必遵守）
- 张量布局：模型输出与中间张量均采用 NCHW（例如 `dst[c * dst_w * dst_h + y * dst_w + x]`）。预处理应产生符合 TensorRT 要求的 NCHW 内存布局。
- 输入通道顺序：原始图像为 HWC BGR；预处理核中通常做 BGR -> RGB 转换（见 `preprocess.cu::normalize_kernel_optimized`）。修改时请保持通道顺序一致。
- 归一化：分类/回归使用 ImageNet 风格的 `means` / `stds`（见 `crop_resize_normalize_kernel`）。YOLO 等任务使用简单的 1/255 缩放，定义为 `SCALE` 常量。
- CUDA 风格：所有 Kernel 接口接受 `cudaStream_t stream`，并在主机端按流启动。禁止在 Kernel 内部调用同步或进行显存分配（cudaMalloc/Free）。若需要边界保护，使用 `get_pix_safe` 或在 launch 侧确保边界。
- 内存安全：任何对原始图像像素的访问要防止越界（`get_pix_safe` 为示例）。非法访问会导致整个 CUDA 上下文崩溃。

常见修改模式与示例
- 添加新的预处理步骤：修改 `preprocess.cu`，遵循 block 16x16 或 32x32 布局，保留流参数并在 `extern "C"` 的 launch 函数中设置 grid/block 和 stream。参考 `launch_normalize_kernel` / `launch_crop_resize_kernel`。
- 更改归一化参数：在 `crop_resize_normalize_kernel` 更新 `means`/`stds`；若只影响单任务（例如 YOLO），调整 `SCALE` 或新增参数并更新调用方。
- 新增源文件：更新 `CMakeLists.txt`，确保在 CUDA 源加入到相应 target。（保持最小变更）

调试与性能诊断
- 本地 CPU 调试：使用 `gdb` 或 `printf` 风格输出（按项目习惯，避免在性能关键路径大量打印）。
- CUDA 调试/检查：推荐使用 `cuda-memcheck` 检测越界/非法访问；使用 NVIDIA Nsight / `nvprof` / `ncu` 做性能剖析。
- 流与同步错误：Kernel 启动后主机会立即返回；如需保证完成请在外部调用 `cudaStreamSynchronize(stream)`。

风格与限制
- 变更要尽量小且原子：在修改设备/内存布局相关代码时，请同时更新所有调用点并运行简单验证（尽可能在主机侧验证输出形状/数值范围）。
- 不要在 Kernel 内做复杂控制流或动态分配。尽量保持内存访问合并与对齐。
- 不要添加跨平台的全局配置或重构，除非能保证向后兼容。

交付与 PR 指南
- 描述变更时指出影响范围（例如：仅预处理、影响模型输入张量、或需要重新训练归一化参数）。
- 包含复现步骤（如何构建、如何运行一个短示例来验证改动）。

如果你需要更多上下文来修改某处代码，优先查看并引用：
- `InferencePipeline.cpp` / `InferencePipeline.hpp` 以理解数据流和调用点
- `preprocess.cu` 以理解现有的预处理实现和约定
- `MemoryState.*` 以确认缓冲区分配与同步

请告知是否需要我将此文件进一步细化（例如加入典型运行命令、示例输入/输出检查脚本，或补充更多文件级别引用）。
