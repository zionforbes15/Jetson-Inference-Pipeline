# Jetson Orin Nano Deep Learning Inference Pipeline Accelerator
# Jetson Orin Nano super深度学习全流程推理加速项目

本项目是一个基于 TensorRT 的高性能视觉处理流水线，专门针对 NVIDIA Jetson Orin Nano 平台优化。实现了 **YOLO 检测 + 裁剪缩放归一化 (GPU Kernel) + 级联分类/回归** 的全流程显存内处理。

This project is a high-performance vision processing pipeline based on TensorRT, specifically optimized for the NVIDIA Jetson Orin Nano platform. It implements an in-GPU memory workflow including **YOLO detection + Crop/Resize/Normalize (CUDA Kernel) + Cascaded Classification/Regression.**

## 🚀 核心模块介绍
- **全流程 GPU 预处理**：手写 CUDA Kernel 替代 OpenCV CPU 缩放，消除大尺度图像 (本项目为1600x1600) 的 CPU 瓶颈。
- **显存不落地 (Memory Efficiency)**：原始图像一次拷贝至显存后，后续裁剪与推理直接在显存内完成。
- **异步流优化**：利用 CUDA Stream 掩盖推理延迟，实现多模型串联下的高吞吐。
- **系统监控**：实时输出 CPU/内存/GPU 负载及频率，方便排查过流保护等硬件限制。

## 🛠️ 环境配置要求
- **硬件**: NVIDIA Jetson Orin Nano (推荐原装适配器 45W及以上)
- **系统**: JetPack 5.x / 6.x
- **依赖**:
  - TensorRT 8.x+
  - CUDA 11.4+
  - OpenCV 4.x+ (带 C++ 支持)
  - CMake 3.10+

## 📂 项目大体结构
- `main.cpp`: 推理主循环，包含多线程异步存图逻辑。
- `InferencePipeline.cpp`: 核心调度逻辑，管理模型生命周期。
- `preprocess.cu`: 自定义 CUDA 算子（Crop, Resize, Normalize, BGR2RGB）。
- `MemoryState.cpp`: 系统性能监控组件。
- `VisionDate.hpp`: 视觉工具函数，包含坐标复原与 Letterbox。

## ⚙️ 编译运行
```bash
# 1. 创建编译目录
mkdir build && cd build

# 2. 编译项目
cmake ..
make -j$(nproc)

# 3. 运行前锁定频率 (可选15w，25w，MAXN SUPER，建议锁定以稳定性能)
sudo jetson_clocks

# 4. 执行推理
./inference_app
```

## 📊 Performance / 性能表现 (Orin Nano @ 918MHz-25w锁频)
**Input Size / 输入尺度**: 1600 x 1600

**YOLO Inference / 检测耗时**: ~30-35ms

**Secondary Inference / 二级推理**: ~2ms per object / 每个目标约 2ms

**Total Pipeline Latency / 总延迟**: Stable at 40ms - 90ms. / 稳定在 39ms - 48ms 之间。

## ⚠️ Notes / 注意事项
**Engine Files / 模型文件**: Ensure .engine files are generated on the target device. / 请确保模型文件在目标设备上生成。

**Power Modes / 功耗模式**: If overcurrent warnings occur, lock the frequency at 918MHz or check the 45W adapter connection. / 若报过流警告，建议锁定频率在 918MHz 并检查电源。

**Log Filtering / 日志过滤**: Non-critical TensorRT warnings (cross-device plan files) are filtered for clarity. / 已屏蔽非关键性跨设备警告，保持输出整洁。