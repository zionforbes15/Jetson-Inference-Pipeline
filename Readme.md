# Jetson-Inference-Pipeline: Medical Object Detection & Analysis
基于 Jetson Orin Nano 的医学目标识别、分类与回归推理系统
# 📌 项目简介 | Introduction
本项目旨在将原本基于 Windows 平台的医学影像识别算法移植到 NVIDIA Jetson Orin Nano 边缘计算平台。系统实现了显微镜倍镜下特定医学对象的全流程自动化分析，包含目标检测、二次分类及几何回归三个阶段的深度学习推理。

This project focuses on migrating medical image recognition algorithms from Windows to the NVIDIA Jetson Orin Nano edge platform. The system implements a complete automated analysis pipeline for specific medical objects under magnification, incorporating Object Detection, Secondary Classification, and Geometric Regression.

# 🚀 核心功能 | Key Features
多级推理流水线 (Multi-stage Pipeline): * Detection: 基于 YOLO 执行高分辨率 (1600x1600) 目标的初步定位。

Classification: 对检测框进行抠图并进行二次鉴别，过滤复杂背景下的误检。

Regression: 针对特定医学对象执行像素级的几何参数回归。

硬件加速优化 (Hardware Acceleration): * 使用 TensorRT 对 ONNX 模型进行序列化加速。

利用 CUDA Kernel 自定义编写图像预处理 (Normalization/HWC2CHW)，减少 CPU 负载。

跨平台移植 (Cross-Platform Migration): 优化了原始 Windows 代码的内存管理，实现了显存预分配与零拷贝逻辑。

实时监控 (Real-time Monitoring): 内置 GPU 负载、频率及内存占用监测模块。

# 🛠️ 环境要求 | Requirements
Hardware: NVIDIA Jetson Orin Nano (JetPack 5.x / 6.x)

Libraries:

CUDA 11.4 / 12.x

TensorRT 8.5.x+

OpenCV 4.5.4+

C++ 17

# 🔨 构建指南 | Build Instructions
Bash

克隆项目 | Clone the repo

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name

创建构建目录 | Build

mkdir build && cd build

cmake ..

make -j4

运行推理 | Run

./inference_app

# 📝 开发备注 | Development Notes
Transform: transform/ 文件夹内包含用于在 PC 端验证模型精度以及导出 Engine 的 Python 脚本。

Optimization: 当前版本重点优化了显存与内存间的拷贝带宽，通过 cudaMemcpyAsync 与自定义 Kernel 配合，显著提升了推理吞吐量。

Clustering: dbscan 相关代码已预留，用于后续处理高密度目标的聚类分析，当前版本暂未启用。
