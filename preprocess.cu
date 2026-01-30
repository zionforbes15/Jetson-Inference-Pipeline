/**
 * @file preprocess.cu
 * @brief CUDA Preprocessing Kernels for Jetson Inference Pipeline
 * * ============================================================================
 * CUDA 编程与优化规范
 * ============================================================================
 * * 1. 指针安全 (Pointer Safety):
 * - 严禁在 Kernel 内部直接解引用 CPU 内存指针。
 * - 必须在入口函数(launch_xxx)中执行边界检查，或在 Kernel 内部使用 get_pix_safe。
 * - 非法内存访问 (Illegal Access) 会导致整个 CUDA Context 崩溃，使后续推理全部失效。
 * * 2. 内存布局 (Memory Layout):
 * - 输出 Tensor 必须符合 TensorRT 要求的 NCHW 格式 (RRR...GGG...BBB...)。
 * - 输入原图通常为 HWC (BGRBGR...)，访问时计算公式: (y * width + x) * channels + c。
 * * 3. 异步流 (Async & Streams):
 * - 所有 Kernel 必须绑定 cudaStream_t。严禁在 Kernel 内部使用同步指令。
 * - CPU 在启动 Kernel 后会立即返回，若需获取计算结果，必须在外部调用 cudaStreamSynchronize。
 * * 4. 线程调度 (Thread Scheduling):
 * - 图像处理推荐 Block 尺寸: 16x16 或 32x32，确保线程束 (Warp) 满载运行。
 * - 尽量保持内存对齐合并访问，避免在 Kernel 内部进行复杂的条件分支 (Branch Divergence)。
 * * 5. 性能敏感性 (Performance Sensitivity):
 * - 禁止在 Kernel 内部进行显存申请/释放 (cudaMalloc/Free)。
 * - 归一化系数 (Means/Stds) 需与模型训练时严格一致，否则会导致推理精度断崖。
 * ============================================================================
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// 安全像素读取函数
__device__ float get_pix_safe(const uint8_t* src, int x, int y, int c, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0.0f;
    return (float)src[(y * w + x) * 3 + c];
}

//全图缩放 + BGR转RGB + 1/255 归一化
__global__ void yolo_normalize_kernel(const uint8_t* src, float* d_dst, int w, int h) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < w && dy < h) {
        for (int c = 0; c < 3; ++c) {
            float val = (float)src[(dy * w + dx) * 3 + (2 - c)]; 
            d_dst[c * w * h + dy * w + dx] = val / 255.0f;
        }
    }
}

//ROI裁剪 + 缩放 + BGR转RGB + ImageNet 标准归一化
__global__ void cls_crop_resize_normalize_kernel(
    const uint8_t* src, float* d_dst, 
    int src_w, int src_h, int dst_w, int dst_h,
    int crop_x, int crop_y, int crop_w, int crop_h) 
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h) {
        float sx = crop_x + (float)dx * crop_w / dst_w;
        float sy = crop_y + (float)dy * crop_h / dst_h;

        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float ux = sx - x0;
        float uy = sy - y0;

        float means[3] = {0.485f, 0.456f, 0.406f};
        float stds[3] = {0.229f, 0.224f, 0.225f};

        for (int c = 0; c < 3; ++c) {
            float v00 = get_pix_safe(src, x0, y0, 2 - c, src_w, src_h);
            float v10 = get_pix_safe(src, x1, y0, 2 - c, src_w, src_h);
            float v01 = get_pix_safe(src, x0, y1, 2 - c, src_w, src_h);
            float v11 = get_pix_safe(src, x1, y1, 2 - c, src_w, src_h);

            float val = (1.0f - ux) * (1.0f - uy) * v00 + 
                        ux * (1.0f - uy) * v10 + 
                        (1.0f - ux) * uy * v01 + 
                        ux * uy * v11;

            d_dst[c * dst_w * dst_h + dy * dst_w + dx] = ((val / 255.0f) - means[c]) / stds[c];
        }
    }
}


extern "C" {
    void launch_normalize_kernel(const uint8_t* d_src, float* d_dst, int w, int h, cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        yolo_normalize_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, w, h);
    }

    void launch_crop_resize_kernel(
        const uint8_t* d_src, float* d_dst, 
        int src_w, int src_h, int dst_w, int dst_h,
        int crop_x, int crop_y, int crop_w, int crop_h, 
        cudaStream_t stream) 
    {
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
        cls_crop_resize_normalize_kernel<<<grid, block, 0, stream>>>(
            d_src, d_dst, src_w, src_h, dst_w, dst_h, crop_x, crop_y, crop_w, crop_h);
    }
}