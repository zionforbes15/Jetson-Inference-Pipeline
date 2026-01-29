#include <cuda_runtime.h>
#include <stdint.h>

__device__ const float SCALE = 1.0f / 255.0f;

__global__ void normalize_kernel_optimized(const uint8_t* __restrict__ src, float* __restrict__ dst, int w, int h) {
    // 每一个线程处理一个像素 (x, y)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int spatial_size = w * h;
        int target_pos = y * w + x;
        int src_pos = target_pos * 3;

        uint8_t b = src[src_pos + 0];
        uint8_t g = src[src_pos + 1];
        uint8_t r = src[src_pos + 2];

        dst[target_pos] = (float)r * SCALE;
        dst[spatial_size + target_pos] = (float)g * SCALE;
        dst[2 * spatial_size + target_pos] = (float)b * SCALE;
    }
}

extern "C" void launch_normalize_kernel(const uint8_t* d_src, float* d_dst, int w, int h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    normalize_kernel_optimized<<<grid, block, 0, stream>>>(d_src, d_dst, w, h);
}