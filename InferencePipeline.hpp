#ifndef INFERENCE_PIPELINE_HPP
#define INFERENCE_PIPELINE_HPP

#include "VisionDate.hpp"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <iostream>

struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const { 
        if (obj) delete obj; 
    }
};

extern "C" void launch_normalize_kernel(const uint8_t* d_src, float* d_dst, int w, int h, cudaStream_t stream);

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};


class InferencePipeline {
public:
    InferencePipeline(const std::string& yolo_path, 
                      const std::string& cls_path, 
                      const std::string& reg_path,
                      const std::string& seg_path); 
    ~InferencePipeline();

    // 全流程接口
    std::vector<Detection> run(cv::Mat& frame);

    // 阶段化接口
    std::vector<Detection> run_yolo_only(cv::Mat& frame); //
    
    //二级推理接口
    void run_secondary_inference(cv::Mat& frame, std::vector<Detection>& dets);

    // 预留接口
    void run_segmentation(cv::Mat& frame, Detection& det); 
    double calculateSharpnessWithMask(const cv::Mat& crop, const cv::Mat& mask);

    void draw_results(cv::Mat& frame, const std::vector<Detection>& dets);

private:
    struct ModelResource {
        std::shared_ptr<nvinfer1::ICudaEngine> engine;
        std::shared_ptr<nvinfer1::IExecutionContext> context;
        std::map<std::string, void*> buffers;
        std::map<std::string, size_t> bufferSizes;
    };

    void loadModel(const std::string& path, ModelResource& res); //
    void blobFromImage(const cv::Mat& img, float* d_input);
    void preprocess_simple(const cv::Mat& img, float* d_input, int w, int h);
    std::vector<Detection> parseYoloOutput(float* output, const AffineInfo& info, cv::Size raw_shape);

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    uint8_t* d_raw_input = nullptr;
    ModelResource yolo, cls, reg, seg;
    cudaStream_t stream;

    // Host 内存缓冲区指针
    float* host_output_yolo;
    float* host_output_cls;
    float* host_output_reg;
    float* host_output_seg; 
};

#endif