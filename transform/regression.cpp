#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <map>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

struct TRTDestroy {
    template <typename T> void operator()(T* obj) const { if (obj) delete obj; }
};

struct Binding {
    size_t size;
    void* d_ptr;
    std::string name;
};

int main() {
    
    std::string enginePath = "/home/zion/model/engine/regression_240524.engine";
    std::string inputNodeName = "image";                   
    std::string outputNodeName = "class";                   
    int inputH = 224;                                       
    int inputW = 224;                                       
    int outputLen = 5;                                      

    // 1. 加载引擎
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) { std::cerr << "无法打开引擎文件！" << std::endl; return -1; }
    file.seekg(0, file.end); size_t size = file.tellg(); file.seekg(0, file.beg);
    std::vector<char> engineData(size); file.read(engineData.data(), size);

    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime{nvinfer1::createInferRuntime(gLogger)};
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine{runtime->deserializeCudaEngine(engineData.data(), size)};
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context{engine->createExecutionContext()};

    // 2. 动态分配所有 Tensor 显存
    std::map<std::string, Binding> bindings;
    int nbIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
        size_t byteSize = vol * sizeof(float);

        void* d_ptr;
        cudaMalloc(&d_ptr, byteSize);
        bindings[name] = {byteSize, d_ptr, name};
        context->setTensorAddress(name, d_ptr);
    }

    // 3. 图像预处理
    cv::Mat img = cv::imread("/home/zion/model/yolo/test.jpg"); 
    if (img.empty()) { std::cerr << "图片读取失败！" << std::endl; return -1; }
    
    cv::Mat resized, floatImg;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // HWC 转 CHW
    std::vector<float> hostInput(3 * inputH * inputW);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(inputH, inputW, CV_32FC1, hostInput.data() + i * inputH * inputW);
    }
    cv::split(floatImg, channels);

    // 4. 执行推理
    cudaMemcpy(bindings[inputNodeName].d_ptr, hostInput.data(), bindings[inputNodeName].size, cudaMemcpyHostToDevice);
    
    std::cout << "正在执行回归模型推理 (" << inputW << "x" << inputH << ")..." << std::endl;
    context->enqueueV3(0); 

    // 5. 提取输出 [class] 的结果
    std::vector<float> hostOutput(outputLen);
    cudaMemcpy(hostOutput.data(), bindings[outputNodeName].d_ptr, outputLen * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 输出结果
    std::cout << "\n推理成功 [" << outputNodeName << "] 结果如下:" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    for (int i = 0; i < outputLen; ++i) {
        printf("Index[%d]: %10.6f\n", i, hostOutput[i]);
    }
    std::cout << "-----------------------------------------------" << std::endl;

    // 清理
    for (auto& pair : bindings) { cudaFree(pair.second.d_ptr); }

    return 0;
}