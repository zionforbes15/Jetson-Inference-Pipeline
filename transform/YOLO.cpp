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
    //加载引擎
    std::string enginePath = "/home/zion/model/yolo/2cls.engine";
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) { std::cerr << "无法打开引擎文件！" << std::endl; return -1; }
    file.seekg(0, file.end); size_t size = file.tellg(); file.seekg(0, file.beg);
    std::vector<char> engineData(size); file.read(engineData.data(), size);

    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime{nvinfer1::createInferRuntime(gLogger)};
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine{runtime->deserializeCudaEngine(engineData.data(), size)};
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context{engine->createExecutionContext()};

    //动态分配所有 Tensor 的显存
    std::vector<Binding> inputBindings;
    std::vector<Binding> outputBindings;
    std::map<std::string, void*> tensorAddresses;

    int nbIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        
        //计算元素总量
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
        size_t byteSize = vol * sizeof(float);

        void* d_ptr;
        cudaMalloc(&d_ptr, byteSize);
        
        Binding b{byteSize, d_ptr, name};
        tensorAddresses[name] = d_ptr;
        context->setTensorAddress(name, d_ptr);

        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            inputBindings.push_back(b);
        else
            outputBindings.push_back(b);
    }

    //图像预处理 (1600x1600, BGR->RGB, HWC->CHW)
    cv::Mat img = cv::imread("/home/zion/model/yolo/test.jpg"); // 确保路径正确
    if (img.empty()) { std::cerr << "图片读取失败！" << std::endl; return -1; }
    
    cv::Mat resized, floatImg;
    cv::resize(img, resized, cv::Size(1600, 1600));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    //HWC 转 CHW 排布
    std::vector<float> hostInput(1 * 3 * 1600 * 1600);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(1600, 1600, CV_32FC1, hostInput.data() + i * 1600 * 1600);
    }
    cv::split(floatImg, channels);

    //执行推理
    cudaMemcpy(inputBindings[0].d_ptr, hostInput.data(), inputBindings[0].size, cudaMemcpyHostToDevice);
    
    std::cout << "正在执行推理 (1600x1600)..." << std::endl;
    context->enqueueV3(0); 

    //提取主输出 [output] 的结果进行对比
    void* d_out_ptr = tensorAddresses["output"];
    size_t outSize = 157500 * 6 * sizeof(float);
    std::vector<float> hostOutput(157500 * 6);
    cudaMemcpy(hostOutput.data(), d_out_ptr, outSize, cudaMemcpyDeviceToHost);

    //完整输出对比结果
    std::cout << "\n推理成功 [output] Tensor 前 20 个数值如下:" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    for (int i = 0; i < 20; ++i) {
        printf("Index[%2d]: %10.6f\n", i, hostOutput[i]);
    }
    std::cout << "-----------------------------------------------" << std::endl;

    // 清理显存
    for (auto& pair : tensorAddresses) { cudaFree(pair.second); }

    return 0;
}