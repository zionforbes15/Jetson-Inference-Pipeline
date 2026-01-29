#include "InferencePipeline.hpp"
#include <fstream>
#include <iostream>

static Logger gLogger;

InferencePipeline::InferencePipeline(const std::string& yolo_path, const std::string& cls_path, const std::string& reg_path, const std::string& seg_path) {
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger), TRTDeleter());
    cudaStreamCreate(&stream);
    loadModel(yolo_path, yolo);
    loadModel(cls_path, cls);
    loadModel(reg_path, reg);

    host_output_yolo = new float[157500 * 6];
    host_output_cls = new float[2];
    host_output_reg = new float[5];
    d_raw_input = nullptr; 
}

InferencePipeline::~InferencePipeline() {
    delete[] host_output_yolo;
    delete[] host_output_cls;
    delete[] host_output_reg;
    auto freeRes = [](ModelResource& res) {
        for (auto& pair : res.buffers) cudaFree(pair.second);
        res.context.reset();
        res.engine.reset();
    };
    freeRes(yolo); freeRes(cls); freeRes(reg);
    if (d_raw_input) cudaFree(d_raw_input);
    mRuntime.reset(); 
    cudaStreamDestroy(stream);
}

void InferencePipeline::loadModel(const std::string& path, ModelResource& res) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to open engine: " + path);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    res.engine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(data.data(), size), TRTDeleter());
    res.context = std::shared_ptr<nvinfer1::IExecutionContext>(res.engine->createExecutionContext(), TRTDeleter());
    for (int i = 0; i < res.engine->getNbIOTensors(); ++i) {
        const char* name = res.engine->getIOTensorName(i);
        auto dims = res.engine->getTensorShape(name);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
        cudaMalloc(&res.buffers[name], vol * sizeof(float));
        res.bufferSizes[name] = vol * sizeof(float);
        res.context->setTensorAddress(name, res.buffers[name]);
    }
}

void InferencePipeline::blobFromImage(const cv::Mat& img, float* d_input) {
    int img_size = img.rows * img.cols * 3;
    if (d_raw_input == nullptr) cudaMalloc((void**)&d_raw_input, 1600 * 1600 * 3); 
    cudaMemcpyAsync(d_raw_input, img.data, img_size, cudaMemcpyHostToDevice, stream);
    launch_normalize_kernel(d_raw_input, d_input, img.cols, img.rows, stream);
}
void InferencePipeline::preprocess_simple(const cv::Mat& img, float* d_input, int w, int h) {
    if (img.empty()) return;
    cv::Mat rgb, resized, float_img;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(w, h));
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    cv::subtract(float_img, cv::Scalar(0.485, 0.456, 0.406), float_img);
    cv::divide(float_img, cv::Scalar(0.229, 0.224, 0.225), float_img);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    size_t plane_bytes = w * h * sizeof(float);
    for (int i = 0; i < 3; ++i) {
        cudaMemcpyAsync(d_input + i * w * h, channels[i].ptr<float>(), plane_bytes, cudaMemcpyHostToDevice, stream);
    }
}

std::vector<Detection> InferencePipeline::parseYoloOutput(float* output, const AffineInfo& info, cv::Size raw_shape) {
    std::vector<Detection> results;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    float conf_threshold = 0.35f; 
    float nms_threshold = 0.50f;

    for (int i = 0; i < 157500; ++i) {
        float* ptr = output + i * 6;
        float score = ptr[4] * ptr[5];
        if (score > conf_threshold) {
            float x1 = (ptr[0] - ptr[2] / 2.0f - info.pad_w) / info.scale;
            float y1 = (ptr[1] - ptr[3] / 2.0f - info.pad_h) / info.scale;
            int l = std::max(0, (int)x1);
            int t = std::max(0, (int)y1);
            int w = std::min((int)(ptr[2] / info.scale), raw_shape.width - l);
            int h = std::min((int)(ptr[3] / info.scale), raw_shape.height - t);
            if (w > 5 && h > 5) {
                boxes.emplace_back(l, t, w, h);
                confs.push_back(score);
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_threshold, nms_threshold, indices);
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.conf = confs[idx];
        results.push_back(det);
    }
    return results;
}

void InferencePipeline::run_secondary_inference(cv::Mat& frame, std::vector<Detection>& dets) {
    for (auto& det : dets) {
        cv::Rect safe_box = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_box.width < 4 || safe_box.height < 4) continue;

        cv::Mat crop = frame(safe_box).clone();
        
        preprocess_simple(crop, (float*)cls.buffers["image"], 224, 224);
        cls.context->enqueueV3(stream);
        cudaMemcpyAsync(host_output_cls, cls.buffers["class"], 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        // 打印原始得分
        printf("Box at [%d,%d] -> Score0: %.4f, Score1: %.4f\n", 
                det.box.x, det.box.y, host_output_cls[0], host_output_cls[1]);

        if (host_output_cls[1] - host_output_cls[0] > 2.0f) {
            det.cls_result = 1; 
        } else {
            det.cls_result = 0;
        }

        //回归推理
        preprocess_simple(crop, (float*)reg.buffers["image"], 224, 224);
        reg.context->enqueueV3(stream);
        cudaMemcpyAsync(host_output_reg, reg.buffers["class"], 5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        det.reg_result.assign(host_output_reg, host_output_reg + 5);
    }
}

void InferencePipeline::draw_results(cv::Mat& frame, const std::vector<Detection>& dets) {
    for (const auto& det : dets) {
        cv::Scalar color = (det.cls_result == 1) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        int thickness = 3;       
        float font_scale = 0.8;  
        int font_thickness = 2;  

        cv::rectangle(frame, det.box, color, thickness);

        std::string label = cv::format("YOLO:%.2f", det.conf);
        if (det.cls_result != -1) {
            label += cv::format(" | CLS:%d", det.cls_result);
        }
        if (!det.reg_result.empty()) {
            // 取回归的前两个关键偏移值显示
            label += cv::format(" | REG:[%.1f,%.1f]", det.reg_result[0], det.reg_result[1]);
        }
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseLine);
        int ty = std::max(det.box.y, labelSize.height + 10);
        cv::Point bg_top_left(det.box.x, ty - labelSize.height - 10);
        cv::Point bg_bottom_right(det.box.x + labelSize.width + 10, ty + baseLine);
        cv::rectangle(frame, bg_top_left, bg_bottom_right, color, cv::FILLED);
        cv::putText(frame, label, cv::Point(det.box.x + 5, ty - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);
    }
}

std::vector<Detection> InferencePipeline::run_yolo_only(cv::Mat& frame) {
    AffineInfo info; cv::Mat pr_img;
    VisionUtils::letterbox_v4(frame, pr_img, cv::Size(1600, 1600), info);
    blobFromImage(pr_img, (float*)yolo.buffers["images"]);
    yolo.context->enqueueV3(stream);
    cudaMemcpyAsync(host_output_yolo, yolo.buffers["output"], yolo.bufferSizes["output"], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return parseYoloOutput(host_output_yolo, info, frame.size());
}

std::vector<Detection> InferencePipeline::run(cv::Mat& frame) {
    auto dets = run_yolo_only(frame);
    run_secondary_inference(frame, dets);
    return dets;
}