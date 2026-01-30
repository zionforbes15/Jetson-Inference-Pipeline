#include "InferencePipeline.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

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
    
    cudaMalloc((void**)&d_raw_input, 4000 * 4000 * 3); 
    //预分配 Letterbox 临时显存
    cudaMalloc((void**)&d_letterbox_tmp, 1600 * 1600 * 3); 
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
    if (d_letterbox_tmp) cudaFree(d_letterbox_tmp);
    mRuntime.reset(); 
    cudaStreamDestroy(stream);
}

void InferencePipeline::blobFromImage(const cv::Mat& img, float* d_input) {
    size_t img_size = img.rows * img.cols * 3;
    cudaMemcpyAsync(d_letterbox_tmp, img.data, img_size, cudaMemcpyHostToDevice, stream);
    launch_normalize_kernel(d_letterbox_tmp, d_input, img.cols, img.rows, stream);
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

std::vector<Detection> InferencePipeline::run_yolo_only(cv::Mat& frame) {
    //将原图存入 d_raw_input，给后面的二级推理用
    cudaMemcpyAsync(d_raw_input, frame.data, frame.cols * frame.rows * 3, cudaMemcpyHostToDevice, stream);

    AffineInfo info; cv::Mat pr_img;
    VisionUtils::letterbox_v4(frame, pr_img, cv::Size(1600, 1600), info);
    
    blobFromImage(pr_img, (float*)yolo.buffers["images"]);
    yolo.context->enqueueV3(stream);
    cudaMemcpyAsync(host_output_yolo, yolo.buffers["output"], yolo.bufferSizes["output"], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); 
    return parseYoloOutput(host_output_yolo, info, frame.size());
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
    if (dets.empty() || d_raw_input == nullptr) return;
    
    for (size_t i = 0; i < dets.size(); ++i) {
        auto& det = dets[i];
        int cx = std::max(0, det.box.x);
        int cy = std::max(0, det.box.y);
        int cw = std::min(det.box.width, frame.cols - cx);
        int ch = std::min(det.box.height, frame.rows - cy);

        if (cw < 4 || ch < 4) continue;

        //分类推理
        launch_crop_resize_kernel(d_raw_input, (float*)cls.buffers["image"], 
                                  frame.cols, frame.rows, 224, 224, 
                                  cx, cy, cw, ch, stream);
        cls.context->enqueueV3(stream);
        cudaMemcpyAsync(host_output_cls, cls.buffers["class"], 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); 

        if (host_output_cls[0] > host_output_cls[1]) { 
            det.cls_result = 0; // 标记为目标

            //执行回归推理
            launch_crop_resize_kernel(d_raw_input, (float*)reg.buffers["image"], 
                                      frame.cols, frame.rows, 224, 224, 
                                      cx, cy, cw, ch, stream);
            reg.context->enqueueV3(stream);
            cudaMemcpyAsync(host_output_reg, reg.buffers["class"], 5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            det.reg_result.assign(host_output_reg, host_output_reg + 5);
        } else {
            det.cls_result = 1;
        }
    }
}

std::vector<Detection> InferencePipeline::run(cv::Mat& frame) {
    auto start = std::chrono::high_resolution_clock::now();

    auto dets = run_yolo_only(frame);
    if (!dets.empty()) {
        run_secondary_inference(frame, dets);
    }

    cudaStreamSynchronize(stream); //确保所有异步任务完成
    auto end = std::chrono::high_resolution_clock::now();
    last_time.total_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return dets;
}

// void InferencePipeline::run_secondary_inference(cv::Mat& frame, std::vector<Detection>& dets) {
//     for (auto& det : dets) {
//         cv::Rect safe_box = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
//         if (safe_box.width < 4 || safe_box.height < 4) continue;

//         cv::Mat crop = frame(safe_box).clone();
        
//         preprocess_simple(crop, (float*)cls.buffers["image"], 224, 224);
//         cls.context->enqueueV3(stream);
//         cudaMemcpyAsync(host_output_cls, cls.buffers["class"], 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
//         cudaStreamSynchronize(stream);
//         // 打印原始得分
//         printf("Box at [%d,%d] -> Score0: %.4f, Score1: %.4f\n", 
//                 det.box.x, det.box.y, host_output_cls[0], host_output_cls[1]);

//         if (host_output_cls[1] - host_output_cls[0] > 2.0f) {
//             det.cls_result = 1; 
//         } else {
//             det.cls_result = 0;
//         }

//         //回归推理
//         preprocess_simple(crop, (float*)reg.buffers["image"], 224, 224);
//         reg.context->enqueueV3(stream);
//         cudaMemcpyAsync(host_output_reg, reg.buffers["class"], 5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
//         cudaStreamSynchronize(stream);
//         det.reg_result.assign(host_output_reg, host_output_reg + 5);
//     }
// }

void InferencePipeline::draw_results(cv::Mat& frame, const std::vector<Detection>& dets) {
    for (const auto& det : dets) {
        cv::Scalar color = (det.cls_result == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
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



