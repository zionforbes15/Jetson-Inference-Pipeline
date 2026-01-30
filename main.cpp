#include "InferencePipeline.hpp"
#include "MemoryState.h" 
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>

// --- 异步保存队列相关变量 ---
struct SaveTask {
    cv::Mat image;
    std::string fileName;
};

std::queue<SaveTask> g_saveQueue;
std::mutex g_queueMtx;
std::condition_variable g_queueCv;
bool g_isFinished = false;

// 后台存图工作线程函数
void saveWorker() {
    while (true) {
        SaveTask task;
        {
            std::unique_lock<std::mutex> lock(g_queueMtx);
            // 等待队列有任务或收到结束信号
            g_queueCv.wait(lock, [] { return !g_saveQueue.empty() || g_isFinished; });
            
            if (g_isFinished && g_saveQueue.empty()) break;
            
            task = std::move(g_saveQueue.front());
            g_saveQueue.pop();
        }
        // 执行耗时的磁盘写入操作
        cv::imwrite(task.fileName, task.image);
    }
}

int main() {
    // 初始化推理引擎
    InferencePipeline pipeline(
        "/home/zion/model/engine/2cls_fp16.engine",          
        "/home/zion/model/engine/ch_impurities_v2_fp16.engine", 
        "/home/zion/model/engine/regression_fp16.engine",
        "" 
    );

    // 准备测试图片路径
    std::vector<std::string> image_paths;
    for(int i = 1; i <= 8; ++i) {
        image_paths.push_back(cv::format("/home/zion/model/test%d.jpg", i));
    }

    // 启动后台存图线程
    std::thread writerThread(saveWorker);

    std::cout << "\n[Starting Parallel Optimized Inference for 8 Images]" << std::endl;

    for (size_t i = 0; i < image_paths.size(); ++i) {
        std::cout << "\n--- Processing Image [" << i + 1 << "/8]: " << image_paths[i] << " ---" << std::endl;
        
        cv::Mat img = cv::imread(image_paths[i]);
        if (img.empty()) {
            std::cerr << "Skipping: Could not read " << image_paths[i] << std::endl;
            continue;
        }

        //GPU执行推理
        std::vector<Detection> dets = pipeline.run(img);
        std::cout << "Detected objects: " << dets.size() << std::endl;

        //CPU执行绘图
        pipeline.draw_results(img, dets);

        //性能监控输出
        UseCondition(pipeline.getLastInferenceTime());

        {
            std::lock_guard<std::mutex> lock(g_queueMtx);
            SaveTask task;
            task.image = img.clone(); // 必须 clone，否则下一帧会修改当前内存
            task.fileName = cv::format("result_img_%zu.jpg", i + 1);
            g_saveQueue.push(std::move(task));
        }
        g_queueCv.notify_one(); 
        
        std::cout << "Push to SaveQueue: Image " << i + 1 << " (Main thread continuing...)" << std::endl;
    }

    {
        std::lock_guard<std::mutex> lock(g_queueMtx);
        g_isFinished = true;
    }
    g_queueCv.notify_all();

    std::cout << "\nWaiting for background writer to finish disk I/O..." << std::endl;
    writerThread.join(); 

    std::cout << "[All 8 images processed and saved successfully]" << std::endl;
    return 0;
}