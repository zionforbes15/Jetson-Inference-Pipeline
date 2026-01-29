#include "InferencePipeline.hpp"
#include "MemoryState.h" 
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main() {
    InferencePipeline pipeline(
        "/home/zion/model/engine/2cls.engine",          
        "/home/zion/model/engine/ch_impurities_v2_240828.engine", 
        "/home/zion/model/engine/regression_240524.engine",
        "" 
    );

    std::vector<std::string> image_paths;
    for(int i = 1; i <= 8; ++i) {
        image_paths.push_back(cv::format("/home/zion/model/test%d.jpg", i));
    }

    std::cout << "\n[Starting Sequential Inference for 8 Images]" << std::endl;

    for (size_t i = 0; i < image_paths.size(); ++i) {
        std::cout << "\n--- Processing Image [" << i + 1 << "/8]: " << image_paths[i] << " ---" << std::endl;
        
        cv::Mat img = cv::imread(image_paths[i]);
        if (img.empty()) {
            std::cerr << "Skipping: Could not read " << image_paths[i] << std::endl;
            continue;
        }

        std::vector<Detection> dets = pipeline.run(img);

        std::cout << "Detected objects: " << dets.size() << std::endl;

        pipeline.draw_results(img, dets);

        std::string save_name = cv::format("result_img_%zu.jpg", i + 1);
        cv::imwrite(save_name, img);
        std::cout << "Saved to: " << save_name << std::endl;

        UseCondition(); 
    }

    std::cout << "\n[All 8 images processed successfully]" << std::endl;
    return 0;
}