#ifndef VISION_DATA_HPP
#define VISION_DATA_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

struct Detection {
    cv::Rect box;      
    float conf;         
    int class_id;      
    cv::Mat mask; 
    int cls_result;              
    std::vector<float> reg_result; 
    Detection() : conf(0.0f), class_id(-1), cls_result(-1) {
        reg_result.reserve(5);
    }
};

struct AffineInfo {
    float scale;
    int pad_w;
    int pad_h;
};

namespace VisionUtils {

    inline void letterbox_v4(const cv::Mat& src, cv::Mat& dst, cv::Size target_size, AffineInfo& info) {
        float r = std::min((float)target_size.width / src.cols, (float)target_size.height / src.rows);
        info.scale = r;
        
        int new_unpad_w = int(round(src.cols * r));
        int new_unpad_h = int(round(src.rows * r));
        
        cv::Mat tmp;
        cv::resize(src, tmp, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        
        info.pad_w = (target_size.width - new_unpad_w) / 2;
        info.pad_h = (target_size.height - new_unpad_h) / 2;
        
        int top = info.pad_h;
        int bottom = target_size.height - new_unpad_h - top;
        int left = info.pad_w;
        int right = target_size.width - new_unpad_w - left;
        
        cv::copyMakeBorder(tmp, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    }

    inline void restore_coords(Detection& det, const AffineInfo& info, cv::Size raw_shape) {
        float x = (det.box.x - info.pad_w) / info.scale;
        float y = (det.box.y - info.pad_h) / info.scale;
        float w = det.box.width / info.scale;
        float h = det.box.height / info.scale;

        int x1 = std::max(0, std::min((int)x, raw_shape.width - 1));
        int y1 = std::max(0, std::min((int)y, raw_shape.height - 1));
        
        det.box.x = x1;
        det.box.y = y1;
        det.box.width = std::max(0, std::min((int)w, raw_shape.width - x1));
        det.box.height = std::max(0, std::min((int)h, raw_shape.height - y1));
    }
}

#endif