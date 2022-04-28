#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <torch/torch.h>

namespace high_res_stereo {

class HighResStereoMatcher {
public:
    HighResStereoMatcher(const std::string& model_path, const cv::Size& img_size = cv::Size(),
        const torch::Device& target_device = torch::Device(torch::kCPU), bool traced = false,
        const std::array<double, 3>& norm_mean = { 0.485, 0.456, 0.406 },
        const std::array<double, 3>& norm_std = { 0.229, 0.224, 0.225 });

    bool set_img_size(const cv::Size img_size, bool warm_up = true);
    inline const cv::Size& img_size() const { return img_size_; }
    inline const cv::Size& net_img_size() const { return net_img_size_; }

    bool set_target_device(const torch::Device& target_device, bool warm_up = true);
    inline const torch::Device& target_device() const { return target_device_; }

    inline bool set_clean(float clean)
    {
        if (!traced_ && set_clean_) {
            set_clean_.value()({ clean });
            return true;
        }
        return false;
    }
    inline float get_clean() const
    {
        if (get_clean_) {
            return get_clean_.value()({}).to<float>();
        }
        return -1.0f;
    }

    inline bool set_level(float level, bool warm_up = true)
    {
        bool updated = false;
        if (!traced_ && set_level_) {
            updated = set_level_.value()({ level }).toBool();
            if (updated) {
                warmed_up_ = false;
                if (warm_up) {
                    WarmUpModel();
                }
            }
        }
        return updated;
    }
    inline float get_level() const
    {
        if (get_level_) {
            return get_level_.value()({}).to<float>();
        }
        return -1.0f;
    }

    inline bool set_max_disp(int max_disp, bool warm_up = true)
    {
        bool updated = false;
        if (!traced_ && set_max_disp_) {
            updated = set_max_disp_.value()({ max_disp }).toBool();
            if (updated) {
                warmed_up_ = false;
                if (warm_up) {
                    WarmUpModel();
                }
            }
        }
        return updated;
    }
    inline int get_max_disp() const
    {
        if (get_max_disp_) {
            return get_max_disp_.value()({}).toInt();
        }
        return -1;
    }

    inline bool traced() const { return traced_; }
    inline bool warmed_up() const { return warmed_up_; }

    bool WarmUpModel();

    bool CalculateDisparity(cv::InputArray img_left, cv::InputArray img_right, cv::OutputArray disparity);

private:
    const std::array<double, 3> norm_mean_;
    const std::array<double, 3> norm_std_;

    torch::jit::Module model_;
    torch::Device target_device_;
    cv::Size img_size_, net_img_size_;

    std::optional<torch::jit::Method> set_clean_, get_clean_;
    std::optional<torch::jit::Method> set_level_, get_level_;
    std::optional<torch::jit::Method> set_max_disp_, get_max_disp_;

    bool traced_, warmed_up_;

    void PrepareImage(const cv::Mat& img, torch::Tensor& out_tensor);
};
}
