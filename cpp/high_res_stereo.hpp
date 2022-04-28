#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <torch/torch.h>

namespace high_res_stereo {

class HighResStereoMatcher {
public:
    HighResStereoMatcher(const std::string& model_path, const torch::Device& target_device = torch::Device(torch::kCPU),
        bool traced = false, const std::array<double, 3>& norm_mean = { 0.485, 0.456, 0.406 },
        const std::array<double, 3>& norm_std = { 0.229, 0.224, 0.225 });

    bool set_target_device(const torch::Device& target_device);
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

    inline bool set_level(int level)
    {
        bool updated = false;
        if (!traced_ && set_level_) {
            updated = set_level_.value()({ level }).toBool();
        }
        return updated;
    }
    inline int get_level() const
    {
        if (get_level_) {
            return get_level_.value()({}).toInt();
        }
        return -1;
    }

    inline bool set_max_disp(int max_disp)
    {
        bool updated = false;
        if (!traced_ && set_max_disp_) {
            updated = set_max_disp_.value()({ max_disp }).toBool();
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

    bool WarmUpModel(cv::Size img_size, unsigned int n_runs = 2);

    void CalculateDisparity(cv::InputArray img_left, cv::InputArray img_right, cv::OutputArray disparity,
        cv::OutputArray entropy = cv::noArray());

private:
    const std::array<double, 3> norm_mean_;
    const std::array<double, 3> norm_std_;

    torch::jit::Module model_;
    torch::Device target_device_;

    std::optional<torch::jit::Method> set_clean_, get_clean_;
    std::optional<torch::jit::Method> set_level_, get_level_;
    std::optional<torch::jit::Method> set_max_disp_, get_max_disp_;

    bool traced_;
};
}
