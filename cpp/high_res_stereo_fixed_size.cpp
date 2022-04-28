#include "high_res_stereo_fixed_size.hpp"
#include "utils.hpp"
/* #include <source_location> */
#include <torch/script.h>

namespace high_res_stereo {
HighResStereoMatcher::HighResStereoMatcher(const std::string& model_path, const cv::Size& img_size,
    const torch::Device& target_device, bool traced, const std::array<double, 3>& norm_mean,
    const std::array<double, 3>& norm_std)
    : norm_mean_(norm_mean)
    , norm_std_(norm_std)
    , target_device_(torch::kCPU)
    , set_clean_(std::nullopt)
    , set_level_(std::nullopt)
    , set_max_disp_(std::nullopt)
    , traced_(traced)
    , warmed_up_(false)
{
    if (target_device.type() == torch::kCUDA && torch::cuda::is_available()) {
        target_device_ = target_device;
    } else {
        utils::PrintError("CUDA device chosen, but CUDA not available -> target device set to CPU" + model_path,
            __func__, __FILE__, __LINE__);
        /* auto loc = std::source_location::current(); */
        /* PrintError("CUDA device chosen, but CUDA not available -> target device set to CPU", loc.function_name(),
         * loc.file_name(), loc.line()); */
    }
    try {
        model_ = torch::jit::load(model_path, target_device_);
        model_.eval();
    } catch (const torch::Error& e) {
        utils::PrintError("Error loading model from " + model_path, __func__, __FILE__, __LINE__);
        /* auto loc = std::source_location::current(); */
        /* PrintError("Error loading model from " + model_path, loc.function_name(), loc.file_name(), loc.line()); */

        throw;
    }

    try {
        get_clean_ = model_.get_method("get_clean");
        get_level_ = model_.get_method("get_level");
        get_max_disp_ = model_.get_method("get_max_disp");

        if (!traced_) {
            set_clean_ = model_.get_method("set_clean");
            set_level_ = model_.get_method("set_level");
            set_max_disp_ = model_.get_method("set_max_disp");
        }
    } catch (const torch::Error& e) {
        utils::PrintError("Error loading functions from model", __func__, __FILE__, __LINE__);
        /* auto loc = std::source_location::current(); */
        /* PrintError("Error loading functions from model", loc.function_name(), loc.file_name(), loc.line()); */
        throw;
    }

    if (!img_size.empty()) {
        set_img_size(img_size);
    }
}

bool HighResStereoMatcher::set_img_size(const cv::Size img_size, bool warm_up)
{
    if (this->img_size_ != img_size) {
        this->img_size_ = img_size;
        auto net_size = utils::CalculateNetworkImgSize(this->img_size_);
        if (this->net_img_size_ != net_size) {
            this->net_img_size_ = net_size;
            warmed_up_ = false;
            if (warm_up) {
                WarmUpModel();
            }
        }
        return true;
    }
    return false;
}

bool HighResStereoMatcher::set_target_device(const torch::Device& target_device, bool warm_up)
{
    bool updated = false;
    if (target_device.type() == torch::kCUDA && !torch::cuda::is_available()) {
        utils::PrintError(
            "CUDA device chosen, but CUDA not available -> target device not changed", __func__, __FILE__, __LINE__);
        /* auto loc = std::source_location::current(); */
        /* PrintError("CUDA device chosen, but CUDA not available -> target device not changed", loc.function_name(),
         * loc.file_name(), loc.line()); */
    } else {
        try {
            model_.to(target_device);
            this->target_device_ = target_device;
            updated = true;
        } catch (const torch::Error& e) {
            utils::PrintError(
                "Error moving model to device -> target device not changed", __func__, __FILE__, __LINE__);
            /* auto loc = std::source_location::current(); */
            /* PrintError("Error moving model to device -> target device not changed", loc.function_name(),
             * loc.file_name(), loc.line(), e.what()); */
        }
    }
    if (updated) {
        warmed_up_ = false;
        if (warm_up) {
            WarmUpModel();
        }
    }
    return updated;
}

bool HighResStereoMatcher::WarmUpModel()
{
    if (!warmed_up_ && !img_size_.empty() && !net_img_size_.empty()) {
        torch::NoGradGuard no_grad;
        try {
            for (auto i = 0; i < 2; i++) {
                torch::zeros({ 1, 3, net_img_size_.height, net_img_size_.width });
                torch::Tensor rTensTmp = torch::zeros({ 1, 3, net_img_size_.height, net_img_size_.width });
                lTensTmp = lTensTmp.to(target_device_);
                rTensTmp = rTensTmp.to(target_device_);

                double processingTime;
                if (target_device_ == torch::kCUDA)
                    torch::cuda::synchronize();
                auto start = std::chrono::high_resolution_clock::now();
                model_.forward({ lTensTmp, rTensTmp });
                if (target_device_ == torch::kCUDA)
                    torch::cuda::synchronize();
                auto stop = std::chrono::high_resolution_clock::now();
                processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                std::cout << "Warm up - run " << i << " - Runtime: " << processingTime << std::endl;
            }
            warmed_up_ = true;
        } catch(const torch::Error& e) {
            utils::PrintError(
                "Error running warm up for model", __func__, __FILE__, __LINE__, e.what());
            warmed_up_ = false;
        }
    }
    return warmed_up_;
}

bool HighResStereoMatcher::CalculateDisparity(
    cv::InputArray img_left, cv::InputArray img_right, cv::OutputArray disparity)
{
    cv::Mat img_l = img_left.getMat();
    cv::Mat img_r = img_right.getMat();

    if (img_l.size() != img_r.size()) {
            throw std::invalid_argument(
                utils::InitErrorMsg("left and right image have different size", __func__, __FILE__, __LINE__));
    }

    // Disable gradient calculation
    torch::NoGradGuard no_grad;

    return true;
}

void HighResStereoMatcher::PrepareImage(const cv::Mat& img, torch::Tensor& out_tensor) {
    if (img_size_.empty() || img_size_.width != img.cols || img_size_.height != img.rows) {
        auto net_img_size = utils::CalculateNetworkImgSize({img.cols, img.rows});
        if (net_img_size != net_img_size_) {
            warmed_up_ = false;
        }
    }
}
}
