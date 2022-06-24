#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <torch/script.h>
#include <torch/torch.h>

namespace high_res_stereo {

namespace utils {
    inline int FloorDivide(float a, float b) { return static_cast<int>(std::floor(a / b)); }

    /**
     * Calculates padding (left, top) to get from input image size to network image size
     */
    std::tuple<int, int> CalculateImgPadding(const cv::Size& img_size, const cv::Size& net_img_size);

    cv::Size CalculateNetworkImgSize(const cv::Size& img_size_in);

    cv::Mat TorchTensorToCVMat(const torch::Tensor& in, int rtype, int channels, const int pad_left = 0, const int pad_top = 0);

    void TorchTensorToCVMats(const torch::Tensor& in, cv::OutputArrayOfArrays out, int rtype, int channels, const int pad_left = 0, const int pad_top = 0);

    void InputTensorFromImage(const cv::Mat& img, torch::Tensor& out_tensor, const std::array<double, 3>& norm_mean,
        const std::array<double, 3>& norm_std, const int pad_left = 0, const int pad_top = 0);

    void InputTensorFromImages(cv::InputArrayOfArrays images, torch::Tensor& out_tensor, const std::array<double, 3>& norm_mean,
        const std::array<double, 3>& norm_std, const int pad_left = 0, const int pad_top = 0);

    void InputBlobFromImage(const cv::Mat& img, cv::Mat& out, const cv::Scalar& norm_mean, const cv::Scalar& norm_std,
        const int pad_left = 0, const int pad_top = 0);

    inline std::string InitErrorMsg(const std::string& message, const char* function, const char* file, int line,
        const std::optional<std::string>& post_msg = std::nullopt)
    {
        std::stringstream msg_str;
        msg_str << message << ": in function '" << function << "', file '" << file << "', line " << line << "."
                << std::endl;
        if (post_msg) {
            msg_str << post_msg.value() << std::endl;
        }

        return msg_str.str();
    }

    inline void PrintError(const std::string& message, const char* function, const char* file, int line,
        const std::optional<std::string>& post_msg = std::nullopt)
    {
        std::cerr << InitErrorMsg(message, function, file, line, post_msg);
    }
}
}
