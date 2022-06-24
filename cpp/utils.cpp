#include "utils.hpp"

#include <algorithm>
#include <execution>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace high_res_stereo {
namespace utils {

    cv::Size CalculateNetworkImgSize(const cv::Size& img_size)
    {
        if (img_size.empty()) {
            throw std::invalid_argument(InitErrorMsg("img_size is empty", __func__, __FILE__, __LINE__));
        }
        int h_out = FloorDivide(img_size.height, 64.0f) * 64;
        int w_out = FloorDivide(img_size.width, 64.0f) * 64;

        h_out = h_out < img_size.height ? h_out + 64 : h_out;
        w_out = w_out < img_size.width ? w_out + 64 : w_out;

        return cv::Size(w_out, h_out);
    }

    std::tuple<int, int> CalculateImgPadding(const cv::Size& img_size, const cv::Size& net_img_size)
    {
        cv::Size size;
        if (img_size.empty() || net_img_size.empty()) {
            throw std::invalid_argument(InitErrorMsg("one of the passed sizes is empty", __func__, __FILE__, __LINE__));
        }
        if (img_size.width > net_img_size.width || img_size.height > net_img_size.height) {
            throw std::invalid_argument(
                InitErrorMsg("image size larger as network image size", __func__, __FILE__, __LINE__));
        }
        return { net_img_size.width - img_size.width, net_img_size.height - img_size.height };
    }

    cv::Mat TorchTensorToCVMat(const torch::Tensor& in, int rtype, int channels, const int pad_left, const int pad_top)
    {
        auto tensor = in.squeeze().detach();
        // used to remove padding added during generation of tensor from cv mats (pad_left and pad_top need to be
        // negative)
        tensor = torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({ pad_left, 0, pad_top, 0 }).mode(torch::kConstant).value(0));
        if (channels == 3) {
            tensor = tensor.permute({ 1, 2, 0 });
        }
        tensor = tensor.contiguous().to(torch::kCPU, tensor.dtype(), false, true);
        return cv::Mat(tensor.size(0), tensor.size(1), rtype, tensor.data_ptr()).clone();
    }

    void TorchTensorToCVMats(const torch::Tensor& in, cv::OutputArrayOfArrays out, int rtype, int channels,
        const int pad_left, const int pad_top)
    {
        auto tensor = in.detach();
        // used to remove padding added during generation of tensor from cv mats (pad_left and pad_top need to be
        // negative)
        tensor = torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({ pad_left, 0, pad_top, 0 }).mode(torch::kConstant).value(0));
        if (channels == 3) {
            tensor = tensor.permute({ 0, 2, 3, 1 });
        }
        tensor = tensor.contiguous().to(torch::kCPU, tensor.dtype(), false, true);

        auto t_imgs = tensor.split(1, 0);
        auto n_imgs = t_imgs.size();

        std::vector<cv::Mat> out_vec(n_imgs);
        /* for (auto i = 0u; i < n_imgs; i++) { */
        /*     /1* auto tmp = t_imgs[i].squeeze(); *1/ */
        /*     /1* out[i] = cv::Mat(tmp.size(0), tmp.size(1), rtype, tmp.data_ptr()).clone(); *1/ */
        /*     t_imgs[i] = t_imgs[i].squeeze(); */
        /*     out_vec[i] = cv::Mat(t_imgs[i].size(0), t_imgs[i].size(1), rtype, t_imgs[i].data_ptr()).clone(); */
        /* } */
        std::transform(
            std::execution::par, t_imgs.begin(), t_imgs.end(), out_vec.begin(), [rtype](const torch::Tensor& t_img) {
                auto tmp = t_img.squeeze();
                return cv::Mat(tmp.size(0), tmp.size(1), rtype, tmp.data_ptr()).clone();
            });
        out.create(n_imgs, 1, rtype);
        out.assign(out_vec);
    }

    void InputTensorFromImage(const cv::Mat& img, torch::Tensor& out_tensor, const std::array<double, 3>& norm_mean,
        const std::array<double, 3>& norm_std, const int pad_left, const int pad_top)
    {
        cv::Mat tmp;
        cv::cvtColor(img, tmp, cv::ColorConversionCodes::COLOR_BGR2RGB);
        tmp.convertTo(tmp, CV_32FC3, 1.0f / 255.0f);

        out_tensor = torch::from_blob(
            tmp.data, { 1, tmp.rows, tmp.cols, tmp.channels() }, torch::TensorOptions().dtype(torch::kFloat))
                         .clone();
        /* std::cout << "tensor size: " << out_tensor.sizes() << std::endl; */
        out_tensor = out_tensor.permute({ 0, 3, 1, 2 });
        /* std::cout << "tensor size after permute: " << out_tensor.sizes() << std::endl; */
        out_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(out_tensor);
        /* std::cout << "tensor size after normalization: " << out_tensor.sizes() << std::endl; */

        out_tensor = torch::nn::functional::pad(out_tensor,
            torch::nn::functional::PadFuncOptions({ pad_left, 0, pad_top, 0 }).mode(torch::kConstant).value(0));
        /* std::cout << "tensor size after padding: " << out_tensor.sizes() << std::endl; */
    }

    void InputTensorFromImages(cv::InputArrayOfArrays images, torch::Tensor& out_tensor,
        const std::array<double, 3>& norm_mean, const std::array<double, 3>& norm_std, const int pad_left,
        const int pad_top)
    {
        std::vector<cv::Mat> imgs;
        images.getMatVector(imgs);
        std::vector<torch::Tensor> tensors(imgs.size());

        std::transform(std::execution::par, imgs.begin(), imgs.end(), tensors.begin(), [](const cv::Mat& first) {
            cv::Mat tmp;
            cv::cvtColor(first, tmp, cv::ColorConversionCodes::COLOR_BGR2RGB);
            tmp.convertTo(tmp, CV_32FC3, 1.0f / 255.0f);
            return torch::from_blob(
                tmp.data, { 1, tmp.rows, tmp.cols, tmp.channels() }, torch::TensorOptions().dtype(torch::kFloat))
                .clone();
        });

        /* for (auto i = 0u; i < imgs.size(); i++) { */
        /*     cv::Mat tmp; */
        /*     cv::cvtColor(imgs[i], tmp, cv::ColorConversionCodes::COLOR_BGR2RGB); */
        /*     tmp.convertTo(tmp, CV_32FC3, 1.0f / 255.0f); */
        /*     tensors[i] = torch::from_blob( */
        /*         tmp.data, { 1, tmp.rows, tmp.cols, tmp.channels() },
         * torch::TensorOptions().dtype(torch::kFloat)).clone(); */
        /* } */

        if (tensors.size() > 1)
            out_tensor = torch::cat(tensors, 0);
        else
            out_tensor = tensors[0];

        /* std::cout << "tensor size: " << out_tensor.sizes() << std::endl; */
        out_tensor = out_tensor.permute({ 0, 3, 1, 2 });
        /* std::cout << "tensor size after permute: " << out_tensor.sizes() << std::endl; */
        out_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(out_tensor);
        /* std::cout << "tensor size after normalization: " << out_tensor.sizes() << std::endl; */

        out_tensor = torch::nn::functional::pad(out_tensor,
            torch::nn::functional::PadFuncOptions({ pad_left, 0, pad_top, 0 }).mode(torch::kConstant).value(0));
        /* std::cout << "tensor size after padding: " << out_tensor.sizes() << std::endl; */
    }

    void InputBlobFromImage(const cv::Mat& img, cv::Mat& out, const cv::Scalar& norm_mean, const cv::Scalar& norm_std,
        const int pad_left, const int pad_top)
    {
        /* std::cout << "prepareImage() - size input: " << img.rows << "x" << img.cols << "x" << img.channels() <<
         * std::endl;
         */
        cv::Scalar mean(norm_mean[2], norm_mean[1], norm_mean[0]);
        mean *= 255.0;
        cv::copyMakeBorder(img, out, pad_top, 0, pad_left, 0, cv::BorderTypes::BORDER_CONSTANT, mean);
        /* std::cout << "prepareImage() - size after pad: " << out.rows << "x" << out.cols << "x" << out.channels() <<
         * std::endl; */
        cv::dnn::blobFromImage(out, out, 1.0 / 255.0, cv::Size(), norm_mean * 255.0, true, false, CV_32F);
        /* std::cout << "blob size: " << out.size << std::endl; */
        cv::divide(out, norm_std, out);
    }
}
}
