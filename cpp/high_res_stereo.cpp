#include "high_res_stereo.hpp"
#include "utils.hpp"
/* #include <source_location> */
#include <torch/script.h>

namespace high_res_stereo {
HighResStereoMatcher::HighResStereoMatcher(const std::string& model_path, const torch::Device& target_device,
    bool traced, const std::array<double, 3>& norm_mean, const std::array<double, 3>& norm_std)
    : model_file_path_(model_path)
    , norm_mean_(norm_mean)
    , norm_std_(norm_std)
    , target_device_(torch::kCPU)
    , set_clean_(std::nullopt)
    , set_level_(std::nullopt)
    , set_max_disp_(std::nullopt)
    , traced_(traced)
{
    if (target_device.type() == torch::kCUDA && !torch::cuda::is_available()) {
        utils::PrintError("CUDA device chosen, but CUDA not available -> target device set to CPU" + model_path,
            __func__, __FILE__, __LINE__);
        /* auto loc = std::source_location::current(); */
        /* PrintError("CUDA device chosen, but CUDA not available -> target device set to CPU", loc.function_name(),
         * loc.file_name(), loc.line()); */
    } else {
        target_device_ = target_device;
    }
    try {
        model_ = torch::jit::load(model_path, target_device_);
        model_.eval();
    } catch (const torch::Error& e) {
        utils::PrintError("Error loading model", __func__, __FILE__, __LINE__);
        /* auto loc = std::source_location::current(); */
        /* PrintError("Error loading model", loc.function_name(), loc.file_name(), loc.line()); */
        throw;
    }
    try {
        get_clean_ = model_.get_method("get_clean");
        get_level_ = model_.get_method("get_level");
        get_max_disp_ = model_.get_method("get_max_disp");

        if (!traced) {
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
}

bool HighResStereoMatcher::set_target_device(const torch::Device& target_device)
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
    return updated;
}

bool HighResStereoMatcher::WarmUpModel(cv::Size img_size, unsigned int n_runs)
{
    bool warmed_up = false;
    try {
        auto net_img_size = utils::CalculateNetworkImgSize(img_size);

        torch::NoGradGuard no_grad;

        for (auto i = 0u; i < n_runs; i++) {
            torch::Tensor lTensTmp = torch::zeros({ 1, 3, net_img_size.height, net_img_size.width });
            torch::Tensor rTensTmp = torch::zeros({ 1, 3, net_img_size.height, net_img_size.width });
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
        warmed_up = true;
    } catch (const torch::Error& e) {
        utils::PrintError("Error running warm up for model", __func__, __FILE__, __LINE__, e.what());
        warmed_up = false;
    }
    return warmed_up;
}

void HighResStereoMatcher::CalculateDisparity(
    cv::InputArray img_left, cv::InputArray img_right, cv::OutputArray disparity, cv::OutputArray entropy)
{
    cv::Mat img_l = img_left.getMat();
    cv::Mat img_r = img_right.getMat();

    if (img_l.size() != img_r.size()) {
        throw std::invalid_argument(
            utils::InitErrorMsg("left and right images have different sizes", __func__, __FILE__, __LINE__));
    }
    if (img_l.empty()) {
        throw std::invalid_argument(utils::InitErrorMsg("input images empty", __func__, __FILE__, __LINE__));
    }
    if (img_l.channels() != 3 || img_l.channels() != img_r.channels()) {
        throw std::invalid_argument(utils::InitErrorMsg(
            "number of channels != 3 for at least one of the input images", __func__, __FILE__, __LINE__));
    }

    // Disable gradient calculation
    torch::NoGradGuard no_grad;

    cv::Size img_size(img_l.cols, img_l.rows);
    auto net_img_size = utils::CalculateNetworkImgSize(img_size);
    std::tuple<int, int> padding = utils::CalculateImgPadding(img_size, net_img_size);

    torch::Tensor t_img_l, t_img_r;

    utils::InputTensorFromImage(img_l, t_img_l, norm_mean_, norm_std_, std::get<0>(padding), std::get<1>(padding));
    utils::InputTensorFromImage(img_r, t_img_r, norm_mean_, norm_std_, std::get<0>(padding), std::get<1>(padding));
    t_img_l = t_img_l.to(target_device_);
    t_img_r = t_img_r.to(target_device_);

    auto result = model_.forward({ t_img_l, t_img_r }).toTuple();

    auto t_disp = result->elements()[0].toTensor();
    auto disp = utils::TorchTensorToCVMat(t_disp, CV_32FC1, 1, std::get<0>(padding) * -1, std::get<1>(padding) * -1);
    // already done in TorchTensorToCVMat
    /* disp = cv::Mat(disp, cv::Rect(std::get<0>(padding), std::get<1>(padding), img_size.width, img_size.height)); */

    /* // set inf to -1 */
    /* cv::Mat inf_mask = disp == std::numeric_limits<float>::infinity(); */
    /* disp.setTo(-1.0f, inf_mask); */
    disparity.assign(disp);

    if (entropy.needed()) {
        if (get_clean() >= 0.0) {
            auto t_ent = result->elements()[1].toTensor();
            auto ent = utils::TorchTensorToCVMat(t_ent, CV_32FC1, 1, std::get<0>(padding) * -1, std::get<1>(padding) * -1);
            // already done in TorchTensorToCVMat
            /* ent = cv::Mat(ent, cv::Rect(std::get<0>(padding), std::get<1>(padding), img_size.width, img_size.height)); */
            entropy.assign(ent);
        } else {
            entropy.clear();
        }
    }
}

void HighResStereoMatcher::CalculateDisparities(cv::InputArrayOfArrays imgs_left, cv::InputArrayOfArrays imgs_right,
    cv::OutputArrayOfArrays disparities, cv::OutputArrayOfArrays entropies)
{
    std::vector<cv::Mat> imgs_l;
    imgs_left.getMatVector(imgs_l);
    std::vector<cv::Mat> imgs_r;
    imgs_right.getMatVector(imgs_r);

    if (imgs_l.empty() || imgs_r.empty()) {
        throw std::invalid_argument(
            utils::InitErrorMsg("one of the input image batches is empty", __func__, __FILE__, __LINE__));
    }
    if (imgs_l.size() != imgs_r.size()) {
        throw std::invalid_argument(utils::InitErrorMsg(
            "number of input images differ between left and right batches", __func__, __FILE__, __LINE__));
    }
    cv::Size img_size = imgs_l[0].size();
    if (img_size.empty()) {
        throw std::invalid_argument(utils::InitErrorMsg("input image size empty", __func__, __FILE__, __LINE__));
    }
    for (auto i=0u; i<imgs_l.size(); i++) {
        if (imgs_r[i].empty() || imgs_l[i].empty()) {
            throw std::invalid_argument(
                utils::InitErrorMsg("at least one input image is empty", __func__, __FILE__, __LINE__));
        }
        if (imgs_l[i].size() != img_size && imgs_l[i].size() != imgs_r[i].size()) {
            throw std::invalid_argument(utils::InitErrorMsg(
                "image size differs for at least one input image", __func__, __FILE__, __LINE__));
        }
        if (imgs_l[i].channels() != 3 || imgs_l[i].channels() != imgs_r[i].channels()) {
            throw std::invalid_argument(utils::InitErrorMsg(
                "number of channels != 3 for at least one of the input images", __func__, __FILE__, __LINE__));
        }
    }
    // Disable gradient calculation
    torch::NoGradGuard no_grad;

    auto net_img_size = utils::CalculateNetworkImgSize(img_size);
    std::tuple<int, int> padding = utils::CalculateImgPadding(img_size, net_img_size);

    torch::Tensor t_imgs_l, t_imgs_r;

    utils::InputTensorFromImages(imgs_l, t_imgs_l, norm_mean_, norm_std_, std::get<0>(padding), std::get<1>(padding));
    utils::InputTensorFromImages(imgs_r, t_imgs_r, norm_mean_, norm_std_, std::get<0>(padding), std::get<1>(padding));
    t_imgs_l = t_imgs_l.to(target_device_);
    t_imgs_r = t_imgs_r.to(target_device_);

    auto result = model_.forward({ t_imgs_l, t_imgs_r }).toTuple();

    auto t_disp = result->elements()[0].toTensor();
    std::vector<cv::Mat> disps;
    utils::TorchTensorToCVMats(t_disp, disps, CV_32FC1, 1, std::get<0>(padding) * -1, std::get<1>(padding) * -1);
    // already done in TorchTensorToCVMat
    /* std::for_each(std::execution::par, disps.begin(), disps.end(), [&padding, &img_size](cv::Mat& disp) { */
    /*     disp = cv::Mat(disp, cv::Rect(std::get<0>(padding), std::get<1>(padding), img_size.width, img_size.height)); */
    /* }); */

    disparities.create(disps.size(), 1, CV_32FC1);
    disparities.assign(disps);

    if (entropies.needed()) {
        if (get_clean() >= 0.0) {
            auto t_ent = result->elements()[1].toTensor();
            std::vector<cv::Mat> ents;
            utils::TorchTensorToCVMats(t_ent, ents, CV_32FC1, 1, std::get<0>(padding) * -1, std::get<1>(padding) * -1);
            // already done in TorchTensorToCVMat
            /* std::for_each(std::execution::par, ents.begin(), ents.end(), [&padding, &img_size](cv::Mat& ent) { */
            /*     ent = cv::Mat(ent, cv::Rect(std::get<0>(padding), std::get<1>(padding), img_size.width, img_size.height)); */
            /* }); */

            entropies.create(ents.size(), 1, CV_32FC1);
            entropies.assign(ents);
        }
    }
}
}
