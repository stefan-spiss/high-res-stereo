#include "high_res_stereo.hpp"
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void GetDisparityVisualize(const cv::Mat& disparity, cv::Mat& disparity_visualized, int max_disp, int min_disp = 0)
{
    double num_disp = static_cast<double>(max_disp - min_disp);
    double scale = 256.0 / num_disp;
    disparity.convertTo(disparity_visualized, CV_8UC1, scale, scale * static_cast<double>(-1 * min_disp));
    cv::cvtColor(disparity_visualized, disparity_visualized, cv::ColorConversionCodes::COLOR_GRAY2RGB);
}

int main(int argc, const char* argv[])
{
    const std::array<double, 3> image_net_mean = { 0.485, 0.456, 0.406 };
    const std::array<double, 3> image_net_std = { 0.229, 0.224, 0.225 };

    const std::string keys
        = "{help h usage ?      |      |print this message}"
          "{@model              |<none>|path to scripted or traced model file (.pt)}"
          "{@left_image         |<none>|path to input image (.png, .jpg, ...)}"
          "{@right_image        |<none>|path to input image (.png, .jpg, ...)}"
          "{traced              |false |must be true if traced model used}"
          "{clean               |1.0   |entropy threshold to be used for cleaning (filtering) the "
          "disparity, < 0.0: no cleaning}"
          "{cuda                |true  |if true, cuda is used when possible, otherwise cpu}"
          "{level               |1     |level of decoder network to use as output (level 1: highest "
          "resolution, level 3: lowest resolution)}"
          "{max_disp            |-1    |maximum disparity, if -1, default value of model is used)}"
          "{res_scale           |1.0   |output resolution multiplier}"
          "{n_runs              |1     |number of runs, the matching should be performed (for runtime measurement)}"
          "{output_folder_path  |      |if not empty, disparity images are stored there as disparity.png and "
          "disparity.pfm}";
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string model_file_path = parser.get<std::string>(0);
    std::string image_file_path_left = parser.get<std::string>(1);
    std::string image_file_path_right = parser.get<std::string>(2);
    bool traced = parser.get<bool>("traced");
    float clean = parser.get<float>("clean");
    bool run_cuda = parser.get<bool>("cuda");
    int level = parser.get<int>("level");
    int max_disp = parser.get<int>("max_disp");
    float resolution_scale = parser.get<float>("res_scale");
    int n_runs = parser.get<int>("n_runs");
    std::string output_folder_path = parser.get<std::string>("output_folder_path");

    if (!parser.check()) {
        parser.printMessage();
        parser.printErrors();
        return 0;
    }

    high_res_stereo::HighResStereoMatcher stereo_matcher(model_file_path,
        run_cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU), traced, image_net_mean, image_net_std);

    auto initial_clean = stereo_matcher.get_clean();
    if (stereo_matcher.set_clean(clean)) {
        std::cout << "clean set from " << initial_clean << " to " << stereo_matcher.get_clean() << std::endl;
    } else {
        std::cout << "clean not updated: " << initial_clean << std::endl;
    }

    auto initial_level = stereo_matcher.get_level();
    if (stereo_matcher.set_level(level)) {
        std::cout << "level set from " << initial_level << " to " << stereo_matcher.get_level() << std::endl;
    } else {
        std::cout << "level not updated: " << initial_level << std::endl;
    }

    auto initial_max_disp = stereo_matcher.get_max_disp();
    if (stereo_matcher.set_max_disp(static_cast<int>(max_disp * resolution_scale))) {
        std::cout << "max_disp set from " << initial_max_disp << " to " << stereo_matcher.get_max_disp();
        if (std::abs(resolution_scale - 1.0) > std::numeric_limits<float>::epsilon()) {
            std::cout << " -- scaled disparity used" << std::endl;
        } else {
            std::cout << std::endl;
        }
    } else {
        std::cout << "max_disp not updated: " << initial_max_disp << std::endl;
    }

    cv::Mat left_img = cv::imread(image_file_path_left, cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread(image_file_path_right, cv::IMREAD_COLOR);

    cv::Size input_img_size(left_img.cols, left_img.rows);
    std::cout << "input image size: " << input_img_size << std::endl;

    if (std::abs(resolution_scale - 1.0f) > std::numeric_limits<float>::epsilon()) {
        cv::resize(
            left_img, left_img, cv::Size(), resolution_scale, resolution_scale, cv::InterpolationFlags::INTER_CUBIC);
        cv::resize(
            right_img, right_img, cv::Size(), resolution_scale, resolution_scale, cv::InterpolationFlags::INTER_CUBIC);
    }
    cv::Size rescaled_img_size(left_img.cols, left_img.rows);
    std::cout << "rescaled image size: " << rescaled_img_size << std::endl;

    auto warmed_up = stereo_matcher.WarmUpModel(rescaled_img_size);
    if (warmed_up) {
        std::cout << "model warmed up" << std::endl;
    } else {
        std::cout << "model warm up failed" << std::endl;
    }

    std::vector<double> times(n_runs);

    cv::Mat disparity, entropy;
    for (auto i = 0; i < n_runs; i++) {
        if (run_cuda)
            torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        stereo_matcher.CalculateDisparity(left_img, right_img, disparity, entropy);
        if (run_cuda)
            torch::cuda::synchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << "run: " << i << " - runtime: " << times[i] << std::endl;
    }

    if (n_runs > 1) {
        std::cout << "mean runtime: " << std::reduce(times.begin(), times.end()) / static_cast<float>(times.size())
                  << std::endl;
    }

    cv::resize(disparity / resolution_scale, disparity, input_img_size);

    cv::Mat disp_vis;
    GetDisparityVisualize(disparity, disp_vis, stereo_matcher.get_max_disp() / resolution_scale, 0);
    cv::imshow("disparity", disp_vis);
    if (!entropy.empty()) {
        cv::resize(entropy, entropy, input_img_size);
        cv::imshow("entropy", entropy);
    }
    cv::waitKey(0);

    std::filesystem::path out_path = output_folder_path;
    if (!output_folder_path.empty() && std::filesystem::is_directory(out_path)) {
        std::stringstream ss;
        ss << "disp_" << std::filesystem::path(image_file_path_left).stem().string() << "_clean-"
           << stereo_matcher.get_clean() << "_level-" << stereo_matcher.get_level() << "_max_disp-"
           << stereo_matcher.get_max_disp() / resolution_scale << "_res-scale-" << resolution_scale;
        cv::imwrite(out_path / (ss.str() + ".png"), disp_vis);
        cv::imwrite(out_path / (ss.str() + ".pfm"), disparity);
    }

    return 0;
}
