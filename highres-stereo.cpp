#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int floorDivide(float a, float b) { return static_cast<int>(std::floor(a / b)); }

void getDisparityVisualize(const cv::Mat& disparity, cv::Mat& disparityVisualize, int outputNumDisp)
{
  double min, max;
  cv::minMaxLoc(disparity, &min, &max);
  /* std::cout << "min = " << min << " max = " << max << std::endl; */
  // +1 since both max and min are included
  double numDispOut = outputNumDisp <= 0 ? max - min + 1.0 : static_cast<double>(outputNumDisp);
  double scale = 256.0 / numDispOut;
  disparity.convertTo(disparityVisualize, CV_8UC1, scale, scale * -1.0 * min);
  cv::cvtColor(disparityVisualize, disparityVisualize, cv::ColorConversionCodes::COLOR_GRAY2RGB);
}

cv::Mat torchTensorToCVMat(torch::Tensor& tensor, int rtype, int channels)
{
  tensor = tensor.squeeze().detach();
  if (channels == 3)
    tensor = tensor.permute({ 1, 2, 0 }).contiguous().to(torch::kCPU);
  else if (channels == 1)
    tensor = tensor.contiguous().to(torch::kCPU);
  return cv::Mat(tensor.size(0), tensor.size(1), rtype, tensor.data_ptr()).clone();
}

void inputTensorFromImage(const cv::Mat& img, at::Tensor& outTensor, const std::array<double, 3>& normMean,
    const std::array<double, 3>& normStd, const int padLeft = 0, const int padRight = 0, const int padTop = 0,
    const int padBottom = 0)
{
  cv::Mat tmp;
  cv::cvtColor(img, tmp, cv::ColorConversionCodes::COLOR_BGR2RGB);
  tmp.convertTo(tmp, CV_32FC3, 1.0f / 255.0f);

  outTensor = torch::from_blob(tmp.data, { 1, tmp.rows, tmp.cols, tmp.channels() }, at::kFloat);
  /* std::cout << "tensor size: " << outTensor.sizes() << std::endl; */
  outTensor = outTensor.permute({ 0, 3, 1, 2 });
  /* std::cout << "tensor size after permute: " << outTensor.sizes() << std::endl; */
  outTensor = torch::data::transforms::Normalize<>(normMean, normStd)(outTensor);
  /* std::cout << "tensor size after normalization: " << outTensor.sizes() << std::endl; */

  outTensor = torch::nn::functional::pad(
      outTensor, torch::nn::functional::PadFuncOptions({ padLeft, 0, padTop, 0 }).mode(torch::kConstant).value(0));
  /* std::cout << "tensor size after padding: " << outTensor.sizes() << std::endl; */
}

void inputBlobFromImage(const cv::Mat& img, cv::Mat& out, const cv::Scalar& normMean, const cv::Scalar& normStd,
    const int padLeft = 0, const int padRight = 0, const int padTop = 0, const int padBottom = 0)
{
  /* std::cout << "prepareImage() - size input: " << img.rows << "x" << img.cols << "x" << img.channels() << std::endl;
   */
  cv::Scalar mean(normMean[2], normMean[1], normMean[0]);
  mean *= 255.0;
  cv::copyMakeBorder(img, out, padTop, padBottom, padLeft, padRight, cv::BorderTypes::BORDER_CONSTANT, mean);
  /* std::cout << "prepareImage() - size after pad: " << out.rows << "x" << out.cols << "x" << out.channels() <<
   * std::endl; */
  cv::dnn::blobFromImage(out, out, 1.0 / 255.0, cv::Size(), normMean * 255.0, true, false, CV_32F);
  /* std::cout << "blob size: " << out.size << std::endl; */
  cv::divide(out, normStd, out);
}

int main(int argc, const char* argv[])
{
  /* const auto imageNetMean = torch::tensor({0.485, 0.456, 0.406}, {torch::kFloat}); */
  /* const auto imageNetStd = torch::tensor({0.229, 0.224, 0.225}, {torch::kFloat}); */
  const std::array<double, 3> imageNetMean = { 0.485, 0.456, 0.406 };
  const std::array<double, 3> imageNetStd = { 0.229, 0.224, 0.225 };

  const std::string keys = "{help h usage ?      |      |print this message}"
                           "{@model              |<none>|path to traced model file (.pt)}"
                           "{@leftImage          |<none>|path to input image (.png, .jpg, ...)}"
                           "{@rightImage         |<none>|path to input image (.png, .jpg, ...)}"
                           "{cuda                |true  |if true, cuda is used when possible, otherwise cpu}"
                           "{resScale            |1.0   |output resolution multiplier}";
  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::string modelFilePath = parser.get<std::string>(0);
  std::string imageFilePathLeft = parser.get<std::string>(1);
  std::string imageFilePathRight = parser.get<std::string>(2);
  bool runCuda = parser.get<bool>("cuda");
  float resolutionScale = parser.get<float>("resScale");

  if (!parser.check()) {
    parser.printMessage();
    parser.printErrors();
    return 0;
  }

  torch::jit::script::Module model;
  torch::Device targetDevice = torch::kCPU;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(modelFilePath);

    if (runCuda && torch::cuda::is_available()) {
      std::cout << "GPU available" << std::endl;
      targetDevice = torch::kCUDA;
    }
    model.eval();
    model.to(targetDevice);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  cv::Mat leftImg = cv::imread(imageFilePathLeft, cv::IMREAD_COLOR);
  cv::Mat rightImg = cv::imread(imageFilePathRight, cv::IMREAD_COLOR);

  CV_Assert(leftImg.channels() == rightImg.channels() && leftImg.channels() == 3);
  CV_Assert(leftImg.rows == rightImg.rows);
  CV_Assert(leftImg.cols == rightImg.cols);

  cv::Size inputImgSize(leftImg.cols, leftImg.rows);
  std::cout << "input image size: " << inputImgSize << std::endl;

  if (std::abs(resolutionScale - 1.0f) > std::numeric_limits<float>::epsilon()) {
    cv::resize(leftImg, leftImg, cv::Size(), resolutionScale, resolutionScale, cv::InterpolationFlags::INTER_CUBIC);
    cv::resize(rightImg, rightImg, cv::Size(), resolutionScale, resolutionScale, cv::InterpolationFlags::INTER_CUBIC);
  }
  cv::Size rescaledImgSize(leftImg.cols, leftImg.rows);
  std::cout << "size after rescale: " << rescaledImgSize << std::endl;

  int hOut = floorDivide(rescaledImgSize.height, 64.0f) * 64;
  int wOut = floorDivide(rescaledImgSize.width, 64.0f) * 64;

  hOut = hOut < rescaledImgSize.height ? hOut + 64 : hOut;
  wOut = wOut < rescaledImgSize.width ? wOut + 64 : wOut;

  int leftPad = wOut - rescaledImgSize.width;
  int rightPad = 0;
  int topPad = hOut - rescaledImgSize.height;
  int bottomPad = 0;

  torch::NoGradGuard no_grad;

  // warmup
  for (auto i = 0; i < 2; i++) {
    cv::Mat lTmp(inputImgSize.height, inputImgSize.width, CV_8UC3, { 0, 0, 0 });
    cv::Mat rTmp = lTmp.clone();
    at::Tensor lTensTmp, rTensTmp;
    inputTensorFromImage(lTmp, lTensTmp, imageNetMean, imageNetStd, leftPad, rightPad, topPad, bottomPad);
    inputTensorFromImage(rTmp, rTensTmp, imageNetMean, imageNetStd, leftPad, rightPad, topPad, bottomPad);
    lTensTmp = lTensTmp.to(targetDevice);
    rTensTmp = rTensTmp.to(targetDevice);

    double processingTime;
    if (targetDevice == torch::kCUDA)
      torch::cuda::synchronize();
    auto start = std::chrono::high_resolution_clock::now();
    model.forward({ lTensTmp, rTensTmp });
    if (targetDevice == torch::kCUDA)
      torch::cuda::synchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Warm up - run " << i << " - Runtime: " << processingTime << std::endl;
  }

  at::Tensor tensorLeftImg, tensorRightImg;

  std::cout << "Prepare image left:" << std::endl;
  inputTensorFromImage(leftImg, tensorLeftImg, imageNetMean, imageNetStd, leftPad, rightPad, topPad, bottomPad);
  std::cout << "Prepare image right:" << std::endl;
  inputTensorFromImage(rightImg, tensorRightImg, imageNetMean, imageNetStd, leftPad, rightPad, topPad, bottomPad);

  /* std::cout << "Prepare image left:" << std::endl; */
  /* cv::Mat outLeft; */
  /* inputBlobFromImage(leftImg, outLeft, cv::Scalar(imageNetMean[0], imageNetMean[1], imageNetMean[2]),
   * cv::Scalar(imageNetStd[0], imageNetStd[1], imageNetStd[2]), leftPad, rightPad, topPad, bottomPad); */
  /* at::Tensor tensorLeftImgTest = torch::from_blob(outLeft.data, {outLeft.size[0], outLeft.size[1], outLeft.size[2],
   * outLeft.size[3]}, at::kFloat); */
  /* std::cout << torch::equal(tensorLeftImg, tensorLeftImgTest) << std::endl; */

  /* tensorLeftImg = tensorLeftImg.squeeze().detach(); */
  /* tensorLeftImg = tensorLeftImg.permute({1, 2, 0}).contiguous(); */
  /* cv::Mat tLeftImg(tensorLeftImg.sizes()[0], tensorLeftImg.sizes()[1], CV_32FC3, tensorLeftImg.data_ptr()); */
  /* tLeftImg.convertTo(tLeftImg, CV_8UC3, 255); */
  /* cv::cvtColor(tLeftImg, tLeftImg, cv::ColorConversionCodes::COLOR_RGB2BGR); */
  /* cv::imshow("torch version", tLeftImg); */

  /* tensorLeftImgTest = tensorLeftImgTest.squeeze().detach().permute({1, 2, 0}).contiguous(); */
  /* cv::Mat tLeftImgTest(tensorLeftImgTest.sizes()[0], tensorLeftImgTest.sizes()[1], CV_32FC3,
   * tensorLeftImgTest.data_ptr()); */
  /* tLeftImgTest.convertTo(tLeftImgTest, CV_8UC3, 255); */
  /* cv::cvtColor(tLeftImgTest, tLeftImgTest, cv::ColorConversionCodes::COLOR_RGB2BGR); */
  /* cv::imshow("cv version", tLeftImgTest); */

  /* cv::imshow("cv diff", tLeftImg - tLeftImgTest); */
  /* cv::waitKey(0); */

  tensorLeftImg = tensorLeftImg.to(targetDevice);
  tensorRightImg = tensorRightImg.to(targetDevice);

  double processingTime;
  if (targetDevice == torch::kCUDA)
    torch::cuda::synchronize();
  auto start = std::chrono::high_resolution_clock::now();
  auto result = model.forward({ tensorLeftImg, tensorRightImg }).toTuple();
  if (targetDevice == torch::kCUDA)
    torch::cuda::synchronize();
  auto stop = std::chrono::high_resolution_clock::now();
  processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  std::cout << "Runtime: " << processingTime << std::endl;

  /* start = std::chrono::high_resolution_clock::now(); */
  /* result = model.forward({ tensorLeftImg, tensorRightImg }).toTuple(); */
  /* stop = std::chrono::high_resolution_clock::now(); */
  /* processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); */
  /* std::cout << "Runtime: " << processingTime << std::endl; */

  auto dispTensor = result->elements()[0].toTensor();
  auto entropyTensor = result->elements()[1].toTensor();
  auto disparity = torchTensorToCVMat(dispTensor, CV_32FC1, 1);
  auto entropy = torchTensorToCVMat(entropyTensor, CV_32FC1, 1);

  /* disparity.forEach<float>([](float &p, const int* position) -> void { */
  /*   if (std::isnan(p) || std::isinf(p)) { */
  /*     p = 0.0f; */
  /*     std::cout << (std::isnan(p) ? "nan" : "inf") << std::endl; */
  /*   } */
  /* }); */

  disparity = cv::Mat(disparity, cv::Rect(leftPad, topPad, rescaledImgSize.width, rescaledImgSize.height));
  entropy = cv::Mat(entropy, cv::Rect(leftPad, topPad, rescaledImgSize.width, rescaledImgSize.height));

  cv::resize(disparity / resolutionScale, disparity, inputImgSize);
  std::cout << disparity.size << std::endl;

  cv::Mat dispVis;
  getDisparityVisualize(disparity, dispVis, 255);
  cv::imshow("disparity", dispVis);
  /* cv::Mat entVis; */
  /* cv::imshow("entropy", entropy); */
  cv::waitKey(0);

  std::cout << "ok\n";
}
