
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>


using json = nlohmann::json;

DEFINE_string(label_path, "../data/imagenet_class_index.json", "imagenet json label path");
DEFINE_string(img_path, "../data/ILSVRC2012_test_00000439.JPEG", "image path");
DEFINE_string(model_path, "../data/resnet101.zip", "model path");
DEFINE_int32(topk, 3, "output topk result");

constexpr int kImgWidth = 224;
constexpr int kImgHeight = 224;
constexpr int kChannels = 3;

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::ifstream ifs(FLAGS_label_path);
  if (!ifs) {
    std::cerr << "invalid imagenet label path" << std::endl;
    return -1;
  }
  json js;
  // read imagenet json label
  ifs >> js;

  // read test image
  cv::Mat img = cv::imread(FLAGS_img_path);
  if (img.empty() || !img.data) {
    std::cerr << "invalid image path" << std::endl;
    return -1;
  }
  // convert color space
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::Size dst_size(kImgWidth, kImgHeight);
  // resize image
  cv::resize(img, img, dst_size);
  // convert to float and scale to [0,1]
  img.convertTo(img, CV_32FC3, 1.0f/255.0f);

  // construct tensor from image data
  auto img_tensor = torch::from_blob(img.data, {1, kImgHeight, kImgWidth, kChannels});
  // convert to NCHW
  img_tensor = img_tensor.permute({0, 3, 1, 2});
  // mean and std
  img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
  img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
  img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
  // to cuda
  img_tensor = img_tensor.to(at::kCUDA);

  // load model
  torch::jit::script::Module model = torch::jit::load(FLAGS_model_path);
  // to cuda
  model.to(at::kCUDA);

  auto output = model.forward({img_tensor}).toTensor();
  // sort the output
  auto ret = output.sort(-1, true);
  auto softmax = std::get<0>(ret)[0].softmax(0);
  auto index = std::get<1>(ret)[0];

  // output topk result
  for (int i = 0; i < FLAGS_topk; ++i) {
    auto idx = index[i].item<int>();
    auto idx_str = std::to_string(idx);
    auto label_name = js[idx_str];
    std::cout << "Top-" << i << " label name: " << label_name << ", probability: " << softmax[i].item<float>() * 100.f << "%" << std::endl;
  }

  return 0;
}

