#pragma once

#include <torch/script.h>
#include <torch/torch.h>

namespace ml_torch_stereo {

    class MLTorchStereoMatcher {
        public:
            inline void set_target_device(torch::Device target_device) { this->target_device_ = target_device; }
            inline torch::Device target_device() { return target_device_; }

        private:
            std::shared_ptr<torch::jit::script::Module> model_;
            torch::Device target_device_;
    };
}
