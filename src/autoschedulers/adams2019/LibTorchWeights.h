#ifndef LIBTORCH_WEIGHTS_H
#define LIBTORCH_WEIGHTS_H

#include <torch/torch.h>
#include "Weights.h"
#include <string>
#include <memory>

namespace Halide {
namespace Internal {
    struct Weights;
}
}

namespace Halide {

// Optimized weight storage for LibTorch cost model
// Stores weights as LibTorch tensors for efficient access
class LibTorchWeights {
public:
    // Network architecture constants
    static constexpr int head1_channels = 8;
    static constexpr int head1_w = 40;
    static constexpr int head1_h = 7;
    static constexpr int head2_channels = 24;
    static constexpr int head2_w = 39;
    static constexpr int conv1_channels = 32;

    // Head1 weights (for pipeline features)
    torch::Tensor head1_filter;  // (head1_channels, head1_w, head1_h) - raw weights
    torch::Tensor head1_filter_sigmoided;  // (head1_channels, 1, head1_h, head1_w) - pre-computed sigmoided for conv
    torch::Tensor head1_bias;  // (head1_channels,)

    // Head2 weights (for schedule features)
    torch::Tensor head2_filter;  // (head2_channels, head2_w, 1) - for Conv1d
    torch::Tensor head2_bias;  // (head2_channels,)

    // Trunk weights (two-stage conv)
    torch::Tensor trunk_filter_stage1;  // (conv1_channels, head1_channels, 1) - for head1
    torch::Tensor trunk_filter_stage2;  // (conv1_channels, head2_channels, 1) - for head2
    torch::Tensor trunk_bias;  // (conv1_channels,) - only for stage1

	torch::Tensor trunk_fc_0;
	torch::Tensor trunk_fc_0_bias;

    LibTorchWeights();
    ~LibTorchWeights() = default;

    // Load weights from original Halide Weights format (for compatibility)
    bool load_from_halide_weights(const Internal::Weights &halide_weights);

    // Load weights from file (supports both .weights and directory format)
    bool load_from_file(const std::string &path);

    // Save weights to file (in original format for compatibility)
    bool save_to_file(const std::string &path) const;

    // Load weights from LibTorch format (direct tensor loading, faster)
    bool load_from_libtorch_file(const std::string &path);
	// same, but for generic set of weights (named parameters) within a map
    // format: "archive" (default, C++ OutputArchive), "torchscript" (Python torch.jit.save)
    bool load_from_libtorch_file_generic(const std::string &path, const std::string &format = "archive");

    // Save weights to LibTorch format (direct tensor saving, faster)
    bool save_to_libtorch_file(const std::string &path) const;
    bool save_to_libtorch_file_generic(const std::string &path) const;

    // Randomize weights (for testing)
    void randomize(uint32_t seed);

    // Randomize generic weights stored in model_weights_
    void randomize_generic(uint32_t seed);

    // Check if weights are loaded
    bool is_loaded() const;

    // Get weights in original Halide format (for saving)
    void to_halide_weights(Internal::Weights &halide_weights) const;

	// generalized weight tensor management
	
    void set_weight(const std::string &name, const torch::Tensor &tensor) {
        model_weights_[name] = tensor.clone();
    }
    
    // Remove a tensor
    void remove_weight(const std::string &name) {
        model_weights_.erase(name);
    }

    
    // Check if tensor exists
    bool has_weight(const std::string &name) const {
        return model_weights_.find(name) != model_weights_.end();
    }

    // Get a tensor by name
    torch::Tensor get_weight(const std::string &name) const {
        auto it = model_weights_.find(name);
        if (it == model_weights_.end()) {
            throw std::runtime_error("Weight not found: " + name);
        }
        return it->second;
    }
    template<typename T>
    static torch::Tensor buffer_to_tensor_public(const Halide::Runtime::Buffer<T> &buf);

private:
    bool loaded = false;
    // Helper: Convert Halide buffer to torch tensor
    template<typename T>
    torch::Tensor buffer_to_tensor(const Halide::Runtime::Buffer<T> &buf);

	// generalized weights list
	std::unordered_map<std::string, torch::Tensor> model_weights_;

};

}  // namespace Halide

#endif  // LIBTORCH_WEIGHTS_H

