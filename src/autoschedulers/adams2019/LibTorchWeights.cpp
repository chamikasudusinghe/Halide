#include "LibTorchWeights.h"
#include "Weights.h"
#include "NetworkSize.h"
#include "Errors.h"
#include "ASLog.h"
#include <random>
#include <fstream>
#include <iostream>

using Halide::Internal::aslog;
using Halide::Internal::Weights;

namespace Halide {

LibTorchWeights::LibTorchWeights() {
    // Initialize tensors with proper shapes
    head1_filter = torch::zeros({head1_channels, head1_w, head1_h}, torch::kFloat32);
    head1_bias = torch::zeros({head1_channels}, torch::kFloat32);
    head2_filter = torch::zeros({head2_channels, head2_w, 1}, torch::kFloat32);
    head2_bias = torch::zeros({head2_channels}, torch::kFloat32);
    trunk_filter_stage1 = torch::zeros({conv1_channels, head1_channels, 1}, torch::kFloat32);
    trunk_filter_stage2 = torch::zeros({conv1_channels, head2_channels, 1}, torch::kFloat32);
    trunk_bias = torch::zeros({conv1_channels}, torch::kFloat32);

	// BHsketch: adding new filters for applying a fully connected layer to each 
	// channel in concat(head1_out, head2_out)
	//trunk_fc_0 = torch::zeros({conv1_channels, 2*conv1_channels, 1}, torch::kFloat32);
	//trunk_fc_0_bias = torch::zeros({conv1_channels}, torch::kFloat32);
	auto opts = torch::TensorOptions().dtype(torch::kFloat32);
	trunk_fc_0 = torch::normal(0.0, 0.1, {conv1_channels, 2 * conv1_channels, 1}, /*generator=*/c10::nullopt, opts);
	trunk_fc_0_bias = torch::normal(0.0, 0.1, {conv1_channels}, /*generator=*/c10::nullopt, opts);
}

template<typename T>
torch::Tensor LibTorchWeights::buffer_to_tensor_public(const Halide::Runtime::Buffer<T> &buf) {
    std::vector<int64_t> shape;
    for (int i = 0; i < buf.dimensions(); i++) {
        shape.push_back(buf.dim(i).extent());
    }
    
    // Create tensor and manually copy data to ensure correct layout
    // Halide buffers may have non-contiguous strides, so we can't use from_blob directly
    auto tensor = torch::zeros(shape, torch::kFloat32);
    auto tensor_data = tensor.data_ptr<float>();
    
    // Copy data respecting Halide buffer's dimension order
    // For a 3D buffer (c, w, h), iterate in that order
    if (buf.dimensions() == 3) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    int64_t tensor_idx = i * shape[1] * shape[2] + j * shape[2] + k;
                    tensor_data[tensor_idx] = buf(i, j, k);
                }
            }
        }
    } else if (buf.dimensions() == 2) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                int64_t tensor_idx = i * shape[1] + j;
                tensor_data[tensor_idx] = buf(i, j);
            }
        }
    } else if (buf.dimensions() == 1) {
        for (int i = 0; i < shape[0]; i++) {
            tensor_data[i] = buf(i);
        }
    } else {
        // Fallback: use from_blob for other dimensions
        tensor = torch::from_blob((void*)buf.data(), shape, torch::kFloat32).clone();
    }
    
    return tensor;
}

template<typename T>
torch::Tensor LibTorchWeights::buffer_to_tensor(const Halide::Runtime::Buffer<T> &buf) {
    std::vector<int64_t> shape;
    for (int i = 0; i < buf.dimensions(); i++) {
        shape.push_back(buf.dim(i).extent());
    }
    
    // Create tensor and manually copy data to ensure correct layout
    // Halide buffers may have non-contiguous strides, so we can't use from_blob directly
    auto tensor = torch::zeros(shape, torch::kFloat32);
    auto tensor_data = tensor.data_ptr<float>();
    
    // Copy data respecting Halide buffer's dimension order
    // For a 3D buffer (c, w, h), iterate in that order
    if (buf.dimensions() == 3) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    int64_t tensor_idx = i * shape[1] * shape[2] + j * shape[2] + k;
                    tensor_data[tensor_idx] = buf(i, j, k);
                }
            }
        }
    } else if (buf.dimensions() == 2) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                int64_t tensor_idx = i * shape[1] + j;
                tensor_data[tensor_idx] = buf(i, j);
            }
        }
    } else if (buf.dimensions() == 1) {
        for (int i = 0; i < shape[0]; i++) {
            tensor_data[i] = buf(i);
        }
    } else {
        // Fallback: use from_blob for other dimensions
        tensor = torch::from_blob((void*)buf.data(), shape, torch::kFloat32).clone();
    }
    
    return tensor;
}

bool LibTorchWeights::load_from_halide_weights(const Internal::Weights &halide_weights) {
    try {
        // Load head1 weights: (head1_channels, head1_w, head1_h)
        head1_filter = buffer_to_tensor(halide_weights.head1_filter);
        
        // Reshape for Conv2d and apply sigmoid: (head1_channels, 1, head1_h, head1_w)
        auto head1_w_reshaped = head1_filter.unsqueeze(1); // (head1_channels, 1, head1_w, head1_h)
        head1_w_reshaped = head1_w_reshaped.permute({0, 1, 3, 2}); // (head1_channels, 1, head1_h, head1_w)
        head1_filter_sigmoided = torch::sigmoid(head1_w_reshaped); // Pre-compute sigmoided weights
        
        head1_bias = buffer_to_tensor(halide_weights.head1_bias);

        // Load head2 weights: (head2_channels, head2_w)
        auto head2_w_tensor = buffer_to_tensor(halide_weights.head2_filter);
        head2_filter = head2_w_tensor.unsqueeze(2); // (head2_channels, head2_w, 1)
        head2_bias = buffer_to_tensor(halide_weights.head2_bias);

        // Load trunk weights: (conv1_channels, head1_channels + head2_channels)
        auto trunk_w_tensor = buffer_to_tensor(halide_weights.conv1_filter);
        
        // Split into two stages
        trunk_filter_stage1 = trunk_w_tensor.slice(1, 0, head1_channels).unsqueeze(2); // (conv1_channels, head1_channels, 1)
        trunk_filter_stage2 = trunk_w_tensor.slice(1, head1_channels, head1_channels + head2_channels).unsqueeze(2); // (conv1_channels, head2_channels, 1)
        
        trunk_bias = buffer_to_tensor(halide_weights.conv1_bias);

        loaded = true;
        return true;
    } catch (...) {
        aslog(0) << "Error loading weights from Halide format\n";
        return false;
    }
}

bool LibTorchWeights::load_from_file(const std::string &path) {
    Internal::Weights halide_weights;
    
    if (path.empty()) {
        aslog(1) << "LibTorchWeights: No weights path specified, using random initialization\n";
        halide_weights.randomize(0);
        return load_from_halide_weights(halide_weights);
    }

    bool loaded_halide = false;
    if (path.size() >= 8 && path.substr(path.size() - 8) == ".weights") {
        loaded_halide = halide_weights.load_from_file(path);
    } else {
        loaded_halide = halide_weights.load_from_dir(path);
    }

    if (!loaded_halide) {
        aslog(1) << "LibTorchWeights: Failed to load weights from " << path << ", using random initialization\n";
        halide_weights.randomize(0);
    } else {
        aslog(1) << "LibTorchWeights: Loaded weights from " << path << "\n";
    }

    return load_from_halide_weights(halide_weights);
}

bool LibTorchWeights::save_to_file(const std::string &path) const {
    if (!loaded) {
        aslog(0) << "LibTorchWeights: Cannot save - weights not loaded\n";
        return false;
    }

    Internal::Weights halide_weights;
    to_halide_weights(halide_weights);

    if (path.size() >= 8 && path.substr(path.size() - 8) == ".weights") {
        return halide_weights.save_to_file(path);
    } else {
        return halide_weights.save_to_dir(path);
    }
}

void LibTorchWeights::randomize(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Randomize all weights
    auto randomize_tensor = [&](torch::Tensor &t) {
        auto data = t.data_ptr<float>();
        for (int64_t i = 0; i < t.numel(); i++) {
            data[i] = dist(rng);
        }
    };

    randomize_tensor(head1_filter);
    randomize_tensor(head1_bias);
    randomize_tensor(head2_filter);
    randomize_tensor(head2_bias);
    randomize_tensor(trunk_filter_stage1);
    randomize_tensor(trunk_filter_stage2);
    randomize_tensor(trunk_bias);

    // Recompute sigmoided weights
    auto head1_w_reshaped = head1_filter.unsqueeze(1);
    head1_w_reshaped = head1_w_reshaped.permute({0, 1, 3, 2});
    head1_filter_sigmoided = torch::sigmoid(head1_w_reshaped);

    loaded = true;
}

bool LibTorchWeights::is_loaded() const {
    return loaded;
}

void LibTorchWeights::to_halide_weights(Internal::Weights &halide_weights) const {
    if (!loaded) {
        internal_error << "LibTorchWeights: Cannot convert - weights not loaded\n";
    }

    // Copy head1 weights
    auto head1_w_data = head1_filter.data_ptr<float>();
    std::memcpy(halide_weights.head1_filter.data(), head1_w_data, 
                head1_filter.numel() * sizeof(float));
    
    auto head1_b_data = head1_bias.data_ptr<float>();
    std::memcpy(halide_weights.head1_bias.data(), head1_b_data, 
                head1_bias.numel() * sizeof(float));

    // Copy head2 weights
    auto head2_w_squeezed = head2_filter.squeeze(2); // (head2_channels, head2_w)
    auto head2_w_data = head2_w_squeezed.data_ptr<float>();
    std::memcpy(halide_weights.head2_filter.data(), head2_w_data, 
                head2_w_squeezed.numel() * sizeof(float));
    
    auto head2_b_data = head2_bias.data_ptr<float>();
    std::memcpy(halide_weights.head2_bias.data(), head2_b_data, 
                head2_bias.numel() * sizeof(float));

    // Combine trunk weights
    auto trunk_w_stage1 = trunk_filter_stage1.squeeze(2); // (conv1_channels, head1_channels)
    auto trunk_w_stage2 = trunk_filter_stage2.squeeze(2); // (conv1_channels, head2_channels)
    auto trunk_w_combined = torch::cat({trunk_w_stage1, trunk_w_stage2}, 1); // (conv1_channels, head1_channels + head2_channels)
    
    auto trunk_w_data = trunk_w_combined.data_ptr<float>();
    std::memcpy(halide_weights.conv1_filter.data(), trunk_w_data, 
                trunk_w_combined.numel() * sizeof(float));
    
    auto trunk_b_data = trunk_bias.data_ptr<float>();
    std::memcpy(halide_weights.conv1_bias.data(), trunk_b_data, 
                trunk_bias.numel() * sizeof(float));
}

bool LibTorchWeights::load_from_libtorch_file(const std::string &path) {
    try {
		std::cerr<<"called load_from_libtorch_file\n";
        // Check if file exists
        std::ifstream file(path);
        if (!file.good()) {
            aslog(0) << "LibTorchWeights: File does not exist: " << path << "\n";
            return false;
        }
        file.close();
        
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        
        archive.read("head1_filter", head1_filter);
        archive.read("head1_bias", head1_bias);
        archive.read("head1_filter_sigmoided", head1_filter_sigmoided);
        archive.read("head2_filter", head2_filter);
        archive.read("head2_bias", head2_bias);
        archive.read("trunk_filter_stage1", trunk_filter_stage1);
        archive.read("trunk_filter_stage2", trunk_filter_stage2);
        archive.read("trunk_bias", trunk_bias);

		if (torch::isnan(head1_filter).any().item<bool>()) {
			std::cout << "head1_filter contains NaN values!" << std::endl;
		}
		if (torch::isnan(head2_filter).any().item<bool>()) {
			std::cout << "head2_filter contains NaN values!" << std::endl;
		}
		if (torch::isnan(trunk_filter_stage1).any().item<bool>()) {
			std::cout << "trunk_filter_stage1 contains NaN values!" << std::endl;
		}
		if (torch::isnan(trunk_filter_stage2).any().item<bool>()) {
			std::cout << "trunk_filter_stage2 contains NaN values!" << std::endl;
		}

		try {
			archive.read("trunk_fc_0", trunk_fc_0);
		} catch (const c10::Error &e) {
			std::cerr << "Warning: trunk_fc_0 not found in archive. Using default (zero) initialization.\n";
			auto opts = torch::TensorOptions().dtype(torch::kFloat32);
			trunk_fc_0 = torch::normal(0.0, 0.1, {conv1_channels, 2 * conv1_channels, 1}, /*generator=*/c10::nullopt, opts);
		}

		try {
			archive.read("trunk_fc_0_bias", trunk_fc_0_bias);
		} catch (const c10::Error &e) {
			std::cerr << "Warning: trunk_fc_0_bias not found in archive. Using default (zero) initialization.\n";
			auto opts = torch::TensorOptions().dtype(torch::kFloat32);
			trunk_fc_0_bias = torch::normal(0.0, 0.1, {conv1_channels}, /*generator=*/c10::nullopt, opts);
		}

        
        // Verify tensors were loaded correctly
        if (head1_filter.numel() == 0 || head2_filter.numel() == 0) {
            aslog(0) << "LibTorchWeights: Loaded tensors are empty\n";
            return false;
        }
        
        loaded = true;
        aslog(0) << "LibTorchWeights: Successfully loaded weights from " << path << "\n";
        return true;
    } catch (const std::exception &e) {
        aslog(0) << "LibTorchWeights: Standard exception loading weights from " << path << ": " << e.what() << "\n";
        return false;
    } catch (...) {
        aslog(0) << "LibTorchWeights: Unknown error loading LibTorch weights from " << path << "\n";
        return false;
    }
}

bool LibTorchWeights::save_to_libtorch_file(const std::string &path) const {
    if (!loaded) {
        aslog(0) << "LibTorchWeights: Cannot save - weights not loaded\n";
        return false;
    }
    
    try {
        torch::serialize::OutputArchive archive;
        
        archive.write("head1_filter", head1_filter);
        archive.write("head1_bias", head1_bias);
        archive.write("head1_filter_sigmoided", head1_filter_sigmoided);
        archive.write("head2_filter", head2_filter);
        archive.write("head2_bias", head2_bias);
        archive.write("trunk_filter_stage1", trunk_filter_stage1);
        archive.write("trunk_filter_stage2", trunk_filter_stage2);
        archive.write("trunk_bias", trunk_bias);
        archive.write("format_version", torch::tensor(1)); // Version 1: LibTorch format
		archive.write("trunk_fc_0", trunk_fc_0);
		archive.write("trunk_fc_0_bias", trunk_fc_0_bias);
        
        archive.save_to(path);
        return true;
    } catch (const std::exception &e) {
        aslog(0) << "Error saving LibTorch weights to " << path << ": " << e.what() << "\n";
        return false;
    }
}

}  // namespace Halide


namespace Halide {

// Explicit template instantiations for buffer_to_tensor_public
template torch::Tensor LibTorchWeights::buffer_to_tensor_public<float>(
    const Halide::Runtime::Buffer<float> &buf);

template torch::Tensor LibTorchWeights::buffer_to_tensor_public<const float>(
    const Halide::Runtime::Buffer<const float> &buf);

}  // namespace Halide
