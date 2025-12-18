#ifndef I_COST_MODEL_NETWORK_H
#define I_COST_MODEL_NETWORK_H

#include "NetworkSize.h"
#include <torch/torch.h>
#include <string>
#include <memory>
#include <vector>

namespace Halide {

// Forward declarations
class LibTorchWeights;

/**
 * Abstract interface for cost model networks.
 * This allows different network architectures to be used interchangeably.
 */
class ICostModelNetwork {
public:
    virtual ~ICostModelNetwork() = default;
    
    /**
     * Forward pass through the network.
     * @param pipeline_features: (num_stages, head1_w, head1_h) tensor
     * @param schedule_features: (batch, head2_w, num_stages) tensor
     * @param num_stages: Number of stages in the pipeline
     * @param batch_size: Batch size for evaluation
     * @return: (batch, num_stages, num_output_channels) tensor of coefficients
     */
    virtual torch::Tensor forward(const torch::Tensor &pipeline_features,
                                 const torch::Tensor &schedule_features,
                                 int num_stages,
                                 int batch_size) = 0;
    
    /**
     * Load weights from LibTorchWeights structure.
     * Used for loading Adams2019-compatible weights.
     */
    virtual void load_weights(const LibTorchWeights &w) = 0;
    
    /**
     * Save weights to LibTorchWeights structure.
     * Used for saving Adams2019-compatible weights.
     */
    virtual void save_weights(LibTorchWeights &w) const = 0;
    
    /**
     * Get the number of output channels (coefficients per stage).
     * This is needed for cost computation.
     */
    virtual int get_num_output_channels() const = 0;
    
    /**
     * Set the network to evaluation mode (no gradients).
     */
    virtual void eval() = 0;
    
    /**
     * Set the network to training mode (with gradients).
     */
    virtual void train() = 0;
    
    /**
     * Get all parameters for optimization.
     */
    virtual std::vector<torch::Tensor> parameters() = 0;
    
    /**
     * Load a full model from a .pt file.
     * This allows loading custom architectures trained separately.
     * @param path: Path to the .pt file
     * @return: true if successful, false otherwise
     */
    virtual bool load_from_file(const std::string &path) {
        // Default implementation: try to load as a torch::nn::Module
        try {
            torch::serialize::InputArchive archive;
            archive.load_from(path);
            // Subclasses should override this to load their specific architecture
            return false;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * Save the full model to a .pt file.
     * This allows saving custom architectures.
     * @param path: Path to save the .pt file
     * @return: true if successful, false otherwise
     */
    virtual bool save_to_file(const std::string &path) const {
        // Default implementation: try to save as a torch::nn::Module
        try {
            torch::serialize::OutputArchive archive;
            // Subclasses should override this to save their specific architecture
            archive.save_to(path);
            return true;
        } catch (...) {
            return false;
        }
    }
};

/**
 * Custom model network that can be configured with different architectures.
 * For now, uses the same architecture as Adams2019 (for easy testing).
 * This can be extended to support different architectures in the future.
 * 
 * Architecture is defined in code, not loaded from file.
 * Uses random weights by default.
 */
class CustomModelNetwork : public torch::nn::Module, public ICostModelNetwork {
public:
    /**
     * Create a custom model network.
     * @param architecture_type: Type of architecture to use (currently only "adams2019" supported)
     * @param use_random_weights: If true, initialize with random weights (default: true)
     */
    CustomModelNetwork(const std::string &architecture_type = "adams2019", 
                       bool use_random_weights = true);
    
    torch::Tensor forward(const torch::Tensor &pipeline_features,
                         const torch::Tensor &schedule_features,
                         int num_stages,
                         int batch_size) override;
    
    void load_weights(const LibTorchWeights &w) override;
    void save_weights(LibTorchWeights &w) const override;
    
    int get_num_output_channels() const override { return conv1_channels; }
    void eval() override { this->torch::nn::Module::eval(); }
    void train() override { this->torch::nn::Module::train(); }
    std::vector<torch::Tensor> parameters() override {
        std::vector<torch::Tensor> params;
        for (auto &param : this->torch::nn::Module::parameters()) {
            params.push_back(param);
        }
        return params;
    }
    
    bool load_from_file(const std::string &path) override;
    bool save_to_file(const std::string &path) const override;

private:
    void initialize_adams2019_architecture(bool use_random_weights);
    void randomize_weights();
    
    std::string architecture_type_;
    int num_output_channels_;
    
    // Adams2019 architecture (same as Adams2019Network for now)
    // This can be extended to support different architectures
    torch::nn::Conv2d head1_conv_raw{nullptr};
    torch::nn::Conv2d head1_conv{nullptr};
    torch::nn::Conv1d head2_conv{nullptr};
    torch::nn::Conv1d trunk_conv_stage1{nullptr};
    torch::nn::Conv1d trunk_conv_stage2{nullptr};
};

/**
 * Factory function to create a network from configuration.
 * Supports:
 * - "adams2019" or "default": Original Adams2019 architecture
 * - Path to .pt file: Load custom model from file
 * - Future: "resnet", "transformer", etc.
 */
std::unique_ptr<ICostModelNetwork> create_cost_model_network(
    const std::string &model_type_or_path = "adams2019",
    const std::string &weights_path = "");

}  // namespace Halide

#endif  // I_COST_MODEL_NETWORK_H

