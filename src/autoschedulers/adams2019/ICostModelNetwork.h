#ifndef I_COST_MODEL_NETWORK_H
#define I_COST_MODEL_NETWORK_H

#include "NetworkSize.h"
#include <torch/torch.h>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace Halide {

// Forward declarations
class LibTorchWeights;

/**
 * Abstract interface for cost model networks.
 * This allows different network architectures to be used interchangeably.
 */
class ICostModelNetwork {
public:

	// BHsketch S ---------
	//std::chrono::duration<float> collectiveInferenceTime{0.0f};	
	std::chrono::duration<float, std::milli> collectiveInferenceDuration = std::chrono::duration<float, std::milli>::zero();
	// BHsketch E ---------
	//
    virtual ~ICostModelNetwork();
    
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


	virtual void initialize_weights(bool randomize_weights, std::string input_weights_path) = 0;
	virtual void sync_weights_from_network() const = 0;
	virtual void sync_weights_to_network() = 0;
	virtual std::shared_ptr<LibTorchWeights> get_weights() = 0;
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

	// ############## Functionality common to all models #####################
	//
    CustomModelNetwork(std::string input_weights_path, const std::string &architecture_type = "adams2019",
                       bool use_random_weights = true);
    
    void eval() override { this->torch::nn::Module::eval(); }
    void train() override { this->torch::nn::Module::train(); }
    std::vector<torch::Tensor> parameters() override {
        std::vector<torch::Tensor> params;
        for (auto &param : this->torch::nn::Module::parameters()) {
            params.push_back(param);
        }
        return params;
    }

	using Creator = std::function<std::unique_ptr<CustomModelNetwork>(std::string, const std::string&, bool)>;
	// function to register a new custom model using the REGISTER_LIBTORCH_MODEL macro
	static void registerModel(std::string name, Creator creator);
	static std::unordered_map<std::string, Creator>& getRegistry();

	// CustomModelNetwork acts as the interface through which we can call 
	// createCustomNetworkFromType, thus creating an instance 
	// of one of the subtypes of this "interface".
	static std::unique_ptr<CustomModelNetwork> createCustomNetworkFromType(const std::string &architecture_type, bool use_random_weights, std::string weights_path);

	void initialize_weights(bool randomize_weights, std::string input_weights_path) override;
	void sync_weights_from_network() const override;
	void sync_weights_to_network() override;
	std::shared_ptr<LibTorchWeights> get_weights() override;


	// ############ Virtual methods specific to the custom model #############
	//
    virtual torch::Tensor forward(const torch::Tensor &pipeline_features,
                         const torch::Tensor &schedule_features,
                         int num_stages,
                         int batch_size) override;
    
    virtual void load_weights(const LibTorchWeights &w) override;
    virtual void save_weights(LibTorchWeights &w) const override;
    
    virtual int get_num_output_channels() const override { return conv1_channels; }
    
    virtual bool load_from_file(const std::string &path) override;
    virtual bool save_to_file(const std::string &path) const override;


protected:
    virtual void initialize_adams2019_architecture(bool use_random_weights);
    virtual void randomize_weights();
    
    std::string architecture_type_;
	bool use_random_weights_;
    int num_output_channels_;
	std::string input_weights_path_;

	// create a LibTorchWeights object that will be shared by our LibTorchCostModel object
	std::shared_ptr<LibTorchWeights> customWeights;
    
    // Adams2019 architecture (same as Adams2019Network for now)
    // This can be extended to support different architectures
    // head1_conv stores raw (pre-sigmoid) weights. Sigmoid is applied dynamically
    // in forward() so that gradients flow through it during training.
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

/*A macro which can be used by a user-defined network to "register" that network into 
 * our CustomModelNetwork class so that the createCustomNetworkFromType function is 
 * automatically informed of this class, and can look up the said class using an identifier.
 * This macro achieves this by creating a helper struct.
 * This helper struct, on program start, calls the static Register() method defined
 * in CustomModelNetwork with the name of our new class, and a corresponding function that calls
 * the constructor of our new class. CustomModelNetwork stores these two things as key and 
 * value in a customModelRegistry hashmap, and whenever it is asked to create a new model
 * via createCustomNetworkFromType, it searches through this hashmap for the corresponding 
 * type, and creates that model if it can.*/
#define REGISTER_LIBTORCH_MODEL(NAME, TYPE) \
	namespace { \
		struct TYPE##Register { \
			TYPE##Register() { \
				CustomModelNetwork::registerModel( \
						NAME, \
						[](std::string input_weights_path, const std::string& arch_type, bool use_random_weights) -> std::unique_ptr<CustomModelNetwork> { \
							return std::make_unique<TYPE>(input_weights_path, arch_type, use_random_weights); \
						}); \
			} \
		}; \
		static TYPE##Register global_##TYPE##Register; \
	}


}  // namespace Halide

#endif  // I_COST_MODEL_NETWORK_H

