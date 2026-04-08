#include "ICostModelNetwork.h"
#include "LibTorchCostModel.h"
#include "LibTorchWeights.h"
#include "NetworkSize.h"
#include "ASLog.h"
#include "Errors.h"
#include <fstream>
#include <ctime>
#include <torch/torch.h>
#include "Timer.h"
#include "CustomNetwork0.h"
#include <unordered_map>

using Halide::Internal::aslog;

namespace Halide {

ICostModelNetwork::~ICostModelNetwork() = default;
// CustomModelNetwork implementation
CustomModelNetwork::CustomModelNetwork(std::string input_weights_path, const std::string &architecture_type, 
                                       bool use_random_weights)
    : architecture_type_(architecture_type), num_output_channels_(conv1_channels), use_random_weights_(use_random_weights), input_weights_path_(input_weights_path) {

	customWeights = std::make_shared<LibTorchWeights>();
	//
    // For now, only support Adams2019 architecture (same as original)
    // This makes it easy to test the flexible system
    // Future: can add other architectures here
    //if (architecture_type == "adams2019" || architecture_type == "default" || architecture_type.empty()) {
        //initialize_adams2019_architecture(use_random_weights);
        //aslog(1) << "CustomModelNetwork: Created with Adams2019 architecture"
                 //<< (use_random_weights ? " (random weights)" : "") << "\n";
    //} else {
        //aslog(0) << "CustomModelNetwork: Unknown architecture type: " << architecture_type 
                 //<< ", using Adams2019 as fallback\n";
        //initialize_adams2019_architecture(use_random_weights);
    //}
}

void CustomModelNetwork::initialize_adams2019_architecture(bool use_random_weights) {
    // Use the same architecture as Adams2019Network
    // Head1: Conv2d for pipeline features
    // Only one conv layer — sigmoid is applied dynamically in forward()
    head1_conv = register_module("head1_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, head1_channels, {head1_h, head1_w})
            .stride({head1_h, head1_w}).bias(true)));

    // Head2: Conv1d for schedule features
    head2_conv = register_module("head2_conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head2_w, head2_channels, 1).bias(true)));

    // Trunk: Two-stage conv matching original architecture
    trunk_conv_stage1 = register_module("trunk_conv_stage1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head1_channels, conv1_channels, 1).bias(true)));
    trunk_conv_stage2 = register_module("trunk_conv_stage2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head2_channels, conv1_channels, 1).bias(false)));

    if (use_random_weights) {
        randomize_weights();
    }
}

void CustomModelNetwork::randomize_weights() {
    // Initialize with random weights
    auto seed = time(nullptr);
    torch::manual_seed(seed);

    // Randomize head1 raw weights (sigmoid applied dynamically in forward)
    for (auto &param : head1_conv->parameters()) {
        torch::nn::init::normal_(param, 0.0, 0.1);
    }

    // Randomize head2 weights
    for (auto &param : head2_conv->parameters()) {
        torch::nn::init::normal_(param, 0.0, 0.1);
    }

    // Randomize trunk weights
    for (auto &param : trunk_conv_stage1->parameters()) {
        torch::nn::init::normal_(param, 0.0, 0.1);
    }
    for (auto &param : trunk_conv_stage2->parameters()) {
        torch::nn::init::normal_(param, 0.0, 0.1);
    }

    aslog(1) << "CustomModelNetwork: Initialized with random weights (seed=" << seed << ")\n";
}

torch::Tensor CustomModelNetwork::forward(const torch::Tensor &pipeline_features,
                                          const torch::Tensor &schedule_features,
                                          int num_stages,
                                          int batch_size) {
    // Use the same forward pass as Adams2019Network
    // This makes it easy to test - same architecture, just different initialization

    // Head1: Process pipeline features
    auto pf_batch = pipeline_features.contiguous().unsqueeze(1); // (num_stages, 1, head1_w, head1_h)
    pf_batch = pf_batch.permute({0, 1, 3, 2}).contiguous(); // (num_stages, 1, head1_h, head1_w)

    // Apply sigmoid to raw weights dynamically so gradients flow through during training
    auto sigmoided_weight = torch::sigmoid(head1_conv->weight);
    auto head1_out = torch::nn::functional::conv2d(pf_batch, sigmoided_weight,
        torch::nn::functional::Conv2dFuncOptions()
            .bias(head1_conv->bias)
            .stride({head1_h, head1_w})); // (num_stages, head1_channels, 1, 1)
    head1_out = head1_out.squeeze(-1).squeeze(-1); // (num_stages, head1_channels)
    int num_stages_dim = head1_out.size(0);
    int head1_channels_dim = head1_out.size(1);
    head1_out = head1_out.unsqueeze(0).expand({batch_size, num_stages_dim, head1_channels_dim});
    
    // Head2: Process schedule features
    auto normalized_sf = torch::log1p(schedule_features.contiguous()); // log(x + 1)
    auto head2_out = head2_conv->forward(normalized_sf); // (batch, head2_channels, num_stages)
    head2_out = torch::relu(head2_out);
    head2_out = head2_out.permute({0, 2, 1}).contiguous(); // (batch, num_stages, head2_channels)
    
    // Trunk: Two-stage conv
    auto head1_for_conv = head1_out.permute({0, 2, 1}).contiguous(); // (batch, head1_channels, num_stages)
    auto trunk_stage1 = trunk_conv_stage1->forward(head1_for_conv); // (batch, conv1_channels, num_stages)
    
    auto head2_for_conv = head2_out.permute({0, 2, 1}).contiguous(); // (batch, head2_channels, num_stages)
    auto trunk_stage2 = trunk_conv_stage2->forward(head2_for_conv); // (batch, conv1_channels, num_stages)
    
    auto trunk_out = trunk_stage1 + trunk_stage2; // (batch, conv1_channels, num_stages)
    trunk_out = torch::relu(trunk_out);
    trunk_out = trunk_out.permute({0, 2, 1}).contiguous(); // (batch, num_stages, conv1_channels)
    
    return trunk_out;
}

void CustomModelNetwork::load_weights(const LibTorchWeights &w) {
    // Support loading Adams2019-compatible weights
    // head1_filter contains raw (pre-sigmoid) weights; sigmoid is applied in forward()
    auto head1_w_raw = w.head1_filter.unsqueeze(1); // (head1_channels, 1, head1_w, head1_h)
    head1_w_raw = head1_w_raw.permute({0, 1, 3, 2}); // (head1_channels, 1, head1_h, head1_w)
    head1_conv->weight.data() = head1_w_raw;
    head1_conv->bias.data() = w.head1_bias.clone();
    
    // Load head2 weights
    head2_conv->weight.data() = w.head2_filter;
    head2_conv->bias.data() = w.head2_bias.clone();
    
    // Load trunk weights
    trunk_conv_stage1->weight.data() = w.trunk_filter_stage1;
    trunk_conv_stage1->bias.data() = w.trunk_bias.clone();
    trunk_conv_stage2->weight.data() = w.trunk_filter_stage2;
    
    aslog(1) << "CustomModelNetwork: Loaded weights from LibTorchWeights\n";
}

void CustomModelNetwork::save_weights(LibTorchWeights &w) const {
    // Save weights in Adams2019-compatible format
    // head1_conv stores raw weights; save them and recompute sigmoided for compatibility
    auto head1_w = head1_conv->weight.data();
    head1_w = head1_w.permute({0, 1, 3, 2}); // (head1_channels, 1, head1_w, head1_h)
    w.head1_filter = head1_w.squeeze(1).clone(); // (head1_channels, head1_w, head1_h)
    w.head1_bias = head1_conv->bias.data().clone();

    // Recompute sigmoided weights for compatibility with code that reads head1_filter_sigmoided
    auto head1_w_reshaped = w.head1_filter.unsqueeze(1);
    head1_w_reshaped = head1_w_reshaped.permute({0, 1, 3, 2});
    w.head1_filter_sigmoided = torch::sigmoid(head1_w_reshaped);
    
    // Save head2 weights
    w.head2_filter = head2_conv->weight.data().clone();
    w.head2_bias = head2_conv->bias.data().clone();
    
    // Save trunk weights
    w.trunk_filter_stage1 = trunk_conv_stage1->weight.data().clone();
    w.trunk_filter_stage2 = trunk_conv_stage2->weight.data().clone();
    w.trunk_bias = trunk_conv_stage1->bias.data().clone();
    
    aslog(1) << "CustomModelNetwork: Saved weights to LibTorchWeights\n";
}

// eval(), train(), parameters(), and get_num_output_channels() are implemented inline in the header

bool CustomModelNetwork::load_from_file(const std::string &path) {
    // For now, try to load as LibTorchWeights format
    // This allows loading pre-trained weights into the custom model
    try {
        LibTorchWeights weights;
        bool loaded = false;
        
        // Try LibTorch format first
        if (path.size() >= 3 && path.substr(path.size() - 3) == ".pt") {
            loaded = weights.load_from_libtorch_file(path);
        }
        
        // Fall back to Halide format
        if (!loaded) {
            loaded = weights.load_from_file(path);
        }
        
        if (loaded) {
            load_weights(weights);
            aslog(1) << "CustomModelNetwork: Loaded weights from " << path << "\n";
            return true;
        } else {
            aslog(0) << "CustomModelNetwork: Failed to load weights from " << path << "\n";
            return false;
        }
    } catch (const std::exception &e) {
        aslog(0) << "CustomModelNetwork: Exception loading from " << path << ": " << e.what() << "\n";
        return false;
    } catch (...) {
        aslog(0) << "CustomModelNetwork: Unknown error loading from " << path << "\n";
        return false;
    }
}

bool CustomModelNetwork::save_to_file(const std::string &path) const {
    // Save weights in LibTorchWeights format
    try {
        LibTorchWeights weights;
        save_weights(weights);
        
        // Save to file
        bool saved = false;
        if (path.size() >= 3 && path.substr(path.size() - 3) == ".pt") {
            saved = weights.save_to_libtorch_file(path);
        } else {
            saved = weights.save_to_file(path);
        }
        
        if (saved) {
            aslog(1) << "CustomModelNetwork: Saved weights to " << path << "\n";
            return true;
        } else {
            aslog(0) << "CustomModelNetwork: Failed to save weights to " << path << "\n";
            return false;
        }
    } catch (const std::exception &e) {
        aslog(0) << "CustomModelNetwork: Exception saving to " << path << ": " << e.what() << "\n";
        return false;
    } catch (...) {
        aslog(0) << "CustomModelNetwork: Unknown error saving to " << path << "\n";
        return false;
    }
}

using Creator = std::function<std::unique_ptr<CustomModelNetwork>(std::string, const std::string&, bool)>;
void CustomModelNetwork::registerModel(std::string name, Creator creator) {
	auto& model_registry = 	getRegistry();
	model_registry[name] = std::move(creator);
}

std::unordered_map<std::string, Creator>& CustomModelNetwork::getRegistry() {
	static std::unordered_map<std::string, Creator> model_registry; 
	return model_registry;
}

std::shared_ptr<LibTorchWeights> CustomModelNetwork::get_weights() {
	return customWeights;
}

std::unique_ptr<CustomModelNetwork> CustomModelNetwork::createCustomNetworkFromType(const std::string &architecture_type, bool use_random_weights, std::string weights_path) {

	// searching registry to see if we have registered this model before
	auto& model_registry = getRegistry();
	auto it = model_registry.find(architecture_type);
	if (it == model_registry.end()) {
		aslog(0) << "Invalid architecture input to createCustomNetworkFromType. Defaulting to Adams2019" << "\n";
		return std::make_unique<Adams2019Network>(weights_path, architecture_type, use_random_weights);
	}

	aslog(0) << "Creating a model of type "<< architecture_type << "through the registry!!" << "\n";
	return (it->second)(weights_path, architecture_type, use_random_weights);
}

void CustomModelNetwork::initialize_weights(bool use_random_weights, std::string input_weights_path) {

		// make sure all the layer parameters have been pushed into the corresponding unordered map entries 
		// within LibTorchWeights.
		sync_weights_from_network();
		bool need_randomize = use_random_weights;
	
		// initialize customWeights either from a file or randomly
		
        if (need_randomize) {
            auto seed = time(nullptr);
            aslog(1) << "CustomModelNetwork::initialize_weights: Randomizing customWeights using seed = " << seed << "\n";
            customWeights->randomize_generic((uint32_t)seed);
        } else {

			if (!input_weights_path.empty()) {
				aslog(1) << "CustomModelNetwork::initialize_weights: Attempting to load weights from: " << input_weights_path << "\n";

				bool loaded = false;
				if (input_weights_path.size() >= 3 && 
						input_weights_path.substr(input_weights_path.size() - 3) == ".pt") {
					// the weight_load_format argument tells the load function that the weights file was stored using python, 
					// and not using this repository's C++ code. It hence reads a little differently.
					// to load C++-written weights, this argument should be "archive" (It uses the InputArchive API)
					std::string weight_load_format = Internal::get_env_variable("HL_WEIGHTS_INPUT_FORMAT");
					loaded = customWeights->load_from_libtorch_file_generic(input_weights_path, weight_load_format);
					if(!loaded) 
					{
						aslog(0) << "CustomModelNetwork::initialize_weights: could not load weights into generic format. Falling back to random weights\n";
					} else if (loaded) {
						aslog(1) << "CustomModelNetwork::initialize_weights: Loaded weights from LibTorch format (.pt)\n";
					}
				}

			} else {
				aslog(1) << "CustomModelNetwork::initialize_weights: No weights path specified (weights_in_path empty, HL_WEIGHTS_DIR not set), using random initialization\n";
				need_randomize = true;
			}

			if (need_randomize) {
				auto seed = time(nullptr);
				aslog(1) << "CustomModelNetwork::initialize_weights: Randomizing customWeights using seed = " << seed << "\n";
				customWeights->randomize_generic((uint32_t)seed);
			}
		}
        
        // copy tensors from the customWeights object into the corresponding libtorch layer parameters
        sync_weights_to_network();
        eval();  // Ensure still in eval mode after loading weights
}

void CustomModelNetwork::sync_weights_from_network() const {
	for (const auto &pair : this->named_parameters()) {
		const std::string &name = pair.key();
		const torch::Tensor &param = pair.value();
		customWeights->set_weight(name, param.data());
	}
}

// Sync weights → network (GENERIC!)
void CustomModelNetwork::sync_weights_to_network() {
	int not_loaded = 0;
	for (auto &pair : named_parameters()) {
		if (customWeights->has_weight(pair.key())) {
			pair.value().data() = customWeights->get_weight(pair.key());
		} else {
			not_loaded++;
		}
	}

	if(not_loaded > 0) {
		aslog(0) << "CustomModelNetwork::sync_weights_to_network  " << not_loaded <<" weights could not be loaded from LibTorchWeights::model_weights_\n";
	}
}


// Factory function
std::unique_ptr<ICostModelNetwork> create_cost_model_network(
    const std::string &model_type_or_path,
    const std::string &weights_path) {

	// BHsketch | NOTE: we're currently treating ALL libtorch models as being "custom" models.
	// so to create the OG hailde model, the user must specify "adams2019", but to create a 
	// libtorch model, specify "custom". 
	// Then use HL_CUSTOM_MODEL_TYPE env variable to specify what kind of custom model. 
    
	bool need_random_weights = false;
	Internal::Autoscheduler::ScopedTimer model_creation_timer("timing model creation time"); 

    // === If HL_COST_MODEL_TYPE is a path, treat it as "custom network + load weights/model from that path".
	// ======================================================================================================
    const bool ends_with_pt = (model_type_or_path.size() >= 3 &&
                              model_type_or_path.substr(model_type_or_path.size() - 3) == ".pt");
    const bool ends_with_weights = (model_type_or_path.size() >= 8 &&
                                   model_type_or_path.substr(model_type_or_path.size() - 8) == ".weights");
    if (ends_with_pt) {
		aslog(0) << "create_cost_model_network: model_type_or_path is a path. Assuming adams2019 libtorch model with weights from the given path\n";
        auto custom_model = CustomModelNetwork::createCustomNetworkFromType("adams2019", false, model_type_or_path);
        return std::move(custom_model);
    } else if(ends_with_weights) {
		aslog(0) << "create_cost_model_network: Error: The .weights format is not supported for libtorch model. \
							Please provide weights as an archive or in a torchscript format. Using random weights for now. \n";
		need_random_weights = true;
	}
	// ======================================================================================================
    
    // Check for "custom" model type (uses Adams2019 architecture with random weights)
    if (model_type_or_path == "custom" || model_type_or_path == "CustomModelNetwork") 
	{

		// find out which custom model, and create it 
		// weight initialization is hardcoded to not random for now... 
		std::string which_custom_model = Internal::get_env_variable("HL_CUSTOM_MODEL_TYPE");
		auto custom_model = CustomModelNetwork::createCustomNetworkFromType(which_custom_model, need_random_weights, weights_path);
		aslog(0) << "ICostModelNetwork::create_cost_model_network: Created CustomModelNetwork with "<< which_custom_model <<" architecture\n";
		// ---------------------------------------------------------------------------------

        return std::move(custom_model);
    }
    
	// suppose HL_USE_LIBTORCH_COST_MODEL is true (else we wouldn't be in this function) 
	// but type is not custom and it's not a path either. Check for known model types to
	// infer what the user may want ====================================================
    if (model_type_or_path == "adams2019" || 
        model_type_or_path == "default" || 
        model_type_or_path == "" || 
        model_type_or_path == "Adams2019") {

		auto network = CustomModelNetwork::createCustomNetworkFromType("adams2019", false, weights_path);
        
        // If weights_path is provided and it's a .weights file, load weights
        // (Custom models loaded from .pt files are handled above)
        if (!weights_path.empty() && 
            weights_path.size() >= 8 && 
            weights_path.substr(weights_path.size() - 8) == ".weights") {
            // Load weights will be handled by LibTorchCostModel constructor
            // This is just creating the network structure
        }
        
        aslog(0) << "Created Adams2019 network\n";
        return std::move(network);
    }
    
    // Unknown model type
    aslog(0) << "Unknown model type: " << model_type_or_path 
             << ", falling back to Adams2019\n";
    aslog(0) << "Available model types: adams2019, custom, or path to .pt/.weights file\n";
	return CustomModelNetwork::createCustomNetworkFromType("adams2019", false, weights_path);
}

}  // namespace Halide

