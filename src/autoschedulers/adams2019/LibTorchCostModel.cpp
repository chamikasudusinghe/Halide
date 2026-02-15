#include "LibTorchCostModel.h"
#include "ASLog.h"
#include "Errors.h"
#include "Featurization.h"
#include "FunctionDAG.h"
#include "HalideBuffer.h"
#include "NetworkSize.h"
#include "LibTorchCostModelOptimizations.h"
#include "LibTorchWeights.h"
#include "LibTorchFeatureConverter.h"
#include "ICostModelNetwork.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <thread>
//BHsketch S ----
#include <chrono>
#include "CustomNetwork0.h"
//BHsketch E ----

using Halide::Internal::aslog;
using Halide::Internal::PipelineFeatures;
using Halide::Internal::ScheduleFeatures;
using Halide::Runtime::Buffer;

namespace Halide {

using namespace Internal::Autoscheduler;

// Helper function to get environment variable
static std::string get_env_variable(const std::string &name) {
    const char *val = std::getenv(name.c_str());
    return val ? std::string(val) : std::string();
}

Adams2019Network::Adams2019Network(const std::string &architecture_type, bool use_random_weights) : CustomModelNetwork(architecture_type, use_random_weights) {
    // Head1: Conv2d for pipeline features
    // Input: (batch, 1, head1_w=40, head1_h=7) -> Output: (batch, head1_channels=8, 1, 1)
    // Use two conv layers: one for raw weights (for saving), one for sigmoided weights (for forward)
    head1_conv_raw = register_module("head1_conv_raw", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, head1_channels, {head1_h, head1_w})
            .stride({head1_h, head1_w}).bias(true)));
    head1_conv = register_module("head1_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, head1_channels, {head1_h, head1_w})
            .stride({head1_h, head1_w}).bias(true)));
    
    // Head2: Conv1d for schedule features  
    // Input: (batch, head2_w=39, num_stages) -> Output: (batch, head2_channels=24, num_stages)
    head2_conv = register_module("head2_conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head2_w, head2_channels, 1).bias(true)));
    
    // Trunk: Two-stage conv matching original architecture
    // Stage1: processes head1 (8 channels) -> 32 channels
    trunk_conv_stage1 = register_module("trunk_conv_stage1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head1_channels, conv1_channels, 1).bias(true)));
    // Stage2: processes head2 (24 channels) -> 32 channels (no bias, adds to stage1)
    trunk_conv_stage2 = register_module("trunk_conv_stage2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head2_channels, conv1_channels, 1).bias(false)));

	if(use_random_weights_) {
		// currently just uses the implementation in its parent class
		randomize_weights();
	}
}

Adams2019Network::Adams2019Network() {
    // Head1: Conv2d for pipeline features
    // Input: (batch, 1, head1_w=40, head1_h=7) -> Output: (batch, head1_channels=8, 1, 1)
    // Use two conv layers: one for raw weights (for saving), one for sigmoided weights (for forward)
    head1_conv_raw = register_module("head1_conv_raw", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, head1_channels, {head1_h, head1_w})
            .stride({head1_h, head1_w}).bias(true)));
    head1_conv = register_module("head1_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, head1_channels, {head1_h, head1_w})
            .stride({head1_h, head1_w}).bias(true)));
    
    // Head2: Conv1d for schedule features  
    // Input: (batch, head2_w=39, num_stages) -> Output: (batch, head2_channels=24, num_stages)
    head2_conv = register_module("head2_conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head2_w, head2_channels, 1).bias(true)));
    
    // Trunk: Two-stage conv matching original architecture
    // Stage1: processes head1 (8 channels) -> 32 channels
    trunk_conv_stage1 = register_module("trunk_conv_stage1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head1_channels, conv1_channels, 1).bias(true)));
    // Stage2: processes head2 (24 channels) -> 32 channels (no bias, adds to stage1)
    trunk_conv_stage2 = register_module("trunk_conv_stage2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(head2_channels, conv1_channels, 1).bias(false)));

}


void Adams2019Network::load_weights(const LibTorchWeights &w) {
    // Use pre-computed weights from LibTorchWeights (already optimized)
    // Store raw weights in head1_conv_raw (for saving)
    auto head1_w_raw = w.head1_filter.unsqueeze(1); // (head1_channels, 1, head1_w, head1_h)
    head1_w_raw = head1_w_raw.permute({0, 1, 3, 2}); // (head1_channels, 1, head1_h, head1_w)
    head1_conv_raw->weight.data() = head1_w_raw;
    head1_conv_raw->bias.data() = w.head1_bias.clone();
    
    // Use pre-computed sigmoided weights (for forward pass - no swapping needed!)
    head1_conv->weight.data() = w.head1_filter_sigmoided;
    head1_conv->bias.data() = w.head1_bias.clone();
    
    // Load head2 weights (already in correct shape)
    head2_conv->weight.data() = w.head2_filter;
    head2_conv->bias.data() = w.head2_bias.clone();
    
    // Load trunk weights (already split into two stages)
    trunk_conv_stage1->weight.data() = w.trunk_filter_stage1;
    trunk_conv_stage1->bias.data() = w.trunk_bias.clone();
    trunk_conv_stage2->weight.data() = w.trunk_filter_stage2;
}

void Adams2019Network::save_weights(LibTorchWeights &w) const {
    // Save head1 weights from raw conv (not sigmoided)
    auto head1_w = head1_conv_raw->weight.data();
    head1_w = head1_w.permute({0, 1, 3, 2}); // (head1_channels, 1, head1_w, head1_h)
    w.head1_filter = head1_w.squeeze(1).clone(); // (head1_channels, head1_w, head1_h)
    w.head1_bias = head1_conv_raw->bias.data().clone();
    
    // Recompute sigmoided weights
    auto head1_w_reshaped = w.head1_filter.unsqueeze(1);
    head1_w_reshaped = head1_w_reshaped.permute({0, 1, 3, 2});
    w.head1_filter_sigmoided = torch::sigmoid(head1_w_reshaped);
    
    // Save head2 weights
    w.head2_filter = head2_conv->weight.data().clone();
    w.head2_bias = head2_conv->bias.data().clone();
    
    // Save trunk weights (already split into two stages)
    w.trunk_filter_stage1 = trunk_conv_stage1->weight.data().clone();
    w.trunk_filter_stage2 = trunk_conv_stage2->weight.data().clone();
    w.trunk_bias = trunk_conv_stage1->bias.data().clone();
}

torch::Tensor Adams2019Network::forward(const torch::Tensor &pipeline_features,
                                       const torch::Tensor &schedule_features,
                                       int num_stages,
                                       int batch_size) {
	
    // Head1: Process pipeline features
    // pipeline_features: (num_stages, head1_w, head1_h)
    // Batch all stages at once for better performance
    // Reshape: (num_stages, head1_w, head1_h) -> (num_stages, 1, head1_h, head1_w)
    // Ensure contiguous for better performance
    auto pf_batch = pipeline_features.contiguous().unsqueeze(1); // (num_stages, 1, head1_w, head1_h)
    pf_batch = pf_batch.permute({0, 1, 3, 2}).contiguous(); // (num_stages, 1, head1_h, head1_w)
    
    // Apply sigmoid to weights before conv (matching original: squashed_head1_filter)
    // head1_conv already has sigmoided weights - no swapping needed!
    auto head1_out = head1_conv->forward(pf_batch); // (num_stages, head1_channels, 1, 1)
    head1_out = head1_out.squeeze(-1).squeeze(-1); // (num_stages, head1_channels)
    // Expand to batch: (batch, num_stages, head1_channels)
    // Use expand (views) for better performance - no memory copies needed
    int num_stages_dim = head1_out.size(0);
    int head1_channels_dim = head1_out.size(1);
    head1_out = head1_out.unsqueeze(0).expand({batch_size, num_stages_dim, head1_channels_dim});
    
    // Head2: Process schedule features
    // schedule_features: (batch, head2_w, num_stages)
    // Ensure contiguous for better performance
    auto normalized_sf = torch::log1p(schedule_features.contiguous()); // log(x + 1)
    auto head2_out = head2_conv->forward(normalized_sf); // (batch, head2_channels, num_stages)
    head2_out = torch::relu(head2_out);
    head2_out = head2_out.permute({0, 2, 1}).contiguous(); // (batch, num_stages, head2_channels)
    
    // Trunk: Two-stage conv matching original architecture
    // Stage1: Process head1 (matching original conv1_stage1)
    // head1_out: (batch, num_stages, head1_channels) -> (batch, head1_channels, num_stages)
    auto head1_for_conv = head1_out.permute({0, 2, 1}).contiguous(); // (batch, head1_channels, num_stages)
    auto trunk_stage1 = trunk_conv_stage1->forward(head1_for_conv); // (batch, conv1_channels, num_stages)
    
    // Stage2: Add head2 processing (matching original conv1_stage2)
    // head2_out: (batch, num_stages, head2_channels) -> (batch, head2_channels, num_stages)
    auto head2_for_conv = head2_out.permute({0, 2, 1}).contiguous(); // (batch, head2_channels, num_stages)
    auto trunk_stage2 = trunk_conv_stage2->forward(head2_for_conv); // (batch, conv1_channels, num_stages)
    
    // Add stage2 to stage1 (matching original: conv1_stage2 = conv1_stage1 + ...)
    auto trunk_out = trunk_stage1 + trunk_stage2; // (batch, conv1_channels, num_stages)
    
    // Apply ReLU (matching original relu1)
    trunk_out = torch::relu(trunk_out);
    trunk_out = trunk_out.permute({0, 2, 1}).contiguous(); // (batch, num_stages, conv1_channels)
    
    return trunk_out;
}

// LibTorchCostModel implementation
LibTorchCostModel::LibTorchCostModel(const std::string &weights_in_path,
                                     const std::string &weights_out_path,
                                     bool randomize_weights)
    : weights_in_path(weights_in_path),
      weights_out_path(weights_out_path),
      randomize_weights(randomize_weights) {
    
    // Set thread settings for performance
    // Use single thread to avoid contention with Halide's threading
    // Halide autoscheduler manages its own parallelism, so LibTorch should not compete
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    
    // For better performance, this can be an option to consider
    // int num_threads = std::min(8, (int)std::thread::hardware_concurrency());
    // f (num_threads == 0) num_threads = 4;  // Fallback if hardware_concurrency fails
    
    // Determine which model to use based on environment variable or default
    std::string model_type = get_env_variable("HL_COST_MODEL_TYPE");
    if (model_type.empty()) {
        model_type = "adams2019";  // Default to Adams2019
    }
    
    // Create the appropriate network
    network = create_cost_model_network(model_type, weights_in_path);
    if (!network) {
        aslog(0) << "LibTorchCostModel: Failed to create network, falling back to Adams2019\n";
        network = std::make_unique<Adams2019Network>();
    }
    network->eval();  // Set to evaluation mode immediately
    
    // Load weights using optimized LibTorchWeights
    // Only load weights if it's an Adams2019 network (custom models are loaded in factory)
    bool need_randomize = randomize_weights;
    string actual_weights_path = weights_in_path;
    
    // Check if we have an Adams2019 network that needs weight loading
    bool is_adams2019 = (dynamic_cast<Adams2019Network*>(network.get()) != nullptr);
    bool is_custom0 = (dynamic_cast<CustomNetwork0*>(network.get()) != nullptr);
    
    // If weights_in_path is empty, try environment variable
    if (actual_weights_path.empty()) {
        actual_weights_path = get_env_variable("HL_WEIGHTS_DIR");
    }
    
    // Only load weights for Adams2019 networks (custom models are already loaded)
    if (is_adams2019 || is_custom0) {
        if (!actual_weights_path.empty()) {
            aslog(1) << "LibTorchCostModel: Attempting to load weights from: " << actual_weights_path << "\n";
			std::cerr << "LibTorchCostModel: Attempting to load weights from: " << actual_weights_path << "\n";
            
            // Try LibTorch format first (faster, direct loading)
            // Check if file ends with .pt (PyTorch/LibTorch format)
            bool loaded = false;
            if (actual_weights_path.size() >= 3 && 
                actual_weights_path.substr(actual_weights_path.size() - 3) == ".pt") {
                loaded = weights.load_from_libtorch_file(actual_weights_path);
                if (loaded) {
                    aslog(1) << "LibTorchCostModel: Loaded weights from LibTorch format (.pt)\n";
					std::cerr << "LibTorchCostModel: Loaded weights from LibTorch format (.pt)\n";
                }
            }
            
            // Fall back to Halide format if LibTorch format failed or not .pt file
            if (!loaded) {
                loaded = weights.load_from_file(actual_weights_path);
                if (loaded) {
                    aslog(1) << "LibTorchCostModel: Loaded weights from Halide format\n";
					std::cerr << "LibTorchCostModel: Loaded weights from Halide format\n";
                }
            }
            
            if (!loaded) {
                aslog(1) << "LibTorchCostModel: Failed to load weights from " << actual_weights_path << ", using random initialization\n";
				std::cerr << "LibTorchCostModel: Failed to load weights from " << actual_weights_path << ", using random initialization\n";
                need_randomize = true;
            }
        } else {
            aslog(1) << "LibTorchCostModel: No weights path specified (weights_in_path empty, HL_WEIGHTS_DIR not set), using random initialization\n";
            need_randomize = true;
        }
        
        if (need_randomize) {
            auto seed = time(nullptr);
            aslog(1) << "Randomizing weights using seed = " << seed << "\n";
            weights.randomize((uint32_t)seed);
        }
        
        // Load weights into network
		//std::cerr<<"LibTorchCostModel: trunk_fc_0->weight: "<<weights.trunk_fc_0<<"\n";
		//std::cerr<<"LibTorchCostModel: trunk_fc_0->bias: "<<weights.trunk_fc_0_bias<<"\n";
		if (torch::isnan(weights.trunk_fc_0).any().item<bool>()) {
			std::cerr << "LibTorchCostModel constructor, trunk_fc_0 contains NaN values!\n";
		}else{
			std::cerr << "LibTorchCostModel constructor, trunk_fc_0 does NOT NaN values\n";
		}


        network->load_weights(weights);
        network->eval();  // Ensure still in eval mode after loading weights
    } else {
        aslog(1) << "LibTorchCostModel: Using custom model, weights already loaded\n";
    }

    // Warm up LibTorch with a dummy forward pass to avoid first-call overhead
    // This initializes any lazy operations and can prevent hangs
    // Use a small warm-up to minimize overhead
    try {
        torch::NoGradGuard no_grad;
        auto dummy_pf = torch::zeros({1, head1_w, head1_h}, torch::kFloat32);
        auto dummy_sf = torch::zeros({1, head2_w, 1}, torch::kFloat32);
        network->forward(dummy_pf, dummy_sf, 1, 1);
        // Operations are synchronous on CPU, so no explicit sync needed
    } catch (...) {
        // Ignore errors in warm-up - it's just for initialization
    }
}

LibTorchCostModel::~LibTorchCostModel() {
    if (optimizer) {
        delete optimizer;
    }
}

void LibTorchCostModel::set_pipeline_features(const FunctionDAG &dag,
                                               const Adams2019Params &params) {
    num_cores = params.parallelism;
    
    // Use optimized feature converter
    int max_num_stages = 0;
    auto pf_tensor = LibTorchFeatureConverter::convert_pipeline_features(dag, max_num_stages);
    
    pipeline_feat_queue.clear();
    pipeline_feat_queue.push_back(pf_tensor);
    num_stages = 0; // Will be set in enqueue() based on schedule_feats.size()
}

void LibTorchCostModel::set_pipeline_features(const torch::Tensor &pf_tensor, int n) {
	
	pipeline_feat_queue.clear();
	pipeline_feat_queue.push_back(pf_tensor);
	num_cores=n;

}

void LibTorchCostModel::enqueue(const FunctionDAG &dag,
                               const StageMapOfScheduleFeatures &schedule_feats,
                               double *cost_ptr) {
	//BHsketch S ----
	// timing the forward inference and adding it to a collective duration variable collectiveInferenceDuration 
	auto enqueueStartTime = std::chrono::high_resolution_clock::now();
	//BHsketch E ----
    // Set num_stages from schedule_feats.size() (like DefaultCostModel does)
    // This can vary between different calls to enqueue
    num_stages = (int)schedule_feats.size();
    
    internal_assert(num_stages > 0) << "enqueue called with empty schedule_feats";
    internal_assert(pipeline_feat_queue.size() > 0) 
        << "enqueue called before set_pipeline_features";
    
    // Use optimized feature converter (iterates in DAG order)
    auto sf_tensor = LibTorchFeatureConverter::convert_schedule_features(dag, schedule_feats, num_stages);
    // sf_tensor is (head2_w, num_stages), transpose to (num_stages, head2_w) for batching
    schedule_feat_queue.push_back(sf_tensor.transpose(0, 1)); // (num_stages, head2_w)
    cost_ptrs.push_back(cost_ptr);
    cursor++;
    
	//BHsketch S ----
	auto enqueueEndTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> enqueueDuration = enqueueEndTime - enqueueStartTime;
	this->collectiveEnqueueDuration += enqueueDuration;
	//aslog(1) << "Updated collective enqueue time: " << this->collectiveEnqueueDuration.count() << "ms \n";
	//BHsketch E ----
	//
    if (cursor == batch_size) {
        evaluate_costs();
    }
}

std::vector<torch::Tensor>& LibTorchCostModel::enqueue(int ns, double *cost_ptr) {
	num_stages = ns;

	// BHsketch S ---
	// the only difference between LibTorchCostModel::enqueue and DefaultCostModel::enqueue is the
	// fact that for libtorch, our queues are no longer halide buffers, but rather vectors
	// of torch tensors. Thus, we call the appropriate APIs to read/store information from/to them.
	// We don't need to do any bounds checking or return appropriate slices of the vector. Rather, 
	// we could just pass a reference to the vector and the caller can push_back a tensor to it.
	// BHsketch E ---
	
	internal_assert(pipeline_feat_queue.size() && "LibTorchCostModel::enqueue: Call set_pipeline_features before calling enqueue\n");
	const int max_num_stages = pipeline_feat_queue[0].size(0);
	internal_assert(num_stages <= max_num_stages)
		<< "LibTorchCostModel::enqueue: schedule features has more stages (" << num_stages 
		<< ") than pipeline features (" << max_num_stages << ")\n";

	const int batch_size = 1024;
	if (!schedule_feat_queue.size()) {
		internal_assert(cursor == 0);
		//schedule_feat_queue = Runtime::Buffer<float>(batch_size, head2_w, max_num_stages);
		//if(!costs.data()) {
			//internal_assert(!cost_ptrs.data());
			//costs = Runtime::Buffer<float>(batch_size);
			//cost_ptrs = Runtime::Buffer<double *>(batch_size);
		//}

	}

	if (cursor == batch_size) {
		evaluate_costs();
	}

	// schedule_feats is a reference to our internal queue, that will be used by the caller
	// (say, the training pipeline) to push back a new tensor to the queue.
	//schedule_feats = schedule_feat_queue;

	// stores, within the cost model class, what location to store the predictions in when we are
	// done evaluating this schedule with our cost model.
	cost_ptrs.push_back(cost_ptr);

	cursor++;
	
	return schedule_feat_queue;
}

torch::Tensor LibTorchCostModel::compute_cost_from_coefficients(const torch::Tensor &coefficients,
                                                                 const torch::Tensor &schedule_features_tensor,
                                                                 int num_stages,
                                                                 int batch_size,
                                                                 int num_cores) {
    // This implements the hand-designed cost formula from cost_model_generator.cpp
    // schedule_features_tensor: (batch, head2_w, num_stages)
    // coefficients: (batch, num_stages, conv1_channels=32)
    
    // Optimized: Use UnpackedScheduleFeatures to avoid many select() + transpose() operations
    auto features = UnpackedScheduleFeatures::unpack(schedule_features_tensor);
    
    // In original Halide: relu1(c, w, n) where c=channel, w=stage, n=batch
    // In LibTorch: coefficients is (batch, num_stages, conv1_channels)
    // We need to permute to (conv1_channels, num_stages, batch) to match relu1(c, w, n)
    // Then access as relu1[channel][stage][batch]
    auto relu1 = coefficients.permute({2, 1, 0}).contiguous(); // (conv1_channels, num_stages, batch)
    
    // Compute cost per stage, then sum
    // For each stage, compute cost terms
    std::vector<torch::Tensor> stage_costs;
    
    for (int s = 0; s < num_stages; s++) {
        auto inlined = features.inlined_calls.select(0, s); // (batch,)
        auto vec_size = features.vector_size.select(0, s);
        auto n_vec = features.num_vectors.select(0, s);
        auto n_scal = features.num_scalars.select(0, s);
        
        // Extract coefficients for this stage: relu1[:, s, :] = (conv1_channels, batch)
        // Then transpose to (batch, conv1_channels) for easier indexing
        auto relu1_s = relu1.select(1, s).transpose(0, 1); // (batch, conv1_channels)
        
        // Compute cost - relu1(c, w, n) where c is channel index
        auto compute_cost = torch::where(inlined == 0,
            vec_size * n_vec * relu1_s.select(1, 0) + n_scal * relu1_s.select(1, 1),
            vec_size * n_vec * relu1_s.select(1, 2) + n_scal * relu1_s.select(1, 3));
        
        // Idle core wastage
        auto inner_par = features.inner_parallelism.select(0, s);
        auto outer_par = features.outer_parallelism.select(0, s);
        auto num_tasks = torch::clamp_min(inner_par * outer_par, 1.0f);
        //auto tasks_per_core = num_tasks / (float)num_cores;
		auto tasks_per_core = num_tasks / std::max((float)num_cores, 1.0f);
        auto idle_core_wastage = torch::ceil(tasks_per_core) / torch::clamp_min(tasks_per_core, 1.0f);
        compute_cost = compute_cost * idle_core_wastage;
        
        // Load cost
        auto n_real = features.num_realizations.select(0, s);
        auto n_prod = features.num_productions.select(0, s);
        auto load_cost = (n_real * features.unique_lines_read_per_realization.select(0, s) * relu1_s.select(1, 5) +
                         n_real * features.unique_bytes_read_per_realization.select(0, s) * relu1_s.select(1, 6) +
                         n_vec * features.scalar_loads_per_vector.select(0, s) * relu1_s.select(1, 7) +
                         n_scal * features.scalar_loads_per_scalar.select(0, s) * relu1_s.select(1, 8) +
                         n_vec * features.vector_loads_per_vector.select(0, s) * relu1_s.select(1, 9) +
                         n_scal * features.unique_bytes_read_per_vector.select(0, s) * relu1_s.select(1, 10) +
                         n_vec * features.unique_bytes_read_per_vector.select(0, s) * relu1_s.select(1, 11) +
                         n_scal * features.unique_lines_read_per_vector.select(0, s) * relu1_s.select(1, 12) +
                         n_vec * features.unique_lines_read_per_vector.select(0, s) * relu1_s.select(1, 13) +
                         num_tasks * features.unique_bytes_read_per_task.select(0, s) * relu1_s.select(1, 14) +
                         num_tasks * features.unique_lines_read_per_task.select(0, s) * relu1_s.select(1, 15));
        
        // Store cost
        auto bytes_task = features.bytes_at_task.select(0, s);
        auto innermost_bytes_task = features.innermost_bytes_at_task.select(0, s);
        auto lines_written = inner_par * (bytes_task / torch::clamp_min(innermost_bytes_task, 1.0f));
        auto bytes_real = features.bytes_at_realization.select(0, s);
        
        // Alpha: select coefficient based on inner_parallelism and stage index
        // Original logic: select(inner_parallelism > 1, relu1(16), 
        //                      w == 0, relu1(17), relu1(18))
        auto alpha_cond1 = (inner_par > 1).to(torch::kFloat32);
        auto alpha_cond2 = torch::ones_like(alpha_cond1) * (s == 0 ? 1.0f : 0.0f);
        auto alpha = alpha_cond1 * relu1_s.select(1, 16) +
                    (1.0f - alpha_cond1) * alpha_cond2 * relu1_s.select(1, 17) +
                    (1.0f - alpha_cond1) * (1.0f - alpha_cond2) * relu1_s.select(1, 18);
        
        // Beta: same logic
        auto beta = alpha_cond1 * relu1_s.select(1, 19) +
                   (1.0f - alpha_cond1) * alpha_cond2 * relu1_s.select(1, 20) +
                   (1.0f - alpha_cond1) * (1.0f - alpha_cond2) * relu1_s.select(1, 21);
        
        auto store_cost = n_real * (lines_written * alpha + bytes_real * beta);
        
        // False sharing
        auto false_sharing = torch::where(inner_par > 1,
            relu1_s.select(1, 22) * (n_vec + n_scal) / torch::clamp_min(innermost_bytes_task, 1.0f),
            torch::zeros_like(store_cost));
        store_cost = store_cost + false_sharing;
        
        // Page faults
        auto max_threads = torch::min(inner_par, 4096.0f / torch::clamp_min(innermost_bytes_task, 1.0f));
        auto bytes_prod = features.bytes_at_production.select(0, s);
        auto page_faults = bytes_prod * max_threads * inner_par * outer_par * relu1_s.select(1, 23);
        store_cost = store_cost + page_faults;
        
        // Malloc cost
        auto malloc_cost = relu1_s.select(1, 24) * n_real;
        
        // Parallelism costs
        auto parallel_launches = n_prod * torch::where(inner_par > 1, relu1_s.select(1, 25), torch::zeros_like(n_prod));
        auto parallel_tasks = n_prod * (inner_par - 1) * relu1_s.select(1, 26);
        auto parallelism_cost = parallel_tasks + parallel_launches;
        
        // Working set cost
        auto ws = features.working_set.select(0, s);
        auto ws_cost = ws * relu1_s.select(1, 27);
        
        // Total cost for this stage (store_cost doubled in original)
        auto stage_cost = compute_cost + store_cost * 2 + load_cost +
                         malloc_cost + parallelism_cost + ws_cost;
        stage_costs.push_back(stage_cost);
    }
    
    // Stack and sum across stages, then convert to runtime
    auto total_cost = torch::stack(stage_costs, 0); // (num_stages, batch)
    total_cost = torch::sum(total_cost, 0); // (batch,)
    auto prediction = total_cost * 1e-9f;
    
    return prediction;
}

void LibTorchCostModel::evaluate_costs() {
    if (cursor == 0 || schedule_feat_queue.empty()) {
        return;
    }
    
    internal_assert(pipeline_feat_queue.size() > 0);
    internal_assert(num_stages > 0) << "num_stages must be set before evaluating costs";
    
    // Batch all schedule features using optimized converter
    // schedule_feat_queue contains (num_stages, head2_w) tensors
    // Need to convert to (batch, head2_w, num_stages)
    std::vector<torch::Tensor> transposed;
    for (const auto &t : schedule_feat_queue) {
        transposed.push_back(t.transpose(0, 1).contiguous()); // (head2_w, num_stages)
    }
    auto schedule_features_batch = torch::stack(transposed, 0).contiguous(); // (batch, head2_w, num_stages)
    
    // Slice pipeline features to only use the first num_stages stages
    // pipeline_feat_queue[0] has shape (max_num_stages, head1_w, head1_h)
    // We need (num_stages, head1_w, head1_h)
    auto pipeline_features_full = pipeline_feat_queue[0];
    int max_stages = pipeline_features_full.size(0);
    internal_assert(num_stages <= max_stages) 
        << "num_stages (" << num_stages << ") exceeds max_stages (" << max_stages << ")";
    auto pipeline_features = pipeline_features_full.slice(0, 0, num_stages); // (num_stages, head1_w, head1_h)
    
    // Forward pass through network
    // Disable gradient computation for inference (faster and uses less memory)
    torch::NoGradGuard no_grad;
    network->eval();

	//BHsketch S ----
	// timing the forward inference and adding it to a collective duration variable collectiveInferenceDuration 
	auto inferenceStartTime = std::chrono::high_resolution_clock::now();
	//BHsketch E ----
													   //
    auto coefficients = network->forward(pipeline_features, schedule_features_batch, num_stages, cursor);
    
    // Compute costs from coefficients
    auto predictions = compute_cost_from_coefficients(coefficients, schedule_features_batch, num_stages, cursor, num_cores);

	//BHsketch S ----
	auto inferenceEndTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> inferenceDuration = inferenceEndTime - inferenceStartTime;
	network->collectiveInferenceDuration += inferenceDuration;
	//aslog(1) << "Updated collective inference time: " << network->collectiveInferenceDuration.count() << "ms \n";
	//BHsketch E ----
    
    // Copy results back
    // Ensure tensor is contiguous and on CPU for efficient access
    auto predictions_cpu = predictions.contiguous().cpu();
    auto predictions_data = predictions_cpu.data_ptr<float>();
    for (int i = 0; i < cursor; i++) {
        internal_assert(cost_ptrs[i]);
        *(cost_ptrs[i]) = (double)predictions_data[i];
    }

    // Clear queues
    cursor = 0;
    schedule_feat_queue.clear();
    cost_ptrs.clear();
}

void LibTorchCostModel::reset() {
    // Evaluate any pending items before clearing
    if (cursor > 0 && !schedule_feat_queue.empty()) {
        evaluate_costs();
    }
    
    cursor = 0;
    schedule_feat_queue.clear();
    cost_ptrs.clear();
    num_stages = 0;
}

float LibTorchCostModel::backprop(const Runtime::Buffer<const float> &true_runtimes, float learning_rate) {
    internal_assert(cursor > 0) << "backprop called with no queued schedules";
    internal_assert(pipeline_feat_queue.size() > 0) << "backprop called before set_pipeline_features";
    
    if (!training_mode) {
        training_mode = true;
        network->train();
        if (optimizer) {
            delete optimizer;
        }
        optimizer = new torch::optim::Adam(network->parameters(), learning_rate);
    }
    
    // Batch schedule features
	// changed cat to stack, so that it batches properly to: (num_schedules, 39, num_stages)
	// instead of merely concatenating to get (1, 39xN, num_stages)
    //auto schedule_features_batch = torch::cat(schedule_feat_queue, 0);
	
	std::cerr<<"in backprop: weights.loaded is "<<weights.is_loaded() <<"\n";
    std::vector<torch::Tensor> transposed;
    for (const auto &t : schedule_feat_queue) {
        transposed.push_back(t.transpose(0, 1).contiguous()); // (head2_w, num_stages)
    }
    auto schedule_features_batch = torch::stack(transposed, 0).contiguous(); // (batch, head2_w, num_stages)
    //auto schedule_features_batch = torch::stack(schedule_feat_queue);

    // Slice pipeline features to only use the first num_stages stages
    auto pipeline_features_full = pipeline_feat_queue[0];
    auto pipeline_features = pipeline_features_full.slice(0, 0, num_stages); // (num_stages, head1_w, head1_h)
    
    // Forward pass
    auto coefficients = network->forward(pipeline_features, schedule_features_batch, num_stages, cursor);
	if (torch::isnan(coefficients).any().item<bool>()) {
		std::cerr << "LibTorchCostModel::backprop, result of forward contains NaN values!\n";
	}else{
		std::cerr << "LibTorchCostModel::backprop, result of forward does NOT contain NaN values!\n";
	}
	std::cerr<<"LibTorchCostModel::backprop: before compute_costs_from_coeff, num cores is "<<num_cores<<"\n";
    auto predictions = compute_cost_from_coefficients(coefficients, schedule_features_batch, num_stages, cursor, num_cores);
    
    // Convert true runtimes to tensor
    auto true_runtimes_tensor = torch::from_blob((void*)true_runtimes.data(), {cursor}, torch::kFloat32).clone();
    
    // Find fastest (reference)
    auto fastest_idx = torch::argmin(true_runtimes_tensor).item<int>();
    auto scale = 1.0f / true_runtimes_tensor[fastest_idx].item<float>();
    
    // Compute relative throughput loss (L2 on relative throughput)
    auto p1 = predictions * scale;
    auto r1 = true_runtimes_tensor * scale;
    auto delta = torch::pow(1.0f / torch::clamp_min(p1, 1e-10f) - 1.0f / r1, 2);
    
    // Regularization term (penalize negative pre-ReLU values)
    // This is simplified - full implementation would need access to pre-ReLU activations
    auto loss = torch::mean(delta);
	std::cerr<<"LibTorchCostModel::backprop, loss: "<<loss<<"\n";

	std::cerr << "=== PRE-BACKWARD DEBUG ===\n";
	std::cerr << "predictions: " << predictions << "\n";
	std::cerr << "predictions min: " << predictions.min().item<float>() << "\n";
	std::cerr << "predictions max: " << predictions.max().item<float>() << "\n";
	std::cerr << "predictions has NaN: " << torch::isnan(predictions).any().item<bool>() << "\n";
	std::cerr << "predictions has inf: " << torch::isinf(predictions).any().item<bool>() << "\n";
	std::cerr << "predictions has negatives: " << (predictions < 0).any().item<bool>() << "\n";
	std::cerr << "predictions has zeros: " << (predictions == 0).any().item<bool>() << "\n";

	std::cerr << "true_runtimes: " << true_runtimes_tensor << "\n";
	std::cerr << "fastest_idx: " << fastest_idx << "\n";
	std::cerr << "fastest_runtime: " << true_runtimes_tensor[fastest_idx].item<float>() << "\n";
	std::cerr << "scale: " << scale << "\n";

	std::cerr << "p1 (scaled predictions): " << p1 << "\n";
	std::cerr << "r1 (scaled true runtimes): " << r1 << "\n";

	auto reciprocal_p1 = 1.0f / torch::clamp_min(p1, 1e-10f);
	auto reciprocal_r1 = 1.0f / r1;
	std::cerr << "1/p1: " << reciprocal_p1 << "\n";
	std::cerr << "1/r1: " << reciprocal_r1 << "\n";
	std::cerr << "1/p1 has inf: " << torch::isinf(reciprocal_p1).any().item<bool>() << "\n";
	std::cerr << "1/r1 has inf: " << torch::isinf(reciprocal_r1).any().item<bool>() << "\n";

	std::cerr << "delta: " << delta << "\n";
	std::cerr << "delta has NaN: " << torch::isnan(delta).any().item<bool>() << "\n";
	std::cerr << "delta has inf: " << torch::isinf(delta).any().item<bool>() << "\n";
	std::cerr << "delta min: " << delta.min().item<float>() << "\n";
	std::cerr << "delta max: " << delta.max().item<float>() << "\n";

	std::cerr << "loss: " << loss.item<float>() << "\n";
	std::cerr << "loss is NaN: " << std::isnan(loss.item<float>()) << "\n";
	std::cerr << "loss is inf: " << std::isinf(loss.item<float>()) << "\n";
	std::cerr << "=========================\n";

    
    // Backward pass
    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
    timestep++;
    
    // Update weights in LibTorchWeights structure
    network->save_weights(weights);
	if (torch::isnan(weights.trunk_fc_0).any().item<bool>()) {
		std::cerr << "LibTorchCostModel::backprop, weights.trunk_fc_0 contains NaN values!\n";
	}else{
		std::cerr << "LibTorchCostModel::backprop, weights.trunk_fc_0 does NOT contain NaN values!\n";
	}
    
    // Copy predictions back
    auto predictions_cpu = predictions.detach().cpu();
    for (int i = 0; i < cursor; i++) {
        internal_assert(cost_ptrs[i]);
        *(cost_ptrs[i]) = predictions_cpu[i].item<double>();
    }
    
    return loss.item<float>();
}

void LibTorchCostModel::save_weights() {
    internal_assert(!weights_out_path.empty())
        << "Unable to save weights: no output path specified\n";
    
	// --------- CHANGED THIS -------------------
    // Save weights from network to LibTorchWeights
    //network->save_weights(weights);
    
    //// Save to file using optimized LibTorchWeights
    //internal_assert(weights.save_to_file(weights_out_path))
        //<< "Unable to save weights to: " << weights_out_path << "\n";
	// -------------------------------------------
	// --------- TO THIS -------------------------
    //internal_assert(network->save_to_file(weights_out_path))
        //<< "Unable to save weights to: " << weights_out_path << "\n";

	// pasting code from CustomModelNetwork->save_weights here. Basically, we can't just create 
	// a new weights object because the weights object within LibTorchModel must persist during 
	// training (duh). Since that weights object is here, it doesn't make sense sending it to
	// network->save_to_file(...) and doing all this over there.
    try {
        network->save_weights(weights);
        
        // Save to file
        bool saved = false;
        if (weights_out_path.size() >= 3 && weights_out_path.substr(weights_out_path.size() - 3) == ".pt") {
            saved = weights.save_to_libtorch_file(weights_out_path);
        } else {
            saved = weights.save_to_file(weights_out_path);
        }
        
        if (saved) {
            aslog(1) << "CustomModelNetwork: Saved weights to " << weights_out_path << "\n";
			std::cerr << "CustomModelNetwork: Saved weights to " << weights_out_path << "\n";
        } else {
            aslog(0) << "CustomModelNetwork: Failed to save weights to " << weights_out_path << "\n";
        }
    } catch (const std::exception &e) {
        aslog(0) << "CustomModelNetwork: Exception saving to " << weights_out_path << ": " << e.what() << "\n";
    } catch (...) {
        aslog(0) << "CustomModelNetwork: Unknown error saving to " << weights_out_path << "\n";
    }
	// -------------------------------------------
	// save_weights stores model weights into our LibTorchWeights object
	// weights.save_to_file saves these weights to a file BY FIRST CONVERTING
	// TO HALIDE WEIGHTS, which is not what we want.
	// directly calling network->save_to_file calls save_weights first internally,
	// but then calls save_to_libtorch_file if our weights were from a .pt file. 
	// This function stores weights in the libtorch format as needed.
}

std::unique_ptr<LibTorchCostModel> make_libtorch_cost_model(const std::string &weights_in_path,
                                                           const std::string &weights_out_path,
                                                           bool randomize_weights) {
    return std::unique_ptr<LibTorchCostModel>(new LibTorchCostModel(weights_in_path, weights_out_path, randomize_weights));
}

}

