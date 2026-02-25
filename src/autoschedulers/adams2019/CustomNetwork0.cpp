#include "CustomNetwork0.h"
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

CustomNetwork0::CustomNetwork0(std::string input_weights_path, const std::string &architecture_type, bool use_random_weights) : CustomModelNetwork(input_weights_path, architecture_type, use_random_weights) {
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

	trunk_fc_0 = register_module("trunk_fc_0",
		torch::nn::Conv1d(torch::nn::Conv1dOptions(2*conv1_channels, conv1_channels, 1).bias(true)));

	//weights = std::make_shared<LibTorchWeights>(use_random_weights);

	initialize_weights(use_random_weights, input_weights_path);
}

torch::Tensor CustomNetwork0::forward(const torch::Tensor &pipeline_features,
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
    
	 ////Add stage2 to stage1 (matching original: conv1_stage2 = conv1_stage1 + ...)
    //auto trunk_out = trunk_stage1 + trunk_stage2; // (batch, conv1_channels, num_stages)
	
	auto combined = torch::cat({trunk_stage1, trunk_stage2}, /*dim=*/1); // (batch, 2*conv1_channels, num_stages)
	//std::cerr<<"CustomNetwork0::backprop combined:"<<combined<<"\n";

	//if (torch::isnan(combined).any().item<bool>()) {
		//std::cerr << "CustomModelNetwork::forward, 'combined' contains NaN values!\n";
	//}else{

		//std::cerr << "CustomModelNetwork::forward, 'combined' does NOT contain NaN values\n";
	//}

	// Single FC layer
	auto trunk_out = trunk_fc_0->forward(combined);  // (batch, conv1_channels, num_stages)
    // Apply ReLU (matching original relu1)
    trunk_out = torch::relu(trunk_out);
    trunk_out = trunk_out.permute({0, 2, 1}).contiguous(); // (batch, num_stages, conv1_channels)

    
    return trunk_out;
}

REGISTER_LIBTORCH_MODEL("custom0",CustomNetwork0)

void CustomNetwork0::load_weights(const LibTorchWeights &w) {
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

	trunk_fc_0->weight.data() = w.trunk_fc_0;
	trunk_fc_0->bias.data() = w.trunk_fc_0_bias;
	//std::cerr<<"loading weights from network->weights...\n";
	//std::cerr<<"CustomNetwork0::load_weights: trunk_fc_0->weight: "<<trunk_fc_0->weight.data()<<"\n";
	//std::cerr<<"CustomNetwork0::load_weights: trunk_fc_0->bias: "<<w.trunk_fc_0_bias<<"\n";
}

void CustomNetwork0::save_weights(LibTorchWeights &w) const {
	sync_weights_from_network();
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

	// save final 1d conv weights
	w.trunk_fc_0 = trunk_fc_0->weight.data().clone();
	w.trunk_fc_0_bias = trunk_fc_0->bias.data().clone();
}

void CustomNetwork0::randomize_weights() {
	auto seed = time(nullptr);
    torch::manual_seed(seed);
    
    // Randomize head1 weights (raw and sigmoided)
    for (auto &param : head1_conv_raw->parameters()) {
        torch::nn::init::normal_(param, 0.0, 0.1);
    }
    // Copy raw weights and apply sigmoid for sigmoided version
    head1_conv->weight.data() = torch::sigmoid(head1_conv_raw->weight.data());
    head1_conv->bias.data() = head1_conv_raw->bias.data().clone();
    
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

	for (auto &param : trunk_fc_0->parameters()) {
		torch::nn::init::normal_(param, 0.0, 0.1);
	}
    
    aslog(1) << "CustomNetwork0: Initialized with random weights (seed=" << seed << ")\n";

}



}
