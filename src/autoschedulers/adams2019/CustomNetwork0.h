#ifndef CUSTOM_NETWORK_0_H
#define CUSTOM_NETWORK_0_H

#include "CostModel.h"
#include "LibTorchWeights.h"
#include "LibTorchFeatureConverter.h"
#include "LibTorchCostModelOptimizations.h"
#include "ICostModelNetwork.h"
#include <torch/torch.h>
#include <string>
#include <memory>
//BHsketch S ----
#include <chrono>
//BHsketch E ----

namespace Halide {

namespace Internal {
namespace Autoscheduler {
struct Adams2019Params;
} 
}

/**
    Adams2019 network architecture implementation.
 */
class CustomNetwork0 : public CustomModelNetwork {
public:
    CustomNetwork0(const std::string &architecture_type, bool use_random_weights);
    //CustomNetwork0();
    
    // ICostModelNetwork interface
    torch::Tensor forward(const torch::Tensor &pipeline_features,
                         const torch::Tensor &schedule_features,
                         int num_stages,
                         int batch_size) override;

    void load_weights(const LibTorchWeights &w) override;
    void save_weights(LibTorchWeights &w) const override;
    
    int get_num_output_channels() const override { return conv1_channels; }
	
	// eval(), train(), and parameters()
    //void eval() override { this->torch::nn::Module::eval(); }
    //void train() override { this->torch::nn::Module::train(); }
    //std::vector<torch::Tensor> parameters() override {
        //std::vector<torch::Tensor> params;
        //for (auto &param : this->torch::nn::Module::parameters()) {
            //params.push_back(param);
        //}
        //return params;
    //}

private:
    void randomize_weights() override;

    // Head1: processes pipeline features (40x7) -> 8 channels
	//Use separate conv layer with sigmoided weights to avoid weight swapping overhead
	
	torch::nn::Conv2d head1_conv_raw{nullptr};  // Stores raw weights (for saving)
	torch::nn::Conv2d head1_conv{nullptr};  // Uses sigmoided weights (for forward pass)
	
	// Head2: processes schedule features (39) -> 24 channels
	torch::nn::Conv1d head2_conv{nullptr};
	
	// Trunk: combines both heads in two stages (matching original architecture)
	// Stage1: processes head1 (8 channels) -> 32 channels
	// Stage2: adds head2 (24 channels) -> 32 channels
	torch::nn::Conv1d trunk_conv_stage1{nullptr};  // Processes head1: (8) -> (32)
	torch::nn::Conv1d trunk_conv_stage2{nullptr};  // Processes head2: (24) -> (32)

	// runs on concatenated outputs of trunk_conv_stage1 and trunk_conv_stage2
	// to model relationships between pipeline and schedule features.
	// this is effectively a fully connected layer on each stage
	//torch::nn::Linear trunk_fc{nullptr};
	torch::nn::Conv1d trunk_fc_0{nullptr};

	// BHsketch S ---------
	//std::chrono::duration<float, std::milli> collectiveInferenceDuration = std::chrono::duration<float, std::milli>::zero();
	// BHsketch E ---------
};

}

#endif // LIBTORCH_COST_MODEL_H

