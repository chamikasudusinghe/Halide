#ifndef LIBTORCH_COST_MODEL_H
#define LIBTORCH_COST_MODEL_H

#include "CostModel.h"
#include "LibTorchWeights.h"
#include "LibTorchFeatureConverter.h"
#include "LibTorchCostModelOptimizations.h"
#include <torch/torch.h>
#include <string>
#include <memory>

namespace Halide {

namespace Internal {
namespace Autoscheduler {
struct Adams2019Params;
} 
}

// Neural network module for the cost model
class CostModelNetwork : public torch::nn::Module {
public:
    CostModelNetwork();
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor &pipeline_features,
                         const torch::Tensor &schedule_features,
                         int num_stages,
                         int batch_size);

    void load_weights(const LibTorchWeights &w);
    void save_weights(LibTorchWeights &w) const;

private:
    // Head1: processes pipeline features (40x7) -> 8 channels
    // Use separate conv layer with sigmoided weights to avoid weight swapping overhead
    torch::nn::Conv2d head1_conv_raw{nullptr};  // Stores raw weights (for saving)
    torch::nn::Conv2d head1_conv{nullptr};  // Uses sigmoided weights (for forward pass)
    
    // Head2: processes schedule features (39) -> 24 channels
    torch::nn::Conv1d head2_conv{nullptr};
    
    // Trunk: combines both heads in two stages (matching original architecture)
    // Stage1: processes head1 (8 channels) -> 32 channels
    // Stage2: adds head2 (24 channels) -> 32 channels
    torch::nn::Conv1d trunk_conv_stage1{nullptr};  // Processes head1: (8) -> (32)
    torch::nn::Conv1d trunk_conv_stage2{nullptr};  // Processes head2: (24) -> (32)
};

class LibTorchCostModel : public CostModel {
private:
    std::unique_ptr<CostModelNetwork> network;
    LibTorchWeights weights;  // Use optimized LibTorch weights instead of Halide Weights
    std::vector<torch::Tensor> pipeline_feat_queue;
    std::vector<torch::Tensor> schedule_feat_queue;
    std::vector<double *> cost_ptrs;
    int cursor = 0;
    int num_stages = 0;
    int num_cores = 0;
    int batch_size = 1024;  // Match original DefaultCostModel batch size for better performance
    
    // Memory pool for tensor reuse
    TensorMemoryPool tensor_pool;
    
    const std::string weights_in_path, weights_out_path;
    const bool randomize_weights;
    
    // Training state
    torch::optim::Adam *optimizer = nullptr;
    int timestep = 0;
    bool training_mode = false;

public:
    LibTorchCostModel(const std::string &weights_in_path,
                     const std::string &weights_out_path,
                     bool randomize_weights);
    ~LibTorchCostModel() override;

    // Configure the cost model for the algorithm to be scheduled.
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                               const Internal::Autoscheduler::Adams2019Params &params) override;

    // Enqueue a schedule to be evaluated.
    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                 const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                 double *cost_ptr) override;

    // Evaluate all schedules in the queue.
    void evaluate_costs() override;

    // Discard all schedules in the queue.
    void reset() override;
    
    // Update model weights using true measured runtimes (training).
    float backprop(const Runtime::Buffer<const float> &true_runtimes, float learning_rate);
    
    // Save the model weights to disk.
    void save_weights();
    
    // Compute cost from network coefficients (matching the hand-designed formula)
    torch::Tensor compute_cost_from_coefficients(const torch::Tensor &coefficients,
                                                  const torch::Tensor &schedule_features_tensor,
                                                  int num_stages,
                                                  int batch_size,
                                                  int num_cores);
};

std::unique_ptr<LibTorchCostModel> make_libtorch_cost_model(const std::string &weights_in_path = "",
                                                           const std::string &weights_out_path = "",
                                                           bool randomize_weights = false);

}

#endif  // LIBTORCH_COST_MODEL_H



