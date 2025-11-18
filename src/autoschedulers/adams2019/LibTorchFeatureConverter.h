#ifndef LIBTORCH_FEATURE_CONVERTER_H
#define LIBTORCH_FEATURE_CONVERTER_H

#include <torch/torch.h>
#include "Featurization.h"
#include "FunctionDAG.h"
#include "PerfectHashMap.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {
    struct FunctionDAG;
    struct Adams2019Params;
    typedef PerfectHashMap<FunctionDAG::Node::Stage, ScheduleFeatures> StageMapOfScheduleFeatures;
}  // namespace Autoscheduler
}  // namespace Internal

// Optimized feature conversion for LibTorch
// Converts Halide features to LibTorch tensors efficiently
class LibTorchFeatureConverter {
public:
    static constexpr int head1_w = 40;
    static constexpr int head1_h = 7;
    static constexpr int head2_w = 39;

    // Convert pipeline features from DAG to LibTorch tensor
    // Returns: (max_num_stages, head1_w, head1_h)
    static torch::Tensor convert_pipeline_features(
        const Internal::Autoscheduler::FunctionDAG &dag,
        int &max_num_stages_out);

    // Convert schedule features to LibTorch tensor (iterating in DAG order)
    // Returns: (head2_w, num_stages)
    // Must iterate in DAG order to match DefaultCostModel
    static torch::Tensor convert_schedule_features(
        const Internal::Autoscheduler::FunctionDAG &dag,
        const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
        int num_stages);

    // Batch multiple schedule feature tensors
    // Input: vector of (head2_w, num_stages) tensors
    // Returns: (batch, head2_w, num_stages)
    static torch::Tensor batch_schedule_features(
        const std::vector<torch::Tensor> &schedule_feat_tensors);

private:
    // Helper: Extract pipeline features from a stage
    static void extract_pipeline_features_from_stage(
        const Internal::PipelineFeatures &pf,
        torch::Tensor &output,
        int stage_idx);
};

}  // namespace Halide

#endif  // LIBTORCH_FEATURE_CONVERTER_H

