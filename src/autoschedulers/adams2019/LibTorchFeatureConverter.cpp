#include "LibTorchFeatureConverter.h"
#include "FunctionDAG.h"
#include "Featurization.h"
#include "NetworkSize.h"
#include "Errors.h"
#include "PerfectHashMap.h"
#include <algorithm>

namespace Halide {

torch::Tensor LibTorchFeatureConverter::convert_pipeline_features(
    const Internal::Autoscheduler::FunctionDAG &dag,
    int &max_num_stages_out) {
    
    // Count total number of stages
    int max_num_stages = 0;
    for (const auto &n : dag.nodes) {
        if (!n.is_input) max_num_stages += (int)n.stages.size();
    }
    max_num_stages_out = max_num_stages;

    // Allocate tensor: (max_num_stages, head1_w, head1_h)
    auto pf_tensor = torch::zeros({max_num_stages, head1_w, head1_h}, torch::kFloat32);
    
    const int pipeline_feat_size = head1_w * head1_h;
    int stage = 0;
    
    // Extract features from each stage
    for (const auto &n : dag.nodes) {
        if (n.is_input) continue;
        for (auto it = n.stages.rbegin(); it != n.stages.rend(); it++) {
            const auto &s = *it;
            extract_pipeline_features_from_stage(s.features, pf_tensor, stage);
            stage += 1;
        }
    }
    
    internal_assert(stage == max_num_stages);
    return pf_tensor.contiguous(); // Ensure contiguous for better performance
}

void LibTorchFeatureConverter::extract_pipeline_features_from_stage(
    const Internal::PipelineFeatures &pf,
    torch::Tensor &output,
    int stage_idx) {
    
    const int pipeline_feat_size = head1_w * head1_h;
    const int *pipeline_feats = (const int *)(&pf) + 7; // skip first 7 features
    
    // Direct memory copy for efficiency
    auto stage_tensor = output[stage_idx]; // (head1_w, head1_h)
    auto data_ptr = stage_tensor.data_ptr<float>();
    
    for (int i = 0; i < pipeline_feat_size; i++) {
        int x = i / 7;
        int y = i % 7;
        data_ptr[x * head1_h + y] = (float)pipeline_feats[i];
    }
}

torch::Tensor LibTorchFeatureConverter::convert_schedule_features(
    const Internal::Autoscheduler::FunctionDAG &dag,
    const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
    int num_stages) {
    
    // Allocate tensor: (head2_w, num_stages)
    auto sf_tensor = torch::zeros({head2_w, num_stages}, torch::kFloat32);
    
    // Extract features in DAG order (matching DefaultCostModel)
    // This matches the order in cost_model_generator.cpp
    int stage_idx = 0;
    for (const auto &n : dag.nodes) {
        if (n.is_input) continue;
        if (stage_idx >= num_stages) break;
        
        for (auto it = n.stages.rbegin(); it != n.stages.rend(); it++) {
            if (stage_idx >= num_stages) break;
            internal_assert(schedule_feats.contains(&*it)) << n.func.name() << "\n";
            const auto &feat = schedule_feats.get(&*it);
            
            int feat_idx = 0;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.num_realizations;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.num_productions;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.points_computed_per_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.points_computed_per_production;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.points_computed_total;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.points_computed_minimum;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.innermost_loop_extent;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.innermost_pure_loop_extent;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unrolled_loop_extent;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.inner_parallelism;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.outer_parallelism;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.bytes_at_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.bytes_at_production;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.bytes_at_root;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.innermost_bytes_at_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.innermost_bytes_at_production;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.innermost_bytes_at_root;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.inlined_calls;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unique_bytes_read_per_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unique_lines_read_per_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.allocation_bytes_read_per_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.working_set;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.vector_size;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.native_vector_size;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.num_vectors;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.num_scalars;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.scalar_loads_per_vector;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.vector_loads_per_vector;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.scalar_loads_per_scalar;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.bytes_at_task;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.innermost_bytes_at_task;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unique_bytes_read_per_vector;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unique_lines_read_per_vector;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unique_bytes_read_per_task;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.unique_lines_read_per_task;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.working_set_at_task;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.working_set_at_production;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.working_set_at_realization;
        sf_tensor[feat_idx++][stage_idx] = (float)feat.working_set_at_root;
        
            stage_idx++;
        }
    }
    
    internal_assert(stage_idx == num_stages);
    
    // Transpose to (num_stages, head2_w) then add batch dimension will be (batch, head2_w, num_stages)
    // Actually, we want (head2_w, num_stages) for now - will be batched later
    return sf_tensor.contiguous();
}

torch::Tensor LibTorchFeatureConverter::batch_schedule_features(
    const std::vector<torch::Tensor> &schedule_feat_tensors) {
    
    if (schedule_feat_tensors.empty()) {
        return torch::empty({0, head2_w, 0}, torch::kFloat32);
    }
    
    // Each tensor is (head2_w, num_stages)
    // We need to stack them along a new batch dimension: (batch, head2_w, num_stages)
    // First, transpose each to (num_stages, head2_w), then stack, then transpose back
    
    std::vector<torch::Tensor> transposed;
    for (const auto &t : schedule_feat_tensors) {
        transposed.push_back(t.transpose(0, 1).contiguous()); // (num_stages, head2_w)
    }
    
    // Stack along batch dimension: (batch, num_stages, head2_w)
    auto batched = torch::stack(transposed, 0);
    
    // Transpose to (batch, head2_w, num_stages)
    return batched.permute({0, 2, 1}).contiguous();
}

}  // namespace Halide

