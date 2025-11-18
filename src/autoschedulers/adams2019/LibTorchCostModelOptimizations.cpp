#include "LibTorchCostModelOptimizations.h"
#include "NetworkSize.h"
#include <algorithm>

namespace Halide {

torch::Tensor TensorMemoryPool::get_tensor(const std::vector<int64_t> &shape, torch::ScalarType dtype) {
    // Try to find a cached tensor with matching shape
    for (size_t i = 0; i < cached_shapes.size(); i++) {
        if (cached_shapes[i] == shape) {
            // Found matching shape, return it (will be overwritten by caller)
            return cached_tensors[i];
        }
    }
    
    // No match found, create new tensor
    torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
    cached_tensors.push_back(tensor);
    cached_shapes.push_back(shape);
    return tensor;
}

void TensorMemoryPool::clear() {
    cached_tensors.clear();
    cached_shapes.clear();
}

UnpackedScheduleFeatures UnpackedScheduleFeatures::unpack(const torch::Tensor &schedule_features_tensor) {
    // schedule_features_tensor: (batch, head2_w=39, num_stages)
    // Optimized: use narrow() and transpose() once instead of many select() + transpose()
    
    UnpackedScheduleFeatures features;
    int batch_size = schedule_features_tensor.size(0);
    int num_stages = schedule_features_tensor.size(2);
    
    // Transpose once: (batch, head2_w, num_stages) -> (num_stages, head2_w, batch)
    auto transposed = schedule_features_tensor.permute({2, 1, 0}); // (num_stages, head2_w, batch)
    
    // Now use narrow() to extract each feature dimension (much faster than select + transpose)
    int idx = 0;
    features.num_realizations = transposed.narrow(1, idx++, 1).squeeze(1); // (num_stages, batch)
    features.num_productions = transposed.narrow(1, idx++, 1).squeeze(1);
    features.points_computed_per_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.points_computed_per_production = transposed.narrow(1, idx++, 1).squeeze(1);
    features.points_computed_total = transposed.narrow(1, idx++, 1).squeeze(1);
    features.points_computed_minimum = transposed.narrow(1, idx++, 1).squeeze(1);
    features.innermost_loop_extent = transposed.narrow(1, idx++, 1).squeeze(1);
    features.innermost_pure_loop_extent = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unrolled_loop_extent = transposed.narrow(1, idx++, 1).squeeze(1);
    features.inner_parallelism = transposed.narrow(1, idx++, 1).squeeze(1);
    features.outer_parallelism = transposed.narrow(1, idx++, 1).squeeze(1);
    features.bytes_at_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.bytes_at_production = transposed.narrow(1, idx++, 1).squeeze(1);
    features.bytes_at_root = transposed.narrow(1, idx++, 1).squeeze(1);
    features.innermost_bytes_at_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.innermost_bytes_at_production = transposed.narrow(1, idx++, 1).squeeze(1);
    features.innermost_bytes_at_root = transposed.narrow(1, idx++, 1).squeeze(1);
    features.inlined_calls = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unique_bytes_read_per_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unique_lines_read_per_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.allocation_bytes_read_per_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.working_set = transposed.narrow(1, idx++, 1).squeeze(1);
    features.vector_size = transposed.narrow(1, idx++, 1).squeeze(1);
    features.native_vector_size = transposed.narrow(1, idx++, 1).squeeze(1);
    features.num_vectors = transposed.narrow(1, idx++, 1).squeeze(1);
    features.num_scalars = transposed.narrow(1, idx++, 1).squeeze(1);
    features.scalar_loads_per_vector = transposed.narrow(1, idx++, 1).squeeze(1);
    features.vector_loads_per_vector = transposed.narrow(1, idx++, 1).squeeze(1);
    features.scalar_loads_per_scalar = transposed.narrow(1, idx++, 1).squeeze(1);
    features.bytes_at_task = transposed.narrow(1, idx++, 1).squeeze(1);
    features.innermost_bytes_at_task = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unique_bytes_read_per_vector = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unique_lines_read_per_vector = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unique_bytes_read_per_task = transposed.narrow(1, idx++, 1).squeeze(1);
    features.unique_lines_read_per_task = transposed.narrow(1, idx++, 1).squeeze(1);
    features.working_set_at_task = transposed.narrow(1, idx++, 1).squeeze(1);
    features.working_set_at_production = transposed.narrow(1, idx++, 1).squeeze(1);
    features.working_set_at_realization = transposed.narrow(1, idx++, 1).squeeze(1);
    features.working_set_at_root = transposed.narrow(1, idx++, 1).squeeze(1);
    
    // Transpose back to (batch, num_stages) for easier use
    // Actually, keep as (num_stages, batch) for now - we'll transpose when needed
    // This avoids another transpose operation
    
    return features;
}

}

