#ifndef LIBTORCH_COST_MODEL_OPTIMIZATIONS_H
#define LIBTORCH_COST_MODEL_OPTIMIZATIONS_H

#include <torch/torch.h>
#include <vector>

namespace Halide {

// Tensor memory pool to avoid repeated allocations
class TensorMemoryPool {
public:
    // Get or create a tensor of the specified shape
    torch::Tensor get_tensor(const std::vector<int64_t> &shape, torch::ScalarType dtype = torch::kFloat32);
    
    // Clear all cached tensors
    void clear();
    
private:
    std::vector<torch::Tensor> cached_tensors;
    std::vector<std::vector<int64_t>> cached_shapes;
};

// Optimized cost computation using pre-unpacked features
// This avoids many select() and transpose() operations
struct UnpackedScheduleFeatures {
    torch::Tensor num_realizations;           // (batch, num_stages)
    torch::Tensor num_productions;
    torch::Tensor points_computed_per_realization;
    torch::Tensor points_computed_per_production;
    torch::Tensor points_computed_total;
    torch::Tensor points_computed_minimum;
    torch::Tensor innermost_loop_extent;
    torch::Tensor innermost_pure_loop_extent;
    torch::Tensor unrolled_loop_extent;
    torch::Tensor inner_parallelism;
    torch::Tensor outer_parallelism;
    torch::Tensor bytes_at_realization;
    torch::Tensor bytes_at_production;
    torch::Tensor bytes_at_root;
    torch::Tensor innermost_bytes_at_realization;
    torch::Tensor innermost_bytes_at_production;
    torch::Tensor innermost_bytes_at_root;
    torch::Tensor inlined_calls;
    torch::Tensor unique_bytes_read_per_realization;
    torch::Tensor unique_lines_read_per_realization;
    torch::Tensor allocation_bytes_read_per_realization;
    torch::Tensor working_set;
    torch::Tensor vector_size;
    torch::Tensor native_vector_size;
    torch::Tensor num_vectors;
    torch::Tensor num_scalars;
    torch::Tensor scalar_loads_per_vector;
    torch::Tensor vector_loads_per_vector;
    torch::Tensor scalar_loads_per_scalar;
    torch::Tensor bytes_at_task;
    torch::Tensor innermost_bytes_at_task;
    torch::Tensor unique_bytes_read_per_vector;
    torch::Tensor unique_lines_read_per_vector;
    torch::Tensor unique_bytes_read_per_task;
    torch::Tensor unique_lines_read_per_task;
    torch::Tensor working_set_at_task;
    torch::Tensor working_set_at_production;
    torch::Tensor working_set_at_realization;
    torch::Tensor working_set_at_root;
    
    // Unpack from (batch, head2_w, num_stages) tensor
    static UnpackedScheduleFeatures unpack(const torch::Tensor &schedule_features_tensor);
};

}

#endif  // LIBTORCH_COST_MODEL_OPTIMIZATIONS_H

