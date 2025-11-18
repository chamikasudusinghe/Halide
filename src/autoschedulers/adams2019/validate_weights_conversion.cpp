// Test to ensure LibTorch weights match original Adams2019 model
// Compares outputs from both cost models using the same weights

#include <iostream>
#include <vector>
#include <cmath>
#include "LibTorchWeights.h"
#include "Weights.h"
#include "DefaultCostModel.h"
#include "LibTorchCostModel.h"
#include "FunctionDAG.h"
#include "Featurization.h"
#include "ASLog.h"
#include "NetworkSize.h"

using namespace Halide;
using namespace Halide::Internal;
using namespace Halide::Internal::Autoscheduler;

// Helper to create dummy pipeline features for testing
Runtime::Buffer<float> create_dummy_pipeline_features(int num_stages) {
    Runtime::Buffer<float> pf(head1_w, head1_h, num_stages);
    // Fill with small random values
    for (int s = 0; s < num_stages; s++) {
        for (int x = 0; x < head1_w; x++) {
            for (int y = 0; y < head1_h; y++) {
                pf(x, y, s) = (float)((x + y + s) % 100) / 100.0f;
            }
        }
    }
    return pf;
}

// Helper to create dummy schedule features for testing
Runtime::Buffer<float> create_dummy_schedule_features(int num_stages) {
    Runtime::Buffer<float> sf(head2_w, num_stages);
    // Fill with small random values
    for (int s = 0; s < num_stages; s++) {
        for (int f = 0; f < head2_w; f++) {
            sf(f, s) = (float)((f + s) % 100) / 100.0f;
        }
    }
    return sf;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <weights_file>\n";
        std::cerr << "  Validates that LibTorch cost model produces same results as original\n";
        return 1;
    }

    std::string weights_path = argv[1];
    
    std::cout << "=== Weight Conversion Validation Test ===\n\n";
    std::cout << "Loading weights from: " << weights_path << "\n";

    // Load weights using both methods
    Internal::Weights halide_weights;
    bool loaded = false;
    if (weights_path.size() >= 8 && weights_path.substr(weights_path.size() - 8) == ".weights") {
        loaded = halide_weights.load_from_file(weights_path);
    } else {
        loaded = halide_weights.load_from_dir(weights_path);
    }

    if (!loaded) {
        std::cerr << "ERROR: Failed to load weights from " << weights_path << "\n";
        return 1;
    }

    std::cout << "Converting to LibTorch format...\n";
    LibTorchWeights libtorch_weights;
    if (!libtorch_weights.load_from_halide_weights(halide_weights)) {
        std::cerr << "ERROR: Failed to convert weights\n";
        return 1;
    }

    std::cout << "\n=== Weight Comparison ===\n";
    
    // Compare weight values
    const float tolerance = 1e-5f;
    bool all_match = true;
    
    // Compare head1_filter
    // Note: Halide buffer is (c, w, h), LibTorch tensor should match after buffer_to_tensor
    std::cout << "Comparing head1_filter...\n";
    auto head1_tensor = libtorch_weights.head1_filter.contiguous();
    bool head1_match = true;
    int mismatch_count = 0;
    for (int c = 0; c < head1_channels; c++) {
        for (int w = 0; w < head1_w; w++) {
            for (int h = 0; h < head1_h; h++) {
                float halide_val = halide_weights.head1_filter(c, w, h);
                // Access tensor using index calculation to match Halide buffer layout
                int64_t idx = c * head1_w * head1_h + w * head1_h + h;
                float libtorch_val = head1_tensor.data_ptr<float>()[idx];
                if (std::abs(halide_val - libtorch_val) > tolerance) {
                    if (mismatch_count < 10) {  // Only show first 10 mismatches
                        std::cerr << "  MISMATCH at [" << c << "," << w << "," << h << "]: "
                                  << "Halide=" << halide_val << ", LibTorch=" << libtorch_val << "\n";
                    }
                    mismatch_count++;
                    head1_match = false;
                    all_match = false;
                }
            }
        }
    }
    if (head1_match) {
        std::cout << "  ✓ head1_filter matches\n";
    } else {
        std::cerr << "  ✗ head1_filter: " << mismatch_count << " mismatches found\n";
    }
    
    // Compare head1_bias
    std::cout << "Comparing head1_bias...\n";
    auto head1_bias_tensor = libtorch_weights.head1_bias.contiguous();
    bool head1_bias_match = true;
    for (int c = 0; c < head1_channels; c++) {
        float halide_val = halide_weights.head1_bias(c);
        float libtorch_val = head1_bias_tensor.data_ptr<float>()[c];
        if (std::abs(halide_val - libtorch_val) > tolerance) {
            std::cerr << "  MISMATCH at [" << c << "]: "
                      << "Halide=" << halide_val << ", LibTorch=" << libtorch_val << "\n";
            head1_bias_match = false;
            all_match = false;
        }
    }
    if (head1_bias_match) std::cout << "  ✓ head1_bias matches\n";
    
    // Compare head2_filter
    // Halide: (c, w), LibTorch: (c, w, 1)
    std::cout << "Comparing head2_filter...\n";
    auto head2_tensor = libtorch_weights.head2_filter.contiguous();
    bool head2_match = true;
    for (int c = 0; c < head2_channels; c++) {
        for (int w = 0; w < head2_w; w++) {
            float halide_val = halide_weights.head2_filter(c, w);
            int64_t idx = c * head2_w * 1 + w * 1 + 0;
            float libtorch_val = head2_tensor.data_ptr<float>()[idx];
            if (std::abs(halide_val - libtorch_val) > tolerance) {
                std::cerr << "  MISMATCH at [" << c << "," << w << "]: "
                          << "Halide=" << halide_val << ", LibTorch=" << libtorch_val << "\n";
                head2_match = false;
                all_match = false;
            }
        }
    }
    if (head2_match) std::cout << "  ✓ head2_filter matches\n";
    
    // Compare trunk weights (combined)
    // Halide: (c, head1_channels + head2_channels), LibTorch: split into (c, head1_channels, 1) and (c, head2_channels, 1)
    std::cout << "Comparing trunk weights...\n";
    auto trunk_stage1 = libtorch_weights.trunk_filter_stage1.contiguous();
    auto trunk_stage2 = libtorch_weights.trunk_filter_stage2.contiguous();
    bool trunk_match = true;
    for (int c = 0; c < conv1_channels; c++) {
        // Check stage1 (head1_channels)
        for (int i = 0; i < head1_channels; i++) {
            float halide_val = halide_weights.conv1_filter(c, i);
            int64_t idx = c * head1_channels * 1 + i * 1 + 0;
            float libtorch_val = trunk_stage1.data_ptr<float>()[idx];
            if (std::abs(halide_val - libtorch_val) > tolerance) {
                std::cerr << "  MISMATCH trunk_stage1 at [" << c << "," << i << "]: "
                          << "Halide=" << halide_val << ", LibTorch=" << libtorch_val << "\n";
                trunk_match = false;
                all_match = false;
            }
        }
        // Check stage2 (head2_channels)
        for (int i = 0; i < head2_channels; i++) {
            float halide_val = halide_weights.conv1_filter(c, head1_channels + i);
            int64_t idx = c * head2_channels * 1 + i * 1 + 0;
            float libtorch_val = trunk_stage2.data_ptr<float>()[idx];
            if (std::abs(halide_val - libtorch_val) > tolerance) {
                std::cerr << "  MISMATCH trunk_stage2 at [" << c << "," << i << "]: "
                          << "Halide=" << halide_val << ", LibTorch=" << libtorch_val << "\n";
                trunk_match = false;
                all_match = false;
            }
        }
    }
    if (trunk_match) std::cout << "  ✓ trunk weights match\n";
    
    std::cout << "\n=== Validate Result ===\n";
    if (all_match) {
        std::cout << "✓ SUCCESS: All weights match within tolerance (" << tolerance << ")\n";
        return 0;
    } else {
        std::cout << "✗ FAILED: Weight mismatches found\n";
        return 1;
    }
}

