// Convert Halide baseline.weights to LibTorch format
// This allows direct loading without conversion overhead

#include <iostream>
#include <fstream>
#include <string>
#include "LibTorchWeights.h"
#include "Weights.h"
#include "ASLog.h"

using namespace Halide;
using namespace Halide::Internal;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.weights> <output_libtorch.weights>\n";
        std::cerr << "  Converts Halide .weights file to LibTorch-compatible format\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::cout << "Loading weights from: " << input_path << "\n";

    // Load Halide weights
    Internal::Weights halide_weights;
    bool loaded = false;
    if (input_path.size() >= 8 && input_path.substr(input_path.size() - 8) == ".weights") {
        loaded = halide_weights.load_from_file(input_path);
    } else {
        loaded = halide_weights.load_from_dir(input_path);
    }

    if (!loaded) {
        std::cerr << "ERROR: Failed to load weights from " << input_path << "\n";
        return 1;
    }

    std::cout << "Converting to LibTorch format...\n";

    // Convert to LibTorch weights
    LibTorchWeights libtorch_weights;
    if (!libtorch_weights.load_from_halide_weights(halide_weights)) {
        std::cerr << "ERROR: Failed to convert weights to LibTorch format\n";
        return 1;
    }

    std::cout << "Saving LibTorch weights to: " << output_path << "\n";

    // Save in LibTorch format using the method from LibTorchWeights
    if (!libtorch_weights.save_to_libtorch_file(output_path)) {
        std::cerr << "ERROR: Failed to save LibTorch weights\n";
        return 1;
    }
    
    std::cout << "SUCCESS: LibTorch weights saved to " << output_path << "\n";
    std::cout << "  - Head1 filter: " << libtorch_weights.head1_filter.sizes() << "\n";
    std::cout << "  - Head2 filter: " << libtorch_weights.head2_filter.sizes() << "\n";
    std::cout << "  - Trunk stage1: " << libtorch_weights.trunk_filter_stage1.sizes() << "\n";
    std::cout << "  - Trunk stage2: " << libtorch_weights.trunk_filter_stage2.sizes() << "\n";
    
    return 0;
}

