##
# Build rules for the Adams2019 autoscheduler library
##

# =================================================================
# weights
set(WF_CPP baseline.cpp)
configure_file(baseline.weights baseline.weights COPYONLY)
add_custom_command(OUTPUT ${WF_CPP}
                   COMMAND binary2cpp baseline_weights < baseline.weights > ${WF_CPP}
                   DEPENDS baseline.weights binary2cpp
                   VERBATIM)

add_library(adams2019_weights_obj OBJECT ${WF_CPP})

# cost_model, train_cost_model
add_executable(adams2019_cost_model.generator cost_model_generator.cpp)
target_link_libraries(adams2019_cost_model.generator PRIVATE Halide::Generator)

add_halide_library(adams2019_cost_model FROM adams2019_cost_model.generator
                   GENERATOR cost_model
                   FUNCTION_NAME cost_model
                   TARGETS cmake)
add_halide_library(adams2019_train_cost_model FROM adams2019_cost_model.generator
                   GENERATOR train_cost_model
                   FUNCTION_NAME train_cost_model
                   TARGETS cmake
                   USE_RUNTIME adams2019_cost_model.runtime)

# TODO: replace when we support "fat" objects in generators
list(LENGTH CMAKE_OSX_ARCHITECTURES num_archs)
if (APPLE AND num_archs GREATER 1)
    if (NOT "x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES OR
        NOT "arm64" IN_LIST CMAKE_OSX_ARCHITECTURES)
        message(FATAL_ERROR "Don't know how to compile for ${CMAKE_OSX_ARCHITECTURES}")
    endif ()

    if (Halide_CMAKE_TARGET MATCHES "x86")
        set(arch arm)
    else ()
        set(arch x86)
    endif ()

    add_halide_library(adams2019_cost_model-arch FROM adams2019_cost_model.generator
                       GENERATOR cost_model
                       FUNCTION_NAME cost_model
                       TARGETS osx-${arch}-64)

    add_halide_library(adams2019_train_cost_model-arch FROM adams2019_cost_model.generator
                       GENERATOR train_cost_model
                       FUNCTION_NAME train_cost_model
                       TARGETS osx-${arch}-64
                       USE_RUNTIME adams2019_cost_model-arch.runtime)

    add_custom_command(
        TARGET adams2019_cost_model POST_BUILD
        COMMAND lipo -create $<TARGET_FILE:adams2019_cost_model-arch> $<TARGET_FILE:adams2019_cost_model> -output $<TARGET_FILE:adams2019_cost_model>
        VERBATIM
    )

    add_custom_command(
        TARGET adams2019_cost_model.runtime POST_BUILD
        COMMAND lipo -create $<TARGET_FILE:adams2019_cost_model-arch.runtime> $<TARGET_FILE:adams2019_cost_model.runtime> -output $<TARGET_FILE:adams2019_cost_model.runtime>
        VERBATIM
    )

    add_custom_command(
        TARGET adams2019_train_cost_model POST_BUILD
        COMMAND lipo -create $<TARGET_FILE:adams2019_train_cost_model-arch> $<TARGET_FILE:adams2019_train_cost_model> -output $<TARGET_FILE:adams2019_train_cost_model>
        VERBATIM
    )
endif ()

# retrain_cost_model
if (WITH_UTILS)
    add_executable(adams2019_retrain_cost_model
                   DefaultCostModel.cpp
                   Weights.cpp
                   retrain_cost_model.cpp
                   $<TARGET_OBJECTS:adams2019_weights_obj>)
    target_link_libraries(adams2019_retrain_cost_model PRIVATE ASLog adams2019_cost_model adams2019_train_cost_model Halide::Halide Halide::Plugin)
endif ()

# =================================================================
##
# Main autoscheduler library
##

# Try to find libtorch (CPU version) Need to set this manually in the environment (I used min -> chamika2)
if(DEFINED ENV{LIBTORCH_ROOT})
    set(LIBTORCH_ROOT $ENV{LIBTORCH_ROOT})
    message(STATUS "Using LIBTORCH_ROOT from environment: ${LIBTORCH_ROOT}")
elseif(DEFINED LIBTORCH_ROOT)
    message(STATUS "Using LIBTORCH_ROOT from CMake: ${LIBTORCH_ROOT}")
else()
    # Try common locations relative to Halide source directory
    # Check if libtorch-cpu is in the parent directory of Halide
    if(EXISTS "${CMAKE_SOURCE_DIR}/../libtorch-cpu/libtorch/share/cmake/Torch")
        set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/../libtorch-cpu/libtorch")
        message(STATUS "Found libtorch-cpu relative to Halide source: ${LIBTORCH_ROOT}")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/../libtorch-cpu/share/cmake/Torch")
        set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/../libtorch-cpu")
        message(STATUS "Found libtorch-cpu relative to Halide source: ${LIBTORCH_ROOT}")
    elseif(EXISTS "/home/chamika2/upstream/libtorch-cpu/libtorch/share/cmake/Torch")
        set(LIBTORCH_ROOT "/home/chamika2/upstream/libtorch-cpu/libtorch")
        message(STATUS "Found libtorch-cpu at default location: ${LIBTORCH_ROOT}")
    elseif(EXISTS "/home/chamika2/upstream/libtorch-cpu/share/cmake/Torch")
        set(LIBTORCH_ROOT "/home/chamika2/upstream/libtorch-cpu")
        message(STATUS "Found libtorch-cpu at default location: ${LIBTORCH_ROOT}")
    endif()
endif()

# Check if libtorch directory exists and set Torch_DIR
if(DEFINED LIBTORCH_ROOT AND EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch")
    set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")
    message(STATUS "Setting Torch_DIR to: ${Torch_DIR}")
    message(STATUS "Using CPU-only libtorch")
    find_package(Torch QUIET)
elseif(DEFINED LIBTORCH_ROOT)
    message(WARNING "LIBTORCH_ROOT is set to ${LIBTORCH_ROOT} but Torch CMake config not found")
    message(WARNING "Expected: ${LIBTORCH_ROOT}/share/cmake/Torch")
endif()

if(NOT Torch_FOUND)
    find_package(Torch QUIET)
endif()

if(Torch_FOUND)
    message(STATUS "Found libtorch:")
    message(STATUS "  TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "  TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    set(USE_LIBTORCH ON)
else()
    message(STATUS "libtorch not found, using default Halide-based cost model")
    message(STATUS "  To use libtorch, set LIBTORCH_ROOT environment variable or CMake variable")
    set(USE_LIBTORCH OFF)
endif()

add_autoscheduler(
    NAME Adams2019
    SOURCES
    AutoSchedule.cpp
    Cache.cpp
    DefaultCostModel.cpp
    FunctionDAG.cpp
    LoopNest.cpp
    State.cpp
    Weights.cpp
    $<TARGET_OBJECTS:adams2019_weights_obj>
)

target_link_libraries(Halide_Adams2019 PRIVATE ASLog ParamParser adams2019_cost_model adams2019_train_cost_model)

# Add libtorch cost model if available
if(USE_LIBTORCH)
    target_sources(Halide_Adams2019 PRIVATE 
        LibTorchCostModel.cpp
        LibTorchCostModelOptimizations.cpp
        LibTorchWeights.cpp
        LibTorchFeatureConverter.cpp
    )
    target_compile_definitions(Halide_Adams2019 PRIVATE USE_LIBTORCH_COST_MODEL)
    
    # Link against torch library first (before setting compile options)
    # This ensures INTERFACE properties from Halide targets are applied first
    target_link_libraries(Halide_Adams2019 PRIVATE ${TORCH_LIBRARIES})
    
    # Add rpath so the library can find LibTorch dependencies at runtime
    # Get the LibTorch lib directory from LIBTORCH_ROOT
    if(DEFINED LIBTORCH_ROOT)
        get_filename_component(TORCH_LIB_DIR "${LIBTORCH_ROOT}/lib" ABSOLUTE)
        if(EXISTS "${TORCH_LIB_DIR}")
            set_target_properties(Halide_Adams2019 PROPERTIES
                INSTALL_RPATH "${TORCH_LIB_DIR}"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
            # Also set runtime path for build tree
            set_target_properties(Halide_Adams2019 PROPERTIES
                BUILD_RPATH "${TORCH_LIB_DIR}"
            )
        endif()
    endif()
    
    # Add include directories
    target_include_directories(Halide_Adams2019 PRIVATE ${TORCH_INCLUDE_DIRS})
    
    # Enable RTTI for libtorch files (libtorch requires RTTI, but Halide is built with -fno-rtti)
    # The -fno-rtti flag comes from INTERFACE_COMPILE_OPTIONS of Halide::Plugin.
    # We need to ensure -frtti comes AFTER -fno-rtti in the compile command (last flag wins).
    # File-level options are typically added last, so we use those to override -fno-rtti.
    
    # Set compile options at the file level - these are added last in the command line
    # and should override -fno-rtti from INTERFACE properties
    set_source_files_properties(
        LibTorchCostModel.cpp
        LibTorchCostModelOptimizations.cpp
        LibTorchWeights.cpp
        LibTorchFeatureConverter.cpp
        AutoSchedule.cpp
        PROPERTIES COMPILE_OPTIONS "-frtti"
    )
    
    # Also add -frtti at target level after linking (this adds it to the end of target options)
    # The combination of file-level and target-level should ensure -frtti appears last
    target_compile_options(Halide_Adams2019 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-frtti>
    )
    
    # Set C++ standard (libtorch requires C++14 or later)
    set_property(TARGET Halide_Adams2019 PROPERTY CXX_STANDARD 14)
    
    # Set torch library properties
    set_property(TARGET Halide_Adams2019 PROPERTY CXX_STANDARD_REQUIRED ON)

    # Build weight conversion utility
    add_executable(adams2019_convert_weights_to_libtorch
        convert_weights_to_libtorch.cpp
        LibTorchWeights.cpp
        Weights.cpp
    )
    target_link_libraries(adams2019_convert_weights_to_libtorch PRIVATE 
        ASLog 
        ${TORCH_LIBRARIES}
        Halide::Halide
        Halide::Tools
    )
    target_include_directories(adams2019_convert_weights_to_libtorch PRIVATE ${TORCH_INCLUDE_DIRS})
    set_source_files_properties(
        convert_weights_to_libtorch.cpp
        LibTorchWeights.cpp
        PROPERTIES COMPILE_OPTIONS "-frtti"
    )
    target_compile_options(adams2019_convert_weights_to_libtorch PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-frtti>
    )
    set_property(TARGET adams2019_convert_weights_to_libtorch PROPERTY CXX_STANDARD 14)
    set_property(TARGET adams2019_convert_weights_to_libtorch PROPERTY CXX_STANDARD_REQUIRED ON)

    # Build weight validation utility
    add_executable(adams2019_validate_weights_conversion
        validate_weights_conversion.cpp
        LibTorchWeights.cpp
        Weights.cpp
    )
    target_link_libraries(adams2019_validate_weights_conversion PRIVATE 
        ASLog 
        ${TORCH_LIBRARIES}
        Halide::Halide
        Halide::Tools
    )
    target_include_directories(adams2019_validate_weights_conversion PRIVATE ${TORCH_INCLUDE_DIRS})
    set_source_files_properties(
        validate_weights_conversion.cpp
        LibTorchWeights.cpp
        PROPERTIES COMPILE_OPTIONS "-frtti"
    )
    target_compile_options(adams2019_validate_weights_conversion PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-frtti>
    )
    set_property(TARGET adams2019_validate_weights_conversion PROPERTY CXX_STANDARD 14)
    set_property(TARGET adams2019_validate_weights_conversion PROPERTY CXX_STANDARD_REQUIRED ON)
endif()

# ====================================================
# Auto-tuning support utilities.
# TODO(#4053): implement auto-tuning support in CMake?

if (WITH_UTILS)
    add_executable(adams2019_featurization_to_sample featurization_to_sample.cpp)

    add_executable(adams2019_get_host_target get_host_target.cpp)
    target_link_libraries(adams2019_get_host_target PRIVATE Halide::Halide)

    add_executable(adams2019_weightsdir_to_weightsfile weightsdir_to_weightsfile.cpp Weights.cpp)
    target_link_libraries(adams2019_weightsdir_to_weightsfile PRIVATE Halide::Runtime)
endif ()

# =================================================================
# Tests for private/internal functionality of Adams2019 (vs for public functionality,
# which is handled in tests/autoschedulers/Adams2019)

if (WITH_TESTS)

    add_executable(adams2019_test_perfect_hash_map test_perfect_hash_map.cpp)
    add_test(NAME adams2019_test_perfect_hash_map COMMAND adams2019_test_perfect_hash_map)
    set_tests_properties(adams2019_test_perfect_hash_map PROPERTIES LABELS "adams2019;autoschedulers;auto_schedule")

    add_executable(adams2019_test_function_dag test_function_dag.cpp FunctionDAG.cpp)
    target_link_libraries(adams2019_test_function_dag PRIVATE ASLog Halide::Halide Halide::Tools Halide::Plugin)
    add_test(NAME adams2019_test_function_dag COMMAND adams2019_test_function_dag)
    set_tests_properties(adams2019_test_function_dag PROPERTIES LABELS "adams2019;autoschedulers;auto_schedule")

endif()
