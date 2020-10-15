#include "Halide.h"
#include "common_halide.h"

namespace interpret_nn {

using Halide::_0;
using Halide::_1;
using Halide::_2;
using Halide::_3;
using Halide::Generator;
using Halide::Target;
using Halide::Type;
using Halide::BoundaryConditions::constant_exterior;
using Halide::ConciseCasts::i16;
using Halide::ConciseCasts::i16_sat;
using Halide::ConciseCasts::i32;
using Halide::ConciseCasts::u32;
using Halide::ConciseCasts::u8_sat;

int GetVectorReduction(const Target &target, Type t) {
    if (target.has_feature(Target::ARMDotProd)) {
        // ARM dot products can do 4-way reductions.
        return 4;
    }
    if (target.arch == Target::Hexagon) {
        // Hexagon can reduce 32-bits of inputs at once.
        return 32 / t.bits();
    }

    // Most targets can do 2-way horizontal reductions well.
    return 2;
}

int GetRecommendedAccumulators(const Target &target) {
    if (target.has_feature(Target::AVX512_Skylake) ||
        (target.arch == Target::ARM && target.bits == 64)) {
        // 32 registers total.
        return 20;
    } else {
        // 16 reigsters total.
        return 12;
    }
}

class Convolution : public Generator<Convolution> {
public:
    // Unsigned 8-bit input tensor, indexed by input_depth, input_x, input_y,
    // input_batch.
    Input<Buffer<uint8_t>> input_{"input", 4};

    // A 4D array of 8-bit filter coefficients indexed by filter_depth, filter_x,
    // filter_y, filter_batch (aka. output_depth).
    Input<Buffer<uint8_t>> filter_{"filter", 4};

    // A 1D array of 32-bit biases. The bias should be added to the c
    // dimension of the output (i.e., # filter batches).
    Input<Buffer<int32_t>> bias_{"bias", 1};

    // Offsets for the input and filter.
    Input<uint8_t> input_offset_{"input_offset"};
    Input<uint8_t> filter_offset_{"filter_offset"};

    // The stride specifies how the input [x, y] is sub-subsampled. For every
    // spatial location [x, y] in the output buffer, the input buffer is sampled
    // spatially at [x * stride, y * stride]. The caller is responsible for
    // allocating the correct output memory.
    Input<int> stride_x_{"stride_x", 1, 1, 4};
    Input<int> stride_y_{"stride_y", 1, 1, 4};
    Input<int> dilation_x_{"dilation_x", 1, 1, 4};
    Input<int> dilation_y_{"dilation_y", 1, 1, 4};

    // Parameters for pointwise operations on the output.
    Input<int> output_multiplier_{"output_multiplier"};
    Input<int> output_shift_{"output_shift"};
    Input<uint8_t> output_offset_{"output_offset"};
    Input<uint8_t> output_min_{"output_min"};
    Input<uint8_t> output_max_{"output_max"};

    Output<Buffer<uint8_t>> output_{"output", 4};

    void generate() {
        // The algorithm.

        // Some free variables, where x and y represent the spatial dimensions.
        Var x("x"), y("y"), c("c"), b("b");

        // Add a "zero" boundary condition to x and y dimensions of the input.
        Func input_bounded = ConstantExteriorTensor(input_, input_offset_);
        // And to c of the filter. This lets us align the inner reduction loop
        // however we want.
        Func filter_bounded =
            constant_exterior(filter_, filter_offset_,
                              {{filter_.dim(0).min(), filter_.dim(0).extent()}});

        // Align the reduction loop of filter.
        int vector_reduction = GetVectorReduction(get_target(), UInt(8));

        // Create a wrapper of the filter that we can reorder the storage of to be
        // more convenient for the inner loop.
        Var ci("ci"), co("co");
        Func filter_tiled("filter_tiled");
        filter_tiled(ci, co, x, y, c) =
            filter_bounded(co * vector_reduction + ci, x, y, c);

        // Set up the reduction loop and inputs.
        Expr reduce_c_extent =
            ((filter_.dim(0).extent() + vector_reduction - 1) / vector_reduction) *
            vector_reduction;
        filter_.dim(1).set_min(0);
        filter_.dim(2).set_min(0);
        RDom r(0, reduce_c_extent, 0, filter_.dim(1).extent(), 0,
               filter_.dim(2).extent());
        RVar rc = r[0];
        RVar rx = r[1];
        RVar ry = r[2];
        Expr filter_rdxyc =
            filter_tiled(rc % vector_reduction, rc / vector_reduction, rx, ry, c);
        Expr input_rdxyc = input_bounded(rc, x * stride_x_ + rx * dilation_x_,
                                         y * stride_y_ + ry * dilation_y_, b);

        // We want to compute the reduction:
        // convolved(c, x, y, b) = bias_(c)
        // convolved(c, x, y, b) +=
        //    (i32(input_rdxyc) - i32(input_offset_)) *
        //    (i32(filter_rdxyc) - i32(filter_offset_))
        //
        // However, this precludes using efficient dot product instructions. To
        // fix this, expand the expression:
        //
        // convolved(c, x, y, b) = bias_(c)
        // convolved(c, x, y, b) +=
        //    i32(filter_rdxyc) * i32(input_rdxyc) -
        //    i32(filter_rdxyc) * i32(input_offset_) -
        //    i32(filter_offset_) * i32(input_rdxyc) +
        //    i32(filter_offset_) * i32(input_offset_)
        //
        // We can then separate this into several reductions. First, the terms that
        // depend only on c.
        Func offset_c("offset_c");
        offset_c(c) = bias_(c);
        offset_c(c) += i32(filter_rdxyc) * i32(input_offset_) -
                       i32(filter_offset_) * i32(input_offset_);

        // Next, the terms that depend only on x, y, b.
        Func offset_xyb("offset_xyb");
        offset_xyb(x, y, b) += i32(filter_offset_) * i32(input_rdxyc);

        // Finally, the terms that depend on all of c, x, y, b.
        Func convolved("convolved");
        convolved(c, x, y, b) = offset_c(c) - offset_xyb(x, y, b);
        convolved(c, x, y, b) += i32(filter_rdxyc) * i32(input_rdxyc);

        // Saturate and narrow the output.
        Expr output =
            MultiplyByQuantizedMultiplierSmallerThanOne(i32(convolved(c, x, y, b)),
                                                        output_multiplier_, output_shift_) +
            output_offset_;
        output_(c, x, y, b) = clamp(u8_sat(output), output_min_, output_max_);

        // Schedule
        InterpretAsTensor(input_);
        InterpretAsTensor(filter_);
        InterpretAsTensor(bias_);
        InterpretAsTensor(output_);

        output_.compute_root();

        // Figure out how big the tile should be by getting the total number of
        // accumulators best for this target and figuring out a tile size.
        int tile_c_max = 4;
        int tile_x = GetRecommendedAccumulators(get_target()) / tile_c_max;
        if (tile_c_max > tile_x) {
            // Prefer bigger x tiles to c tiles.
            std::swap(tile_c_max, tile_x);
        }

        // We need to tile the output, but we can't use GuardWithIf because we need
        // things computed at the tile to have constant size. We can't assume the
        // output is bigger than a minimum size. So, we specialize for decreasing
        // tile sizes, and have a degenerate tile case to handle the rest.
        const int vector_size = natural_vector_size<uint8_t>() / vector_reduction;
        Var xo("xo");
        Expr output_channels = output_.dim(0).extent();
        Expr output_width = output_.dim(1).extent();
        for (int tile_c = tile_c_max; tile_c >= 1; tile_c /= 2) {
            output_
                .specialize(output_channels >= tile_c * vector_size &&
                            output_width >= tile_x)
                .tile(c, x, co, xo, c, x, tile_c * vector_size, tile_x,
                      TailStrategy::ShiftInwards)
                .reorder(c, x, co, xo, y, b)
                .vectorize(c)
                .unroll(x);
        }

        // In case there are no suitable tile sizes, just make a dummy split so the
        // rest of the schedule still works.
        output_
            .tile(c, x, co, xo, c, x, 1, 1, TailStrategy::RoundUp)
            .reorder(c, x, co, xo, y, b);

        // These GuardWithIf splits simplify for the constant-tile specializations,
        // but probably generate poor code for the general case.
        convolved.compute_at(output_, co)
            .store_in(MemoryType::Stack)
            .reorder(x, c, y, b)
            .vectorize(c)
            .unroll(x);

        RVar rco, rci;
        convolved.update()
            .split(rc, rco, rci, vector_reduction)
            .reorder(rci, x, c, rco, rx, ry, y, b)
            .vectorize(c)
            .atomic()
            .vectorize(rci)
            .unroll(x);

        // Precompute the channel offset at root.
        // TODO: This gets recomputed often when the op is split up into small
        // pieces.
        offset_c.compute_root();
        offset_c.update().specialize(input_offset_ == 0);

        // Compute the batch offsets outside the loops over channels.
        offset_xyb.compute_at(output_, xo);
        offset_xyb.update().specialize(filter_offset_ == 0);

        // Pretranspose the filter, so we don't need to do it in the inner loop.
        // TODO: This gets recomputed often when the op is split up into small
        // pieces.
        filter_tiled.compute_root()
            .reorder_storage(ci, c, co, x, y)
            .reorder(ci, c, x, y, co)
            .bound(ci, 0, vector_reduction)
            .align_storage(ci, vector_reduction)
            .align_storage(c, vector_size * tile_c_max)
            .unroll(ci);
    }
};

}  // namespace interpret_nn

HALIDE_REGISTER_GENERATOR(interpret_nn::Convolution, Convolution)
