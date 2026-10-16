"""A2 UB-vector raw preparation with explicit FP32 and RNE boundaries."""
from ascriptor.a2 import *


@func
def _broadcast64(destination, source, block):
    brcb(block, source, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
    brcb(destination, block[0:1, 0:8], repeat=1, dst_blk_stride=1, dst_rep_stride=8)


def make_norm(name, input_dtype, output_dtype):
    @kernel(mode="vec")
    def norm_kernel(source: GM[input_dtype, (1, "N")],
                    destination: GM[output_dtype, (1, "N")], N: i32):
        incoming = Tensor(input_dtype, [32, 128], Position.UB)
        outgoing = Tensor(output_dtype, [32, 128], Position.UB)
        x = Tensor(f32, [1, 128], Position.UB)
        square = Tensor(f32, [1, 128], Position.UB)
        partial = Tensor(f32, [1, 64], Position.UB)
        total = Tensor(f32, [1, 64], Position.UB)
        block = Tensor(f32, [8, 8], Position.UB)
        denom = Tensor(f32, [1, 64], Position.UB)
        result = Tensor(f32, [1, 128], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                offset = Var(work * 4096)
                count = Min(4096, N-offset)
                gm_to_ub_pad(incoming, source[0, offset:offset+count], n_burst=1,
                             burst_len_element=count, src_stride_element=0, dst_stride=0)
                for row in range(count / 128):
                    if input_dtype == bf16:
                        cast(x, incoming[row:row+1, :], count=128, round_mode=RoundMode.NONE)
                    else:
                        ub_to_ub(x, incoming[row:row+1, :], n_burst=1, burst_len=16, src_stride=0, dst_stride=0)
                    mul(square, x, x, count=128)
                    dup(partial, 0., count=64)
                    dup(total, 0., count=64)
                    cadd(partial, square, repeat=2, src_blk_stride=1, src_rep_stride=8, count_per_rep=64)
                    cadd(total, partial, repeat=1, count_per_rep=2)
                    _broadcast64(denom, total, block)
                    adds(denom, denom, 1e-6, count=64)
                    sqrt(denom, denom, count=64)
                    for half in unroll(2):
                        div(result[0, half*64:half*64+64], x[0, half*64:half*64+64], denom, count=64)
                    if output_dtype == bf16:
                        cast(outgoing[row:row+1, :], result, count=128, round_mode=RoundMode.TO_EVEN)
                    else:
                        ub_to_ub(outgoing[row:row+1, :], result, n_burst=1, burst_len=16, src_stride=0, dst_stride=0)
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination
    norm_kernel.name = name
    return norm_kernel


def make_gate(name, input_dtype, a_dtype, bias_dtype):
    @kernel(mode="vec")
    def gate_kernel(source: GM[input_dtype, (1, "N")],
                    alog: GM[a_dtype, (1, "HV")], bias: GM[bias_dtype, (1, "Channels")],
                    destination: GM[f32, (1, "N")], N: i32, HV: i32, Channels: i32, BT: i32):
        incoming = Tensor(input_dtype, [32, 128], Position.UB)
        outgoing = Tensor(f32, [32, 128], Position.UB)
        avec = Tensor(a_dtype, [1, 128], Position.UB)
        bvec = Tensor(bias_dtype, [1, 128], Position.UB)
        af = Tensor(f32, [1, 64], Position.UB)
        bf = Tensor(f32, [1, 128], Position.UB)
        block = Tensor(f32, [8, 8], Position.UB)
        decay = Tensor(f32, [1, 64], Position.UB)
        value = Tensor(f32, [1, 128], Position.UB)
        u = Tensor(f32, [1, 64], Position.UB)
        argument = Tensor(f32, [1, 64], Position.UB)
        exponential = Tensor(f32, [1, 64], Position.UB)
        plus_one = Tensor(f32, [1, 64], Position.UB)
        denominator = Tensor(f32, [1, 64], Position.UB)
        correction = Tensor(f32, [1, 64], Position.UB)
        logarithm = Tensor(f32, [1, 64], Position.UB)
        product = Tensor(f32, [1, 64], Position.UB)
        small_result = Tensor(f32, [1, 64], Position.UB)
        polynomial = Tensor(f32, [1, 64], Position.UB)
        series_argument = Tensor(f32, [1, 64], Position.UB)
        series_result = Tensor(f32, [1, 64], Position.UB)
        accurate_small = Tensor(f32, [1, 64], Position.UB)
        use_series = Tensor(u8, [1, 32], Position.UB)
        result = Tensor(f32, [1, 64], Position.UB)
        one = Tensor(f32, [1, 64], Position.UB)
        threshold = Tensor(f32, [1, 64], Position.UB)
        upper = Tensor(u8, [1, 32], Position.UB)
        tiny = Tensor(u8, [1, 32], Position.UB)
        mask_address = Tensor(u32, [1, 8], Position.UB)
        chunks = CeilDiv(BT, 32)
        works = Var(HV * chunks)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            dup(one, 1., count=64)
            dup(threshold, 20., count=64)
            for work in range(begin, end):
                head = Var(work / chunks)
                start_row = Var((work % chunks) * 32)
                rows = Min(32, BT-start_row)
                offset = Var(start_row*Channels+head*128)
                span = Var((rows-1)*Channels+128)
                dup(avec.reinterpret(u32), 0, count=64 if a_dtype == bf16 else 128)
                gm_to_ub_pad(avec, alog[0, head:head+1], n_burst=1, burst_len_element=1,
                             src_stride_element=0, dst_stride=0, pad=0.)
                gm_to_ub_pad(bvec, bias[0, head*128:head*128+128], n_burst=1, burst_len_element=128,
                             src_stride_element=0, dst_stride=0)
                gm_to_ub_pad(incoming, source[0, offset:offset+span], n_burst=rows,
                             burst_len_element=128, src_stride_element=Channels-128, dst_stride=0)
                if a_dtype == bf16:
                    cast(af, avec, count=64, round_mode=RoundMode.NONE)
                else:
                    ub_to_ub(af, avec, n_burst=1, burst_len=8, src_stride=0, dst_stride=0)
                exp(af, af, count=64)
                _broadcast64(decay, af, block)
                muls(decay, decay, -1., count=64)
                if bias_dtype == bf16:
                    cast(bf, bvec, count=128, round_mode=RoundMode.NONE)
                else:
                    ub_to_ub(bf, bvec, n_burst=1, burst_len=16, src_stride=0, dst_stride=0)
                for row in range(rows):
                    if input_dtype == bf16:
                        cast(value, incoming[row:row+1, :], count=128, round_mode=RoundMode.NONE)
                    else:
                        ub_to_ub(value, incoming[row:row+1, :], n_burst=1, burst_len=16, src_stride=0, dst_stride=0)
                    for half in unroll(2):
                        add(u, value[0, half*64:half*64+64], bf[0, half*64:half*64+64], count=64)
                        compare_scalar(upper, u, 20., CompareMode.GT, repeat=1)
                        select(argument, upper, threshold, u, SelectMode.TENSOR_TENSOR, tmp_addr_buf=mask_address, repeat=1)
                        exp(exponential, argument, count=64)
                        adds(plus_one, exponential, 1., count=64)
                        adds(product, plus_one, -1., count=64)
                        compare_scalar(tiny, plus_one, 1., CompareMode.EQ, repeat=1)
                        select(denominator, tiny, one, product, SelectMode.TENSOR_TENSOR, tmp_addr_buf=mask_address, repeat=1)
                        div(correction, exponential, denominator, count=64)
                        ln(logarithm, plus_one, count=64)
                        mul(product, logarithm, correction, count=64)
                        select(small_result, tiny, exponential, product, SelectMode.TENSOR_TENSOR, tmp_addr_buf=mask_address, repeat=1)
                        # log1p(x) avoids the small-x cancellation in 1+x.
                        # Ten terms bound relative truncation below 1e-10 for x <= 1/8.
                        vmins(series_argument, exponential, 0.125, count=64)
                        dup(polynomial, -0.1, count=64)
                        for coefficient in (1./9., -1./8., 1./7., -1./6., 1./5., -1./4., 1./3., -0.5, 1.):
                            mul(polynomial, polynomial, series_argument, count=64)
                            adds(polynomial, polynomial, coefficient, count=64)
                        mul(series_result, polynomial, series_argument, count=64)
                        compare_scalar(use_series, exponential, 0.125, CompareMode.LE, repeat=1)
                        select(accurate_small, use_series, series_result, small_result, SelectMode.TENSOR_TENSOR, tmp_addr_buf=mask_address, repeat=1)
                        select(result, upper, u, accurate_small, SelectMode.TENSOR_TENSOR, tmp_addr_buf=mask_address, repeat=1)
                        mul(outgoing[row:row+1, half*64:half*64+64], decay, result, count=64)
                ub_to_gm_pad(destination[0, offset:offset+span], outgoing, n_burst=rows,
                             burst_len_element=128, src_stride=0, dst_stride_element=Channels-128)
        return destination
    gate_kernel.name = name
    return gate_kernel


def make_beta(name, input_dtype):
    @kernel(mode="vec")
    def beta_kernel(source: GM[input_dtype, (1, "N")],
                    destination: GM[f32, (1, "N")], N: i32):
        incoming = Tensor(input_dtype, [1, 4096], Position.UB)
        value = Tensor(f32, [1, 4096], Position.UB)
        outgoing = Tensor(f32, [1, 4096], Position.UB)
        one = Tensor(f32, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin+per_vec, works)
        with auto_sync():
            dup(one, 1., count=4096)
            for work in range(begin, end):
                offset = Var(work*4096)
                count = Min(4096, N-offset)
                padded = Align64(count)
                dup(incoming.reinterpret(u32), 0, count=2048 if input_dtype == bf16 else 4096)
                gm_to_ub_pad(incoming, source[0, offset:offset+count], n_burst=1,
                             burst_len_element=count, src_stride_element=0, dst_stride=0)
                if input_dtype == bf16:
                    cast(value, incoming, count=padded, round_mode=RoundMode.NONE)
                else:
                    ub_to_ub(value, incoming, n_burst=1, burst_len=padded/8, src_stride=0, dst_stride=0)
                muls(value, value, -1., count=padded)
                exp(value, value, count=padded)
                adds(value, value, 1., count=padded)
                div(outgoing, one, value, count=padded)
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination
    beta_kernel.name = name
    return beta_kernel
