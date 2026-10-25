"""A2 layout copies use padded DMA and byte-addressed UB gathers.

Each work owns 64 output elements. The full-mask BF16 gather writes 128
elements into owned storage; its extra offsets repeat the first 64 elements.
Only the 64 logical elements reach GM. No host tensor transformation is used.
"""
from ascriptor.a2 import *


def make_move(name, input_dtype, output_dtype):
    @kernel(mode="vec")
    def move_kernel(
        source: GM[input_dtype, (1, "Storage")],
        destination: GM[output_dtype, (1, "N")],
        Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
        S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
        multiply: i32, factor: f32,
    ):
        padded = Tensor(input_dtype, [64, 16 if input_dtype == bf16 else 8], Position.UB)
        incoming = Tensor(input_dtype, [1, 128], Position.UB)
        outgoing = Tensor(output_dtype, [1, 128], Position.UB)
        scaled = Tensor(f32, [1, 128], Position.UB)
        offsets = Tensor(u32, [1, 128], Position.UB)
        works = Var(N / 64)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for lane in range(128):
                address = Var(0, u32)
                if S4 != 0:
                    address.set((lane % 64) * 32)
                address.SetValueTo(offsets[0, lane:lane+1])
            # The A2 autosync planner cannot publish scalar stores with an
            # event. Complete offset initialization before vector consumption.
            barrier(Pipe.ALL)
            for work in range(begin, end):
                offset = Var(work * 64)
                i4 = Var(offset % D4)
                rest3 = Var(offset / D4)
                i3 = Var(rest3 % D3)
                rest2 = Var(rest3 / D3)
                i2 = Var(rest2 % D2)
                rest1 = Var(rest2 / D2)
                i1 = Var(rest1 % D1)
                i0 = Var(rest1 / D1)
                source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
                if S4 == 1:
                    gm_to_ub_pad(incoming, source[0, source_offset:source_offset+64],
                                 n_burst=1, burst_len_element=64,
                                 src_stride_element=0, dst_stride=0)
                else:
                    if S4 == 0:
                        gm_to_ub_pad(padded, source[0, source_offset:source_offset+1],
                                     n_burst=1, burst_len_element=1,
                                     src_stride_element=0, dst_stride=0)
                    else:
                        gm_to_ub_pad(padded, source[0, source_offset:source_offset+63*S4+1],
                                     n_burst=64, burst_len_element=1,
                                     src_stride_element=S4-1, dst_stride=0)
                    # ResetMask changes bits, not the count/normal mode. An
                    # uncounted gather must establish normal mode explicitly.
                    set_mask_normal()
                    gather(incoming, padded, offsets, repeat=1, dst_rep_stride=8)
                if input_dtype == f32:
                    if multiply != 0:
                        muls(scaled, incoming, factor, count=64)
                    else:
                        ub_to_ub(scaled, incoming, n_burst=1, burst_len=8,
                                 src_stride=0, dst_stride=0)
                    if output_dtype == bf16:
                        cast(outgoing, scaled, count=64, round_mode=RoundMode.TO_EVEN)
                    else:
                        ub_to_ub(outgoing, scaled, n_burst=1, burst_len=8,
                                 src_stride=0, dst_stride=0)
                else:
                    if output_dtype == f32:
                        cast(outgoing, incoming, count=64, round_mode=RoundMode.NONE)
                    else:
                        ub_to_ub(outgoing, incoming, n_burst=1, burst_len=4,
                                 src_stride=0, dst_stride=0)
                ub_to_gm_pad(destination[0, offset:offset+64], outgoing,
                             n_burst=1, burst_len_element=64,
                             src_stride=0, dst_stride_element=0)
        return destination

    move_kernel.name = name
    return move_kernel


def make_zero(name, dtype):
    @kernel(mode="vec")
    def zero_kernel(destination: GM[dtype, (1, "N")], N: i32):
        staging = Tensor(dtype, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            # Two BF16 positive-zero encodings fit in each UINT32 carrier.
            dup(staging.reinterpret(u32), 0, count=2048 if dtype == bf16 else 4096)
            for work in range(begin, end):
                offset = Var(work * 4096)
                active = Min(4096, N-offset)
                ub_to_gm_pad(destination[0, offset:offset+active], staging,
                             n_burst=1, burst_len_element=active,
                             src_stride=0, dst_stride_element=0)
        return destination

    zero_kernel.name = name
    return zero_kernel


def entries():
    dtypes = {"bf16": bf16, "f32": f32}
    result = {f"{a}_{b}": make_move(f"kda_layout_a2_{a}_{b}_kernel", x, y)
              for a, x in dtypes.items() for b, y in dtypes.items()}
    result.update({f"zero_{a}": make_zero(f"kda_layout_a2_zero_{a}_kernel", x)
                   for a, x in dtypes.items()})
    return result
