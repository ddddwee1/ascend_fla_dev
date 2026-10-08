"""PGDN: one vector owner per (B,H), one ATK chain and grouped main states.

ATK lookahead is valid because neither v nor the main state feeds A. Each
step retains distinct normalized k_read and preconditioned k_write. Raw q/k
are widened then normalized; q scaling occurs only after normalization.
"""
from ascriptor.a5 import *
D = 128
S_MAX = 16
SCALE = 128**-.5
LOG_X = 0.4054651081081644


@vf()
def zero_state(state: Tensor):
    zero = RegList(DT.float, 2)
    zero <<= 0.0
    for row in range(D):
        state[row:row+1, 0:D] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def zero_atk(state: Tensor):
    value = RegList(DT.float, 2)
    value <<= 0.0
    state[0:1, 0:D] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def atk_tile(q: Tensor, k: Tensor, write: Tensor, gate: Tensor,
             beta: Tensor, state: Tensor, scalar: Tensor, steps: Var):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    ar = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    r = RegList(DT.float, 2)
    denom = RegList(DT.float, 2)
    norm = Reg(DT.float)
    decay = Reg(DT.float)
    update = Reg(DT.float)
    gate_value = Reg(DT.float)
    subnormal_factor = Reg(DT.float)
    mantissa = Reg(DT.int32)
    small_decay = MaskReg(DT.float)
    round_even = CastConfig(round_mode=RoundMode.TO_EVEN)
    ar <<= state[0:1, 0:D]
    for i in range(steps):
        qr <<= q[i:i+1, 0:D]
        tmp <<= qr * qr
        norm <<= tmp.cadd()
        norm <<= norm.sqrt()
        norm <<= norm.vmaxs(1e-12)
        scalar[0:1, 0:1] <<= norm.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        qr <<= qr / norm
        qr <<= qr * SCALE
        q[i:i+1, 0:D] <<= qr

        kr <<= k[i:i+1, 0:D]
        tmp <<= kr * kr
        norm <<= tmp.cadd()
        norm <<= norm.sqrt()
        norm <<= norm.vmaxs(1e-12)
        scalar[0:1, 0:1] <<= norm.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        kr <<= kr / norm
        k[i:i+1, 0:D] <<= kr

        gate_value <<= gate[i:i+1, 0:1].single()
        decay <<= gate_value.exp()
        compare(small_decay, gate_value, -87.0, CompareMode.LT)
        # Preserve the FP32 exponential's subnormal mantissa before multiplying
        # a potentially large A. Materializing exp(g) directly flushes it on A5.
        # Clamp only the unused alternate path to keep its int32 cast in range.
        subnormal_factor <<= gate_value.vmins(-87.0)
        subnormal_factor <<= subnormal_factor + 103.2789306640625
        subnormal_factor <<= subnormal_factor + (-7.606306642760963e-7)
        subnormal_factor <<= subnormal_factor.exp()
        mantissa <<= subnormal_factor.astype(DT.int32, round_even)
        subnormal_factor <<= mantissa.astype(DT.float, round_even)
        subnormal_factor <<= subnormal_factor * 2.5849394142282115e-26
        tmp <<= ar * 5.421010862427522e-20
        tmp <<= tmp * subnormal_factor
        ar <<= ar * decay
        for half in unroll(2):
            ar[half] <<= small_decay.select(tmp[half], ar[half])
        update <<= beta[i:i+1, 0:1].single()
        tmp <<= kr * kr
        tmp <<= tmp * update
        ar <<= ar + tmp
        r <<= ar + 1e-6
        r <<= r.ln()
        r <<= r + 0.2
        denom <<= r.abs()
        denom <<= denom + 1.0
        r <<= r / denom
        r <<= r * (-LOG_X)
        r <<= r.exp()
        tmp <<= kr * r
        write[i:i+1, 0:D] <<= tmp
    state[0:1, 0:D] <<= ar
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def widen(source: Tensor, dest: Tensor, steps: Var):
    value = Reg(DT.float)
    for t in range(steps):
        for half in unroll(2):
            value <<= source[t:t+1, half*64:half*64+64].unpack()
            dest[t:t+1, half*64:half*64+64] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def advance(state: Tensor, q: Tensor, k: Tensor, write: Tensor, v: Tensor,
            g: Tensor, beta: Tensor, out: Tensor, steps: Var):
    row = RegList(DT.float, 2)
    prediction = RegList(DT.float, 2)
    result = RegList(DT.float, 2)
    delta = RegList(DT.float, 2)
    product = RegList(DT.float, 2)
    decay = Reg(DT.float)
    key = Reg(DT.float)
    query = Reg(DT.float)
    weight = Reg(DT.float)
    for t in range(steps):
        prediction <<= 0.0
        result <<= 0.0
        decay <<= g[t:t+1, 0:1].single()
        decay <<= decay.exp()
        for ki in range(D):
            key <<= k[t:t+1, ki:ki+1].single()
            row <<= state[ki:ki+1, 0:D]
            row <<= row * decay
            state[ki:ki+1, 0:D] <<= row
            product <<= row * key
            prediction <<= prediction + product
        # The second sweep reloads the decayed rows stored by the first.
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        weight <<= beta[t:t+1, 0:1].single()
        delta <<= v[t:t+1, 0:D]
        delta <<= delta - prediction
        delta <<= delta * weight
        for ki in range(D):
            key <<= write[t:t+1, ki:ki+1].single()
            query <<= q[t:t+1, ki:ki+1].single()
            row <<= state[ki:ki+1, 0:D]
            product <<= delta * key
            row <<= row + product
            state[ki:ki+1, 0:D] <<= row
            product <<= row * query
            result <<= result + product
        out[t:t+1, 0:D] <<= result
        # Retire state writes before the next token's first sweep.
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def narrow_output(source: Tensor, out: Tensor, steps: Var):
    value = Reg(DT.float)
    packed = Reg(DT.bfloat16)
    full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
    config = CastConfig(round_mode=RoundMode.TO_EVEN)
    for t in range(steps):
        for half in unroll(2):
            value <<= source[t:t+1, half*64:half*64+64]
            packed <<= value.astype(DT.bfloat16, config)
            # The even lanes compact to64 BF16 elements, not128.
            reg_to_ub_downsample(out[t:t+1, half*64:half*64+64], packed, mask=full)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode="vec")
def pgdn_recurrent_fp32(
    q: GM[f32, ("B", "S", "H", 128)],
    k: GM[f32, ("B", "S", "H", 128)],
    v: GM[f32, ("B", "S", "HV", 128)],
    g_atk: GM[f32, ("B", "S", "H")],
    g: GM[f32, ("B", "S", "HV")],
    beta_atk: GM[f32, ("B", "S", "H")],
    beta: GM[f32, ("B", "S", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    initial_A_state: GM[f32, ("B", "H", 128)],
    o: GM[f32, ("B", "S", "HV", 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    final_A_state: GM[f32, ("B", "H", 128)],
    B: i32, S: i32, H: i32, HV: i32, HAS_INITIAL: i32, HAS_A: i32,
):
    state = Tensor(DT.float, [D, D], Position.UB)
    qu = Tensor(DT.float, [S_MAX, D], Position.UB)
    ku = Tensor(DT.float, [S_MAX, D], Position.UB)
    wu = Tensor(DT.float, [S_MAX, D], Position.UB)
    vu = Tensor(DT.float, [S_MAX, D], Position.UB)
    ou = Tensor(DT.float, [S_MAX, D], Position.UB)
    au = Tensor(DT.float, [1, D], Position.UB)
    gu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    bu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    gau = Tensor(DT.float, [S_MAX, 8], Position.UB)
    bau = Tensor(DT.float, [S_MAX, 8], Position.UB)
    scalar = Tensor(DT.float, [1, 8], Position.UB)
    per_vec = CeilDiv(B * H, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, B * H)
    with auto_sync():
        for work in range(begin, end):
            bi = Var(work // H)
            hi = Var(work % H)
            if HAS_A != 0:
                au[0:1, 0:D] <<= initial_A_state[bi, hi:hi+1, 0:D]
            else:
                zero_atk(au)
            gm_to_ub_pad(qu[0:S, 0:D], q[bi, 0:S, hi, 0:D],
                         S, D, (H - 1) * D, 0)
            gm_to_ub_pad(ku[0:S, 0:D], k[bi, 0:S, hi, 0:D],
                         S, D, (H - 1) * D, 0)
            gau[0:S, 0:1] <<= g_atk[bi, 0:S, hi:hi+1]
            bau[0:S, 0:1] <<= beta_atk[bi, 0:S, hi:hi+1]
            atk_tile(qu, ku, wu, gau, bau, au, scalar, S)
            final_A_state[bi, hi:hi+1, 0:D] <<= au[0:1, 0:D]
            for group in range(HV // H):
                vi = Var(hi * (HV // H) + group)
                if HAS_INITIAL != 0:
                    state[0:D, 0:D] <<= initial_state[bi, vi, 0:D, 0:D]
                else:
                    zero_state(state)
                gm_to_ub_pad(vu[0:S, 0:D], v[bi, 0:S, vi, 0:D],
                             S, D, (HV - 1) * D, 0)
                gu[0:S, 0:1] <<= g[bi, 0:S, vi:vi+1]
                bu[0:S, 0:1] <<= beta[bi, 0:S, vi:vi+1]
                advance(state, qu, ku, wu, vu, gu, bu, ou, S)
                ub_to_gm_pad(o[bi, 0:S, vi, 0:D], ou[0:S, 0:D],
                             S, D, 0, (HV - 1) * D)
                final_state[bi, vi, 0:D, 0:D] <<= state[0:D, 0:D]
    return o, final_state, final_A_state


@kernel(mode="vec")
def pgdn_recurrent_bf16(
    q: GM[bf16, ("B", "S", "H", 128)],
    k: GM[bf16, ("B", "S", "H", 128)],
    v: GM[bf16, ("B", "S", "HV", 128)],
    g_atk: GM[f32, ("B", "S", "H")],
    g: GM[f32, ("B", "S", "HV")],
    beta_atk: GM[f32, ("B", "S", "H")],
    beta: GM[f32, ("B", "S", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    initial_A_state: GM[f32, ("B", "H", 128)],
    o: GM[bf16, ("B", "S", "HV", 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    final_A_state: GM[f32, ("B", "H", 128)],
    B: i32, S: i32, H: i32, HV: i32, HAS_INITIAL: i32, HAS_A: i32,
):
    state = Tensor(DT.float, [D, D], Position.UB)
    qu = Tensor(DT.float, [S_MAX, D], Position.UB)
    ku = Tensor(DT.float, [S_MAX, D], Position.UB)
    wu = Tensor(DT.float, [S_MAX, D], Position.UB)
    vu = Tensor(DT.float, [S_MAX, D], Position.UB)
    ou = Tensor(DT.float, [S_MAX, D], Position.UB)
    au = Tensor(DT.float, [1, D], Position.UB)
    gu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    bu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    gau = Tensor(DT.float, [S_MAX, 8], Position.UB)
    bau = Tensor(DT.float, [S_MAX, 8], Position.UB)
    scalar = Tensor(DT.float, [1, 8], Position.UB)
    qb = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    kb = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    vb = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    ob = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    per_vec = CeilDiv(B * H, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, B * H)
    with auto_sync():
        for work in range(begin, end):
            bi = Var(work // H)
            hi = Var(work % H)
            if HAS_A != 0:
                au[0:1, 0:D] <<= initial_A_state[bi, hi:hi+1, 0:D]
            else:
                zero_atk(au)
            gm_to_ub_pad(qb[0:S, 0:D], q[bi, 0:S, hi, 0:D],
                         S, D, (H - 1) * D, 0)
            gm_to_ub_pad(kb[0:S, 0:D], k[bi, 0:S, hi, 0:D],
                         S, D, (H - 1) * D, 0)
            widen(qb, qu, S)
            widen(kb, ku, S)
            gau[0:S, 0:1] <<= g_atk[bi, 0:S, hi:hi+1]
            bau[0:S, 0:1] <<= beta_atk[bi, 0:S, hi:hi+1]
            atk_tile(qu, ku, wu, gau, bau, au, scalar, S)
            final_A_state[bi, hi:hi+1, 0:D] <<= au[0:1, 0:D]
            for group in range(HV // H):
                vi = Var(hi * (HV // H) + group)
                if HAS_INITIAL != 0:
                    state[0:D, 0:D] <<= initial_state[bi, vi, 0:D, 0:D]
                else:
                    zero_state(state)
                gm_to_ub_pad(vb[0:S, 0:D], v[bi, 0:S, vi, 0:D],
                             S, D, (HV - 1) * D, 0)
                widen(vb, vu, S)
                gu[0:S, 0:1] <<= g[bi, 0:S, vi:vi+1]
                bu[0:S, 0:1] <<= beta[bi, 0:S, vi:vi+1]
                advance(state, qu, ku, wu, vu, gu, bu, ou, S)
                narrow_output(ou, ob, S)
                ub_to_gm_pad(o[bi, 0:S, vi, 0:D], ob[0:S, 0:D],
                             S, D, 0, (HV - 1) * D)
                final_state[bi, vi, 0:D, 0:D] <<= state[0:D, 0:D]
    return o, final_state, final_A_state
