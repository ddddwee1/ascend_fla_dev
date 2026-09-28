"""Raw-key, scalar-gated GDN; one vector participant owns a whole state.

The two state sweeps implement decay/read then update/output. They retain
beta*(v-read) before the outer product. Scalar gates differ from KDA decode.
No state or arithmetic moves to the host; no cross-core state sharing occurs.
"""
from ascriptor.a5 import *

D = 128
S_MAX = 16
SCALE = 128**-0.5


@vf()
def zero_state(state: Tensor):
    zero = RegList(DT.float, 2)
    zero <<= 0.0
    for row in range(D):
        state[row:row+1, 0:D] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def scale_query(query: Tensor, steps: Var):
    row = RegList(DT.float, 2)
    for t in range(steps):
        row <<= query[t:t+1, 0:D]
        row <<= row * SCALE
        query[t:t+1, 0:D] <<= row
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def widen_inputs(qb: Tensor, kb: Tensor, vb: Tensor,
                 q: Tensor, k: Tensor, v: Tensor, steps: Var):
    value = Reg(DT.float)
    for t in range(steps):
        for half in unroll(2):
            # unpack reads exactly64 BF16 elements and widens them to FP32.
            value <<= qb[t:t+1, half*64:half*64+64].unpack()
            value <<= value * SCALE
            q[t:t+1, half*64:half*64+64] <<= value
            value <<= kb[t:t+1, half*64:half*64+64].unpack()
            k[t:t+1, half*64:half*64+64] <<= value
            value <<= vb[t:t+1, half*64:half*64+64].unpack()
            v[t:t+1, half*64:half*64+64] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def advance(state: Tensor, q: Tensor, k: Tensor, v: Tensor,
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
            key <<= k[t:t+1, ki:ki+1].single()
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
def gdn_recurrent_fp32(
    q: GM[f32, ("B", "S", "H", 128)],
    k: GM[f32, ("B", "S", "H", 128)],
    v: GM[f32, ("B", "S", "HV", 128)],
    g: GM[f32, ("B", "S", "HV")],
    beta: GM[f32, ("B", "S", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    o: GM[f32, ("B", "S", "HV", 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32, S: i32, H: i32, HV: i32, HAS_INITIAL: i32,
):
    state = Tensor(DT.float, [D, D], Position.UB)
    qu = Tensor(DT.float, [S_MAX, D], Position.UB)
    ku = Tensor(DT.float, [S_MAX, D], Position.UB)
    vu = Tensor(DT.float, [S_MAX, D], Position.UB)
    gu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    bu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    ou = Tensor(DT.float, [S_MAX, D], Position.UB)
    per_vec = CeilDiv(B * HV, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, B * HV)
    with auto_sync():
        for work in range(begin, end):
            bi = Var(work // HV)
            hi = Var(work % HV)
            qi = Var(hi // (HV // H))
            if HAS_INITIAL != 0:
                state[0:D, 0:D] <<= initial_state[bi, hi, 0:D, 0:D]
            else:
                zero_state(state)
            gm_to_ub_pad(qu[0:S, 0:D], q[bi, 0:S, qi, 0:D],
                         S, D, (H - 1) * D, 0)
            gm_to_ub_pad(ku[0:S, 0:D], k[bi, 0:S, qi, 0:D],
                         S, D, (H - 1) * D, 0)
            gm_to_ub_pad(vu[0:S, 0:D], v[bi, 0:S, hi, 0:D],
                         S, D, (HV - 1) * D, 0)
            gu[0:S, 0:1] <<= g[bi, 0:S, hi:hi+1]
            bu[0:S, 0:1] <<= beta[bi, 0:S, hi:hi+1]
            scale_query(qu, S)
            advance(state, qu, ku, vu, gu, bu, ou, S)
            ub_to_gm_pad(o[bi, 0:S, hi, 0:D], ou[0:S, 0:D],
                         S, D, 0, (HV - 1) * D)
            final_state[bi, hi, 0:D, 0:D] <<= state[0:D, 0:D]
    return o, final_state


@kernel(mode="vec")
def gdn_recurrent_bf16(
    q: GM[bf16, ("B", "S", "H", 128)],
    k: GM[bf16, ("B", "S", "H", 128)],
    v: GM[bf16, ("B", "S", "HV", 128)],
    g: GM[f32, ("B", "S", "HV")],
    beta: GM[f32, ("B", "S", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    o: GM[bf16, ("B", "S", "HV", 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32, S: i32, H: i32, HV: i32, HAS_INITIAL: i32,
):
    state = Tensor(DT.float, [D, D], Position.UB)
    qb = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    kb = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    vb = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    qu = Tensor(DT.float, [S_MAX, D], Position.UB)
    ku = Tensor(DT.float, [S_MAX, D], Position.UB)
    vu = Tensor(DT.float, [S_MAX, D], Position.UB)
    gu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    bu = Tensor(DT.float, [S_MAX, 8], Position.UB)
    ou = Tensor(DT.float, [S_MAX, D], Position.UB)
    ob = Tensor(DT.bfloat16, [S_MAX, D], Position.UB)
    per_vec = CeilDiv(B * HV, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, B * HV)
    with auto_sync():
        for work in range(begin, end):
            bi = Var(work // HV)
            hi = Var(work % HV)
            qi = Var(hi // (HV // H))
            if HAS_INITIAL != 0:
                state[0:D, 0:D] <<= initial_state[bi, hi, 0:D, 0:D]
            else:
                zero_state(state)
            gm_to_ub_pad(qb[0:S, 0:D], q[bi, 0:S, qi, 0:D],
                         S, D, (H - 1) * D, 0)
            gm_to_ub_pad(kb[0:S, 0:D], k[bi, 0:S, qi, 0:D],
                         S, D, (H - 1) * D, 0)
            gm_to_ub_pad(vb[0:S, 0:D], v[bi, 0:S, hi, 0:D],
                         S, D, (HV - 1) * D, 0)
            gu[0:S, 0:1] <<= g[bi, 0:S, hi:hi+1]
            bu[0:S, 0:1] <<= beta[bi, 0:S, hi:hi+1]
            widen_inputs(qb, kb, vb, qu, ku, vu, S)
            advance(state, qu, ku, vu, gu, bu, ou, S)
            narrow_output(ou, ob, S)
            ub_to_gm_pad(o[bi, 0:S, hi, 0:D], ob[0:S, 0:D],
                         S, D, 0, (HV - 1) * D)
            final_state[bi, hi, 0:D, 0:D] <<= state[0:D, 0:D]
    return o, final_state
