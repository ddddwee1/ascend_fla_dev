"""Normalization and complete per-key-head ATK history for the adjoint."""
from ascriptor.a5 import *

C = 64
D = 128
LOG_X = 0.4054651081081644


@vf()
def zero_row(row: Tensor):
    z = RegList(DT.float, 2)
    z <<= 0.0
    row[0:1, 0:D] <<= z


@vf()
def atk_tile(q: Tensor, k: Tensor, write: Tensor, history: Tensor,
             gate: Tensor, beta: Tensor, qnorm: Tensor, knorm: Tensor,
             state: Tensor, scalar: Tensor):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    ar = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    r = RegList(DT.float, 2)
    denom = RegList(DT.float, 2)
    norm = Reg(DT.float)
    norm_acc = Reg(DT.float)
    norm_piece = Reg(DT.float)
    decay = Reg(DT.float)
    update = Reg(DT.float)
    ar <<= state[0:1, 0:D]
    for i in range(C):
        qr <<= q[i:i+1, 0:D]
        # Literal FP32 AVX2 norm: eight lanes, then sequential lane sum.
        # BRCB reads exactly eight FP32 values and repeats each over a block.
        norm_acc <<= 0.0
        for j in range(D // 8):
            norm_piece <<= q[i:i+1, j*8:j*8+8].brcb()
            norm_piece <<= norm_piece * norm_piece
            norm_acc <<= norm_acc + norm_piece
        scalar[0:1, 0:64] <<= norm_acc
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        for j in range(1, 8):
            norm_piece <<= scalar[0:1, j*8:j*8+1].single()
            norm <<= norm + norm_piece
        norm <<= norm.sqrt()
        qnorm[i:i+1, 0:1] <<= norm.single_value()
        norm <<= norm.vmaxs(1e-12)
        scalar[0:1, 0:1] <<= norm.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        qr <<= qr / norm
        q[i:i+1, 0:D] <<= qr
        kr <<= k[i:i+1, 0:D]
        norm_acc <<= 0.0
        for j in range(D // 8):
            norm_piece <<= k[i:i+1, j*8:j*8+8].brcb()
            norm_piece <<= norm_piece * norm_piece
            norm_acc <<= norm_acc + norm_piece
        scalar[0:1, 0:64] <<= norm_acc
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        for j in range(1, 8):
            norm_piece <<= scalar[0:1, j*8:j*8+1].single()
            norm <<= norm + norm_piece
        norm <<= norm.sqrt()
        knorm[i:i+1, 0:1] <<= norm.single_value()
        norm <<= norm.vmaxs(1e-12)
        scalar[0:1, 0:1] <<= norm.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        norm <<= scalar[0:1, 0:1].single()
        kr <<= kr / norm
        k[i:i+1, 0:D] <<= kr
        decay <<= gate[i:i+1, 0:1].single()
        decay <<= decay.exp()
        update <<= beta[i:i+1, 0:1].single()
        ar <<= ar * decay
        tmp <<= kr * kr
        tmp <<= tmp * update
        ar <<= ar + tmp
        history[i:i+1, 0:D] <<= ar
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


@kernel()
def pgdn_bwd_atk(
    q: GM[f32, ('B','T','H',128)], k: GM[f32, ('B','T','H',128)],
    g_atk: GM[f32, ('B','T','H')], beta_atk: GM[f32, ('B','T','H')],
    q_norm: GM[f32, ('B','T','H',128)], k_read: GM[f32, ('B','T','H',128)],
    k_write: GM[f32, ('B','T','H',128)], A_history: GM[f32, ('B','T','H',128)],
    q_raw_norm: GM[f32, ('B','T','H')], k_raw_norm: GM[f32, ('B','T','H')],
    final_A_state: GM[f32, ('B','H',128)], B: i32, T: i32, H: i32, N: i32,
):
    qu = Tensor(DT.float, [C,D], Position.UB)
    ku = Tensor(DT.float, [C,D], Position.UB)
    wu = Tensor(DT.float, [C,D], Position.UB)
    history = Tensor(DT.float, [C,D], Position.UB)
    gu = Tensor(DT.float, [C,8], Position.UB)
    bu = Tensor(DT.float, [C,8], Position.UB)
    qnu = Tensor(DT.float, [C,8], Position.UB)
    knu = Tensor(DT.float, [C,8], Position.UB)
    au = Tensor(DT.float, [1,D], Position.UB)
    scalar = Tensor(DT.float, [1,64], Position.UB)
    per = CeilDiv(B*H, GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per, B*H)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//H)
            hh = Var(item%H)
            zero_row(au)
            for cc in range(N):
                tt = Var(cc*C)
                gm_to_ub_pad(qu,q[bb,tt:tt+C,hh,:],C,D,(H-1)*D,0)
                gm_to_ub_pad(ku,k[bb,tt:tt+C,hh,:],C,D,(H-1)*D,0)
                gu[:,0:1] <<= g_atk[bb,tt:tt+C,hh:hh+1]
                bu[:,0:1] <<= beta_atk[bb,tt:tt+C,hh:hh+1]
                atk_tile(qu,ku,wu,history,gu,bu,qnu,knu,au,scalar)
                ub_to_gm_pad(q_norm[bb,tt:tt+C,hh,:],qu,C,D,0,(H-1)*D)
                ub_to_gm_pad(k_read[bb,tt:tt+C,hh,:],ku,C,D,0,(H-1)*D)
                ub_to_gm_pad(k_write[bb,tt:tt+C,hh,:],wu,C,D,0,(H-1)*D)
                ub_to_gm_pad(A_history[bb,tt:tt+C,hh,:],history,C,D,0,(H-1)*D)
                q_raw_norm[bb,tt:tt+C,hh:hh+1] <<= qnu[:,0:1]
                k_raw_norm[bb,tt:tt+C,hh:hh+1] <<= knu[:,0:1]
            final_A_state[bb,hh:hh+1,:] <<= au[:,:]
    return q_norm,k_read,k_write,A_history,q_raw_norm,k_raw_norm,final_A_state
