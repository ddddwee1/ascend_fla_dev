"""Deterministic grouped adjoints, one ATK cotangent seed per key head."""
from ascriptor.a5 import *
from .atk import zero_row

D = 128
LOG_X = 0.4054651081081644


@vf()
def add_part(accumulator: Tensor, part: Tensor):
    a = RegList(DT.float,2)
    b = RegList(DT.float,2)
    a <<= accumulator[0:1,0:D]
    b <<= part[0:1,0:D]
    a <<= a + b
    accumulator[0:1,0:D] <<= a
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@vf()
def atk_adjoint(key: Tensor, current: Tensor, previous: Tensor, adjoint: Tensor,
                read_grad: Tensor, write_grad: Tensor, gate: Tensor, beta: Tensor,
                dg: Tensor, dbeta: Tensor):
    u = RegList(DT.float,2)
    ar = RegList(DT.float,2)
    z = RegList(DT.float,2)
    r = RegList(DT.float,2)
    den = RegList(DT.float,2)
    mult = RegList(DT.float,2)
    dw = RegList(DT.float,2)
    du = RegList(DT.float,2)
    tmp = RegList(DT.float,2)
    decay = Reg(DT.float)
    beta2 = Reg(DT.float)
    reduced = Reg(DT.float)
    u <<= key[0:1,0:D]
    ar <<= current[0:1,0:D]
    ar <<= ar + 1e-6
    r <<= ar.ln()
    r <<= r + 0.2
    den <<= r.abs()
    den <<= den + 1.0
    mult <<= r / den
    mult <<= mult * (-LOG_X)
    mult <<= mult.exp()
    z <<= adjoint[0:1,0:D]
    dw <<= write_grad[0:1,0:D]
    tmp <<= dw * u
    tmp <<= tmp * mult
    tmp <<= tmp * (-LOG_X)
    den <<= den * den
    tmp <<= tmp / den
    tmp <<= tmp / ar
    z <<= z + tmp
    du <<= read_grad[0:1,0:D]
    tmp <<= dw * mult
    du <<= du + tmp
    beta2 <<= beta[0:1,0:1].single()
    beta2 <<= beta2 * 2.0
    tmp <<= u * beta2
    tmp <<= tmp * z
    du <<= du + tmp
    read_grad[0:1,0:D] <<= du
    decay <<= gate[0:1,0:1].single()
    decay <<= decay.exp()
    ar <<= previous[0:1,0:D]
    ar <<= ar * decay
    tmp <<= z * ar
    reduced <<= tmp.cadd()
    dg[0:1,0:1] <<= reduced.single_value()
    tmp <<= u * u
    tmp <<= tmp * z
    reduced <<= tmp.cadd()
    dbeta[0:1,0:1] <<= reduced.single_value()
    z <<= z * decay
    adjoint[0:1,0:D] <<= z
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@vf()
def norm_adjoint(normalized: Tensor, raw_norm: Tensor, gradient: Tensor, scalar: Tensor):
    y = RegList(DT.float,2)
    h = RegList(DT.float,2)
    tmp = RegList(DT.float,2)
    norm = Reg(DT.float)
    dot = Reg(DT.float)
    zero = Reg(DT.float)
    active = MaskReg(DT.float)
    y <<= normalized[0:1,0:D]
    h <<= gradient[0:1,0:D]
    tmp <<= y * h
    dot <<= tmp.cadd()
    scalar[0:1,0:1] <<= dot.single_value()
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)
    dot <<= scalar[0:1,0:1].single()
    norm <<= raw_norm[0:1,0:1].single()
    compare(active,norm,1e-12,CompareMode.GE)
    zero <<= 0.0
    # The scalar norm and dot are broadcast to all lanes before selection.
    dot <<= active.select(dot,zero)
    norm <<= norm.vmaxs(1e-12)
    tmp <<= y * dot
    h <<= h - tmp
    h <<= h / norm
    gradient[0:1,0:D] <<= h
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@kernel()
def pgdn_bwd_atk_reverse(
    q_norm: GM[f32, ('B','T','H',128)], k_read: GM[f32, ('B','T','H',128)], A_history: GM[f32, ('B','T','H',128)],
    q_raw_norm: GM[f32, ('B','T','H')], k_raw_norm: GM[f32, ('B','T','H')],
    dq_norm_parts: GM[f32, ('B','T','HV',128)], dk_read_parts: GM[f32, ('B','T','HV',128)], dk_write_parts: GM[f32, ('B','T','HV',128)],
    g_atk: GM[f32, ('B','T','H')], beta_atk: GM[f32, ('B','T','H')], dat: GM[f32, ('B','H',128)],
    dq: GM[f32, ('B','T','H',128)], dk: GM[f32, ('B','T','H',128)],
    dg_atk: GM[f32, ('B','T','H')], dbeta_atk: GM[f32, ('B','T','H')],
    B: i32, T: i32, H: i32, HV: i32, has_dat: i32,
):
    qu = Tensor(DT.float,[1,D],Position.UB)
    ku = Tensor(DT.float,[1,D],Position.UB)
    current = Tensor(DT.float,[1,D],Position.UB)
    previous = Tensor(DT.float,[1,D],Position.UB)
    adjoint = Tensor(DT.float,[1,D],Position.UB)
    qacc = Tensor(DT.float,[1,D],Position.UB)
    racc = Tensor(DT.float,[1,D],Position.UB)
    wacc = Tensor(DT.float,[1,D],Position.UB)
    part = Tensor(DT.float,[1,D],Position.UB)
    gate = Tensor(DT.float,[1,8],Position.UB)
    bu = Tensor(DT.float,[1,8],Position.UB)
    qnu = Tensor(DT.float,[1,8],Position.UB)
    knu = Tensor(DT.float,[1,8],Position.UB)
    dgu = Tensor(DT.float,[1,8],Position.UB)
    dbu = Tensor(DT.float,[1,8],Position.UB)
    scalar = Tensor(DT.float,[1,8],Position.UB)
    per = CeilDiv(B*H,GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per,B*H)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//H)
            hh = Var(item%H)
            if has_dat:
                adjoint[:,:] <<= dat[bb,hh:hh+1,:]
            else:
                zero_row(adjoint)
            for rev in range(T):
                tt = Var(T-1-rev)
                qu[:,:] <<= q_norm[bb,tt,hh:hh+1,:]
                ku[:,:] <<= k_read[bb,tt,hh:hh+1,:]
                current[:,:] <<= A_history[bb,tt,hh:hh+1,:]
                if tt > 0:
                    previous[:,:] <<= A_history[bb,tt-1,hh:hh+1,:]
                else:
                    zero_row(previous)
                qnu[:,0:1] <<= q_raw_norm[bb,tt:tt+1,hh:hh+1]
                knu[:,0:1] <<= k_raw_norm[bb,tt:tt+1,hh:hh+1]
                gate[:,0:1] <<= g_atk[bb,tt:tt+1,hh:hh+1]
                bu[:,0:1] <<= beta_atk[bb,tt:tt+1,hh:hh+1]
                zero_row(qacc)
                zero_row(racc)
                zero_row(wacc)
                for group in range(HV//H):
                    vh = Var(hh*(HV//H)+group)
                    part[:,:] <<= dq_norm_parts[bb,tt,vh:vh+1,:]
                    add_part(qacc,part)
                    part[:,:] <<= dk_read_parts[bb,tt,vh:vh+1,:]
                    add_part(racc,part)
                    part[:,:] <<= dk_write_parts[bb,tt,vh:vh+1,:]
                    add_part(wacc,part)
                atk_adjoint(ku,current,previous,adjoint,racc,wacc,gate,bu,dgu,dbu)
                norm_adjoint(qu,qnu,qacc,scalar)
                norm_adjoint(ku,knu,racc,scalar)
                dq[bb,tt,hh:hh+1,:] <<= qacc[:,:]
                dk[bb,tt,hh:hh+1,:] <<= racc[:,:]
                dg_atk[bb,tt:tt+1,hh:hh+1] <<= dgu[:,0:1]
                dbeta_atk[bb,tt:tt+1,hh:hh+1] <<= dbu[:,0:1]
    return dq,dk,dg_atk,dbeta_atk
