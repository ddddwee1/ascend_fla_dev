"""Main-state checkpoint scan and bounded chunk replay with its adjoint.

Each (batch,value-head) has one owner for the entire time axis. Read and write
keys remain distinct in every primal and reverse contraction.
"""
from ascriptor.a5 import *
from .atk import zero_row

C = 64
D = 128
SCALE = 128**-.5


@vf()
def zero_state(state: Tensor):
    z = RegList(DT.float, 2)
    z <<= 0.0
    for i in range(D):
        state[i:i+1,0:D] <<= z


@vf()
def primal(state: Tensor, read: Tensor, write: Tensor, value: Tensor,
           gate: Tensor, beta: Tensor):
    row = RegList(DT.float, 2)
    correction = RegList(DT.float, 2)
    delta = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    weight = Reg(DT.float)
    decay = Reg(DT.float)
    update = Reg(DT.float)
    decay <<= gate[0:1,0:1].single()
    decay <<= decay.exp()
    update <<= beta[0:1,0:1].single()
    correction <<= 0.0
    for i in range(D):
        row <<= state[i:i+1,0:D]
        row <<= row * decay
        state[i:i+1,0:D] <<= row
        weight <<= read[0:1,i:i+1].single()
        tmp <<= row * weight
        correction <<= correction + tmp
    delta <<= value[0:1,0:D]
    delta <<= delta - correction
    delta <<= delta * update
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)
    for i in range(D):
        row <<= state[i:i+1,0:D]
        weight <<= write[0:1,i:i+1].single()
        tmp <<= delta * weight
        row <<= row + tmp
        state[i:i+1,0:D] <<= row
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@kernel()
def pgdn_bwd_checkpoints(
    k_read: GM[f32, ('B','T','H',128)], k_write: GM[f32, ('B','T','H',128)],
    v: GM[f32, ('B','T','HV',128)], g: GM[f32, ('B','T','HV')], beta: GM[f32, ('B','T','HV')],
    checkpoints: GM[f32, ('B','N','HV',128,128)], final_state: GM[f32, ('B','HV',128,128)],
    B: i32, T: i32, H: i32, HV: i32, N: i32,
):
    state = Tensor(DT.float,[D,D],Position.UB)
    read = Tensor(DT.float,[1,D],Position.UB)
    write = Tensor(DT.float,[1,D],Position.UB)
    value = Tensor(DT.float,[1,D],Position.UB)
    gate = Tensor(DT.float,[1,8],Position.UB)
    bu = Tensor(DT.float,[1,8],Position.UB)
    per = CeilDiv(B*HV,GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per,B*HV)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//HV)
            hh = Var(item%HV)
            kh = Var(hh//(HV//H))
            zero_state(state)
            for cc in range(N):
                checkpoints[bb,cc,hh,:,:] <<= state[:,:]
                for ii in range(C):
                    tt = Var(cc*C+ii)
                    read[:,:] <<= k_read[bb,tt,kh:kh+1,:]
                    write[:,:] <<= k_write[bb,tt,kh:kh+1,:]
                    value[:,:] <<= v[bb,tt,hh:hh+1,:]
                    gate[:,0:1] <<= g[bb,tt:tt+1,hh:hh+1]
                    bu[:,0:1] <<= beta[bb,tt:tt+1,hh:hh+1]
                    primal(state,read,write,value,gate,bu)
            final_state[bb,hh,:,:] <<= state[:,:]
    return checkpoints,final_state


@vf()
def reverse_step(state: Tensor, adjoint: Tensor, query: Tensor, read: Tensor,
                 write: Tensor, value: Tensor, dout: Tensor, gate: Tensor,
                 beta: Tensor, dq: Tensor, dkr: Tensor, dkw: Tensor,
                 dv: Tensor, dg: Tensor, dbeta: Tensor):
    row = RegList(DT.float, 2)
    grad = RegList(DT.float, 2)
    residual = RegList(DT.float, 2)
    update = RegList(DT.float, 2)
    dotout = RegList(DT.float, 2)
    dz = RegList(DT.float, 2)
    dr = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    gate_terms = RegList(DT.float, 2)
    decay = Reg(DT.float)
    betav = Reg(DT.float)
    weight = Reg(DT.float)
    reduced = Reg(DT.float)
    decay <<= gate[0:1,0:1].single()
    decay <<= decay.exp()
    betav <<= beta[0:1,0:1].single()
    dotout <<= dout[0:1,0:D]
    tmp <<= 0.0
    # Materialize D and the READ-key correction, using the primal order.
    dz <<= 0.0
    for i in range(D):
        row <<= state[i:i+1,0:D]
        row <<= row * decay
        state[i:i+1,0:D] <<= row
        weight <<= read[0:1,i:i+1].single()
        tmp <<= row * weight
        dz <<= dz + tmp
    residual <<= value[0:1,0:D]
    residual <<= residual - dz
    update <<= residual * betav
    dz <<= 0.0
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)
    for i in range(D):
        row <<= state[i:i+1,0:D]
        weight <<= write[0:1,i:i+1].single()
        tmp <<= update * weight
        row <<= row + tmp
        tmp <<= row * dotout
        reduced <<= tmp.cadd()
        reduced <<= reduced * SCALE
        dq[0:1,i:i+1] <<= reduced.single_value()
        grad <<= adjoint[i:i+1,0:D]
        weight <<= query[0:1,i:i+1].single()
        weight <<= weight * SCALE
        tmp <<= dotout * weight
        grad <<= grad + tmp
        adjoint[i:i+1,0:D] <<= grad
        tmp <<= grad * update
        reduced <<= tmp.cadd()
        dkw[0:1,i:i+1] <<= reduced.single_value()
        weight <<= write[0:1,i:i+1].single()
        tmp <<= grad * weight
        dz <<= dz + tmp
    dr <<= dz * betav
    dv[0:1,0:D] <<= dr
    tmp <<= dz * residual
    reduced <<= tmp.cadd()
    dbeta[0:1,0:1] <<= reduced.single_value()
    gate_terms <<= 0.0
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)
    for i in range(D):
        row <<= state[i:i+1,0:D]
        tmp <<= row * dr
        reduced <<= tmp.cadd()
        reduced <<= reduced * (-1.0)
        dkr[0:1,i:i+1] <<= reduced.single_value()
        grad <<= adjoint[i:i+1,0:D]
        weight <<= read[0:1,i:i+1].single()
        tmp <<= dr * weight
        grad <<= grad - tmp
        tmp <<= grad * row
        gate_terms <<= gate_terms + tmp
        grad <<= grad * decay
        adjoint[i:i+1,0:D] <<= grad
    reduced <<= gate_terms.cadd()
    dg[0:1,0:1] <<= reduced.single_value()
    vf_barrier(VfPipe.STORE,VfPipe.LOAD)


@kernel()
def pgdn_bwd_reverse(
    q_norm: GM[f32, ('B','T','H',128)], k_read: GM[f32, ('B','T','H',128)], k_write: GM[f32, ('B','T','H',128)],
    v: GM[f32, ('B','T','HV',128)], g: GM[f32, ('B','T','HV')], beta: GM[f32, ('B','T','HV')],
    dout: GM[f32, ('B','T','HV',128)], dht: GM[f32, ('B','HV',128,128)], checkpoints: GM[f32, ('B','N','HV',128,128)],
    tape: GM[f32, ('B','HV',64,128,128)],
    dq_norm_parts: GM[f32, ('B','T','HV',128)], dk_read_parts: GM[f32, ('B','T','HV',128)], dk_write_parts: GM[f32, ('B','T','HV',128)],
    dv: GM[f32, ('B','T','HV',128)], dg: GM[f32, ('B','T','HV')], dbeta: GM[f32, ('B','T','HV')],
    B: i32, T: i32, H: i32, HV: i32, N: i32, has_do: i32, has_dht: i32,
):
    state = Tensor(DT.float,[D,D],Position.UB)
    adjoint = Tensor(DT.float,[D,D],Position.UB)
    query = Tensor(DT.float,[1,D],Position.UB)
    read = Tensor(DT.float,[1,D],Position.UB)
    write = Tensor(DT.float,[1,D],Position.UB)
    value = Tensor(DT.float,[1,D],Position.UB)
    dotout = Tensor(DT.float,[1,D],Position.UB)
    dqu = Tensor(DT.float,[1,D],Position.UB)
    dkru = Tensor(DT.float,[1,D],Position.UB)
    dkwu = Tensor(DT.float,[1,D],Position.UB)
    dvu = Tensor(DT.float,[1,D],Position.UB)
    gate = Tensor(DT.float,[1,8],Position.UB)
    bu = Tensor(DT.float,[1,8],Position.UB)
    dgu = Tensor(DT.float,[1,8],Position.UB)
    dbu = Tensor(DT.float,[1,8],Position.UB)
    ready = SEvent(Pipe.MTE3,Pipe.MTE2)
    per = CeilDiv(B*HV,GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per,B*HV)
    with auto_sync():
        for item in range(begin,end):
            bb = Var(item//HV)
            hh = Var(item%HV)
            kh = Var(hh//(HV//H))
            if has_dht:
                adjoint[:,:] <<= dht[bb,hh,:,:]
            else:
                zero_state(adjoint)
            for rev in range(N):
                cc = Var(N-1-rev)
                state[:,:] <<= checkpoints[bb,cc,hh,:,:]
                for ii in range(C):
                    tt = Var(cc*C+ii)
                    tape[bb,hh,ii,:,:] <<= state[:,:]
                    read[:,:] <<= k_read[bb,tt,kh:kh+1,:]
                    write[:,:] <<= k_write[bb,tt,kh:kh+1,:]
                    value[:,:] <<= v[bb,tt,hh:hh+1,:]
                    gate[:,0:1] <<= g[bb,tt:tt+1,hh:hh+1]
                    bu[:,0:1] <<= beta[bb,tt:tt+1,hh:hh+1]
                    primal(state,read,write,value,gate,bu)
                # GM tape is private to this owner; explicitly publish writes.
                ready.set()
                ready.wait()
                for rr in range(C):
                    ii = Var(C-1-rr)
                    tt = Var(cc*C+ii)
                    state[:,:] <<= tape[bb,hh,ii,:,:]
                    query[:,:] <<= q_norm[bb,tt,kh:kh+1,:]
                    read[:,:] <<= k_read[bb,tt,kh:kh+1,:]
                    write[:,:] <<= k_write[bb,tt,kh:kh+1,:]
                    value[:,:] <<= v[bb,tt,hh:hh+1,:]
                    if has_do:
                        dotout[:,:] <<= dout[bb,tt,hh:hh+1,:]
                    else:
                        zero_row(dotout)
                    gate[:,0:1] <<= g[bb,tt:tt+1,hh:hh+1]
                    bu[:,0:1] <<= beta[bb,tt:tt+1,hh:hh+1]
                    reverse_step(state,adjoint,query,read,write,value,dotout,gate,bu,dqu,dkru,dkwu,dvu,dgu,dbu)
                    dq_norm_parts[bb,tt,hh:hh+1,:] <<= dqu[:,:]
                    dk_read_parts[bb,tt,hh:hh+1,:] <<= dkru[:,:]
                    dk_write_parts[bb,tt,hh:hh+1,:] <<= dkwu[:,:]
                    dv[bb,tt,hh:hh+1,:] <<= dvu[:,:]
                    dg[bb,tt:tt+1,hh:hh+1] <<= dgu[:,0:1]
                    dbeta[bb,tt:tt+1,hh:hh+1] <<= dbu[:,0:1]
                # Last tape reader retires before the next chunk overwrites it.
                barrier(Pipe.ALL)
    return tape,dq_norm_parts,dk_read_parts,dk_write_parts,dv,dg,dbeta
