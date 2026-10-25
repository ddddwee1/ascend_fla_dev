"""A2 raw derivatives with fixed-order compensated reductions.

A2 vmla is non-fused: product residuals use a mantissa bit split.
Each helper owns a disjoint region in the reusable scratch allocation.
GM transfers drain local pipes before and after every transfer.
The fixed polynomial coefficient loops preserve the original Horner order.
See scratch-layout.json for offsets and lifetime requirements.
"""
from ascriptor.a2 import *
import math
import struct

@func
def _copy64(scratch, destination, source):
    ub_to_ub(destination, source, n_burst=1, burst_len=8, src_stride=0, dst_stride=0)

@func
def _select64(scratch, destination, yes, no, mask):
    selected = scratch[0:1, :].reinterpret(f32)[0:1, 0:64]
    address = scratch[1:2, :].reinterpret(u32)[0:1, 0:8]
    select(selected, mask, yes, no, SelectMode.TENSOR_TENSOR, tmp_addr_buf=address, repeat=1)
    _copy64(scratch, destination, selected)

@func
def _broadcast64(scratch, destination, source):
    block = scratch[2:3, :].reinterpret(f32)[0:1, 0:64]
    result = scratch[3:4, :].reinterpret(f32)[0:1, 0:64]
    brcb(block, source, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
    brcb(result, block[0:1, 0:8], repeat=1, dst_blk_stride=1, dst_rep_stride=8)
    _copy64(scratch, destination, result)

@func
def _reduce_add64(scratch, destination, source):
    reduced = scratch[4:5, :].reinterpret(f32)[0:1, 0:64]
    dup(reduced, 0.0, count=64)
    cadd(reduced, source, repeat=1, count_per_rep=64)
    _copy64(scratch, destination, reduced)

@func
def _reduce_max64(scratch, destination, source):
    reduced = scratch[5:6, :].reinterpret(f32)[0:1, 0:64]
    dup(reduced, 0.0, count=64)
    cmax(reduced, source, repeat=1, count_per_rep=64)
    _copy64(scratch, destination, reduced)

@func
def _initialize_offsets(scratch, offsets):
    for level in unroll(6):
        for lane in range(64):
            step = 32 >> level
            partner = Var((lane / (step * 2) * (step * 2) + (lane + step) % (step * 2)) * 4)
            partner.SetValueTo(offsets[level:level + 1, lane:lane + 1])
    barrier(Pipe.ALL)

@func
def _gather64(scratch, destination, source, offsets):
    gathered = scratch[6:7, :].reinterpret(f32)[0:1, 0:64]
    set_mask_normal()
    gather(gathered, source, offsets, repeat=1, dst_rep_stride=8)
    _copy64(scratch, destination, gathered)

@func
def _and32(scratch, destination, a, b):
    vand(destination.reinterpret(u16), a.reinterpret(u16), b.reinterpret(u16), count=128)

@func
def _inverse_exponent(scratch, destination, maximum):
    mask = scratch[7:8, :].reinterpret(u32)[0:1, 0:64]
    bits = scratch[8:9, :].reinterpret(u32)[0:1, 0:64]
    power = scratch[9:10, :].reinterpret(i32)[0:1, 0:64]
    dup(mask, 2139095040, count=64)
    _and32(scratch, bits, maximum, mask)
    dup(power, 2130706432, count=64)
    sub(bits.reinterpret(i32), power, bits.reinterpret(i32), count=64)
    _copy64(scratch, destination, bits.reinterpret(f32))

@func
def _load_one_f32(scratch, destination, source):
    value = Var(0.0, f32)
    value.GetValueFrom(source)
    dup(destination, value, count=64)

@func
def _store_one_f32(scratch, destination, source):
    value = Var(0.0, f32)
    barrier(Pipe.ALL)
    value.GetValueFrom(source)
    value.SetValueTo(destination)
    barrier(Pipe.ALL)

@func
def _product_residual(scratch, destination, a, b, product):
    mask = scratch[10:11, :].reinterpret(u32)[0:1, 0:64]
    ah = scratch[11:12, :].reinterpret(f32)[0:1, 0:64]
    al = scratch[12:13, :].reinterpret(f32)[0:1, 0:64]
    bh = scratch[13:14, :].reinterpret(f32)[0:1, 0:64]
    bl = scratch[14:15, :].reinterpret(f32)[0:1, 0:64]
    error = scratch[15:16, :].reinterpret(f32)[0:1, 0:64]
    term = scratch[16:17, :].reinterpret(f32)[0:1, 0:64]
    dup(mask, 4294963200, count=64)
    _and32(scratch, ah, a, mask)
    _and32(scratch, bh, b, mask)
    sub(al, a, ah, count=64)
    sub(bl, b, bh, count=64)
    mul(error, ah, bh, count=64)
    sub(error, error, product, count=64)
    mul(term, ah, bl, count=64)
    add(error, error, term, count=64)
    mul(term, al, bh, count=64)
    add(error, error, term, count=64)
    mul(term, al, bl, count=64)
    add(error, error, term, count=64)
    _copy64(scratch, destination, error)

def _split_constant(value):
    high = struct.unpack('f', struct.pack('f', value))[0]
    return (high, value - high)
_EXP_COEFFICIENTS = tuple((_split_constant(1 / math.factorial(n)) for n in range(13, -1, -1)))
_LOG_COEFFICIENTS = tuple((_split_constant(1 / n) for n in range(39, 0, -2)))
_EXP_FIRST = _split_constant(1 / math.factorial(14))
_LOG_FIRST = _split_constant(1 / 41)
_EXP_INDEXED = tuple(((index, *pair) for index, pair in enumerate(_EXP_COEFFICIENTS)))
_LOG_INDEXED = tuple(((index, *pair) for index, pair in enumerate(_LOG_COEFFICIENTS)))

@func
def _initialize_exp_coefficients(scratch, table):
    for index, high, low in _EXP_INDEXED:
        h = Var(high, f32)
        l = Var(low, f32)
        h.SetValueTo(table[index:index + 1, 0:1])
        l.SetValueTo(table[index:index + 1, 1:2])
    barrier(Pipe.ALL)

@func
def _initialize_log_coefficients(scratch, table):
    for index, high, low in _LOG_INDEXED:
        h = Var(high, f32)
        l = Var(low, f32)
        h.SetValueTo(table[index:index + 1, 0:1])
        l.SetValueTo(table[index:index + 1, 1:2])
    barrier(Pipe.ALL)

@func
def _dd_add(scratch, oh, ol, ah, al, bh, bl):
    s = scratch[17:18, :].reinterpret(DT.float)[0:1, 0:64]
    v = scratch[18:19, :].reinterpret(DT.float)[0:1, 0:64]
    e = scratch[19:20, :].reinterpret(DT.float)[0:1, 0:64]
    t = scratch[20:21, :].reinterpret(DT.float)[0:1, 0:64]
    h = scratch[21:22, :].reinterpret(DT.float)[0:1, 0:64]
    l = scratch[22:23, :].reinterpret(DT.float)[0:1, 0:64]
    overflow = scratch[23:24, :].reinterpret(u8)[0:1, 0:32]
    add(s, ah, bh, count=64)
    sub(v, s, ah, count=64)
    sub(t, s, v, count=64)
    sub(e, ah, t, count=64)
    sub(t, bh, v, count=64)
    add(e, e, t, count=64)
    add(t, al, bl, count=64)
    add(e, e, t, count=64)
    add(h, s, e, count=64)
    sub(t, h, s, count=64)
    sub(l, e, t, count=64)
    dup(e, 0.0, count=64)
    abs(t, h, count=64)
    compare_scalar(overflow, t, 3.4028234663852886e+38, CompareMode.GT, repeat=1)
    _select64(scratch, l, e, l, overflow)
    abs(t, s, count=64)
    compare_scalar(overflow, t, 3.4028234663852886e+38, CompareMode.GT, repeat=1)
    _select64(scratch, h, s, h, overflow)
    _select64(scratch, l, e, l, overflow)
    _copy64(scratch, oh, h)
    _copy64(scratch, ol, l)

@func
def _dd_add_constant(scratch, oh, ol, ah, al, high, low):
    s = scratch[24:25, :].reinterpret(DT.float)[0:1, 0:64]
    e = scratch[25:26, :].reinterpret(DT.float)[0:1, 0:64]
    t = scratch[26:27, :].reinterpret(DT.float)[0:1, 0:64]
    h = scratch[27:28, :].reinterpret(DT.float)[0:1, 0:64]
    l = scratch[28:29, :].reinterpret(DT.float)[0:1, 0:64]
    adds(s, ah, high, count=64)
    adds(t, s, -high, count=64)
    sub(e, ah, t, count=64)
    adds(t, al, low, count=64)
    add(e, e, t, count=64)
    add(h, s, e, count=64)
    sub(t, h, s, count=64)
    sub(l, e, t, count=64)
    _copy64(scratch, oh, h)
    _copy64(scratch, ol, l)

@func
def _dd_mul(scratch, oh, ol, ah, al, bh, bl, protect_overflow=True):
    p = scratch[29:30, :].reinterpret(DT.float)[0:1, 0:64]
    e = scratch[30:31, :].reinterpret(DT.float)[0:1, 0:64]
    t = scratch[31:32, :].reinterpret(DT.float)[0:1, 0:64]
    h = scratch[32:33, :].reinterpret(DT.float)[0:1, 0:64]
    l = scratch[33:34, :].reinterpret(DT.float)[0:1, 0:64]
    mul(p, ah, bh, count=64)
    _product_residual(scratch, e, ah, bh, p)
    mul(t, ah, bl, count=64)
    add(e, e, t, count=64)
    mul(t, al, bh, count=64)
    add(e, e, t, count=64)
    add(h, p, e, count=64)
    sub(t, h, p, count=64)
    sub(l, e, t, count=64)
    if protect_overflow:
        zero = scratch[34:35, :].reinterpret(DT.float)[0:1, 0:64]
        overflow = scratch[35:36, :].reinterpret(u8)[0:1, 0:32]
        dup(zero, 0.0, count=64)
        abs(t, h, count=64)
        compare_scalar(overflow, t, 3.4028234663852886e+38, CompareMode.GT, repeat=1)
        _select64(scratch, l, zero, l, overflow)
        abs(t, p, count=64)
        compare_scalar(overflow, t, 3.4028234663852886e+38, CompareMode.GT, repeat=1)
        _select64(scratch, h, p, h, overflow)
        _select64(scratch, l, zero, l, overflow)
    _copy64(scratch, oh, h)
    _copy64(scratch, ol, l)

@func
def _dd_div(scratch, oh, ol, ah, al, bh, bl):
    q = scratch[36:37, :].reinterpret(DT.float)[0:1, 0:64]
    zero = scratch[37:38, :].reinterpret(DT.float)[0:1, 0:64]
    ph = scratch[38:39, :].reinterpret(DT.float)[0:1, 0:64]
    pl = scratch[39:40, :].reinterpret(DT.float)[0:1, 0:64]
    rh = scratch[40:41, :].reinterpret(DT.float)[0:1, 0:64]
    rl = scratch[41:42, :].reinterpret(DT.float)[0:1, 0:64]
    correction = scratch[42:43, :].reinterpret(DT.float)[0:1, 0:64]
    dup(zero, 0.0, count=64)
    div(q, ah, bh, count=64)
    _dd_mul(scratch, ph, pl, bh, bl, q, zero)
    muls(ph, ph, -1.0, count=64)
    muls(pl, pl, -1.0, count=64)
    _dd_add(scratch, rh, rl, ah, al, ph, pl)
    add(correction, rh, rl, count=64)
    div(correction, correction, bh, count=64)
    _dd_add(scratch, oh, ol, q, zero, correction, zero)

@func
def _dd_exp_negative(scratch, oh, ol, xh, xl, exp_coefficients):
    h = scratch[43:44, :].reinterpret(DT.float)[0:1, 0:64]
    l = scratch[44:45, :].reinterpret(DT.float)[0:1, 0:64]
    zh = scratch[45:46, :].reinterpret(DT.float)[0:1, 0:64]
    zl = scratch[46:47, :].reinterpret(DT.float)[0:1, 0:64]
    fallback = scratch[47:48, :].reinterpret(DT.float)[0:1, 0:64]
    floor = scratch[48:49, :].reinterpret(DT.float)[0:1, 0:64]
    zero = scratch[49:50, :].reinterpret(DT.float)[0:1, 0:64]
    outside = scratch[50:51, :].reinterpret(u8)[0:1, 0:32]
    dup(floor, -80.0, count=64)
    dup(zero, 0.0, count=64)
    compare_scalar(outside, xh, -80.0, CompareMode.LT, repeat=1)
    _select64(scratch, zh, floor, xh, outside)
    _select64(scratch, zl, zero, xl, outside)
    muls(zh, zh, 0.00390625, count=64)
    muls(zl, zl, 0.00390625, count=64)
    dup(h, _EXP_FIRST[0], count=64)
    dup(l, _EXP_FIRST[1], count=64)
    high = Var(0.0, f32)
    low = Var(0.0, f32)
    for coefficient in range(14):
        high.GetValueFrom(exp_coefficients[coefficient:coefficient + 1, 0:1])
        low.GetValueFrom(exp_coefficients[coefficient:coefficient + 1, 1:2])
        _dd_mul(scratch, h, l, h, l, zh, zl, protect_overflow=False)
        _dd_add_constant(scratch, h, l, h, l, high, low)
    for iteration in range(8):
        _dd_mul(scratch, h, l, h, l, h, l, protect_overflow=False)
    exp(fallback, xh, count=64)
    _select64(scratch, oh, fallback, h, outside)
    _select64(scratch, ol, zero, l, outside)

@func
def _dd_softplus_sigmoid(scratch, sh, sl, dh, dl, uh, ul, exp_coefficients, log_coefficients):
    zero = scratch[51:52, :].reinterpret(DT.float)[0:1, 0:64]
    one = scratch[52:53, :].reinterpret(DT.float)[0:1, 0:64]
    two = scratch[53:54, :].reinterpret(DT.float)[0:1, 0:64]
    xh = scratch[54:55, :].reinterpret(DT.float)[0:1, 0:64]
    xl = scratch[55:56, :].reinterpret(DT.float)[0:1, 0:64]
    eh = scratch[56:57, :].reinterpret(DT.float)[0:1, 0:64]
    el = scratch[57:58, :].reinterpret(DT.float)[0:1, 0:64]
    bh = scratch[58:59, :].reinterpret(DT.float)[0:1, 0:64]
    bl = scratch[59:60, :].reinterpret(DT.float)[0:1, 0:64]
    nh = scratch[60:61, :].reinterpret(DT.float)[0:1, 0:64]
    nl = scratch[61:62, :].reinterpret(DT.float)[0:1, 0:64]
    zh = scratch[62:63, :].reinterpret(DT.float)[0:1, 0:64]
    zl = scratch[63:64, :].reinterpret(DT.float)[0:1, 0:64]
    zz_h = scratch[64:65, :].reinterpret(DT.float)[0:1, 0:64]
    zz_l = scratch[65:66, :].reinterpret(DT.float)[0:1, 0:64]
    ph = scratch[66:67, :].reinterpret(DT.float)[0:1, 0:64]
    pl = scratch[67:68, :].reinterpret(DT.float)[0:1, 0:64]
    positive = scratch[68:69, :].reinterpret(u8)[0:1, 0:32]
    upper = scratch[69:70, :].reinterpret(u8)[0:1, 0:32]
    dup(zero, 0.0, count=64)
    dup(one, 1.0, count=64)
    dup(two, 2.0, count=64)
    compare_scalar(positive, uh, 0.0, CompareMode.GE, repeat=1)
    compare_scalar(upper, uh, 20.0, CompareMode.GT, repeat=1)
    muls(xh, uh, -1.0, count=64)
    muls(xl, ul, -1.0, count=64)
    _select64(scratch, xh, xh, uh, positive)
    _select64(scratch, xl, xl, ul, positive)
    _dd_exp_negative(scratch, eh, el, xh, xl, exp_coefficients)
    _dd_add(scratch, bh, bl, one, zero, eh, el)
    _select64(scratch, nh, one, eh, positive)
    _select64(scratch, nl, zero, el, positive)
    _dd_div(scratch, dh, dl, nh, nl, bh, bl)
    _select64(scratch, dh, one, dh, upper)
    _select64(scratch, dl, zero, dl, upper)
    _dd_add(scratch, bh, bl, two, zero, eh, el)
    _dd_div(scratch, zh, zl, eh, el, bh, bl)
    _dd_mul(scratch, zz_h, zz_l, zh, zl, zh, zl, protect_overflow=False)
    dup(ph, _LOG_FIRST[0], count=64)
    dup(pl, _LOG_FIRST[1], count=64)
    high = Var(0.0, f32)
    low = Var(0.0, f32)
    for coefficient in range(20):
        high.GetValueFrom(log_coefficients[coefficient:coefficient + 1, 0:1])
        low.GetValueFrom(log_coefficients[coefficient:coefficient + 1, 1:2])
        _dd_mul(scratch, ph, pl, ph, pl, zz_h, zz_l, protect_overflow=False)
        _dd_add_constant(scratch, ph, pl, ph, pl, high, low)
    _dd_mul(scratch, ph, pl, ph, pl, zh, zl, protect_overflow=False)
    muls(ph, ph, 2.0, count=64)
    muls(pl, pl, 2.0, count=64)
    _select64(scratch, nh, uh, zero, positive)
    _select64(scratch, nl, ul, zero, positive)
    _dd_add(scratch, sh, sl, ph, pl, nh, nl)
    _select64(scratch, sh, uh, sh, upper)
    _select64(scratch, sl, ul, sl, upper)

def make_norm_backward(name, raw_dtype):

    @func
    def differentiate(scratch, source: Tensor, sensitivity: Tensor, destination: Tensor, rows: Var, offsets: Tensor):
        x0 = Tensor(DT.float, [1, 64], Position.UB)
        x1 = Tensor(DT.float, [1, 64], Position.UB)
        d0 = Tensor(DT.float, [1, 64], Position.UB)
        d1 = Tensor(DT.float, [1, 64], Position.UB)
        sh = Tensor(DT.float, [1, 64], Position.UB)
        sl = Tensor(DT.float, [1, 64], Position.UB)
        dh = Tensor(DT.float, [1, 64], Position.UB)
        dl = Tensor(DT.float, [1, 64], Position.UB)
        ph = Tensor(DT.float, [1, 64], Position.UB)
        pl = Tensor(DT.float, [1, 64], Position.UB)
        qh = Tensor(DT.float, [1, 64], Position.UB)
        ql = Tensor(DT.float, [1, 64], Position.UB)
        a_hi = Tensor(DT.float, [1, 64], Position.UB)
        a_lo = Tensor(DT.float, [1, 64], Position.UB)
        b_hi = Tensor(DT.float, [1, 64], Position.UB)
        b_lo = Tensor(DT.float, [1, 64], Position.UB)
        split = Tensor(DT.float, [1, 64], Position.UB)
        temp = Tensor(DT.float, [1, 64], Position.UB)
        summed = Tensor(DT.float, [1, 64], Position.UB)
        virtual_b = Tensor(DT.float, [1, 64], Position.UB)
        virtual_a = Tensor(DT.float, [1, 64], Position.UB)
        error_a = Tensor(DT.float, [1, 64], Position.UB)
        error_b = Tensor(DT.float, [1, 64], Position.UB)
        low = Tensor(DT.float, [1, 64], Position.UB)
        numerator = Tensor(DT.float, [1, 64], Position.UB)
        denominator = Tensor(DT.float, [1, 64], Position.UB)
        result = Tensor(DT.float, [1, 64], Position.UB)
        row_scale = Tensor(DT.float, [1, 64], Position.UB)
        epsilon = Tensor(DT.float, [1, 64], Position.UB)
        large = Tensor(u8, [1, 32], Position.UB)
        forward_overflow = Tensor(u8, [1, 32], Position.UB)
        packed = Tensor(DT.bfloat16, [1, 128], Position.UB)
        for row in range(rows):
            if raw_dtype == bf16:
                cast(x0, source[0, row * 128:row * 128 + 64], count=64, round_mode=RoundMode.NONE)
                cast(x1, source[0, row * 128 + 64:row * 128 + 128], count=64, round_mode=RoundMode.NONE)
            else:
                _copy64(scratch, x0, source[0, row * 128:row * 128 + 64])
                _copy64(scratch, x1, source[0, row * 128 + 64:row * 128 + 128])
            cast(d0, sensitivity[0, row * 128:row * 128 + 64], count=64, round_mode=RoundMode.NONE)
            cast(d1, sensitivity[0, row * 128 + 64:row * 128 + 128], count=64, round_mode=RoundMode.NONE)
            mul(ph, x0, x0, count=64)
            mul(qh, x1, x1, count=64)
            _reduce_add64(scratch, denominator, ph)
            _reduce_add64(scratch, numerator, qh)
            add(temp, denominator, numerator, count=64)
            adds(temp, temp, 1e-06, count=64)
            _broadcast64(scratch, temp, temp)
            compare_scalar(forward_overflow, temp, 3.4028234663852886e+38, CompareMode.GT, repeat=1)
            abs(temp, x0, count=64)
            abs(result, x1, count=64)
            _reduce_max64(scratch, denominator, temp)
            _reduce_max64(scratch, numerator, result)
            vmax(denominator, denominator, numerator, count=64)
            _broadcast64(scratch, denominator, denominator)
            compare_scalar(large, denominator, 16.0, CompareMode.GE, repeat=1)
            _inverse_exponent(scratch, row_scale, denominator)
            vmaxs(row_scale, row_scale, 7.52316384526264e-37, count=64)
            dup(numerator, 1.0, count=64)
            _select64(scratch, row_scale, row_scale, numerator, large)
            mul(x0, x0, row_scale, count=64)
            mul(x1, x1, row_scale, count=64)
            muls(epsilon, row_scale, 1e-06, count=64)
            mul(epsilon, epsilon, row_scale, count=64)
            for left0, left1, right0, right1, high_out, low_out in ((x0, x1, x0, x1, sh, sl), (x0, x1, d0, d1, dh, dl)):
                for a, b, high, residual in ((left0, right0, ph, pl), (left1, right1, qh, ql)):
                    mul(high, a, b, count=64)
                    if raw_dtype == bf16:
                        dup(residual, 0.0, count=64)
                    else:
                        _product_residual(scratch, residual, a, b, high)
                _reduce_add64(scratch, denominator, ph)
                _reduce_add64(scratch, result, qh)
                add(denominator, denominator, result, count=64)
                _broadcast64(scratch, denominator, denominator)
                for level in unroll(7):
                    if level > 0:
                        _gather64(scratch, qh, ph, offsets[level - 1:level, :])
                        _gather64(scratch, ql, pl, offsets[level - 1:level, :])
                    add(summed, ph, qh, count=64)
                    sub(virtual_b, summed, ph, count=64)
                    sub(virtual_a, summed, virtual_b, count=64)
                    sub(error_b, qh, virtual_b, count=64)
                    sub(error_a, ph, virtual_a, count=64)
                    add(temp, error_a, error_b, count=64)
                    add(low, pl, ql, count=64)
                    add(low, low, temp, count=64)
                    add(ph, summed, low, count=64)
                    sub(temp, ph, summed, count=64)
                    sub(pl, low, temp, count=64)
                _broadcast64(scratch, high_out, ph)
                _broadcast64(scratch, low_out, pl)
                sub(temp, high_out, denominator, count=64)
                add(low_out, temp, low_out, count=64)
                _copy64(scratch, high_out, denominator)
            for x, gy, offset in ((x0, d0, 0), (x1, d1, 64)):
                for a, b, high, residual in ((gy, sh, ph, pl), (x, dh, qh, ql)):
                    mul(high, a, b, count=64)
                    _product_residual(scratch, residual, a, b, high)
                muls(qh, qh, -1.0, count=64)
                muls(ql, ql, -1.0, count=64)
                add(summed, ph, qh, count=64)
                sub(virtual_b, summed, ph, count=64)
                sub(virtual_a, summed, virtual_b, count=64)
                sub(error_b, qh, virtual_b, count=64)
                sub(error_a, ph, virtual_a, count=64)
                add(temp, error_a, error_b, count=64)
                add(low, pl, ql, count=64)
                add(low, low, temp, count=64)
                add(ph, summed, low, count=64)
                sub(temp, ph, summed, count=64)
                sub(pl, low, temp, count=64)
                mul(numerator, gy, sl, count=64)
                mul(temp, x, dl, count=64)
                sub(numerator, numerator, temp, count=64)
                mul(temp, gy, epsilon, count=64)
                add(numerator, numerator, temp, count=64)
                add(numerator, numerator, pl, count=64)
                add(numerator, ph, numerator, count=64)
                add(temp, sl, epsilon, count=64)
                add(summed, sh, temp, count=64)
                sqrt(denominator, summed, count=64)
                mul(denominator, summed, denominator, count=64)
                div(result, numerator, denominator, count=64)
                mul(result, result, row_scale, count=64)
                dup(temp, 0.0, count=64)
                _select64(scratch, result, temp, result, forward_overflow)
                if raw_dtype == bf16:
                    cast(packed, result, count=64, round_mode=RoundMode.TO_EVEN)
                    ub_to_ub(destination[0, row * 128 + offset:row * 128 + offset + 64], packed, n_burst=1, burst_len=4, src_stride=0, dst_stride=0)
                else:
                    _copy64(scratch, destination[0, row * 128 + offset:row * 128 + offset + 64], result)
        barrier(Pipe.ALL)

    @kernel(mode='vec')
    def norm_backward(source: GM[raw_dtype, (1, 'N')], sensitivity: GM[bf16, (1, 'N')], destination: GM[raw_dtype, (1, 'N')], N: i32):
        scratch = Tensor(f32, [70, 64], Position.UB)
        incoming = Tensor(raw_dtype, [1, 4096], Position.UB)
        gradient = Tensor(DT.bfloat16, [1, 4096], Position.UB)
        offsets = Tensor(u32, [6, 64], Position.UB)
        outgoing = Tensor(raw_dtype, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        _initialize_offsets(scratch, offsets)
        for work in range(begin, end):
            offset = Var(work * 4096)
            count = Min(4096, N - offset)
            barrier(Pipe.ALL)
            gm_to_ub_pad(incoming, source[0, offset:offset + count], n_burst=1, burst_len_element=count, src_stride_element=0, dst_stride=0)
            barrier(Pipe.ALL)
            barrier(Pipe.ALL)
            gm_to_ub_pad(gradient, sensitivity[0, offset:offset + count], n_burst=1, burst_len_element=count, src_stride_element=0, dst_stride=0)
            barrier(Pipe.ALL)
            differentiate(scratch, incoming, gradient, outgoing, Var(count / 128), offsets)
            barrier(Pipe.ALL)
            ub_to_gm_pad(destination[0, offset:offset + count], outgoing, n_burst=1, burst_len_element=count, src_stride=0, dst_stride_element=0)
            barrier(Pipe.ALL)
        return destination
    norm_backward.name = name
    return norm_backward

def make_gate_backward(name, raw_dtype, a_dtype, bias_dtype):

    @func
    def contributions(scratch, source: Tensor, sensitivity: Tensor, avec: Tensor, bias: Tensor, destination: Tensor, terms: Tensor, partial: Tensor, rows: Var, exp_coefficients: Tensor, log_coefficients: Tensor):
        decay = Tensor(DT.float, [1, 64], Position.UB)
        narrow = Tensor(DT.bfloat16, [1, 128], Position.UB)
        _load_one_f32(scratch, decay, avec[0, 0:1])
        exp(decay, decay, count=64)
        muls(decay, decay, -1.0, count=64)
        bias0 = Tensor(DT.float, [1, 64], Position.UB)
        bias1 = Tensor(DT.float, [1, 64], Position.UB)
        if bias_dtype == bf16:
            cast(bias0, bias[0, 0:64], count=64, round_mode=RoundMode.NONE)
            cast(bias1, bias[0, 64:128], count=64, round_mode=RoundMode.NONE)
        else:
            _copy64(scratch, bias0, bias[0, 0:64])
            _copy64(scratch, bias1, bias[0, 64:128])
        value = Tensor(DT.float, [1, 64], Position.UB)
        gy = Tensor(DT.float, [1, 64], Position.UB)
        uh = Tensor(DT.float, [1, 64], Position.UB)
        ul = Tensor(DT.float, [1, 64], Position.UB)
        sh = Tensor(DT.float, [1, 64], Position.UB)
        sl = Tensor(DT.float, [1, 64], Position.UB)
        dh = Tensor(DT.float, [1, 64], Position.UB)
        dl = Tensor(DT.float, [1, 64], Position.UB)
        ah = Tensor(DT.float, [1, 64], Position.UB)
        al = Tensor(DT.float, [1, 64], Position.UB)
        bh = Tensor(DT.float, [1, 64], Position.UB)
        bl = Tensor(DT.float, [1, 64], Position.UB)
        result = Tensor(DT.float, [1, 64], Position.UB)
        zero = Tensor(DT.float, [1, 64], Position.UB)
        dup(zero, 0.0, count=64)
        for row in range(rows):
            for half in unroll(2):
                if raw_dtype == bf16:
                    cast(value, source[0, row * 128 + half * 64:row * 128 + half * 64 + 64], count=64, round_mode=RoundMode.NONE)
                else:
                    _copy64(scratch, value, source[0, row * 128 + half * 64:row * 128 + half * 64 + 64])
                _copy64(scratch, gy, sensitivity[0, row * 128 + half * 64:row * 128 + half * 64 + 64])
                if half == 0:
                    _dd_add(scratch, uh, ul, value, zero, bias0, zero)
                else:
                    _dd_add(scratch, uh, ul, value, zero, bias1, zero)
                _dd_softplus_sigmoid(scratch, sh, sl, dh, dl, uh, ul, exp_coefficients, log_coefficients)
                _dd_mul(scratch, ah, al, gy, zero, sh, sl)
                _dd_mul(scratch, bh, bl, gy, zero, dh, dl)
                _copy64(scratch, terms[0, row * 128 + half * 64:row * 128 + half * 64 + 64], ah)
                _copy64(scratch, terms[0, 4096 + row * 128 + half * 64:4096 + row * 128 + half * 64 + 64], al)
                _copy64(scratch, terms[0, 8192 + row * 128 + half * 64:8192 + row * 128 + half * 64 + 64], bh)
                _copy64(scratch, terms[0, 12288 + row * 128 + half * 64:12288 + row * 128 + half * 64 + 64], bl)
                _dd_mul(scratch, dh, dl, bh, bl, decay, zero)
                add(result, dh, dl, count=64)
                if raw_dtype == bf16:
                    cast(narrow, result, count=64, round_mode=RoundMode.TO_EVEN)
                    ub_to_ub(destination[0, row * 128 + half * 64:row * 128 + half * 64 + 64], narrow, n_burst=1, burst_len=4, src_stride=0, dst_stride=0)
                else:
                    _copy64(scratch, destination[0, row * 128 + half * 64:row * 128 + half * 64 + 64], result)
        for row in range(rows, 32):
            for part in unroll(4):
                for half in unroll(2):
                    _copy64(scratch, terms[0, part * 4096 + row * 128 + half * 64:part * 4096 + row * 128 + half * 64 + 64], zero)
        barrier(Pipe.ALL)
        for level in unroll(5):
            for pair in range(32 // (2 << level)):
                for part in unroll(2):
                    for half in unroll(2):
                        _copy64(scratch, ah, terms[0, part * 8192 + pair * (256 << level) + half * 64:part * 8192 + pair * (256 << level) + half * 64 + 64])
                        _copy64(scratch, al, terms[0, part * 8192 + 4096 + pair * (256 << level) + half * 64:part * 8192 + 4096 + pair * (256 << level) + half * 64 + 64])
                        _copy64(scratch, bh, terms[0, part * 8192 + pair * (256 << level) + (128 << level) + half * 64:part * 8192 + pair * (256 << level) + (128 << level) + half * 64 + 64])
                        _copy64(scratch, bl, terms[0, part * 8192 + 4096 + pair * (256 << level) + (128 << level) + half * 64:part * 8192 + 4096 + pair * (256 << level) + (128 << level) + half * 64 + 64])
                        _dd_add(scratch, ah, al, ah, al, bh, bl)
                        _copy64(scratch, terms[0, part * 8192 + pair * (256 << level) + half * 64:part * 8192 + pair * (256 << level) + half * 64 + 64], ah)
                        _copy64(scratch, terms[0, part * 8192 + 4096 + pair * (256 << level) + half * 64:part * 8192 + 4096 + pair * (256 << level) + half * 64 + 64], al)
            barrier(Pipe.ALL)
        for part in unroll(4):
            for half in unroll(2):
                _copy64(scratch, result, terms[0, part * 4096 + half * 64:part * 4096 + half * 64 + 64])
                _copy64(scratch, partial[0, part * 128 + half * 64:part * 128 + half * 64 + 64], result)
        barrier(Pipe.ALL)

    @kernel(mode='vec')
    def gate_backward(source: GM[raw_dtype, (1, 'N')], sensitivity: GM[f32, (1, 'N')], alog: GM[a_dtype, (1, 'HV')], bias: GM[bias_dtype, (1, 'Channels')], destination: GM[raw_dtype, (1, 'N')], partials: GM[f32, (1, 'P')], N: i32, HV: i32, Channels: i32, BT: i32, P: i32):
        scratch = Tensor(f32, [70, 64], Position.UB)
        incoming = Tensor(raw_dtype, [1, 4096], Position.UB)
        gradient = Tensor(DT.float, [1, 4096], Position.UB)
        outgoing = Tensor(raw_dtype, [1, 4096], Position.UB)
        avec_raw = Tensor(a_dtype, [1, 128], Position.UB)
        avec = Tensor(f32, [1, 64], Position.UB)
        bvec = Tensor(bias_dtype, [1, 128], Position.UB)
        terms = Tensor(DT.float, [1, 16384], Position.UB)
        partial = Tensor(DT.float, [1, 512], Position.UB)
        chunks = CeilDiv(BT, 32)
        exp_coefficients = Tensor(f32, [14, 8], Position.UB)
        log_coefficients = Tensor(f32, [20, 8], Position.UB)
        works = Var(HV * chunks)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        _initialize_exp_coefficients(scratch, exp_coefficients)
        _initialize_log_coefficients(scratch, log_coefficients)
        for work in range(begin, end):
            head = Var(work / chunks)
            start = Var(work % chunks * 32)
            rows = Min(32, BT - start)
            offset = Var(start * Channels + head * 128)
            span = Var((rows - 1) * Channels + 128)
            barrier(Pipe.ALL)
            dup(avec_raw.reinterpret(u32), 0, count=64 if a_dtype == bf16 else 128)
            barrier(Pipe.ALL)
            gm_to_ub_pad(avec_raw, alog[0, head:head + 1], n_burst=1, burst_len_element=1, src_stride_element=0, dst_stride=0, pad=0.0)
            barrier(Pipe.ALL)
            if a_dtype == bf16:
                cast(avec, avec_raw, count=64, round_mode=RoundMode.NONE)
            else:
                _copy64(scratch, avec, avec_raw)
            barrier(Pipe.ALL)
            barrier(Pipe.ALL)
            gm_to_ub_pad(bvec, bias[0, head * 128:head * 128 + 128], n_burst=1, burst_len_element=128, src_stride_element=0, dst_stride=0)
            barrier(Pipe.ALL)
            barrier(Pipe.ALL)
            gm_to_ub_pad(incoming, source[0, offset:offset + span], n_burst=rows, burst_len_element=128, src_stride_element=Channels - 128, dst_stride=0)
            barrier(Pipe.ALL)
            barrier(Pipe.ALL)
            gm_to_ub_pad(gradient, sensitivity[0, offset:offset + span], n_burst=rows, burst_len_element=128, src_stride_element=Channels - 128, dst_stride=0)
            barrier(Pipe.ALL)
            contributions(scratch, incoming, gradient, avec, bvec, outgoing, terms, partial, Var(rows), exp_coefficients, log_coefficients)
            barrier(Pipe.ALL)
            ub_to_gm_pad(destination[0, offset:offset + span], outgoing, n_burst=rows, burst_len_element=128, src_stride=0, dst_stride_element=Channels - 128)
            barrier(Pipe.ALL)
            barrier(Pipe.ALL)
            ub_to_gm_pad(partials[0, work * 512:work * 512 + 512], partial, n_burst=1, burst_len_element=512, src_stride=0, dst_stride_element=0)
            barrier(Pipe.ALL)
        return (destination, partials)
    gate_backward.name = name
    return gate_backward

def make_gate_reduce(name, a_dtype, bias_dtype):
    heads_per_work = 16 if a_dtype == bf16 else 8

    @func
    def initialize(scratch, state: Tensor):
        zero = Tensor(DT.float, [1, 64], Position.UB)
        dup(zero, 0.0, count=64)
        for part in unroll(8):
            _copy64(scratch, state[0, part * 64:part * 64 + 64], zero)
        barrier(Pipe.ALL)

    @func
    def accumulate(scratch, partials: Tensor, state: Tensor, rows: Var):
        ah = Tensor(DT.float, [1, 64], Position.UB)
        al = Tensor(DT.float, [1, 64], Position.UB)
        bh = Tensor(DT.float, [1, 64], Position.UB)
        bl = Tensor(DT.float, [1, 64], Position.UB)
        for part in unroll(2):
            for half in unroll(2):
                _copy64(scratch, ah, state[0, part * 256 + half * 64:part * 256 + half * 64 + 64])
                _copy64(scratch, al, state[0, part * 256 + 128 + half * 64:part * 256 + 128 + half * 64 + 64])
                for row in range(rows):
                    _copy64(scratch, bh, partials[0, row * 512 + part * 256 + half * 64:row * 512 + part * 256 + half * 64 + 64])
                    _copy64(scratch, bl, partials[0, row * 512 + part * 256 + 128 + half * 64:row * 512 + part * 256 + 128 + half * 64 + 64])
                    _dd_add(scratch, ah, al, ah, al, bh, bl)
                _copy64(scratch, state[0, part * 256 + half * 64:part * 256 + half * 64 + 64], ah)
                _copy64(scratch, state[0, part * 256 + 128 + half * 64:part * 256 + 128 + half * 64 + 64], al)
        barrier(Pipe.ALL)

    @func
    def finish(scratch, state: Tensor, avec: Tensor, aout: Tensor, bout: Tensor, slot: Var, offsets: Tensor):
        ah = Tensor(DT.float, [1, 64], Position.UB)
        al = Tensor(DT.float, [1, 64], Position.UB)
        bh = Tensor(DT.float, [1, 64], Position.UB)
        bl = Tensor(DT.float, [1, 64], Position.UB)
        result = Tensor(DT.float, [1, 64], Position.UB)
        decay = Tensor(DT.float, [1, 64], Position.UB)
        zero = Tensor(DT.float, [1, 64], Position.UB)
        narrow = Tensor(DT.bfloat16, [1, 128], Position.UB)
        dup(zero, 0.0, count=64)
        _load_one_f32(scratch, decay, avec[0, slot:slot + 1])
        exp(decay, decay, count=64)
        muls(decay, decay, -1.0, count=64)
        _copy64(scratch, ah, state[0, 0:64])
        _copy64(scratch, al, state[0, 128:192])
        _copy64(scratch, bh, state[0, 64:128])
        _copy64(scratch, bl, state[0, 192:256])
        _dd_add(scratch, ah, al, ah, al, bh, bl)
        for level in unroll(6):
            _gather64(scratch, bh, ah, offsets[5 - level:6 - level, :])
            _gather64(scratch, bl, al, offsets[5 - level:6 - level, :])
            _dd_add(scratch, ah, al, ah, al, bh, bl)
        _dd_mul(scratch, ah, al, ah, al, decay, zero)
        add(result, ah, al, count=64)
        _store_one_f32(scratch, aout[0, slot:slot + 1], result)
        for half in unroll(2):
            _copy64(scratch, ah, state[0, 256 + half * 64:320 + half * 64])
            _copy64(scratch, al, state[0, 384 + half * 64:448 + half * 64])
            _dd_mul(scratch, ah, al, ah, al, decay, zero)
            add(result, ah, al, count=64)
            if bias_dtype == bf16:
                cast(narrow, result, count=64, round_mode=RoundMode.TO_EVEN)
                ub_to_ub(bout[0, half * 64:half * 64 + 64], narrow, n_burst=1, burst_len=4, src_stride=0, dst_stride=0)
            else:
                _copy64(scratch, bout[0, half * 64:half * 64 + 64], result)
        barrier(Pipe.ALL)

    @kernel(mode='vec')
    def gate_reduce(partials: GM[f32, (1, 'P')], alog: GM[a_dtype, (1, 'HV')], da: GM[a_dtype, (1, 'HV')], db: GM[bias_dtype, (1, 'Channels')], P: i32, HV: i32, Channels: i32, Chunks: i32):
        scratch = Tensor(f32, [70, 64], Position.UB)
        incoming = Tensor(DT.float, [1, 4096], Position.UB)
        state = Tensor(DT.float, [1, 512], Position.UB)
        offsets = Tensor(u32, [6, 64], Position.UB)
        avec_raw = Tensor(a_dtype, [1, 128], Position.UB)
        avec = Tensor(f32, [1, 64], Position.UB)
        aout = Tensor(f32, [1, 64], Position.UB)
        aout_typed = Tensor(a_dtype, [1, 128], Position.UB)
        bout = Tensor(bias_dtype, [1, 128], Position.UB)
        works = CeilDiv(HV, heads_per_work)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        _initialize_offsets(scratch, offsets)
        for work in range(begin, end):
            head_begin = Var(work * heads_per_work)
            heads = Min(heads_per_work, HV - head_begin)
            barrier(Pipe.ALL)
            dup(avec_raw.reinterpret(u32), 0, count=64 if a_dtype == bf16 else 128)
            barrier(Pipe.ALL)
            gm_to_ub_pad(avec_raw, alog[0, head_begin:head_begin + heads], n_burst=1, burst_len_element=heads, src_stride_element=0, dst_stride=0, pad=0.0)
            barrier(Pipe.ALL)
            if a_dtype == bf16:
                cast(avec, avec_raw, count=64, round_mode=RoundMode.NONE)
            else:
                _copy64(scratch, avec, avec_raw)
            barrier(Pipe.ALL)
            dup(aout, 0.0, count=64)
            for slot in range(heads):
                head = Var(head_begin + slot)
                initialize(scratch, state)
                for tile in range(CeilDiv(Chunks, 8)):
                    start = Var(tile * 8)
                    rows = Min(8, Chunks - start)
                    offset = Var((head * Chunks + start) * 512)
                    count = Var(rows * 512)
                    barrier(Pipe.ALL)
                    gm_to_ub_pad(incoming, partials[0, offset:offset + count], n_burst=1, burst_len_element=count, src_stride_element=0, dst_stride=0)
                    barrier(Pipe.ALL)
                    accumulate(scratch, incoming, state, Var(rows))
                finish(scratch, state, avec, aout, bout, Var(slot), offsets)
                barrier(Pipe.ALL)
                ub_to_gm_pad(db[0, head * 128:head * 128 + 128], bout, n_burst=1, burst_len_element=128, src_stride=0, dst_stride_element=0)
                barrier(Pipe.ALL)
            barrier(Pipe.ALL)
            if a_dtype == bf16:
                cast(aout_typed, aout, count=64, round_mode=RoundMode.TO_EVEN)
            else:
                _copy64(scratch, aout_typed, aout)
            barrier(Pipe.ALL)
            ub_to_gm_pad(da[0, head_begin:head_begin + heads], aout_typed, n_burst=1, burst_len_element=heads, src_stride=0, dst_stride_element=0)
            barrier(Pipe.ALL)
        return (da, db)
    gate_reduce.name = name
    return gate_reduce

def make_beta_backward(name, raw_dtype):

    @func
    def differentiate(scratch, source: Tensor, probability: Tensor, sensitivity: Tensor, destination: Tensor, count: Var, exp_coefficients: Tensor):
        s = Tensor(DT.float, [1, 64], Position.UB)
        x = Tensor(DT.float, [1, 64], Position.UB)
        gy = Tensor(DT.float, [1, 64], Position.UB)
        eh = Tensor(DT.float, [1, 64], Position.UB)
        el = Tensor(DT.float, [1, 64], Position.UB)
        dh = Tensor(DT.float, [1, 64], Position.UB)
        dl = Tensor(DT.float, [1, 64], Position.UB)
        nh = Tensor(DT.float, [1, 64], Position.UB)
        nl = Tensor(DT.float, [1, 64], Position.UB)
        result = Tensor(DT.float, [1, 64], Position.UB)
        one = Tensor(DT.float, [1, 64], Position.UB)
        zero = Tensor(DT.float, [1, 64], Position.UB)
        saturated_zero = Tensor(u8, [1, 32], Position.UB)
        saturated_one = Tensor(u8, [1, 32], Position.UB)
        dup(one, 1.0, count=64)
        dup(zero, 0.0, count=64)
        packed = Tensor(DT.bfloat16, [1, 128], Position.UB)
        for index in range(count / 64):
            if raw_dtype == bf16:
                cast(x, source[0, index * 64:index * 64 + 64], count=64, round_mode=RoundMode.NONE)
            else:
                _copy64(scratch, x, source[0, index * 64:index * 64 + 64])
            _copy64(scratch, s, probability[0, index * 64:index * 64 + 64])
            _copy64(scratch, gy, sensitivity[0, index * 64:index * 64 + 64])
            compare_scalar(saturated_zero, s, 0.0, CompareMode.EQ, repeat=1)
            compare_scalar(saturated_one, s, 1.0, CompareMode.EQ, repeat=1)
            abs(x, x, count=64)
            muls(x, x, -1.0, count=64)
            _dd_exp_negative(scratch, eh, el, x, zero, exp_coefficients)
            _dd_add(scratch, dh, dl, one, zero, eh, el)
            _dd_mul(scratch, dh, dl, dh, dl, dh, dl)
            _dd_mul(scratch, nh, nl, gy, zero, eh, el)
            _dd_div(scratch, nh, nl, nh, nl, dh, dl)
            add(result, nh, nl, count=64)
            _select64(scratch, result, zero, result, saturated_zero)
            _select64(scratch, result, zero, result, saturated_one)
            if raw_dtype == bf16:
                cast(packed, result, count=64, round_mode=RoundMode.TO_EVEN)
                ub_to_ub(destination[0, index * 64:index * 64 + 64], packed, n_burst=1, burst_len=4, src_stride=0, dst_stride=0)
            else:
                _copy64(scratch, destination[0, index * 64:index * 64 + 64], result)
        barrier(Pipe.ALL)

    @kernel(mode='vec')
    def beta_backward(source: GM[raw_dtype, (1, 'N')], probability: GM[f32, (1, 'N')], sensitivity: GM[f32, (1, 'N')], destination: GM[raw_dtype, (1, 'N')], N: i32):
        scratch = Tensor(f32, [70, 64], Position.UB)
        source_ub = Tensor(raw_dtype, [1, 4096], Position.UB)
        probability_ub = Tensor(DT.float, [1, 4096], Position.UB)
        gradient = Tensor(DT.float, [1, 4096], Position.UB)
        outgoing = Tensor(raw_dtype, [1, 4096], Position.UB)
        exp_coefficients = Tensor(f32, [14, 8], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        _initialize_exp_coefficients(scratch, exp_coefficients)
        for work in range(begin, end):
            offset = Var(work * 4096)
            count = Min(4096, N - offset)
            padded = Align64(count)
            dup(source_ub.reinterpret(u32), 0, count=2048 if raw_dtype == bf16 else 4096)
            barrier(Pipe.ALL)
            gm_to_ub_pad(source_ub, source[0, offset:offset + count], n_burst=1, burst_len_element=count, src_stride_element=0, dst_stride=0)
            barrier(Pipe.ALL)
            dup(probability_ub, 0.0, count=4096)
            barrier(Pipe.ALL)
            gm_to_ub_pad(probability_ub, probability[0, offset:offset + count], n_burst=1, burst_len_element=count, src_stride_element=0, dst_stride=0)
            barrier(Pipe.ALL)
            dup(gradient, 0.0, count=4096)
            barrier(Pipe.ALL)
            gm_to_ub_pad(gradient, sensitivity[0, offset:offset + count], n_burst=1, burst_len_element=count, src_stride_element=0, dst_stride=0)
            barrier(Pipe.ALL)
            differentiate(scratch, source_ub, probability_ub, gradient, outgoing, Var(padded), exp_coefficients)
            barrier(Pipe.ALL)
            ub_to_gm_pad(destination[0, offset:offset + count], outgoing, n_burst=1, burst_len_element=count, src_stride=0, dst_stride_element=0)
            barrier(Pipe.ALL)
        return destination
    beta_backward.name = name
    return beta_backward
