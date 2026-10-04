use achainsaw_ir::ast::{
    BinaryOp, CastOp, Constant, Function, Instruction, Terminator, UnaryOp, VBinOp, VCmpOp,
    VectorReduceOp,
};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{
    AbiParam, Block as ClifBlock, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind,
    Value as ClifValue,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Module as ClifModule};
use std::collections::HashMap;

/// Width of the vectors Cranelift generates code for.
pub const VECTOR_PART_BITS: u32 = 128;

/// Canonical Cranelift type of one 128-bit vector part. AIR vectors are untyped, so parts
/// are kept as F32X4 and bitcast to the lane type each op names (a no-op on x86 and ARM).
const VEC_PART: types::Type = types::F32X4;

/// Cranelift values making up one AIR register: one for scalars, `v128`, and `vx` (which is
/// 128 bits on this backend), two for `v256`, four for `v512`.
type Parts = Vec<ClifValue>;
type Values = HashMap<String, (Parts, Type)>;

/// Number of 128-bit parts an AIR value occupies.
pub fn part_count(ty: Type) -> usize {
    match ty {
        Type::V256 => 2,
        Type::V512 => 4,
        _ => 1,
    }
}

/// Cranelift type of an AIR scalar, or of each 128-bit part of an AIR vector.
pub fn to_clif_type(ty: Type) -> types::Type {
    match ty {
        Type::I8 => types::I8,
        Type::I16 => types::I16,
        Type::I32 => types::I32,
        Type::I64 => types::I64,
        Type::F32 => types::F32,
        Type::F64 => types::F64,
        // Storage-only half types are carried as their raw bits.
        Type::F16 | Type::BF16 => types::I16,
        Type::Ptr => types::I64,
        Type::V128 | Type::V256 | Type::V512 | Type::Vx => VEC_PART,
    }
}

/// Appends the ABI parameters for an AIR type (one per part) to a signature list.
pub fn push_abi_params(params: &mut Vec<AbiParam>, ty: Type) {
    for _ in 0..part_count(ty) {
        params.push(AbiParam::new(to_clif_type(ty)));
    }
}

/// Cranelift vector type of one 128-bit part viewed with `lane` lanes.
fn lane_vec_type(lane: Type) -> types::Type {
    match lane {
        Type::I8 => types::I8X16,
        Type::I16 => types::I16X8,
        Type::I32 => types::I32X4,
        Type::I64 | Type::Ptr => types::I64X2,
        Type::F64 => types::F64X2,
        _ => types::F32X4,
    }
}

fn as_lanes(builder: &mut FunctionBuilder, part: ClifValue, lane: Type) -> ClifValue {
    let ty = lane_vec_type(lane);
    if builder.func.dfg.value_type(part) == ty {
        part
    } else {
        builder.ins().bitcast(ty, bitcast_flags(), part)
    }
}

fn to_part(builder: &mut FunctionBuilder, v: ClifValue) -> ClifValue {
    if builder.func.dfg.value_type(v) == VEC_PART {
        v
    } else {
        builder.ins().bitcast(VEC_PART, bitcast_flags(), v)
    }
}

/// Memory flags for vector and bulk (`ldm`/`stm`/`mm`) accesses. Unlike `trusted()`, these
/// do not claim alignment: AIR pointers into user buffers can be arbitrarily aligned, and
/// an aligned load may be folded into an SSE instruction that faults on unaligned addresses.
fn user_mem_flags() -> MemFlagsData {
    MemFlagsData::new().with_notrap()
}

fn c32(builder: &mut FunctionBuilder, v: u32) -> ClifValue {
    builder.ins().iconst(types::I32, v as i64)
}

fn c64(builder: &mut FunctionBuilder, v: i64) -> ClifValue {
    builder.ins().iconst(types::I64, v)
}

/// Integer type with the same width as `lane` (for copying lanes as raw bits).
fn lane_int_type(lane: Type) -> types::Type {
    match lane.byte_size() {
        1 => types::I8,
        2 => types::I16,
        4 => types::I32,
        _ => types::I64,
    }
}

/// binary16 bits (I16) -> f32. Exact; NaN payloads are preserved.
fn f16_to_f32(builder: &mut FunctionBuilder, h: ClifValue) -> ClifValue {
    let x = builder.ins().uextend(types::I32, h);
    let k8000 = c32(builder, 0x8000);
    let sign = builder.ins().band(x, k8000);
    let sign = builder.ins().ishl_imm_u(sign, 16);
    let exp = builder.ins().ushr_imm_u(x, 10);
    let k1f = c32(builder, 0x1f);
    let exp = builder.ins().band(exp, k1f);
    let k3ff = c32(builder, 0x3ff);
    let mant = builder.ins().band(x, k3ff);
    let mant13 = builder.ins().ishl_imm_u(mant, 13);
    // Normal: rebias the exponent from 15 to 127.
    let k112 = c32(builder, 112);
    let exp32 = builder.ins().iadd(exp, k112);
    let exp32 = builder.ins().ishl_imm_u(exp32, 23);
    let normal = builder.ins().bor(exp32, mant13);
    // Inf/NaN.
    let kinf = c32(builder, 0x7f80_0000);
    let infnan = builder.ins().bor(mant13, kinf);
    // Zero/subnormal: mant * 2^-24, exact in f32.
    let mant_f = builder.ins().fcvt_from_uint(types::F32, mant);
    let scale = builder.ins().f32const(f32::from_bits(0x3380_0000)); // 2^-24
    let sub_f = builder.ins().fmul(mant_f, scale);
    let sub = builder.ins().bitcast(types::I32, bitcast_flags(), sub_f);
    let is_sub = builder.ins().icmp_imm_u(IntCC::Equal, exp, 0);
    let is_max = builder.ins().icmp_imm_u(IntCC::Equal, exp, 31);
    let mag = builder.ins().select(is_max, infnan, normal);
    let mag = builder.ins().select(is_sub, sub, mag);
    let bits = builder.ins().bor(mag, sign);
    builder.ins().bitcast(types::F32, bitcast_flags(), bits)
}

/// f32 -> binary16 bits (I16), rounding to nearest-even (F. Giesen's
/// `float_to_half_fast3_rtne`). NaNs become the canonical quiet NaN 0x7e00.
fn f32_to_f16(builder: &mut FunctionBuilder, f: ClifValue) -> ClifValue {
    let x = builder.ins().bitcast(types::I32, bitcast_flags(), f);
    let ksign = c32(builder, 0x8000_0000);
    let sign = builder.ins().band(x, ksign);
    let a = builder.ins().bxor(x, sign);
    // |f| >= 65536.0 (2^16): Inf, or NaN when above the f32 Inf pattern.
    let is_big = builder
        .ins()
        .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, a, 0x4780_0000);
    let is_nan = builder
        .ins()
        .icmp_imm_u(IntCC::UnsignedGreaterThan, a, 0x7f80_0000);
    let qnan = c32(builder, 0x7e00);
    let inf = c32(builder, 0x7c00);
    let big = builder.ins().select(is_nan, qnan, inf);
    // Result is subnormal or zero: let the FPU round by adding 0.5 (denorm magic).
    let is_sub = builder
        .ins()
        .icmp_imm_u(IntCC::UnsignedLessThan, a, 0x3880_0000);
    let af = builder.ins().bitcast(types::F32, bitcast_flags(), a);
    let magic = builder.ins().f32const(f32::from_bits(0x3f00_0000));
    let sum = builder.ins().fadd(af, magic);
    let sum_bits = builder.ins().bitcast(types::I32, bitcast_flags(), sum);
    let kmagic = c32(builder, 0x3f00_0000);
    let sub = builder.ins().isub(sum_bits, kmagic);
    // Normal: rebias and round to nearest-even on the 13 dropped mantissa bits.
    let odd = builder.ins().ushr_imm_u(a, 13);
    let k1 = c32(builder, 1);
    let odd = builder.ins().band(odd, k1);
    let bias = c32(builder, (((15i32 - 127) << 23) + 0xfff) as u32);
    let t = builder.ins().iadd(a, bias);
    let t = builder.ins().iadd(t, odd);
    let normal = builder.ins().ushr_imm_u(t, 13);
    let mag = builder.ins().select(is_sub, sub, normal);
    let mag = builder.ins().select(is_big, big, mag);
    let sign16 = builder.ins().ushr_imm_u(sign, 16);
    let bits = builder.ins().bor(mag, sign16);
    builder.ins().ireduce(types::I16, bits)
}

/// bfloat16 bits (I16) -> f32. Exact.
fn bf16_to_f32(builder: &mut FunctionBuilder, h: ClifValue) -> ClifValue {
    let x = builder.ins().uextend(types::I32, h);
    let x = builder.ins().ishl_imm_u(x, 16);
    builder.ins().bitcast(types::F32, bitcast_flags(), x)
}

/// f32 -> bfloat16 bits (I16), rounding to nearest-even; NaNs are quieted.
fn f32_to_bf16(builder: &mut FunctionBuilder, f: ClifValue) -> ClifValue {
    let x = builder.ins().bitcast(types::I32, bitcast_flags(), f);
    let hi = builder.ins().ushr_imm_u(x, 16);
    let k1 = c32(builder, 1);
    let lsb = builder.ins().band(hi, k1);
    let k7fff = c32(builder, 0x7fff);
    let r = builder.ins().iadd(x, k7fff);
    let r = builder.ins().iadd(r, lsb);
    let rounded = builder.ins().ushr_imm_u(r, 16);
    let k40 = c32(builder, 0x40);
    let quiet = builder.ins().bor(hi, k40);
    let kabs = c32(builder, 0x7fff_ffff);
    let abs = builder.ins().band(x, kabs);
    let is_nan = builder
        .ins()
        .icmp_imm_u(IntCC::UnsignedGreaterThan, abs, 0x7f80_0000);
    let bits = builder.ins().select(is_nan, quiet, rounded);
    builder.ins().ireduce(types::I16, bits)
}

/// Converts a loaded `mm` element (`dtype`) to the accumulator type (f32, or i32 for i8).
fn mm_widen(builder: &mut FunctionBuilder, dtype: Type, v: ClifValue) -> ClifValue {
    match dtype {
        Type::BF16 => bf16_to_f32(builder, v),
        Type::F16 => f16_to_f32(builder, v),
        Type::I8 => builder.ins().sextend(types::I32, v),
        _ => v,
    }
}

/// Bytes a plain `ld`/`st` of `ty` touches.
fn access_bytes(ty: Type) -> usize {
    if ty.is_vector() {
        part_count(ty) * 16
    } else {
        ty.byte_size()
    }
}

/// Number of `lane` lanes in a vector of type `ty` on this backend.
fn lanes_of(ty: Type, lane: Type) -> i64 {
    (part_count(ty) * 16 / lane.byte_size()) as i64
}

/// `min(max(count, 0), lanes)`.
fn clamp_count(builder: &mut FunctionBuilder, count: ClifValue, lanes: i64) -> ClifValue {
    let zero = c64(builder, 0);
    let lanes_v = c64(builder, lanes);
    let n = builder.ins().smax(count, zero);
    builder.ins().smin(n, lanes_v)
}

/// Emits `for i in 0..n { *(dst + i*size) = *(src + i*size) }` with `size`-byte lanes,
/// leaving the builder in the block after the loop.
fn emit_lane_copy(
    builder: &mut FunctionBuilder,
    src: ClifValue,
    dst: ClifValue,
    n: ClifValue,
    lane: Type,
) {
    let header = builder.create_block();
    builder.append_block_param(header, types::I64);
    let body = builder.create_block();
    let exit = builder.create_block();
    let zero = c64(builder, 0);
    builder.ins().jump(header, &[BlockArg::Value(zero)]);

    builder.switch_to_block(header);
    let i = builder.block_params(header)[0];
    let more = builder.ins().icmp(IntCC::SignedLessThan, i, n);
    builder.ins().brif(more, body, &[], exit, &[]);

    builder.switch_to_block(body);
    let size = c64(builder, lane.byte_size() as i64);
    let off = builder.ins().imul(i, size);
    let from = builder.ins().iadd(src, off);
    let to = builder.ins().iadd(dst, off);
    let v = builder
        .ins()
        .load(lane_int_type(lane), user_mem_flags(), from, 0);
    builder.ins().store(user_mem_flags(), v, to, 0);
    let one = c64(builder, 1);
    let next = builder.ins().iadd(i, one);
    builder.ins().jump(header, &[BlockArg::Value(next)]);

    builder.switch_to_block(exit);
}

/// Stack slot big enough for a vector of type `ty`, and its address.
fn vector_scratch(builder: &mut FunctionBuilder, ty: Type) -> ClifValue {
    let bytes = (part_count(ty) * 16) as u32;
    let slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        bytes,
        4, // 16-byte alignment
    ));
    builder.ins().stack_addr(types::I64, slot, 0)
}

fn load_parts(builder: &mut FunctionBuilder, ptr: ClifValue, ty: Type) -> Parts {
    (0..part_count(ty))
        .map(|k| {
            builder
                .ins()
                .load(VEC_PART, user_mem_flags(), ptr, (k * 16) as i32)
        })
        .collect()
}

fn store_parts(builder: &mut FunctionBuilder, ptr: ClifValue, parts: &[ClifValue]) {
    for (k, part) in parts.iter().enumerate() {
        builder
            .ins()
            .store(user_mem_flags(), *part, ptr, (k * 16) as i32);
    }
}

/// `v = ldm p:ty, n:lane`. Full vectors use a plain load; partial ones copy the first
/// `n` lanes into a zeroed stack slot, so memory past them is never touched.
fn emit_masked_load(
    builder: &mut FunctionBuilder,
    guard: Option<Guard>,
    ptr: ClifValue,
    count: ClifValue,
    ty: Type,
    lane: Type,
) -> Parts {
    let lanes = lanes_of(ty, lane);
    let n = clamp_count(builder, count, lanes);
    if guard.is_some() {
        let bytes = builder.ins().imul_imm_s(n, lane.byte_size() as i64);
        emit_bounds_check(builder, guard, ptr, bytes);
    }
    let full_blk = builder.create_block();
    let part_blk = builder.create_block();
    let merge = builder.create_block();
    for _ in 0..part_count(ty) {
        builder.append_block_param(merge, VEC_PART);
    }
    let is_full = builder.ins().icmp_imm_u(IntCC::Equal, n, lanes);
    builder.ins().brif(is_full, full_blk, &[], part_blk, &[]);

    builder.switch_to_block(full_blk);
    let parts = load_parts(builder, ptr, ty);
    jump_with(builder, merge, &parts);

    builder.switch_to_block(part_blk);
    let scratch = vector_scratch(builder, ty);
    let zero_f = builder.ins().f32const(0.0);
    let zero = builder.ins().splat(VEC_PART, zero_f);
    store_parts(builder, scratch, &vec![zero; part_count(ty)]);
    emit_lane_copy(builder, ptr, scratch, n, lane);
    let parts = load_parts(builder, scratch, ty);
    jump_with(builder, merge, &parts);

    builder.switch_to_block(merge);
    builder.block_params(merge).to_vec()
}

/// `stm p, v, n:lane`: writes only the first `n` lanes.
fn emit_masked_store(
    builder: &mut FunctionBuilder,
    guard: Option<Guard>,
    ptr: ClifValue,
    parts: &[ClifValue],
    count: ClifValue,
    ty: Type,
    lane: Type,
) {
    let lanes = lanes_of(ty, lane);
    let n = clamp_count(builder, count, lanes);
    if guard.is_some() {
        let bytes = builder.ins().imul_imm_s(n, lane.byte_size() as i64);
        emit_bounds_check(builder, guard, ptr, bytes);
    }
    let full_blk = builder.create_block();
    let part_blk = builder.create_block();
    let done = builder.create_block();
    let is_full = builder.ins().icmp_imm_u(IntCC::Equal, n, lanes);
    builder.ins().brif(is_full, full_blk, &[], part_blk, &[]);

    builder.switch_to_block(full_blk);
    store_parts(builder, ptr, parts);
    builder.ins().jump(done, &[]);

    builder.switch_to_block(part_blk);
    let scratch = vector_scratch(builder, ty);
    store_parts(builder, scratch, parts);
    emit_lane_copy(builder, scratch, ptr, n, lane);
    builder.ins().jump(done, &[]);

    builder.switch_to_block(done);
}

fn jump_with(builder: &mut FunctionBuilder, target: ClifBlock, vals: &[ClifValue]) {
    let args: Vec<BlockArg> = vals.iter().copied().map(BlockArg::Value).collect();
    builder.ins().jump(target, &args);
}

/// Sandbox arena bounds, and the block that reports a violation `(addr, size)` and unwinds.
#[derive(Clone, Copy)]
struct Guard {
    base: i64,
    len: i64,
    fault: ClifBlock,
}

/// Branches to the fault block unless `[ptr, ptr + size)` lies inside the arena or
/// `size` is 0, leaving the builder in the passing block. `size` must not exceed
/// `MIN_SANDBOX_ARENA_BYTES`.
fn emit_bounds_check(
    builder: &mut FunctionBuilder,
    guard: Option<Guard>,
    ptr: ClifValue,
    size: ClifValue,
) {
    let Some(g) = guard else { return };
    let base = c64(builder, g.base);
    let off = builder.ins().isub(ptr, base);
    let len = c64(builder, g.len);
    let room = builder.ins().isub(len, size);
    let fits = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, off, room);
    let empty = builder.ins().icmp_imm_u(IntCC::Equal, size, 0);
    let ok = builder.ins().bor(fits, empty);
    let pass = builder.create_block();
    builder.ins().brif(
        ok,
        pass,
        &[],
        g.fault,
        &[BlockArg::Value(ptr), BlockArg::Value(size)],
    );
    builder.switch_to_block(pass);
}

fn emit_bounds_check_const(
    builder: &mut FunctionBuilder,
    guard: Option<Guard>,
    ptr: ClifValue,
    size: usize,
) {
    if guard.is_some() {
        let size = c64(builder, size as i64);
        emit_bounds_check(builder, guard, ptr, size);
    }
}

/// Pointer to element `row * cols + col` of a row-major matrix with `esize`-byte elements.
fn elem_ptr(
    builder: &mut FunctionBuilder,
    base: ClifValue,
    row: ClifValue,
    cols: ClifValue,
    col: ClifValue,
    esize: i64,
) -> ClifValue {
    let idx = builder.ins().imul(row, cols);
    let idx = builder.ins().iadd(idx, col);
    let size = c64(builder, esize);
    let off = builder.ins().imul(idx, size);
    builder.ins().iadd(base, off)
}

/// Element type loaded for an `mm` operand.
fn mm_elem_type(dtype: Type) -> types::Type {
    match dtype {
        Type::BF16 | Type::F16 => types::I16,
        Type::I8 => types::I8,
        _ => types::F32,
    }
}

fn scalar(values: &Values, name: &str) -> (ClifValue, Type) {
    let (parts, ty) = &values[name];
    (parts[0], *ty)
}

fn parts(values: &Values, name: &str) -> (Parts, Type) {
    values[name].clone()
}

/// All parts of the named registers, flattened in order (call and branch arguments).
fn flat_args(values: &Values, names: &[String]) -> Vec<ClifValue> {
    names
        .iter()
        .flat_map(|n| values[n].0.iter().copied())
        .collect()
}

/// Combines two lane values for a horizontal reduction.
fn reduce_pair(
    builder: &mut FunctionBuilder,
    op: VectorReduceOp,
    lane: Type,
    a: ClifValue,
    b: ClifValue,
) -> ClifValue {
    match (op, lane.is_float()) {
        (VectorReduceOp::Sum, true) => builder.ins().fadd(a, b),
        (VectorReduceOp::Sum, false) => builder.ins().iadd(a, b),
        (VectorReduceOp::Max, true) => builder.ins().fmax(a, b),
        (VectorReduceOp::Max, false) => builder.ins().smax(a, b),
        (VectorReduceOp::Min, true) => builder.ins().fmin(a, b),
        (VectorReduceOp::Min, false) => builder.ins().smin(a, b),
    }
}

/// Reduces values with the canonical recursive-halves tree:
/// `reduce(v) = op(reduce(lo half), reduce(hi half))`.
fn reduce_tree(
    builder: &mut FunctionBuilder,
    op: VectorReduceOp,
    lane: Type,
    vals: &[ClifValue],
) -> ClifValue {
    if vals.len() == 1 {
        return vals[0];
    }
    let (lo, hi) = vals.split_at(vals.len() / 2);
    let a = reduce_tree(builder, op, lane, lo);
    let b = reduce_tree(builder, op, lane, hi);
    reduce_pair(builder, op, lane, a, b)
}

fn vbinary_part(
    builder: &mut FunctionBuilder,
    op: VBinOp,
    lane: Type,
    l: ClifValue,
    r: ClifValue,
) -> ClifValue {
    if op.is_bitwise() {
        let a = as_lanes(builder, l, Type::I32);
        let b = as_lanes(builder, r, Type::I32);
        let res = match op {
            VBinOp::And => builder.ins().band(a, b),
            VBinOp::Or => builder.ins().bor(a, b),
            _ => builder.ins().bxor(a, b),
        };
        return to_part(builder, res);
    }
    let a = as_lanes(builder, l, lane);
    let b = as_lanes(builder, r, lane);
    let res = match (op, lane.is_float()) {
        (VBinOp::Add, true) => builder.ins().fadd(a, b),
        (VBinOp::Add, false) => builder.ins().iadd(a, b),
        (VBinOp::Sub, true) => builder.ins().fsub(a, b),
        (VBinOp::Sub, false) => builder.ins().isub(a, b),
        (VBinOp::Mul, true) => builder.ins().fmul(a, b),
        (VBinOp::Mul, false) => builder.ins().imul(a, b),
        (VBinOp::Min, true) => builder.ins().fmin(a, b),
        (VBinOp::Min, false) => builder.ins().smin(a, b),
        (VBinOp::Max, true) => builder.ins().fmax(a, b),
        (VBinOp::Max, false) => builder.ins().smax(a, b),
        // The validator only allows float lanes for vdiv.
        (VBinOp::Div, _) => builder.ins().fdiv(a, b),
        (VBinOp::And | VBinOp::Or | VBinOp::Xor, _) => unreachable!(),
    };
    to_part(builder, res)
}

fn vcmp_part(
    builder: &mut FunctionBuilder,
    op: VCmpOp,
    lane: Type,
    l: ClifValue,
    r: ClifValue,
) -> ClifValue {
    let a = as_lanes(builder, l, lane);
    let b = as_lanes(builder, r, lane);
    let res = if lane.is_float() {
        let cc = match op {
            VCmpOp::Eq => FloatCC::Equal,
            VCmpOp::Ne => FloatCC::NotEqual,
            VCmpOp::Lt => FloatCC::LessThan,
            VCmpOp::Gt => FloatCC::GreaterThan,
            VCmpOp::Le => FloatCC::LessThanOrEqual,
            VCmpOp::Ge => FloatCC::GreaterThanOrEqual,
        };
        builder.ins().fcmp(cc, a, b)
    } else {
        let cc = match op {
            VCmpOp::Eq => IntCC::Equal,
            VCmpOp::Ne => IntCC::NotEqual,
            VCmpOp::Lt => IntCC::SignedLessThan,
            VCmpOp::Gt => IntCC::SignedGreaterThan,
            VCmpOp::Le => IntCC::SignedLessThanOrEqual,
            VCmpOp::Ge => IntCC::SignedGreaterThanOrEqual,
        };
        builder.ins().icmp(cc, a, b)
    };
    to_part(builder, res)
}

fn bitcast_flags() -> MemFlagsData {
    let mut flags = MemFlagsData::new();
    flags.set_endianness(cranelift_codegen::ir::Endianness::Little);
    flags
}

/// `mm pc, pa, pb, m, n, k:dtype`: `C[m x n] += A[m x k] * B[k x n]` as an inline loop
/// nest accumulating in f32 (i32 for i8) over `k` in order. Non-positive dimensions are a
/// no-op. With `bounds`, `rt_sandbox_check_mm` validates all three matrices first. With
/// `fuel`, charges one unit per 1024 multiply-adds before starting. Either one failing
/// branches to its block and leaves C untouched.
fn emit_matmul(
    builder: &mut FunctionBuilder,
    regs: &[ClifValue],
    dtype: Type,
    bounds: Option<(cranelift_codegen::ir::FuncRef, ClifBlock)>,
    fuel: Option<(cranelift_codegen::ir::FuncRef, ClifBlock)>,
) {
    let (pc, pa, pb, m, n, k) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    let is_int = dtype == Type::I8;
    let acc_ty = if is_int { types::I32 } else { types::F32 };
    let esize = dtype.byte_size() as i64;
    let elem_ty = mm_elem_type(dtype);

    let start = builder.create_block();
    let i_hdr = builder.create_block();
    let j_hdr = builder.create_block();
    let k_init = builder.create_block();
    let k_hdr = builder.create_block();
    let k_body = builder.create_block();
    let k_done = builder.create_block();
    let i_next = builder.create_block();
    let done = builder.create_block();
    builder.append_block_param(i_hdr, types::I64); // i
    builder.append_block_param(j_hdr, types::I64); // i
    builder.append_block_param(j_hdr, types::I64); // j
    for _ in 0..3 {
        builder.append_block_param(k_hdr, types::I64); // i, j, kk
    }
    builder.append_block_param(k_hdr, acc_ty); // acc

    let m_pos = builder.ins().icmp_imm_s(IntCC::SignedGreaterThan, m, 0);
    let n_pos = builder.ins().icmp_imm_s(IntCC::SignedGreaterThan, n, 0);
    let k_pos = builder.ins().icmp_imm_s(IntCC::SignedGreaterThan, k, 0);
    let mn = builder.ins().band(m_pos, n_pos);
    let any = builder.ins().band(mn, k_pos);
    builder.ins().brif(any, start, &[], done, &[]);

    builder.switch_to_block(start);
    let zero = c64(builder, 0);
    if let Some((check, trap)) = bounds {
        let es = c64(builder, esize);
        let call = builder.ins().call(check, &[pc, pa, pb, m, n, k, es]);
        let bad = builder.inst_results(call)[0];
        let checked = builder.create_block();
        builder.ins().brif(bad, trap, &[], checked, &[]);
        builder.switch_to_block(checked);
    }
    if let Some((consume, trap)) = fuel {
        // units = m*n*k / 1024 + 1, computed in f64 and saturated so huge shapes cannot wrap.
        let mf = builder.ins().fcvt_from_sint(types::F64, m);
        let nf = builder.ins().fcvt_from_sint(types::F64, n);
        let kf = builder.ins().fcvt_from_sint(types::F64, k);
        let work = builder.ins().fmul(mf, nf);
        let work = builder.ins().fmul(work, kf);
        let per_unit = builder.ins().f64const(1024.0);
        let units_f = builder.ins().fdiv(work, per_unit);
        let units = builder.ins().fcvt_to_sint_sat(types::I64, units_f);
        let units = builder.ins().iadd_imm_s(units, 1);
        let call = builder.ins().call(consume, &[units]);
        let exhausted = builder.inst_results(call)[0];
        builder
            .ins()
            .brif(exhausted, trap, &[], i_hdr, &[BlockArg::Value(zero)]);
    } else {
        builder.ins().jump(i_hdr, &[BlockArg::Value(zero)]);
    }

    builder.switch_to_block(i_hdr);
    let i = builder.block_params(i_hdr)[0];
    let more_i = builder.ins().icmp(IntCC::SignedLessThan, i, m);
    builder.ins().brif(
        more_i,
        j_hdr,
        &[BlockArg::Value(i), BlockArg::Value(zero)],
        done,
        &[],
    );

    builder.switch_to_block(j_hdr);
    let (ji, j) = (
        builder.block_params(j_hdr)[0],
        builder.block_params(j_hdr)[1],
    );
    let more_j = builder.ins().icmp(IntCC::SignedLessThan, j, n);
    builder.ins().brif(more_j, k_init, &[], i_next, &[]);

    builder.switch_to_block(i_next);
    let i2 = builder.ins().iadd_imm_s(ji, 1);
    builder.ins().jump(i_hdr, &[BlockArg::Value(i2)]);

    builder.switch_to_block(k_init);
    let c_ptr = elem_ptr(builder, pc, ji, n, j, 4);
    let acc0 = builder.ins().load(acc_ty, user_mem_flags(), c_ptr, 0);
    jump_with(builder, k_hdr, &[ji, j, zero, acc0]);

    builder.switch_to_block(k_hdr);
    let p = builder.block_params(k_hdr).to_vec();
    let (ki, kj, kk, acc) = (p[0], p[1], p[2], p[3]);
    let more_k = builder.ins().icmp(IntCC::SignedLessThan, kk, k);
    builder.ins().brif(more_k, k_body, &[], k_done, &[]);

    builder.switch_to_block(k_body);
    let a_ptr = elem_ptr(builder, pa, ki, k, kk, esize);
    let b_ptr = elem_ptr(builder, pb, kk, n, kj, esize);
    let a_raw = builder.ins().load(elem_ty, user_mem_flags(), a_ptr, 0);
    let b_raw = builder.ins().load(elem_ty, user_mem_flags(), b_ptr, 0);
    let a = mm_widen(builder, dtype, a_raw);
    let b = mm_widen(builder, dtype, b_raw);
    let acc2 = if is_int {
        let prod = builder.ins().imul(a, b);
        builder.ins().iadd(acc, prod)
    } else {
        let prod = builder.ins().fmul(a, b);
        builder.ins().fadd(acc, prod)
    };
    let kk2 = builder.ins().iadd_imm_s(kk, 1);
    jump_with(builder, k_hdr, &[ki, kj, kk2, acc2]);

    builder.switch_to_block(k_done);
    let c_out = elem_ptr(builder, pc, ki, n, kj, 4);
    builder.ins().store(user_mem_flags(), acc, c_out, 0);
    let j2 = builder.ins().iadd_imm_s(kj, 1);
    jump_with(builder, j_hdr, &[ki, j2]);

    builder.switch_to_block(done);
}

/// Inline fuel check before a branch: decrements the engine's counter in place and, only
/// when it reaches zero, asks `rt_fuel_exhausted` whether to unwind (budget spent, or a
/// failure was recorded). Leaves the builder in the block where execution continues.
fn emit_fuel_check<M: ClifModule>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    (counter, exhausted): (i64, FuncId),
    trap_block: ClifBlock,
) {
    let addr = c64(builder, counter);
    let fuel = builder
        .ins()
        .load(types::I64, MemFlagsData::trusted(), addr, 0);
    let left = builder.ins().iadd_imm_s(fuel, -1);
    builder.ins().store(MemFlagsData::trusted(), left, addr, 0);
    let out = builder
        .ins()
        .icmp_imm_s(IntCC::SignedLessThanOrEqual, left, 0);
    let slow = builder.create_block();
    let cont = builder.create_block();
    builder.set_cold_block(slow);
    builder.ins().brif(out, slow, &[], cont, &[]);

    builder.switch_to_block(slow);
    let callee = module.declare_func_in_func(exhausted, builder.func);
    let call_inst = builder.ins().call(callee, &[]);
    let stop = builder.inst_results(call_inst)[0];
    builder.ins().brif(stop, trap_block, &[], cont, &[]);

    builder.switch_to_block(cont);
}

pub struct LowerConfig {
    /// Inline fuel check at every branch: the address of the engine's i64 fuel counter
    /// (decremented in place) and `rt_fuel_exhausted() -> i32`, called when it reaches zero.
    pub fuel_check: Option<(i64, FuncId)>,
    /// `rt_consume_fuel(units: i64) -> i32`, charged up front by bulk ops (`mm`).
    pub fuel_consume_func_id: Option<FuncId>,
    pub rt_malloc_id: FuncId,
    pub rt_free_id: FuncId,
    /// Bounds-check memory accesses and cap stack depth (JIT only; see `JitEngine::enable_sandbox`).
    pub sandbox: Option<SandboxConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct SandboxConfig {
    pub arena_base: i64,
    pub arena_len: i64,
    /// `rt_sandbox_fault(addr: i64, size: i64)`.
    pub fault_id: FuncId,
    /// `rt_stack_check() -> i32`, called on function entry.
    pub stack_check_id: FuncId,
    /// `rt_sandbox_check_mm(pc, pa, pb, m, n, k, esize) -> i32`.
    pub mm_check_id: FuncId,
}

pub fn lower_function<M: ClifModule>(
    module: &mut M,
    ctx: &mut cranelift_codegen::Context,
    builder_context: &mut FunctionBuilderContext,
    func: &Function,
    func_ids: &HashMap<String, FuncId>,
    func_returns: &HashMap<String, Option<Type>>,
    config: &LowerConfig,
) -> Result<()> {
    ctx.func
        .signature
        .clear(module.target_config().default_call_conv);
    for (_, p_ty) in &func.params {
        push_abi_params(&mut ctx.func.signature.params, *p_ty);
    }
    if let Some(r_ty) = func.ret_type {
        push_abi_params(&mut ctx.func.signature.returns, r_ty);
    }

    let mut builder = FunctionBuilder::new(&mut ctx.func, builder_context);

    // Create a synthetic entry block so entry never has loop back-edges
    let synth_entry = builder.create_block();
    builder.append_block_params_for_function_params(synth_entry);

    // Create blocks for each AIR block
    let mut clif_blocks = HashMap::new();
    for block in &func.blocks {
        let clif_block = builder.create_block();
        clif_blocks.insert(block.label.clone(), clif_block);

        for (_, p_ty) in &block.params {
            for _ in 0..part_count(*p_ty) {
                builder.append_block_param(clif_block, to_clif_type(*p_ty));
            }
        }
    }

    // Returns zeroes once a runtime check fails; `call_typed` then reports the status.
    let fuel_trap_block =
        (config.fuel_check.is_some() || config.sandbox.is_some()).then(|| builder.create_block());

    let guard = config.sandbox.map(|sb| {
        let fault = builder.create_block();
        builder.append_block_param(fault, types::I64); // addr
        builder.append_block_param(fault, types::I64); // size
        Guard {
            base: sb.arena_base,
            len: sb.arena_len,
            fault,
        }
    });

    // Values map: SSA register name -> (Cranelift parts, Type)
    let mut values: Values = HashMap::new();

    // Map function parameters in synthetic entry
    builder.switch_to_block(synth_entry);
    let entry_params = builder.block_params(synth_entry).to_vec();
    let mut next_param = 0;
    for (p_name, p_ty) in &func.params {
        let n = part_count(*p_ty);
        values.insert(
            p_name.clone(),
            (entry_params[next_param..next_param + n].to_vec(), *p_ty),
        );
        next_param += n;
    }

    // Jump from synthetic entry to function's first block
    let first_block_label = &func.blocks[0].label;
    let first_block = *clif_blocks.get(first_block_label).unwrap();
    if let (Some(sb), Some(trap_block)) = (config.sandbox, fuel_trap_block) {
        let callee = module.declare_func_in_func(sb.stack_check_id, builder.func);
        let call_inst = builder.ins().call(callee, &[]);
        let overflow = builder.inst_results(call_inst)[0];
        builder
            .ins()
            .brif(overflow, trap_block, &[], first_block, &[]);
    } else {
        builder.ins().jump(first_block, &[]);
    }

    // Translate each AIR block
    for block in &func.blocks {
        let clif_block = *clif_blocks.get(&block.label).unwrap();
        builder.switch_to_block(clif_block);

        // Bind block parameters to SSA registers
        let block_vals = builder.block_params(clif_block).to_vec();
        let mut next_val = 0;
        for (bp_name, bp_ty) in &block.params {
            let n = part_count(*bp_ty);
            values.insert(
                bp_name.clone(),
                (block_vals[next_val..next_val + n].to_vec(), *bp_ty),
            );
            next_val += n;
        }

        // Translate instructions
        for inst in &block.instructions {
            match inst {
                Instruction::AssignConst { dst, val, ty, .. } => {
                    let clif_ty = to_clif_type(*ty);
                    let v = match val {
                        Constant::Int(n) => builder.ins().iconst(clif_ty, *n),
                        Constant::Float(f) => match ty {
                            Type::F32 => builder.ins().f32const(*f as f32),
                            Type::F64 => builder.ins().f64const(*f),
                            _ => unreachable!(),
                        },
                    };
                    values.insert(dst.clone(), (vec![v], *ty));
                }
                Instruction::Binary {
                    op, dst, lhs, rhs, ..
                } => {
                    let (lhs_val, lhs_ty) = scalar(&values, lhs);
                    let (rhs_val, rhs_ty) = scalar(&values, rhs);

                    let (res_val, res_ty) = match lhs_ty {
                        Type::F32 | Type::F64 => match op {
                            BinaryOp::Add => (builder.ins().fadd(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Sub => (builder.ins().fsub(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Mul => (builder.ins().fmul(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Div => (builder.ins().fdiv(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Eq => {
                                let cmp = builder.ins().fcmp(FloatCC::Equal, lhs_val, rhs_val);
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Ne => {
                                let cmp = builder.ins().fcmp(FloatCC::NotEqual, lhs_val, rhs_val);
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Lt => {
                                let cmp = builder.ins().fcmp(FloatCC::LessThan, lhs_val, rhs_val);
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Gt => {
                                let cmp =
                                    builder.ins().fcmp(FloatCC::GreaterThan, lhs_val, rhs_val);
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Le => {
                                let cmp =
                                    builder
                                        .ins()
                                        .fcmp(FloatCC::LessThanOrEqual, lhs_val, rhs_val);
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Ge => {
                                let cmp = builder.ins().fcmp(
                                    FloatCC::GreaterThanOrEqual,
                                    lhs_val,
                                    rhs_val,
                                );
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Min => (builder.ins().fmin(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Max => (builder.ins().fmax(lhs_val, rhs_val), lhs_ty),
                            _ => return Err(anyhow!("Unsupported float op {:?}", op)),
                        },
                        _ => {
                            let default_res_ty = if lhs_ty == Type::Ptr || rhs_ty == Type::Ptr {
                                Type::Ptr
                            } else {
                                lhs_ty
                            };
                            let clif_ty = to_clif_type(lhs_ty);

                            match op {
                                BinaryOp::Add => {
                                    (builder.ins().iadd(lhs_val, rhs_val), default_res_ty)
                                }
                                BinaryOp::Sub => {
                                    (builder.ins().isub(lhs_val, rhs_val), default_res_ty)
                                }
                                BinaryOp::Mul => (builder.ins().imul(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Div | BinaryOp::Rem => {
                                    // Safe division and remainder: guard against division by zero and INT_MIN / -1
                                    let is_zero =
                                        builder.ins().icmp_imm_s(IntCC::Equal, rhs_val, 0);
                                    let is_neg_one =
                                        builder.ins().icmp_imm_s(IntCC::Equal, rhs_val, -1);
                                    let min_val = match lhs_ty {
                                        Type::I8 => -128i64,
                                        Type::I16 => -32768i64,
                                        Type::I32 => i32::MIN as i64,
                                        Type::I64 | Type::Ptr => i64::MIN,
                                        _ => 0i64,
                                    };
                                    let is_min =
                                        builder.ins().icmp_imm_s(IntCC::Equal, lhs_val, min_val);
                                    let is_ovf = builder.ins().band(is_neg_one, is_min);
                                    let is_bad = builder.ins().bor(is_zero, is_ovf);

                                    let safe_block = builder.create_block();
                                    let bad_block = builder.create_block();
                                    let merge_block = builder.create_block();
                                    builder.append_block_param(merge_block, clif_ty);

                                    builder.ins().brif(is_bad, bad_block, &[], safe_block, &[]);

                                    builder.switch_to_block(safe_block);
                                    let op_res = match op {
                                        BinaryOp::Div => builder.ins().sdiv(lhs_val, rhs_val),
                                        BinaryOp::Rem => builder.ins().srem(lhs_val, rhs_val),
                                        _ => unreachable!(),
                                    };
                                    builder.ins().jump(merge_block, &[BlockArg::Value(op_res)]);

                                    builder.switch_to_block(bad_block);
                                    let zero_res = builder.ins().iconst(clif_ty, 0);
                                    builder
                                        .ins()
                                        .jump(merge_block, &[BlockArg::Value(zero_res)]);

                                    builder.switch_to_block(merge_block);
                                    let res = builder.block_params(merge_block)[0];
                                    (res, lhs_ty)
                                }
                                BinaryOp::And => (builder.ins().band(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Or => (builder.ins().bor(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Xor => (builder.ins().bxor(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Shl => (builder.ins().ishl(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Shr => (builder.ins().sshr(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Eq => {
                                    let cmp = builder.ins().icmp(IntCC::Equal, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Ne => {
                                    let cmp = builder.ins().icmp(IntCC::NotEqual, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Lt => {
                                    let cmp =
                                        builder.ins().icmp(IntCC::SignedLessThan, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Gt => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::SignedGreaterThan,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Le => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::SignedLessThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Ge => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::SignedGreaterThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Min => (builder.ins().smin(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Max => (builder.ins().smax(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Umin => (builder.ins().umin(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Umax => (builder.ins().umax(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Ushr => (builder.ins().ushr(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Ult => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::UnsignedLessThan,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Ugt => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::UnsignedGreaterThan,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Ule => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::UnsignedLessThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Uge => {
                                    let cmp = builder.ins().icmp(
                                        IntCC::UnsignedGreaterThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    );
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Udiv | BinaryOp::Urem => {
                                    let is_zero =
                                        builder.ins().icmp_imm_s(IntCC::Equal, rhs_val, 0);
                                    let safe_block = builder.create_block();
                                    let bad_block = builder.create_block();
                                    let merge_block = builder.create_block();
                                    builder.append_block_param(merge_block, clif_ty);

                                    builder.ins().brif(is_zero, bad_block, &[], safe_block, &[]);

                                    builder.switch_to_block(safe_block);
                                    let op_res = match op {
                                        BinaryOp::Udiv => builder.ins().udiv(lhs_val, rhs_val),
                                        BinaryOp::Urem => builder.ins().urem(lhs_val, rhs_val),
                                        _ => unreachable!(),
                                    };
                                    builder.ins().jump(merge_block, &[BlockArg::Value(op_res)]);

                                    builder.switch_to_block(bad_block);
                                    let zero_res = builder.ins().iconst(clif_ty, 0);
                                    builder
                                        .ins()
                                        .jump(merge_block, &[BlockArg::Value(zero_res)]);

                                    builder.switch_to_block(merge_block);
                                    let res = builder.block_params(merge_block)[0];
                                    (res, lhs_ty)
                                }
                            }
                        }
                    };
                    values.insert(dst.clone(), (vec![res_val], res_ty));
                }
                Instruction::Load { dst, ptr, ty, .. } => {
                    let (ptr_val, _) = scalar(&values, ptr);
                    emit_bounds_check_const(&mut builder, guard, ptr_val, access_bytes(*ty));
                    let mut loaded = Vec::with_capacity(part_count(*ty));
                    if ty.is_vector() {
                        for k in 0..part_count(*ty) {
                            let offset = (k * 16) as i32;
                            loaded.push(builder.ins().load(
                                VEC_PART,
                                user_mem_flags(),
                                ptr_val,
                                offset,
                            ));
                        }
                    } else {
                        loaded.push(builder.ins().load(
                            to_clif_type(*ty),
                            MemFlagsData::trusted(),
                            ptr_val,
                            0,
                        ));
                    }
                    values.insert(dst.clone(), (loaded, *ty));
                }
                Instruction::Store { ptr, val, .. } => {
                    let (ptr_val, _) = scalar(&values, ptr);
                    let (val_parts, val_ty) = parts(&values, val);
                    emit_bounds_check_const(&mut builder, guard, ptr_val, access_bytes(val_ty));
                    let flags = if val_ty.is_vector() {
                        user_mem_flags()
                    } else {
                        MemFlagsData::trusted()
                    };
                    for (k, part) in val_parts.iter().enumerate() {
                        builder.ins().store(flags, *part, ptr_val, (k * 16) as i32);
                    }
                }
                Instruction::Call {
                    dst,
                    func: callee_name,
                    args,
                    ..
                } => {
                    let target_func_id = *func_ids
                        .get(callee_name)
                        .ok_or_else(|| anyhow!("Unknown function '{callee_name}' in call"))?;
                    let callee = module.declare_func_in_func(target_func_id, builder.func);
                    let arg_vals = flat_args(&values, args);
                    let call_inst = builder.ins().call(callee, &arg_vals);
                    if let Some(d) = dst {
                        let results = builder.inst_results(call_inst).to_vec();
                        let ret_ir_ty = func_returns
                            .get(callee_name)
                            .copied()
                            .flatten()
                            .unwrap_or(Type::I32);
                        values.insert(d.clone(), (results, ret_ir_ty));
                    }
                }
                Instruction::Splat { dst, src, ty, .. } => {
                    let (src_val, src_ty) = scalar(&values, src);
                    let splat = builder.ins().splat(lane_vec_type(src_ty), src_val);
                    let part = to_part(&mut builder, splat);
                    values.insert(dst.clone(), (vec![part; part_count(*ty)], *ty));
                }
                Instruction::ExtractLane {
                    dst, vec, lane, ty, ..
                } => {
                    let (vec_parts, _) = parts(&values, vec);
                    let lane_bytes = ty.byte_size() as u32;
                    let byte = lane * lane_bytes;
                    let part = vec_parts[(byte / 16) as usize];
                    let index = ((byte % 16) / lane_bytes) as u8;
                    let lanes = as_lanes(&mut builder, part, *ty);
                    let scalar_val = builder.ins().extractlane(lanes, index);
                    values.insert(dst.clone(), (vec![scalar_val], *ty));
                }
                Instruction::Alloc { dst, size, .. } => {
                    let (size_val, size_ty) = scalar(&values, size);
                    let size_i64 = if size_ty == Type::I64 || size_ty == Type::Ptr {
                        size_val
                    } else {
                        builder.ins().uextend(types::I64, size_val)
                    };
                    let callee = module.declare_func_in_func(config.rt_malloc_id, builder.func);
                    let call_inst = builder.ins().call(callee, &[size_i64]);
                    let ptr_val = builder.inst_results(call_inst)[0];
                    values.insert(dst.clone(), (vec![ptr_val], Type::Ptr));
                }
                Instruction::Free { ptr, .. } => {
                    let (ptr_val, _) = scalar(&values, ptr);
                    let callee = module.declare_func_in_func(config.rt_free_id, builder.func);
                    builder.ins().call(callee, &[ptr_val]);
                }
                Instruction::Select {
                    dst,
                    cond,
                    then_val,
                    else_val,
                    ..
                } => {
                    let (cond_val, _) = scalar(&values, cond);
                    let (then_parts, then_ty) = parts(&values, then_val);
                    let (else_parts, _) = parts(&values, else_val);
                    let cmp = builder.ins().icmp_imm_s(IntCC::NotEqual, cond_val, 0);
                    let res: Parts = then_parts
                        .iter()
                        .zip(&else_parts)
                        .map(|(t, e)| builder.ins().select(cmp, *t, *e))
                        .collect();
                    values.insert(dst.clone(), (res, then_ty));
                }
                Instruction::Unary { op, dst, src, .. } => {
                    let (src_val, src_ty) = scalar(&values, src);
                    let res = match op {
                        UnaryOp::Neg => {
                            if src_ty.is_float() {
                                builder.ins().fneg(src_val)
                            } else {
                                builder.ins().ineg(src_val)
                            }
                        }
                        UnaryOp::Abs => {
                            if src_ty.is_float() {
                                builder.ins().fabs(src_val)
                            } else {
                                let neg = builder.ins().ineg(src_val);
                                builder.ins().smax(src_val, neg)
                            }
                        }
                        UnaryOp::Sqrt => builder.ins().sqrt(src_val),
                    };
                    values.insert(dst.clone(), (vec![res], src_ty));
                }
                Instruction::Cast {
                    op, dst, src, ty, ..
                } if ty.is_vector() => {
                    // Only same-type bitcasts are valid for vectors: a no-op.
                    debug_assert_eq!(*op, CastOp::Bitcast);
                    let (src_parts, _) = parts(&values, src);
                    values.insert(dst.clone(), (src_parts, *ty));
                }
                Instruction::Cast {
                    op, dst, src, ty, ..
                } => {
                    let (src_val, src_ty) = scalar(&values, src);
                    let clif_target_ty = to_clif_type(*ty);
                    let res = match op {
                        CastOp::Fext if src_ty == Type::F16 => f16_to_f32(&mut builder, src_val),
                        CastOp::Fext if src_ty == Type::BF16 => bf16_to_f32(&mut builder, src_val),
                        CastOp::Ftrunc if *ty == Type::F16 => f32_to_f16(&mut builder, src_val),
                        CastOp::Ftrunc if *ty == Type::BF16 => f32_to_bf16(&mut builder, src_val),
                        // f16/bf16 <-> i16 bitcasts: both are carried as I16 already.
                        CastOp::Bitcast if to_clif_type(src_ty) == clif_target_ty => src_val,
                        CastOp::Itof => builder.ins().fcvt_from_sint(clif_target_ty, src_val),
                        CastOp::Ftoi => {
                            if *ty == Type::I8 || *ty == Type::I16 {
                                let i32_val = builder.ins().fcvt_to_sint_sat(types::I32, src_val);
                                builder.ins().ireduce(clif_target_ty, i32_val)
                            } else {
                                builder.ins().fcvt_to_sint_sat(clif_target_ty, src_val)
                            }
                        }
                        CastOp::Sext => builder.ins().sextend(clif_target_ty, src_val),
                        CastOp::Zext => builder.ins().uextend(clif_target_ty, src_val),
                        CastOp::Trunc => builder.ins().ireduce(clif_target_ty, src_val),
                        CastOp::Fext => builder.ins().fpromote(types::F64, src_val),
                        CastOp::Ftrunc => builder.ins().fdemote(types::F32, src_val),
                        CastOp::Bitcast => {
                            builder
                                .ins()
                                .bitcast(clif_target_ty, bitcast_flags(), src_val)
                        }
                    };
                    values.insert(dst.clone(), (vec![res], *ty));
                }
                Instruction::VectorReduce {
                    op, dst, src, ty, ..
                } => {
                    let (src_parts, _) = parts(&values, src);
                    let lanes_per_part = 16 / ty.byte_size();
                    let mut part_results = Vec::with_capacity(src_parts.len());
                    for part in src_parts {
                        let v = as_lanes(&mut builder, part, *ty);
                        let lanes: Vec<ClifValue> = (0..lanes_per_part)
                            .map(|i| builder.ins().extractlane(v, i as u8))
                            .collect();
                        part_results.push(reduce_tree(&mut builder, *op, *ty, &lanes));
                    }
                    let res = reduce_tree(&mut builder, *op, *ty, &part_results);
                    values.insert(dst.clone(), (vec![res], *ty));
                }
                Instruction::VBinary {
                    op,
                    dst,
                    lhs,
                    rhs,
                    lane,
                    ..
                } => {
                    let (l_parts, vec_ty) = parts(&values, lhs);
                    let (r_parts, _) = parts(&values, rhs);
                    let mut out = Vec::with_capacity(l_parts.len());
                    for (l, r) in l_parts.into_iter().zip(r_parts) {
                        out.push(vbinary_part(&mut builder, *op, *lane, l, r));
                    }
                    values.insert(dst.clone(), (out, vec_ty));
                }
                Instruction::VFma {
                    dst, a, b, c, lane, ..
                } => {
                    let (a_parts, vec_ty) = parts(&values, a);
                    let (b_parts, _) = parts(&values, b);
                    let (c_parts, _) = parts(&values, c);
                    let mut out = Vec::with_capacity(a_parts.len());
                    for k in 0..a_parts.len() {
                        let x = as_lanes(&mut builder, a_parts[k], *lane);
                        let y = as_lanes(&mut builder, b_parts[k], *lane);
                        let z = as_lanes(&mut builder, c_parts[k], *lane);
                        let res = builder.ins().fma(x, y, z);
                        out.push(to_part(&mut builder, res));
                    }
                    values.insert(dst.clone(), (out, vec_ty));
                }
                Instruction::VCmp {
                    op,
                    dst,
                    lhs,
                    rhs,
                    lane,
                    ..
                } => {
                    let (l_parts, vec_ty) = parts(&values, lhs);
                    let (r_parts, _) = parts(&values, rhs);
                    let mut out = Vec::with_capacity(l_parts.len());
                    for (l, r) in l_parts.into_iter().zip(r_parts) {
                        out.push(vcmp_part(&mut builder, *op, *lane, l, r));
                    }
                    values.insert(dst.clone(), (out, vec_ty));
                }
                Instruction::VSelect {
                    dst,
                    mask,
                    then_val,
                    else_val,
                    ..
                } => {
                    let (m_parts, vec_ty) = parts(&values, mask);
                    let (t_parts, _) = parts(&values, then_val);
                    let (e_parts, _) = parts(&values, else_val);
                    let mut out = Vec::with_capacity(m_parts.len());
                    for k in 0..m_parts.len() {
                        let m = as_lanes(&mut builder, m_parts[k], Type::I32);
                        let t = as_lanes(&mut builder, t_parts[k], Type::I32);
                        let e = as_lanes(&mut builder, e_parts[k], Type::I32);
                        let res = builder.ins().bitselect(m, t, e);
                        out.push(to_part(&mut builder, res));
                    }
                    values.insert(dst.clone(), (out, vec_ty));
                }
                Instruction::MaskedLoad {
                    dst,
                    ptr,
                    count,
                    ty,
                    lane,
                    ..
                } => {
                    let (ptr_val, _) = scalar(&values, ptr);
                    let (count_val, _) = scalar(&values, count);
                    let parts =
                        emit_masked_load(&mut builder, guard, ptr_val, count_val, *ty, *lane);
                    values.insert(dst.clone(), (parts, *ty));
                }
                Instruction::MaskedStore {
                    ptr,
                    val,
                    count,
                    lane,
                    ..
                } => {
                    let (ptr_val, _) = scalar(&values, ptr);
                    let (count_val, _) = scalar(&values, count);
                    let (val_parts, val_ty) = parts(&values, val);
                    emit_masked_store(
                        &mut builder,
                        guard,
                        ptr_val,
                        &val_parts,
                        count_val,
                        val_ty,
                        *lane,
                    );
                }
                Instruction::MatMul {
                    pc,
                    pa,
                    pb,
                    m,
                    n,
                    k,
                    dtype,
                    ..
                } => {
                    let regs: Vec<ClifValue> = [pc, pa, pb, m, n, k]
                        .iter()
                        .map(|r| scalar(&values, r).0)
                        .collect();
                    let fuel = match (config.fuel_consume_func_id, fuel_trap_block) {
                        (Some(id), Some(trap)) => {
                            Some((module.declare_func_in_func(id, builder.func), trap))
                        }
                        _ => None,
                    };
                    let bounds = match (config.sandbox, fuel_trap_block) {
                        (Some(sb), Some(trap)) => Some((
                            module.declare_func_in_func(sb.mm_check_id, builder.func),
                            trap,
                        )),
                        _ => None,
                    };
                    emit_matmul(&mut builder, &regs, *dtype, bounds, fuel);
                }
                Instruction::VLen { dst, lane, .. } => {
                    let lanes = VECTOR_PART_BITS / lane.bit_width().unwrap_or(32);
                    let v = builder.ins().iconst(types::I64, lanes as i64);
                    values.insert(dst.clone(), (vec![v], Type::I64));
                }
            }
        }

        // Translate terminators
        match &block.terminator {
            Terminator::Jmp { target, args, .. } => {
                let target_block = *clif_blocks.get(target).unwrap();
                let arg_vals: Vec<BlockArg> = flat_args(&values, args)
                    .into_iter()
                    .map(BlockArg::Value)
                    .collect();

                if let (Some(check), Some(trap_block)) = (config.fuel_check, fuel_trap_block) {
                    emit_fuel_check(&mut builder, module, check, trap_block);
                }
                builder.ins().jump(target_block, &arg_vals);
            }
            Terminator::Br {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                let (cond_val, _) = scalar(&values, cond);
                let then_target = *clif_blocks.get(then_block).unwrap();
                let then_vals: Vec<BlockArg> = flat_args(&values, then_args)
                    .into_iter()
                    .map(BlockArg::Value)
                    .collect();
                let else_target = *clif_blocks.get(else_block).unwrap();
                let else_vals: Vec<BlockArg> = flat_args(&values, else_args)
                    .into_iter()
                    .map(BlockArg::Value)
                    .collect();

                if let (Some(check), Some(trap_block)) = (config.fuel_check, fuel_trap_block) {
                    emit_fuel_check(&mut builder, module, check, trap_block);
                }
                builder
                    .ins()
                    .brif(cond_val, then_target, &then_vals, else_target, &else_vals);
            }
            Terminator::Ret { val, .. } => {
                if let Some(v) = val {
                    let (ret_parts, _) = parts(&values, v);
                    builder.ins().return_(&ret_parts);
                } else {
                    builder.ins().return_(&[]);
                }
            }
        }
    }

    if let (Some(g), Some(sb), Some(trap_block)) = (guard, config.sandbox, fuel_trap_block) {
        builder.switch_to_block(g.fault);
        let fault_args = builder.block_params(g.fault).to_vec();
        let callee = module.declare_func_in_func(sb.fault_id, builder.func);
        builder.ins().call(callee, &fault_args);
        builder.ins().jump(trap_block, &[]);
    }

    if let Some(trap_block) = fuel_trap_block {
        builder.switch_to_block(trap_block);
        if let Some(r_ty) = func.ret_type {
            let zero_val = match r_ty {
                Type::F32 => builder.ins().f32const(0.0),
                Type::F64 => builder.ins().f64const(0.0),
                Type::I64 | Type::Ptr => builder.ins().iconst(types::I64, 0),
                Type::I32 => builder.ins().iconst(types::I32, 0),
                Type::I16 => builder.ins().iconst(types::I16, 0),
                Type::I8 => builder.ins().iconst(types::I8, 0),
                Type::F16 | Type::BF16 => builder.ins().iconst(types::I16, 0),
                Type::V128 | Type::V256 | Type::V512 | Type::Vx => {
                    let zero_f = builder.ins().f32const(0.0);
                    builder.ins().splat(VEC_PART, zero_f)
                }
            };
            builder.ins().return_(&vec![zero_val; part_count(r_ty)]);
        } else {
            builder.ins().return_(&[]);
        }
    }

    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RtValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ptr(usize),
}

impl RtValue {
    pub fn to_u64(&self, expected: Type) -> Result<u64> {
        match (*self, expected) {
            (RtValue::I8(n), Type::I8) => Ok((n as u8) as u64),
            (RtValue::I16(n), Type::I16) => Ok((n as u16) as u64),
            (RtValue::I32(n), Type::I32) => Ok((n as u32) as u64),
            (RtValue::I64(n), Type::I64) => Ok(n as u64),
            (RtValue::Ptr(p), Type::Ptr) => Ok(p as u64),
            (RtValue::F32(f), Type::F32) => Ok(f.to_bits() as u64),
            (RtValue::F64(f), Type::F64) => Ok(f.to_bits()),
            // Allow loose integer conversions if within range
            (RtValue::I64(n), Type::I32) if (i32::MIN as i64..=i32::MAX as i64).contains(&n) => {
                Ok((n as i32 as u32) as u64)
            }
            (RtValue::I64(n), Type::I16) if (i16::MIN as i64..=i16::MAX as i64).contains(&n) => {
                Ok((n as i16 as u16) as u64)
            }
            (RtValue::I64(n), Type::I8) if (i8::MIN as i64..=i8::MAX as i64).contains(&n) => {
                Ok((n as i8 as u8) as u64)
            }
            (RtValue::I64(n), Type::Ptr) => Ok(n as u64),
            (RtValue::I32(n), Type::I64) => Ok(n as i64 as u64),
            (RtValue::I32(n), Type::Ptr) => Ok(n as usize as u64),
            (RtValue::Ptr(p), Type::I64) => Ok(p as u64),
            (other, exp) => Err(anyhow!(
                "Type mismatch: argument {:?} cannot be passed to parameter of type {:?}",
                other,
                exp
            )),
        }
    }

    pub fn from_u64(raw: u64, ty: Type) -> Self {
        match ty {
            Type::I8 => RtValue::I8(raw as u8 as i8),
            Type::I16 => RtValue::I16(raw as u16 as i16),
            Type::I32 => RtValue::I32(raw as u32 as i32),
            Type::I64 => RtValue::I64(raw as i64),
            Type::Ptr => RtValue::Ptr(raw as usize),
            Type::F32 => RtValue::F32(f32::from_bits(raw as u32)),
            Type::F64 => RtValue::F64(f64::from_bits(raw)),
            Type::V128 | Type::V256 | Type::V512 | Type::Vx | Type::F16 | Type::BF16 => {
                panic!("{ty} cannot be decoded from scalar u64")
            }
        }
    }
}

pub fn lower_trampoline<M: ClifModule>(
    module: &mut M,
    ctx: &mut cranelift_codegen::Context,
    builder_context: &mut FunctionBuilderContext,
    func_name: &str,
    target_func_id: FuncId,
    param_types: &[Type],
    ret_type: Option<Type>,
) -> Result<FuncId> {
    let trampoline_name = format!("__achainsaw_trampoline_{func_name}");
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // args_ptr
    sig.params.push(AbiParam::new(types::I64)); // ret_ptr

    let tramp_id =
        module.declare_function(&trampoline_name, cranelift_module::Linkage::Export, &sig)?;

    ctx.func.signature = sig;
    let mut builder = FunctionBuilder::new(&mut ctx.func, builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);

    let args_ptr = builder.block_params(entry)[0];
    let ret_ptr = builder.block_params(entry)[1];

    let mut arg_vals = Vec::with_capacity(param_types.len());
    for (i, &ty) in param_types.iter().enumerate() {
        let offset = (i * 8) as i32;
        let raw_val = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), args_ptr, offset);
        let arg_val = match ty {
            Type::I64 | Type::Ptr => raw_val,
            Type::I32 => builder.ins().ireduce(types::I32, raw_val),
            Type::I16 => builder.ins().ireduce(types::I16, raw_val),
            Type::I8 => builder.ins().ireduce(types::I8, raw_val),
            Type::F64 => builder.ins().bitcast(types::F64, bitcast_flags(), raw_val),
            Type::F32 => {
                let r32 = builder.ins().ireduce(types::I32, raw_val);
                builder.ins().bitcast(types::F32, bitcast_flags(), r32)
            }
            Type::V128 | Type::V256 | Type::V512 | Type::Vx | Type::F16 | Type::BF16 => {
                return Err(anyhow!("Cannot pass {ty} directly in scalar trampoline"))
            }
        };
        arg_vals.push(arg_val);
    }

    let callee = module.declare_func_in_func(target_func_id, builder.func);
    let call_inst = builder.ins().call(callee, &arg_vals);

    if let Some(r_ty) = ret_type {
        let res = builder.inst_results(call_inst)[0];
        let res_i64 = match r_ty {
            Type::I64 | Type::Ptr => res,
            Type::I32 | Type::I16 | Type::I8 => builder.ins().uextend(types::I64, res),
            Type::F64 => builder.ins().bitcast(types::I64, bitcast_flags(), res),
            Type::F32 => {
                let b = builder.ins().bitcast(types::I32, bitcast_flags(), res);
                builder.ins().uextend(types::I64, b)
            }
            Type::V128 | Type::V256 | Type::V512 | Type::Vx | Type::F16 | Type::BF16 => {
                return Err(anyhow!(
                    "Cannot return {r_ty} directly in scalar trampoline"
                ))
            }
        };
        builder
            .ins()
            .store(MemFlagsData::trusted(), res_i64, ret_ptr, 0);
    }

    builder.ins().return_(&[]);
    builder.seal_all_blocks();
    builder.finalize(module.target_config());

    module.define_function(tramp_id, ctx)?;
    module.clear_context(ctx);

    Ok(tramp_id)
}
