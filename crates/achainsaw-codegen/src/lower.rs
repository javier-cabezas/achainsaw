use achainsaw_ir::ast::{
    BinaryOp, CastOp, Constant, Function, Instruction, Terminator, UnaryOp, VBinOp, VCmpOp,
    VectorReduceOp,
};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Value as ClifValue};
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

/// Memory flags for vector loads and stores. Unlike `trusted()`, these do not claim
/// alignment: AIR pointers into user buffers can be arbitrarily aligned, and an aligned
/// load may be folded into an SSE instruction that faults on unaligned addresses.
fn vector_mem_flags() -> MemFlagsData {
    MemFlagsData::new().with_notrap()
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

pub struct LowerConfig {
    pub fuel_check_func_id: Option<FuncId>,
    pub rt_malloc_id: FuncId,
    pub rt_free_id: FuncId,
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

    let fuel_trap_block = if config.fuel_check_func_id.is_some() {
        Some(builder.create_block())
    } else {
        None
    };

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
    builder.ins().jump(first_block, &[]);

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
                    let mut loaded = Vec::with_capacity(part_count(*ty));
                    if ty.is_vector() {
                        for k in 0..part_count(*ty) {
                            let offset = (k * 16) as i32;
                            loaded.push(builder.ins().load(
                                VEC_PART,
                                vector_mem_flags(),
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
                    let flags = if val_ty.is_vector() {
                        vector_mem_flags()
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
                    let (src_val, _) = scalar(&values, src);
                    let clif_target_ty = to_clif_type(*ty);
                    let res = match op {
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

                if let (Some(fuel_func_id), Some(trap_block)) =
                    (config.fuel_check_func_id, fuel_trap_block)
                {
                    let callee = module.declare_func_in_func(fuel_func_id, builder.func);
                    let call_inst = builder.ins().call(callee, &[]);
                    let is_exhausted = builder.inst_results(call_inst)[0];
                    builder
                        .ins()
                        .brif(is_exhausted, trap_block, &[], target_block, &arg_vals);
                } else {
                    builder.ins().jump(target_block, &arg_vals);
                }
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

                if let (Some(fuel_func_id), Some(trap_block)) =
                    (config.fuel_check_func_id, fuel_trap_block)
                {
                    let callee = module.declare_func_in_func(fuel_func_id, builder.func);
                    let call_inst = builder.ins().call(callee, &[]);
                    let is_exhausted = builder.inst_results(call_inst)[0];
                    let normal_br_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_exhausted, trap_block, &[], normal_br_block, &[]);

                    builder.switch_to_block(normal_br_block);
                    builder
                        .ins()
                        .brif(cond_val, then_target, &then_vals, else_target, &else_vals);
                } else {
                    builder
                        .ins()
                        .brif(cond_val, then_target, &then_vals, else_target, &else_vals);
                }
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
            Type::V128 | Type::V256 | Type::V512 | Type::Vx => {
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
            Type::V128 | Type::V256 | Type::V512 | Type::Vx => {
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
            Type::V128 | Type::V256 | Type::V512 | Type::Vx => {
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
