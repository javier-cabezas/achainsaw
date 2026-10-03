use achainsaw_ir::ast::{BinaryOp, Constant, Function, Instruction, Terminator};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Value as ClifValue};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Module as ClifModule};
use std::collections::HashMap;

pub fn to_clif_type(ty: Type) -> types::Type {
    match ty {
        Type::I8 => types::I8,
        Type::I16 => types::I16,
        Type::I32 => types::I32,
        Type::I64 => types::I64,
        Type::F32 => types::F32,
        Type::F64 => types::F64,
        Type::Ptr => types::I64,
        Type::V128 => types::F32X4,
    }
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
    ctx.func.signature.clear(module.target_config().default_call_conv);
    for (_, p_ty) in &func.params {
        ctx.func.signature.params.push(AbiParam::new(to_clif_type(*p_ty)));
    }
    if let Some(r_ty) = func.ret_type {
        ctx.func.signature.returns.push(AbiParam::new(to_clif_type(r_ty)));
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
            builder.append_block_param(clif_block, to_clif_type(*p_ty));
        }
    }

    let fuel_trap_block = if config.fuel_check_func_id.is_some() {
        Some(builder.create_block())
    } else {
        None
    };

    // Values map: SSA register name -> (ClifValue, Type)
    let mut values: HashMap<String, (ClifValue, Type)> = HashMap::new();

    // Map function parameters in synthetic entry
    builder.switch_to_block(synth_entry);
    for (i, (p_name, p_ty)) in func.params.iter().enumerate() {
        let val = builder.block_params(synth_entry)[i];
        values.insert(p_name.clone(), (val, *p_ty));
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
        for (i, (bp_name, bp_ty)) in block.params.iter().enumerate() {
            let val = builder.block_params(clif_block)[i];
            values.insert(bp_name.clone(), (val, *bp_ty));
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
                    values.insert(dst.clone(), (v, *ty));
                }
                Instruction::Binary {
                    op, dst, lhs, rhs, ..
                } => {
                    let (lhs_val, lhs_ty) = *values.get(lhs).unwrap();
                    let (rhs_val, rhs_ty) = *values.get(rhs).unwrap();

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
                                let cmp = builder.ins().fcmp(FloatCC::GreaterThan, lhs_val, rhs_val);
                                let ext = builder.ins().uextend(types::I32, cmp);
                                (ext, Type::I32)
                            }
                            BinaryOp::Le => {
                                let cmp =
                                    builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs_val, rhs_val);
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
                            _ => return Err(anyhow!("Unsupported float op {:?}", op)),
                        },
                        Type::V128 => match op {
                            BinaryOp::VfAdd => (builder.ins().fadd(lhs_val, rhs_val), Type::V128),
                            BinaryOp::VfSub => (builder.ins().fsub(lhs_val, rhs_val), Type::V128),
                            BinaryOp::VfMul => (builder.ins().fmul(lhs_val, rhs_val), Type::V128),
                            BinaryOp::VfDiv => (builder.ins().fdiv(lhs_val, rhs_val), Type::V128),
                            BinaryOp::ViAdd => {
                                let lhs_i = builder.ins().bitcast(types::I32X4, bitcast_flags(), lhs_val);
                                let rhs_i = builder.ins().bitcast(types::I32X4, bitcast_flags(), rhs_val);
                                let sum_i = builder.ins().iadd(lhs_i, rhs_i);
                                let res = builder.ins().bitcast(types::F32X4, bitcast_flags(), sum_i);
                                (res, Type::V128)
                            }
                            BinaryOp::ViSub => {
                                let lhs_i = builder.ins().bitcast(types::I32X4, bitcast_flags(), lhs_val);
                                let rhs_i = builder.ins().bitcast(types::I32X4, bitcast_flags(), rhs_val);
                                let diff_i = builder.ins().isub(lhs_i, rhs_i);
                                let res = builder.ins().bitcast(types::F32X4, bitcast_flags(), diff_i);
                                (res, Type::V128)
                            }
                            BinaryOp::ViMul => {
                                let lhs_i = builder.ins().bitcast(types::I32X4, bitcast_flags(), lhs_val);
                                let rhs_i = builder.ins().bitcast(types::I32X4, bitcast_flags(), rhs_val);
                                let prod_i = builder.ins().imul(lhs_i, rhs_i);
                                let res = builder.ins().bitcast(types::F32X4, bitcast_flags(), prod_i);
                                (res, Type::V128)
                            }
                            _ => return Err(anyhow!("Unsupported vector op {:?}", op)),
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
                                    let is_zero = builder.ins().icmp_imm_s(IntCC::Equal, rhs_val, 0);
                                    let is_neg_one = builder.ins().icmp_imm_s(IntCC::Equal, rhs_val, -1);
                                    let min_val = match lhs_ty {
                                        Type::I8 => -128i64,
                                        Type::I16 => -32768i64,
                                        Type::I32 => i32::MIN as i64,
                                        Type::I64 | Type::Ptr => i64::MIN,
                                        _ => 0i64,
                                    };
                                    let is_min = builder.ins().icmp_imm_s(IntCC::Equal, lhs_val, min_val);
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
                                    builder.ins().jump(merge_block, &[BlockArg::Value(zero_res)]);

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
                                    let cmp = builder
                                        .ins()
                                        .icmp(IntCC::SignedGreaterThan, lhs_val, rhs_val);
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
                                _ => return Err(anyhow!("Invalid scalar integer op {:?}", op)),
                            }
                        }
                    };
                    values.insert(dst.clone(), (res_val, res_ty));
                }
                Instruction::Load { dst, ptr, ty, .. } => {
                    let (ptr_val, _) = *values.get(ptr).unwrap();
                    let clif_ty = to_clif_type(*ty);
                    let val = builder
                        .ins()
                        .load(clif_ty, MemFlagsData::trusted(), ptr_val, 0);
                    values.insert(dst.clone(), (val, *ty));
                }
                Instruction::Store { ptr, val, .. } => {
                    let (ptr_val, _) = *values.get(ptr).unwrap();
                    let (val_val, _) = *values.get(val).unwrap();
                    builder
                        .ins()
                        .store(MemFlagsData::trusted(), val_val, ptr_val, 0);
                }
                Instruction::Call {
                    dst, func: callee_name, args, ..
                } => {
                    let target_func_id = *func_ids.get(callee_name).ok_or_else(|| {
                        anyhow!("Unknown function '{callee_name}' in call")
                    })?;
                    let callee = module.declare_func_in_func(target_func_id, builder.func);
                    let arg_vals: Vec<ClifValue> =
                        args.iter().map(|a| values.get(a).unwrap().0).collect();
                    let call_inst = builder.ins().call(callee, &arg_vals);
                    if let Some(d) = dst {
                        let results = builder.inst_results(call_inst);
                        let res_val = results[0];
                        let ret_ir_ty = func_returns
                            .get(callee_name)
                            .copied()
                            .flatten()
                            .unwrap_or(Type::I32);
                        values.insert(d.clone(), (res_val, ret_ir_ty));
                    }
                }
                Instruction::Splat { dst, src, .. } => {
                    let (src_val, src_ty) = *values.get(src).unwrap();
                    let vec_val = match src_ty {
                        Type::F32 => builder.ins().splat(types::F32X4, src_val),
                        Type::I32 => {
                            let s = builder.ins().splat(types::I32X4, src_val);
                            builder.ins().bitcast(types::F32X4, bitcast_flags(), s)
                        }
                        Type::I64 => {
                            let s = builder.ins().splat(types::I64X2, src_val);
                            builder.ins().bitcast(types::F32X4, bitcast_flags(), s)
                        }
                        Type::F64 => {
                            let s = builder.ins().splat(types::F64X2, src_val);
                            builder.ins().bitcast(types::F32X4, bitcast_flags(), s)
                        }
                        _ => builder.ins().splat(types::F32X4, src_val),
                    };
                    values.insert(dst.clone(), (vec_val, Type::V128));
                }
                Instruction::ExtractLane {
                    dst, vec, lane, ty, ..
                } => {
                    let (vec_val, _) = *values.get(vec).unwrap();
                    let scalar_val = match ty {
                        Type::F32 => builder.ins().extractlane(vec_val, *lane as u8),
                        Type::I32 => {
                            let vi = builder.ins().bitcast(types::I32X4, bitcast_flags(), vec_val);
                            builder.ins().extractlane(vi, *lane as u8)
                        }
                        Type::F64 => {
                            let vf = builder.ins().bitcast(types::F64X2, bitcast_flags(), vec_val);
                            builder.ins().extractlane(vf, *lane as u8)
                        }
                        Type::I64 => {
                            let vi = builder.ins().bitcast(types::I64X2, bitcast_flags(), vec_val);
                            builder.ins().extractlane(vi, *lane as u8)
                        }
                        _ => builder.ins().extractlane(vec_val, *lane as u8),
                    };
                    values.insert(dst.clone(), (scalar_val, *ty));
                }
                Instruction::Alloc { dst, size, .. } => {
                    let (size_val, size_ty) = *values.get(size).unwrap();
                    let size_i64 = if size_ty == Type::I64 || size_ty == Type::Ptr {
                        size_val
                    } else {
                        builder.ins().uextend(types::I64, size_val)
                    };
                    let callee = module.declare_func_in_func(config.rt_malloc_id, builder.func);
                    let call_inst = builder.ins().call(callee, &[size_i64]);
                    let ptr_val = builder.inst_results(call_inst)[0];
                    values.insert(dst.clone(), (ptr_val, Type::Ptr));
                }
                Instruction::Free { ptr, .. } => {
                    let (ptr_val, _) = *values.get(ptr).unwrap();
                    let callee = module.declare_func_in_func(config.rt_free_id, builder.func);
                    builder.ins().call(callee, &[ptr_val]);
                }
            }
        }

        // Translate terminators
        match &block.terminator {
            Terminator::Jmp { target, args, .. } => {
                let target_block = *clif_blocks.get(target).unwrap();
                let arg_vals: Vec<BlockArg> = args
                    .iter()
                    .map(|a| BlockArg::Value(values.get(a).unwrap().0))
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
                let (cond_val, _) = *values.get(cond).unwrap();
                let then_target = *clif_blocks.get(then_block).unwrap();
                let then_vals: Vec<BlockArg> = then_args
                    .iter()
                    .map(|a| BlockArg::Value(values.get(a).unwrap().0))
                    .collect();
                let else_target = *clif_blocks.get(else_block).unwrap();
                let else_vals: Vec<BlockArg> = else_args
                    .iter()
                    .map(|a| BlockArg::Value(values.get(a).unwrap().0))
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
                    builder.ins().brif(
                        cond_val,
                        then_target,
                        &then_vals,
                        else_target,
                        &else_vals,
                    );
                } else {
                    builder.ins().brif(
                        cond_val,
                        then_target,
                        &then_vals,
                        else_target,
                        &else_vals,
                    );
                }
            }
            Terminator::Ret { val, .. } => {
                if let Some(v) = val {
                    let (ret_val, _) = *values.get(v).unwrap();
                    builder.ins().return_(&[ret_val]);
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
                Type::V128 => {
                    let zero_f = builder.ins().f32const(0.0);
                    builder.ins().splat(types::F32X4, zero_f)
                }
            };
            builder.ins().return_(&[zero_val]);
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
            (RtValue::F32(f), Type::F32) => Ok((f.to_bits() as u32) as u64),
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
            Type::V128 => panic!("v128 cannot be decoded from scalar u64"),
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
        let raw_val = builder.ins().load(types::I64, MemFlagsData::trusted(), args_ptr, offset);
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
            Type::V128 => return Err(anyhow!("Cannot pass v128 directly in scalar trampoline")),
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
            Type::V128 => return Err(anyhow!("Cannot return v128 directly in scalar trampoline")),
        };
        builder.ins().store(MemFlagsData::trusted(), res_i64, ret_ptr, 0);
    }

    builder.ins().return_(&[]);
    builder.seal_all_blocks();
    builder.finalize(module.target_config());

    module.define_function(tramp_id, ctx)?;
    module.clear_context(ctx);

    Ok(tramp_id)
}
