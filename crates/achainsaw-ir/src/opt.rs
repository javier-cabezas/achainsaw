use crate::ast::*;
use crate::types::Type;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub constants_folded: usize,
    pub algebraic_simplifications: usize,
    pub branches_folded: usize,
    pub dead_instructions_removed: usize,
    pub dead_blocks_removed: usize,
    pub iterations: usize,
}

impl OptimizationStats {
    pub fn total_optimizations(&self) -> usize {
        self.constants_folded
            + self.algebraic_simplifications
            + self.branches_folded
            + self.dead_instructions_removed
            + self.dead_blocks_removed
    }
}

/// Return types of every callable (externs and module functions), by name.
type Signatures = HashMap<String, Option<Type>>;

/// Optimizes an entire IR module in-place, returning summary statistics.
pub fn optimize_module(module: &mut Module) -> OptimizationStats {
    let mut total_stats = OptimizationStats::default();

    let mut sigs: Signatures = HashMap::new();
    for ext in &module.extern_functions {
        sigs.insert(ext.name.clone(), ext.ret_type);
    }
    for func in &module.functions {
        sigs.insert(func.name.clone(), func.ret_type);
    }

    for func in &mut module.functions {
        let stats = optimize_function_with(func, &sigs);
        total_stats.constants_folded += stats.constants_folded;
        total_stats.algebraic_simplifications += stats.algebraic_simplifications;
        total_stats.branches_folded += stats.branches_folded;
        total_stats.dead_instructions_removed += stats.dead_instructions_removed;
        total_stats.dead_blocks_removed += stats.dead_blocks_removed;
        total_stats.iterations += stats.iterations;
    }

    total_stats
}

/// Optimizes a single IR function to a fixpoint.
///
/// Without module context the result types of `call` instructions are unknown, so
/// optimizations that depend on them are skipped. Prefer [`optimize_module`].
pub fn optimize_function(func: &mut Function) -> OptimizationStats {
    optimize_function_with(func, &HashMap::new())
}

fn optimize_function_with(func: &mut Function, sigs: &Signatures) -> OptimizationStats {
    let mut stats = OptimizationStats::default();
    let max_iters = 16;

    for iter in 1..=max_iters {
        stats.iterations = iter;
        let mut changed = false;

        // 1. Constant folding & propagation + Algebraic simplification
        let (folded, simplified) = run_constant_and_algebraic_pass(func, sigs);
        if folded > 0 || simplified > 0 {
            stats.constants_folded += folded;
            stats.algebraic_simplifications += simplified;
            changed = true;
        }

        // 2. Branch folding (br with constant condition -> jmp)
        let branches = run_branch_folding_pass(func);
        if branches > 0 {
            stats.branches_folded += branches;
            changed = true;
        }

        // 3. Dead block pruning (unreachable basic blocks)
        let dead_blocks = run_dead_block_elimination(func);
        if dead_blocks > 0 {
            stats.dead_blocks_removed += dead_blocks;
            changed = true;
        }

        // 4. Dead Code Elimination (DCE)
        let dce = run_dead_code_elimination(func, sigs);
        if dce > 0 {
            stats.dead_instructions_removed += dce;
            changed = true;
        }

        if !changed {
            break;
        }
    }

    stats
}

/// Infers the type of every register in `func` (parameters, block parameters and
/// instruction results). Registers whose type cannot be determined are omitted.
fn infer_reg_types(func: &Function, sigs: &Signatures) -> HashMap<String, Type> {
    let mut types: HashMap<String, Type> = HashMap::new();
    for (name, ty) in &func.params {
        types.insert(name.clone(), *ty);
    }
    for block in &func.blocks {
        for (name, ty) in &block.params {
            types.insert(name.clone(), *ty);
        }
    }

    // Block order is not dominance order, so iterate until nothing new is learned.
    for _ in 0..=func.blocks.len() {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                let (dst, ty) = match inst {
                    Instruction::AssignConst { dst, ty, .. } => (dst, Some(*ty)),
                    Instruction::Load { dst, ty, .. } => (dst, Some(*ty)),
                    Instruction::ExtractLane { dst, ty, .. } => (dst, Some(*ty)),
                    Instruction::Splat { dst, ty, .. } => (dst, Some(*ty)),
                    Instruction::VBinary { dst, lhs, .. }
                    | Instruction::VCmp { dst, lhs, .. }
                    | Instruction::VFma { dst, a: lhs, .. }
                    | Instruction::VSelect {
                        dst, then_val: lhs, ..
                    } => (dst, types.get(lhs).copied()),
                    Instruction::VLen { dst, .. } => (dst, Some(Type::I64)),
                    Instruction::Alloc { dst, .. } => (dst, Some(Type::Ptr)),
                    Instruction::Call {
                        dst: Some(dst),
                        func,
                        ..
                    } => (dst, sigs.get(func).copied().flatten()),
                    Instruction::Binary {
                        op, dst, lhs, rhs, ..
                    } => {
                        let ty = match (types.get(lhs), types.get(rhs)) {
                            (Some(l), Some(r)) => {
                                crate::validator::binary_result_type(*op, *l, *r).ok()
                            }
                            _ => None,
                        };
                        (dst, ty)
                    }
                    Instruction::Select { dst, then_val, .. } => {
                        (dst, types.get(then_val).copied())
                    }
                    Instruction::Unary { dst, src, .. } => (dst, types.get(src).copied()),
                    Instruction::Cast { dst, ty, .. } => (dst, Some(*ty)),
                    Instruction::VectorReduce { dst, ty, .. } => (dst, Some(*ty)),
                    _ => continue,
                };
                if let Some(ty) = ty {
                    if types.insert(dst.clone(), ty).is_none() {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    types
}

/// Runs constant folding, constant propagation, and algebraic simplification.
fn run_constant_and_algebraic_pass(func: &mut Function, sigs: &Signatures) -> (usize, usize) {
    let reg_types = infer_reg_types(func, sigs);
    let mut constants: HashMap<String, (Constant, Type)> = HashMap::new();
    let mut substitutions: HashMap<String, String> = HashMap::new();
    let mut folded_count = 0;
    let mut simplified_count = 0;

    // First collect all explicit constants
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::AssignConst { dst, val, ty, .. } = inst {
                constants.insert(dst.clone(), (val.clone(), *ty));
            }
        }
    }

    // Block order is not dominance order, so a substitution discovered late may still
    // need to be applied to earlier blocks. Rescan until no new substitution appears.
    loop {
        let substitutions_before = substitutions.len();

        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                // Apply variable substitutions to operands
                substitute_instruction_operands(inst, &substitutions);

                match inst {
                    Instruction::AssignConst { dst, val, ty, .. } => {
                        constants.insert(dst.clone(), (val.clone(), *ty));
                        continue;
                    }
                    Instruction::Binary {
                        op,
                        dst,
                        lhs,
                        rhs,
                        span,
                    } => {
                        let (op, dst, lhs, rhs, span) =
                            (*op, dst.clone(), lhs.clone(), rhs.clone(), *span);
                        let (Some(&lhs_ty), Some(&rhs_ty)) =
                            (reg_types.get(&lhs), reg_types.get(&rhs))
                        else {
                            continue;
                        };
                        let Ok(res_ty) = crate::validator::binary_result_type(op, lhs_ty, rhs_ty)
                        else {
                            continue;
                        };

                        // Full constant folding if both operands are constants
                        if let (Some((c1, _)), Some((c2, _))) =
                            (constants.get(&lhs), constants.get(&rhs))
                        {
                            if let Some((folded_const, folded_ty)) =
                                fold_binary_op(op, c1, lhs_ty, c2, rhs_ty)
                            {
                                debug_assert_eq!(folded_ty, res_ty);
                                *inst = Instruction::AssignConst {
                                    dst: dst.clone(),
                                    val: folded_const.clone(),
                                    ty: folded_ty,
                                    span,
                                };
                                constants.insert(dst, (folded_const, folded_ty));
                                folded_count += 1;
                                continue;
                            }
                        }

                        // Algebraic simplification
                        if let Some(res) =
                            simplify_algebraic(op, &lhs, &rhs, &constants, &reg_types, res_ty)
                        {
                            match res {
                                AlgebraicResult::Constant(c, ty) => {
                                    *inst = Instruction::AssignConst {
                                        dst: dst.clone(),
                                        val: c.clone(),
                                        ty,
                                        span,
                                    };
                                    constants.insert(dst, (c, ty));
                                    simplified_count += 1;
                                }
                                AlgebraicResult::Identity(source) => {
                                    let real_source =
                                        substitutions.get(&source).cloned().unwrap_or(source);
                                    if substitutions.insert(dst, real_source).is_none() {
                                        simplified_count += 1;
                                    }
                                }
                            }
                        }
                    }
                    Instruction::Unary { op, dst, src, span } => {
                        let (op, dst, src, span) = (*op, dst.clone(), src.clone(), *span);
                        let Some(&src_ty) = reg_types.get(&src) else {
                            continue;
                        };
                        if let Some((c, _)) = constants.get(&src) {
                            if let Some((folded_const, folded_ty)) = fold_unary_op(op, c, src_ty) {
                                *inst = Instruction::AssignConst {
                                    dst: dst.clone(),
                                    val: folded_const.clone(),
                                    ty: folded_ty,
                                    span,
                                };
                                constants.insert(dst, (folded_const, folded_ty));
                                folded_count += 1;
                                continue;
                            }
                        }
                    }
                    Instruction::Cast {
                        op,
                        dst,
                        src,
                        ty,
                        span,
                    } => {
                        let (op, dst, src, target_ty, span) =
                            (*op, dst.clone(), src.clone(), *ty, *span);
                        let Some(&src_ty) = reg_types.get(&src) else {
                            continue;
                        };
                        if let Some((c, _)) = constants.get(&src) {
                            if let Some((folded_const, folded_ty)) =
                                fold_cast_op(op, c, src_ty, target_ty)
                            {
                                *inst = Instruction::AssignConst {
                                    dst: dst.clone(),
                                    val: folded_const.clone(),
                                    ty: folded_ty,
                                    span,
                                };
                                constants.insert(dst, (folded_const, folded_ty));
                                folded_count += 1;
                                continue;
                            }
                        }
                    }
                    Instruction::Select {
                        dst,
                        cond,
                        then_val,
                        else_val,
                        span,
                    } => {
                        let (dst, cond, then_val, else_val, span) = (
                            dst.clone(),
                            cond.clone(),
                            then_val.clone(),
                            else_val.clone(),
                            *span,
                        );
                        if then_val == else_val {
                            let real_source =
                                substitutions.get(&then_val).cloned().unwrap_or(then_val);
                            if substitutions.insert(dst, real_source).is_none() {
                                simplified_count += 1;
                            }
                            continue;
                        }
                        if let Some((Constant::Int(c), _)) = constants.get(&cond) {
                            let chosen = if *c != 0 { then_val } else { else_val };
                            if let Some((c_val, c_ty)) = constants.get(&chosen) {
                                *inst = Instruction::AssignConst {
                                    dst: dst.clone(),
                                    val: c_val.clone(),
                                    ty: *c_ty,
                                    span,
                                };
                                constants.insert(dst, (c_val.clone(), *c_ty));
                                folded_count += 1;
                            } else {
                                let real_source =
                                    substitutions.get(&chosen).cloned().unwrap_or(chosen);
                                if substitutions.insert(dst, real_source).is_none() {
                                    simplified_count += 1;
                                }
                            }
                            continue;
                        }
                    }
                    _ => {}
                }
            }

            // Substitute terminator operands
            substitute_terminator_operands(&mut block.terminator, &substitutions);
        }

        if substitutions.len() == substitutions_before {
            break;
        }
    }

    (folded_count, simplified_count)
}

enum AlgebraicResult {
    Constant(Constant, Type),
    Identity(String),
}

/// Algebraic identities. Only rewrites that provably preserve both the value and the
/// result type `res_ty` are returned; floating-point values are never simplified
/// (`x - x` is NaN for NaN/inf, `x * 0` is NaN for inf, `x + 0` loses `-0.0`).
fn simplify_algebraic(
    op: BinaryOp,
    lhs: &str,
    rhs: &str,
    constants: &HashMap<String, (Constant, Type)>,
    reg_types: &HashMap<String, Type>,
    res_ty: Type,
) -> Option<AlgebraicResult> {
    // Only integer/pointer results are eligible.
    if !(res_ty.is_int() || res_ty == Type::Ptr) {
        return None;
    }

    let lhs_c = constants.get(lhs);
    let rhs_c = constants.get(rhs);

    let identity = |source: &str| -> Option<AlgebraicResult> {
        if reg_types.get(source) == Some(&res_ty) {
            Some(AlgebraicResult::Identity(source.to_string()))
        } else {
            None
        }
    };
    let zero = |ty: Type| -> Option<AlgebraicResult> {
        if ty == res_ty && res_ty.is_int() {
            Some(AlgebraicResult::Constant(Constant::Int(0), ty))
        } else {
            None
        }
    };
    let is_int_const = |c: Option<&(Constant, Type)>, v: i64| -> bool {
        matches!(c, Some((Constant::Int(n), ty)) if ty.wrap_int(*n) == v)
    };

    let result = match op {
        // x + 0 -> x, 0 + x -> x
        BinaryOp::Add => {
            if is_int_const(rhs_c, 0) {
                identity(lhs)
            } else if is_int_const(lhs_c, 0) {
                identity(rhs)
            } else {
                None
            }
        }
        // x - 0 -> x, x - x -> 0
        BinaryOp::Sub => {
            if is_int_const(rhs_c, 0) {
                identity(lhs)
            } else if lhs == rhs {
                reg_types.get(lhs).copied().and_then(zero)
            } else {
                None
            }
        }
        // x * 1 -> x, 1 * x -> x, x * 0 -> 0, 0 * x -> 0
        BinaryOp::Mul => {
            if is_int_const(rhs_c, 1) {
                identity(lhs)
            } else if is_int_const(lhs_c, 1) {
                identity(rhs)
            } else if is_int_const(rhs_c, 0) {
                rhs_c.and_then(|(_, ty)| zero(*ty))
            } else if is_int_const(lhs_c, 0) {
                lhs_c.and_then(|(_, ty)| zero(*ty))
            } else {
                None
            }
        }
        // x / 1 -> x
        BinaryOp::Div => {
            if is_int_const(rhs_c, 1) {
                identity(lhs)
            } else {
                None
            }
        }
        // x & 0 -> 0, 0 & x -> 0, x & x -> x
        BinaryOp::And => {
            if is_int_const(rhs_c, 0) {
                rhs_c.and_then(|(_, ty)| zero(*ty))
            } else if is_int_const(lhs_c, 0) {
                lhs_c.and_then(|(_, ty)| zero(*ty))
            } else if lhs == rhs {
                identity(lhs)
            } else {
                None
            }
        }
        // x | 0 -> x, 0 | x -> x, x | x -> x
        BinaryOp::Or => {
            if is_int_const(rhs_c, 0) {
                identity(lhs)
            } else if is_int_const(lhs_c, 0) {
                identity(rhs)
            } else if lhs == rhs {
                identity(lhs)
            } else {
                None
            }
        }
        // x ^ 0 -> x, 0 ^ x -> x, x ^ x -> 0
        BinaryOp::Xor => {
            if is_int_const(rhs_c, 0) {
                identity(lhs)
            } else if is_int_const(lhs_c, 0) {
                identity(rhs)
            } else if lhs == rhs {
                reg_types.get(lhs).copied().and_then(zero)
            } else {
                None
            }
        }
        _ => None,
    };
    result
}

/// Folds `op` over two constants of types `ty1`/`ty2` with the exact semantics the
/// code generators implement (two's-complement wrapping at the declared width,
/// shift amounts masked to the width, f32 arithmetic performed in f32).
/// Returns `None` when folding could hide a runtime error (division by zero, overflow).
fn fold_binary_op(
    op: BinaryOp,
    c1: &Constant,
    ty1: Type,
    c2: &Constant,
    ty2: Type,
) -> Option<(Constant, Type)> {
    let bool_const = |b: bool| Some((Constant::Int(b as i64), Type::I32));

    match (c1, c2) {
        (Constant::Int(a), Constant::Int(b)) => {
            // Pointer arithmetic and mixed-type forms are left to the code generator.
            if ty1 != ty2 || !ty1.is_int() {
                return None;
            }
            let ty = ty1;
            let bits = ty.int_bits()?;
            let a = ty.wrap_int(*a);
            let b = ty.wrap_int(*b);
            let min = if bits == 64 {
                i64::MIN
            } else {
                -(1i64 << (bits - 1))
            };
            let shift = (b as u32) & (bits - 1);

            let res = match op {
                BinaryOp::Add => a.wrapping_add(b),
                BinaryOp::Sub => a.wrapping_sub(b),
                BinaryOp::Mul => a.wrapping_mul(b),
                BinaryOp::Div => {
                    if b == 0 || (a == min && b == -1) {
                        return None;
                    }
                    a / b
                }
                BinaryOp::Rem => {
                    if b == 0 || (a == min && b == -1) {
                        return None;
                    }
                    a % b
                }
                BinaryOp::And => a & b,
                BinaryOp::Or => a | b,
                BinaryOp::Xor => a ^ b,
                BinaryOp::Shl => a.wrapping_shl(shift),
                BinaryOp::Shr => a >> shift,
                BinaryOp::Eq => return bool_const(a == b),
                BinaryOp::Ne => return bool_const(a != b),
                BinaryOp::Lt => return bool_const(a < b),
                BinaryOp::Gt => return bool_const(a > b),
                BinaryOp::Le => return bool_const(a <= b),
                BinaryOp::Ge => return bool_const(a >= b),
                BinaryOp::Min => a.min(b),
                BinaryOp::Max => a.max(b),
                BinaryOp::Umin => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    ua.min(ub) as i64
                }
                BinaryOp::Umax => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    ua.max(ub) as i64
                }
                BinaryOp::Udiv => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    if ub == 0 {
                        return None;
                    }
                    (ua / ub) as i64
                }
                BinaryOp::Urem => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    if ub == 0 {
                        return None;
                    }
                    (ua % ub) as i64
                }
                BinaryOp::Ushr => {
                    let ua = to_unsigned(a, bits);
                    (ua >> shift) as i64
                }
                BinaryOp::Ult => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    return bool_const(ua < ub);
                }
                BinaryOp::Ugt => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    return bool_const(ua > ub);
                }
                BinaryOp::Ule => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    return bool_const(ua <= ub);
                }
                BinaryOp::Uge => {
                    let ua = to_unsigned(a, bits);
                    let ub = to_unsigned(b, bits);
                    return bool_const(ua >= ub);
                }
            };
            Some((Constant::Int(ty.wrap_int(res)), ty))
        }
        (Constant::Float(a), Constant::Float(b)) => {
            if ty1 != ty2 || !ty1.is_float() {
                return None;
            }
            if ty1 == Type::F32 {
                let (a, b) = (*a as f32, *b as f32);
                let res = match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Eq => return bool_const(a == b),
                    BinaryOp::Ne => return bool_const(a != b),
                    BinaryOp::Lt => return bool_const(a < b),
                    BinaryOp::Gt => return bool_const(a > b),
                    BinaryOp::Le => return bool_const(a <= b),
                    BinaryOp::Ge => return bool_const(a >= b),
                    BinaryOp::Min => a.min(b),
                    BinaryOp::Max => a.max(b),
                    _ => return None,
                };
                Some((Constant::Float(res as f64), Type::F32))
            } else {
                let (a, b) = (*a, *b);
                let res = match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Eq => return bool_const(a == b),
                    BinaryOp::Ne => return bool_const(a != b),
                    BinaryOp::Lt => return bool_const(a < b),
                    BinaryOp::Gt => return bool_const(a > b),
                    BinaryOp::Le => return bool_const(a <= b),
                    BinaryOp::Ge => return bool_const(a >= b),
                    BinaryOp::Min => a.min(b),
                    BinaryOp::Max => a.max(b),
                    _ => return None,
                };
                Some((Constant::Float(res), Type::F64))
            }
        }
        _ => None,
    }
}

fn to_unsigned(v: i64, bits: u32) -> u64 {
    if bits == 64 {
        v as u64
    } else {
        (v as u64) & ((1u64 << bits) - 1)
    }
}

fn fold_unary_op(op: UnaryOp, c: &Constant, ty: Type) -> Option<(Constant, Type)> {
    match c {
        Constant::Int(n) => {
            if !ty.is_int() {
                return None;
            }
            let bits = ty.int_bits()?;
            let val = ty.wrap_int(*n);
            let min = if bits == 64 {
                i64::MIN
            } else {
                -(1i64 << (bits - 1))
            };
            let res = match op {
                UnaryOp::Neg => val.wrapping_neg(),
                UnaryOp::Abs => {
                    if val == min {
                        return None;
                    }
                    val.abs()
                }
                UnaryOp::Sqrt => return None,
            };
            Some((Constant::Int(ty.wrap_int(res)), ty))
        }
        Constant::Float(f) => {
            if !ty.is_float() {
                return None;
            }
            if ty == Type::F32 {
                let f = *f as f32;
                let res = match op {
                    UnaryOp::Neg => -f,
                    UnaryOp::Abs => f.abs(),
                    UnaryOp::Sqrt => {
                        if f < 0.0 {
                            return None;
                        }
                        f.sqrt()
                    }
                };
                Some((Constant::Float(res as f64), Type::F32))
            } else {
                let f = *f;
                let res = match op {
                    UnaryOp::Neg => -f,
                    UnaryOp::Abs => f.abs(),
                    UnaryOp::Sqrt => {
                        if f < 0.0 {
                            return None;
                        }
                        f.sqrt()
                    }
                };
                Some((Constant::Float(res), Type::F64))
            }
        }
    }
}

fn fold_cast_op(
    op: CastOp,
    c: &Constant,
    src_ty: Type,
    target_ty: Type,
) -> Option<(Constant, Type)> {
    match (op, c) {
        (CastOp::Itof, Constant::Int(n)) => {
            let val = src_ty.wrap_int(*n);
            if target_ty == Type::F32 {
                Some((Constant::Float((val as f32) as f64), Type::F32))
            } else if target_ty == Type::F64 {
                Some((Constant::Float(val as f64), Type::F64))
            } else {
                None
            }
        }
        (CastOp::Ftoi, Constant::Float(f)) => {
            if !target_ty.is_int() {
                return None;
            }
            let f = if src_ty == Type::F32 {
                *f as f32 as f64
            } else {
                *f
            };
            let bits = target_ty.int_bits()?;
            let (min, max) = if bits == 64 {
                (i64::MIN, i64::MAX)
            } else {
                (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
            };
            let int_val = if f.is_nan() {
                0
            } else if f <= min as f64 {
                min
            } else if f >= max as f64 {
                max
            } else {
                f as i64
            };
            Some((Constant::Int(target_ty.wrap_int(int_val)), target_ty))
        }
        (CastOp::Sext, Constant::Int(n)) => {
            if !src_ty.is_int()
                || !target_ty.is_int()
                || target_ty.int_bits()? <= src_ty.int_bits()?
            {
                return None;
            }
            let val = src_ty.wrap_int(*n);
            Some((Constant::Int(target_ty.wrap_int(val)), target_ty))
        }
        (CastOp::Zext, Constant::Int(n)) => {
            if !src_ty.is_int()
                || !target_ty.is_int()
                || target_ty.int_bits()? <= src_ty.int_bits()?
            {
                return None;
            }
            let bits = src_ty.int_bits()?;
            let uval = to_unsigned(*n, bits) as i64;
            Some((Constant::Int(target_ty.wrap_int(uval)), target_ty))
        }
        (CastOp::Trunc, Constant::Int(n)) => {
            if !src_ty.is_int()
                || !target_ty.is_int()
                || target_ty.int_bits()? >= src_ty.int_bits()?
            {
                return None;
            }
            Some((Constant::Int(target_ty.wrap_int(*n)), target_ty))
        }
        (CastOp::Fext, Constant::Float(f)) => {
            if src_ty == Type::F32 && target_ty == Type::F64 {
                Some((Constant::Float((*f as f32) as f64), Type::F64))
            } else {
                None
            }
        }
        (CastOp::Ftrunc, Constant::Float(f)) => {
            if src_ty == Type::F64 && target_ty == Type::F32 {
                Some((Constant::Float((*f as f32) as f64), Type::F32))
            } else {
                None
            }
        }
        (CastOp::Bitcast, Constant::Int(n)) => {
            if src_ty == Type::I32 && target_ty == Type::F32 {
                let f = f32::from_bits(*n as u32);
                Some((Constant::Float(f as f64), Type::F32))
            } else if src_ty == Type::I64 && target_ty == Type::F64 {
                let f = f64::from_bits(*n as u64);
                Some((Constant::Float(f), Type::F64))
            } else {
                None
            }
        }
        (CastOp::Bitcast, Constant::Float(f)) => {
            if src_ty == Type::F32 && target_ty == Type::I32 {
                let bits = (*f as f32).to_bits() as i32 as i64;
                Some((Constant::Int(target_ty.wrap_int(bits)), Type::I32))
            } else if src_ty == Type::F64 && target_ty == Type::I64 {
                let bits = f.to_bits() as i64;
                Some((Constant::Int(bits), Type::I64))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn substitute_instruction_operands(
    inst: &mut Instruction,
    substitutions: &HashMap<String, String>,
) {
    if substitutions.is_empty() {
        return;
    }
    for reg in inst.operands_mut() {
        if let Some(new_reg) = substitutions.get(reg.as_str()) {
            *reg = new_reg.clone();
        }
    }
}

fn substitute_terminator_operands(term: &mut Terminator, substitutions: &HashMap<String, String>) {
    if substitutions.is_empty() {
        return;
    }
    match term {
        Terminator::Jmp { args, .. } => {
            for arg in args {
                if let Some(new_a) = substitutions.get(arg) {
                    *arg = new_a.clone();
                }
            }
        }
        Terminator::Br {
            cond,
            then_args,
            else_args,
            ..
        } => {
            if let Some(new_c) = substitutions.get(cond) {
                *cond = new_c.clone();
            }
            for arg in then_args {
                if let Some(new_a) = substitutions.get(arg) {
                    *arg = new_a.clone();
                }
            }
            for arg in else_args {
                if let Some(new_a) = substitutions.get(arg) {
                    *arg = new_a.clone();
                }
            }
        }
        Terminator::Ret { val, .. } => {
            if let Some(v) = val {
                if let Some(new_v) = substitutions.get(v) {
                    *v = new_v.clone();
                }
            }
        }
    }
}

/// Folds conditional branches with known constant conditions:
/// `br cst_1, then, else -> jmp then`
/// `br cst_0, then, else -> jmp else`
fn run_branch_folding_pass(func: &mut Function) -> usize {
    let mut folded = 0;

    let mut constants: HashMap<String, Constant> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::AssignConst { dst, val, .. } = inst {
                constants.insert(dst.clone(), val.clone());
            }
        }
    }

    for block in &mut func.blocks {
        if let Terminator::Br {
            cond,
            then_block,
            then_args,
            else_block,
            else_args,
            span,
        } = &block.terminator
        {
            if let Some(Constant::Int(val)) = constants.get(cond) {
                if *val != 0 {
                    // Always true branch
                    block.terminator = Terminator::Jmp {
                        target: then_block.clone(),
                        args: then_args.clone(),
                        span: *span,
                    };
                    folded += 1;
                } else {
                    // Always false branch
                    block.terminator = Terminator::Jmp {
                        target: else_block.clone(),
                        args: else_args.clone(),
                        span: *span,
                    };
                    folded += 1;
                }
            }
        }
    }

    folded
}

/// Prunes blocks that are unreachable from the entry block.
fn run_dead_block_elimination(func: &mut Function) -> usize {
    if func.blocks.is_empty() {
        return 0;
    }

    let mut reachable = HashSet::new();
    let entry_label = func.blocks[0].label.clone();
    let mut worklist = vec![entry_label];

    let block_map: HashMap<String, &Block> =
        func.blocks.iter().map(|b| (b.label.clone(), b)).collect();

    while let Some(label) = worklist.pop() {
        if reachable.insert(label.clone()) {
            if let Some(block) = block_map.get(&label) {
                match &block.terminator {
                    Terminator::Jmp { target, .. } => {
                        worklist.push(target.clone());
                    }
                    Terminator::Br {
                        then_block,
                        else_block,
                        ..
                    } => {
                        worklist.push(then_block.clone());
                        worklist.push(else_block.clone());
                    }
                    Terminator::Ret { .. } => {}
                }
            }
        }
    }

    let initial_count = func.blocks.len();
    func.blocks.retain(|b| reachable.contains(&b.label));
    initial_count - func.blocks.len()
}

/// Dead Code Elimination: removes pure instructions whose destination is never used.
fn run_dead_code_elimination(func: &mut Function, sigs: &Signatures) -> usize {
    let mut total_removed = 0;

    loop {
        // Collect all used variables
        let mut used = HashSet::new();

        for block in &func.blocks {
            // Instructions
            for inst in &block.instructions {
                for reg in inst.operands() {
                    used.insert(reg.clone());
                }
            }

            // Terminator
            match &block.terminator {
                Terminator::Jmp { args, .. } => {
                    for a in args {
                        used.insert(a.clone());
                    }
                }
                Terminator::Br {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => {
                    used.insert(cond.clone());
                    for a in then_args {
                        used.insert(a.clone());
                    }
                    for a in else_args {
                        used.insert(a.clone());
                    }
                }
                Terminator::Ret { val, .. } => {
                    if let Some(v) = val {
                        used.insert(v.clone());
                    }
                }
            }
        }

        // Remove pure instructions whose destination is NOT in `used`.
        // Integer div/rem raise runtime errors (zero divisor, MIN / -1), so they are
        // observable even when unused; they are kept unless provably float.
        let reg_types = infer_reg_types(func, sigs);
        let mut pass_removed = 0;
        for block in &mut func.blocks {
            let before_len = block.instructions.len();
            block.instructions.retain(|inst| match inst {
                Instruction::AssignConst { dst, .. } => used.contains(dst),
                Instruction::Binary { op, dst, lhs, .. } => {
                    let can_trap = matches!(
                        op,
                        BinaryOp::Div | BinaryOp::Rem | BinaryOp::Udiv | BinaryOp::Urem
                    ) && !reg_types.get(lhs).is_some_and(|t| t.is_float());
                    used.contains(dst) || can_trap
                }
                Instruction::Select { dst, .. } => used.contains(dst),
                Instruction::Unary { dst, .. } => used.contains(dst),
                Instruction::Cast { dst, .. } => used.contains(dst),
                Instruction::VectorReduce { dst, .. } => used.contains(dst),
                Instruction::Splat { dst, .. } => used.contains(dst),
                Instruction::ExtractLane { dst, .. } => used.contains(dst),
                Instruction::VBinary { dst, .. }
                | Instruction::VFma { dst, .. }
                | Instruction::VCmp { dst, .. }
                | Instruction::VSelect { dst, .. }
                | Instruction::VLen { dst, .. } => used.contains(dst),

                // Effectful instructions must never be eliminated
                Instruction::Load { .. }
                | Instruction::Store { .. }
                | Instruction::Call { .. }
                | Instruction::Alloc { .. }
                | Instruction::Free { .. } => true,
            });
            pass_removed += before_len - block.instructions.len();
        }

        if pass_removed == 0 {
            break;
        }
        total_removed += pass_removed;
    }

    total_removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_and_validate;

    #[test]
    fn test_constant_folding() {
        let code = r#"
        fn test_fold()->i32
          b0:
            a = cst 10:i32
            b = cst 25:i32
            c = add a, b
            ret c
        "#;
        let mut module = parse_and_validate(code).unwrap();
        let stats = optimize_module(&mut module);

        assert!(stats.constants_folded >= 1);
        let fn_0 = &module.functions[0];
        // After constant folding and DCE of dead constants, `c` should be folded to `cst 35:i32`
        let last_inst = &fn_0.blocks[0].instructions.last().unwrap();
        if let Instruction::AssignConst { val, .. } = last_inst {
            assert_eq!(*val, Constant::Int(35));
        } else {
            panic!("Expected AssignConst, found {:?}", last_inst);
        }
    }

    #[test]
    fn test_algebraic_simplification() {
        let code = r#"
        fn test_alg(x:i32)->i32
          b0:
            zero = cst 0:i32
            one = cst 1:i32
            a = add x, zero
            b = mul a, one
            ret b
        "#;
        let mut module = parse_and_validate(code).unwrap();
        let stats = optimize_module(&mut module);

        assert!(stats.algebraic_simplifications >= 1);
        // After simplification and DCE, b should return x
        let fn_0 = &module.functions[0];
        if let Terminator::Ret { val, .. } = &fn_0.blocks[0].terminator {
            assert_eq!(val.as_deref(), Some("x"));
        } else {
            panic!("Expected Ret x");
        }
    }

    #[test]
    fn test_dead_block_elimination() {
        let code = r#"
        fn test_dead()->i32
          b0:
            c = cst 42:i32
            ret c
          b_unreachable:
            dead_val = cst 100:i32
            ret dead_val
        "#;
        let mut module = parse_and_validate(code).unwrap();
        let stats = optimize_module(&mut module);

        assert_eq!(stats.dead_blocks_removed, 1);
        assert_eq!(module.functions[0].blocks.len(), 1);
        assert_eq!(module.functions[0].blocks[0].label, "b0");
    }

    #[test]
    fn test_dead_code_elimination() {
        let code = r#"
        fn test_dce(x:i32)->i32
          b0:
            dead1 = cst 999:i32
            dead2 = add dead1, x
            live = add x, x
            ret live
        "#;
        let mut module = parse_and_validate(code).unwrap();
        let stats = optimize_module(&mut module);

        assert!(stats.dead_instructions_removed >= 2);
        let block = &module.functions[0].blocks[0];
        for inst in &block.instructions {
            match inst {
                Instruction::AssignConst { dst, .. } => assert_ne!(dst, "dead1"),
                Instruction::Binary { dst, .. } => assert_ne!(dst, "dead2"),
                _ => {}
            }
        }
    }
}
