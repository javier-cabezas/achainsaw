use crate::ast::*;
use crate::diag::Span;
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

/// Optimizes an entire IR module in-place, returning summary statistics.
pub fn optimize_module(module: &mut Module) -> OptimizationStats {
    let mut total_stats = OptimizationStats::default();

    for func in &mut module.functions {
        let stats = optimize_function(func);
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
pub fn optimize_function(func: &mut Function) -> OptimizationStats {
    let mut stats = OptimizationStats::default();
    let max_iters = 16;

    for iter in 1..=max_iters {
        stats.iterations = iter;
        let mut changed = false;

        // 1. Constant folding & propagation + Algebraic simplification
        let (folded, simplified) = run_constant_and_algebraic_pass(func);
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
        let dce = run_dead_code_elimination(func);
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

/// Runs constant folding, constant propagation, and algebraic simplification.
fn run_constant_and_algebraic_pass(func: &mut Function) -> (usize, usize) {
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

    // Now scan and simplify instructions
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            // Apply variable substitutions to operands
            substitute_instruction_operands(inst, &substitutions);

            let binary_info = match inst {
                Instruction::AssignConst { dst, val, ty, .. } => {
                    constants.insert(dst.clone(), (val.clone(), *ty));
                    None
                }
                Instruction::Binary {
                    op,
                    dst,
                    lhs,
                    rhs,
                    span,
                } => Some((*op, dst.clone(), lhs.clone(), rhs.clone(), *span)),
                _ => None,
            };

            if let Some((op, dst, lhs, rhs, span)) = binary_info {
                // Try full constant folding if both operands are constants
                let lhs_c = constants.get(&lhs).cloned();
                let rhs_c = constants.get(&rhs).cloned();

                if let (Some((c1, ty1)), Some((c2, _ty2))) = (lhs_c, rhs_c) {
                    if let Some((folded_const, folded_ty)) = fold_binary_op(op, &c1, &c2, ty1) {
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

                // Try algebraic simplification if one operand is a constant or both are identical
                if let Some(res) = simplify_algebraic(op, &dst, &lhs, &rhs, &constants, span) {
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
                            let real_source = substitutions.get(&source).cloned().unwrap_or(source);
                            substitutions.insert(dst, real_source);
                            simplified_count += 1;
                        }
                    }
                }
            }
        }

        // Substitute terminator operands
        substitute_terminator_operands(&mut block.terminator, &substitutions);
    }

    (folded_count, simplified_count)
}

enum AlgebraicResult {
    Constant(Constant, Type),
    Identity(String),
}

fn simplify_algebraic(
    op: BinaryOp,
    _dst: &str,
    lhs: &str,
    rhs: &str,
    constants: &HashMap<String, (Constant, Type)>,
    _span: Span,
) -> Option<AlgebraicResult> {
    let lhs_c = constants.get(lhs);
    let rhs_c = constants.get(rhs);

    match op {
        // x + 0 -> x, 0 + x -> x
        BinaryOp::Add => {
            if let Some((Constant::Int(0), _)) = rhs_c {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
            if let Some((Constant::Int(0), _)) = lhs_c {
                return Some(AlgebraicResult::Identity(rhs.to_string()));
            }
        }
        // x - 0 -> x, x - x -> 0
        BinaryOp::Sub => {
            if let Some((Constant::Int(0), _)) = rhs_c {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
            if lhs == rhs {
                let ty = lhs_c.map(|(_, t)| *t).unwrap_or(Type::I32);
                return Some(AlgebraicResult::Constant(Constant::Int(0), ty));
            }
        }
        // x * 1 -> x, 1 * x -> x, x * 0 -> 0, 0 * x -> 0
        BinaryOp::Mul => {
            if let Some((Constant::Int(1), _)) = rhs_c {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
            if let Some((Constant::Int(1), _)) = lhs_c {
                return Some(AlgebraicResult::Identity(rhs.to_string()));
            }
            if let Some((Constant::Int(0), ty)) = rhs_c {
                return Some(AlgebraicResult::Constant(Constant::Int(0), *ty));
            }
            if let Some((Constant::Int(0), ty)) = lhs_c {
                return Some(AlgebraicResult::Constant(Constant::Int(0), *ty));
            }
        }
        // x / 1 -> x
        BinaryOp::Div => {
            if let Some((Constant::Int(1), _)) = rhs_c {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
        }
        // x & 0 -> 0, 0 & x -> 0
        BinaryOp::And => {
            if let Some((Constant::Int(0), ty)) = rhs_c {
                return Some(AlgebraicResult::Constant(Constant::Int(0), *ty));
            }
            if let Some((Constant::Int(0), ty)) = lhs_c {
                return Some(AlgebraicResult::Constant(Constant::Int(0), *ty));
            }
            if lhs == rhs {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
        }
        // x | 0 -> x, 0 | x -> x
        BinaryOp::Or => {
            if let Some((Constant::Int(0), _)) = rhs_c {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
            if let Some((Constant::Int(0), _)) = lhs_c {
                return Some(AlgebraicResult::Identity(rhs.to_string()));
            }
            if lhs == rhs {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
        }
        // x ^ 0 -> x, 0 ^ x -> x, x ^ x -> 0
        BinaryOp::Xor => {
            if let Some((Constant::Int(0), _)) = rhs_c {
                return Some(AlgebraicResult::Identity(lhs.to_string()));
            }
            if let Some((Constant::Int(0), _)) = lhs_c {
                return Some(AlgebraicResult::Identity(rhs.to_string()));
            }
            if lhs == rhs {
                let ty = lhs_c.map(|(_, t)| *t).unwrap_or(Type::I32);
                return Some(AlgebraicResult::Constant(Constant::Int(0), ty));
            }
        }
        _ => {}
    }

    None
}

fn fold_binary_op(
    op: BinaryOp,
    c1: &Constant,
    c2: &Constant,
    ty: Type,
) -> Option<(Constant, Type)> {
    match (c1, c2) {
        (Constant::Int(a), Constant::Int(b)) => {
            let res = match op {
                BinaryOp::Add => a.wrapping_add(*b),
                BinaryOp::Sub => a.wrapping_sub(*b),
                BinaryOp::Mul => a.wrapping_mul(*b),
                BinaryOp::Div => {
                    if *b == 0 {
                        return None;
                    }
                    a.checked_div(*b)?
                }
                BinaryOp::Rem => {
                    if *b == 0 {
                        return None;
                    }
                    a.checked_rem(*b)?
                }
                BinaryOp::And => a & b,
                BinaryOp::Or => a | b,
                BinaryOp::Xor => a ^ b,
                BinaryOp::Shl => a.wrapping_shl((*b & 63) as u32),
                BinaryOp::Shr => a.wrapping_shr((*b & 63) as u32),
                BinaryOp::Eq => {
                    return Some((Constant::Int(if a == b { 1 } else { 0 }), Type::I32))
                }
                BinaryOp::Ne => {
                    return Some((Constant::Int(if a != b { 1 } else { 0 }), Type::I32))
                }
                BinaryOp::Lt => return Some((Constant::Int(if a < b { 1 } else { 0 }), Type::I32)),
                BinaryOp::Gt => return Some((Constant::Int(if a > b { 1 } else { 0 }), Type::I32)),
                BinaryOp::Le => {
                    return Some((Constant::Int(if a <= b { 1 } else { 0 }), Type::I32))
                }
                BinaryOp::Ge => {
                    return Some((Constant::Int(if a >= b { 1 } else { 0 }), Type::I32))
                }
                _ => return None,
            };
            Some((Constant::Int(res), ty))
        }
        (Constant::Float(a), Constant::Float(b)) => {
            let res = match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                BinaryOp::Eq => {
                    return Some((Constant::Int(if a == b { 1 } else { 0 }), Type::I32))
                }
                BinaryOp::Ne => {
                    return Some((Constant::Int(if a != b { 1 } else { 0 }), Type::I32))
                }
                BinaryOp::Lt => return Some((Constant::Int(if a < b { 1 } else { 0 }), Type::I32)),
                BinaryOp::Gt => return Some((Constant::Int(if a > b { 1 } else { 0 }), Type::I32)),
                BinaryOp::Le => {
                    return Some((Constant::Int(if a <= b { 1 } else { 0 }), Type::I32))
                }
                BinaryOp::Ge => {
                    return Some((Constant::Int(if a >= b { 1 } else { 0 }), Type::I32))
                }
                _ => return None,
            };
            Some((Constant::Float(res), ty))
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
    match inst {
        Instruction::AssignConst { .. } => {}
        Instruction::Binary { lhs, rhs, .. } => {
            if let Some(new_l) = substitutions.get(lhs) {
                *lhs = new_l.clone();
            }
            if let Some(new_r) = substitutions.get(rhs) {
                *rhs = new_r.clone();
            }
        }
        Instruction::Load { ptr, .. } => {
            if let Some(new_p) = substitutions.get(ptr) {
                *ptr = new_p.clone();
            }
        }
        Instruction::Store { ptr, val, .. } => {
            if let Some(new_p) = substitutions.get(ptr) {
                *ptr = new_p.clone();
            }
            if let Some(new_v) = substitutions.get(val) {
                *val = new_v.clone();
            }
        }
        Instruction::Call { args, .. } => {
            for arg in args {
                if let Some(new_a) = substitutions.get(arg) {
                    *arg = new_a.clone();
                }
            }
        }
        Instruction::Splat { src, .. } => {
            if let Some(new_s) = substitutions.get(src) {
                *src = new_s.clone();
            }
        }
        Instruction::ExtractLane { vec, .. } => {
            if let Some(new_v) = substitutions.get(vec) {
                *vec = new_v.clone();
            }
        }
        Instruction::Alloc { size, .. } => {
            if let Some(new_s) = substitutions.get(size) {
                *size = new_s.clone();
            }
        }
        Instruction::Free { ptr, .. } => {
            if let Some(new_p) = substitutions.get(ptr) {
                *ptr = new_p.clone();
            }
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
fn run_dead_code_elimination(func: &mut Function) -> usize {
    let mut total_removed = 0;

    loop {
        // Collect all used variables
        let mut used = HashSet::new();

        for block in &func.blocks {
            // Instructions
            for inst in &block.instructions {
                match inst {
                    Instruction::AssignConst { .. } => {}
                    Instruction::Binary { lhs, rhs, .. } => {
                        used.insert(lhs.clone());
                        used.insert(rhs.clone());
                    }
                    Instruction::Load { ptr, .. } => {
                        used.insert(ptr.clone());
                    }
                    Instruction::Store { ptr, val, .. } => {
                        used.insert(ptr.clone());
                        used.insert(val.clone());
                    }
                    Instruction::Call { args, .. } => {
                        for a in args {
                            used.insert(a.clone());
                        }
                    }
                    Instruction::Splat { src, .. } => {
                        used.insert(src.clone());
                    }
                    Instruction::ExtractLane { vec, .. } => {
                        used.insert(vec.clone());
                    }
                    Instruction::Alloc { size, .. } => {
                        used.insert(size.clone());
                    }
                    Instruction::Free { ptr, .. } => {
                        used.insert(ptr.clone());
                    }
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

        // Remove pure instructions whose destination is NOT in `used`
        let mut pass_removed = 0;
        for block in &mut func.blocks {
            let before_len = block.instructions.len();
            block.instructions.retain(|inst| match inst {
                Instruction::AssignConst { dst, .. } => used.contains(dst),
                Instruction::Binary { dst, .. } => used.contains(dst),
                Instruction::Splat { dst, .. } => used.contains(dst),
                Instruction::ExtractLane { dst, .. } => used.contains(dst),
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
