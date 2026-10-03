use crate::ast::*;
use crate::cfg::Cfg;
use crate::diag::{Diagnostic, Span};
use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// Prefix reserved for compiler runtime symbols; user functions may not use it.
pub const RESERVED_PREFIX: &str = "__achainsaw";

pub struct Validator {
    // Function table: name -> (param types, return type)
    functions: HashMap<String, (Vec<Type>, Option<Type>)>,
}

impl Validator {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the result type of a binary op, or `(error_code, message)` if the
/// operand types are illegal for it. Shared with the optimizer so both agree.
pub fn binary_result_type(
    op: BinaryOp,
    lhs: Type,
    rhs: Type,
) -> Result<Type, (&'static str, String)> {
    use BinaryOp::*;
    let mismatch = |msg: String| Err(("ERR_TYPE_MISMATCH", msg));

    if op.is_vector() {
        if lhs != Type::V128 || rhs != Type::V128 {
            return mismatch(format!(
                "Vector op '{op:?}' requires operands of type 'v128', found '{lhs}' and '{rhs}'"
            ));
        }
        return Ok(Type::V128);
    }
    if lhs == Type::V128 || rhs == Type::V128 {
        return mismatch(format!(
            "Scalar op '{op:?}' cannot be applied to 'v128' (found '{lhs}' and '{rhs}'); use vfadd/vfsub/vfmul/vfdiv/viadd/visub/vimul"
        ));
    }

    // ptr (+|-) i64 and i64 + ptr
    if (lhs == Type::Ptr && rhs == Type::I64) || (lhs == Type::I64 && rhs == Type::Ptr) {
        let ok = op == Add || (op == Sub && lhs == Type::Ptr);
        if !ok {
            return Err((
                "ERR_INVALID_PTR_OP",
                format!(
                    "Invalid pointer operation '{op:?}' on '{lhs}' and '{rhs}'. Only 'add' (either order) and 'sub' (ptr - i64) are allowed"
                ),
            ));
        }
        return Ok(Type::Ptr);
    }

    if lhs != rhs {
        return mismatch(format!(
            "Binary op '{op:?}' operands must have matching types, found '{lhs}' and '{rhs}'"
        ));
    }

    let ty = lhs;
    match ty {
        Type::Ptr => match op {
            Eq | Ne | Lt | Gt | Le | Ge => Ok(Type::I32),
            Sub => Ok(Type::I64), // pointer difference in bytes
            _ => Err((
                "ERR_INVALID_PTR_OP",
                format!("Invalid pointer operation '{op:?}' on two 'ptr' values"),
            )),
        },
        Type::F32 | Type::F64 => match op {
            Add | Sub | Mul | Div => Ok(ty),
            Eq | Ne | Lt | Gt | Le | Ge => Ok(Type::I32),
            _ => Err((
                "ERR_INVALID_OP_FOR_TYPE",
                format!("Operation '{op:?}' is not defined for floating-point type '{ty}'"),
            )),
        },
        _ => {
            // I8 / I16 / I32 / I64
            if op.is_comparison() {
                Ok(Type::I32)
            } else {
                Ok(ty)
            }
        }
    }
}

/// Per-function validation context.
struct FnCtx<'a> {
    func: &'a Function,
    /// Parameter types of every block, keyed by label.
    block_params: HashMap<&'a str, &'a [(String, Type)]>,
    /// Which block defines each register (first definition wins). Used only to
    /// produce `ERR_NON_DOMINATING_USE` instead of a vague "undefined" error.
    def_blocks: HashMap<&'a str, &'a str>,
}

/// Registers visible at a program point (those defined in dominating code).
type Visible = HashMap<String, Type>;

impl Validator {
    pub fn validate_module(&mut self, module: &Module) -> Result<(), Diagnostic> {
        // Collect function signatures
        for ext_fn in &module.extern_functions {
            Self::check_reserved(&ext_fn.name, ext_fn.span)?;
            if self.functions.contains_key(&ext_fn.name) {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_FUNCTION",
                    format!(
                        "Duplicate function name '{}' (already declared)",
                        ext_fn.name
                    ),
                    ext_fn.span,
                ));
            }
            let param_types = ext_fn.params.iter().map(|(_, ty)| *ty).collect();
            self.functions
                .insert(ext_fn.name.clone(), (param_types, ext_fn.ret_type));
        }

        for func in &module.functions {
            Self::check_reserved(&func.name, func.span)?;
            if self.functions.contains_key(&func.name) {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_FUNCTION",
                    format!("Duplicate function name '{}'", func.name),
                    func.span,
                ));
            }
            let param_types = func.params.iter().map(|(_, ty)| *ty).collect();
            self.functions
                .insert(func.name.clone(), (param_types, func.ret_type));
        }

        // Validate each function
        for func in &module.functions {
            self.validate_function(func)?;
        }

        Ok(())
    }

    fn check_reserved(name: &str, span: Span) -> Result<(), Diagnostic> {
        if name.starts_with(RESERVED_PREFIX) {
            return Err(Diagnostic::error(
                "ERR_RESERVED_NAME",
                format!("Function name '{name}' uses the reserved prefix '{RESERVED_PREFIX}'"),
                span,
            ));
        }
        Ok(())
    }

    fn validate_function(&self, func: &Function) -> Result<(), Diagnostic> {
        if func.blocks.is_empty() {
            return Err(Diagnostic::error(
                "ERR_EMPTY_FUNCTION",
                format!("Function '{}' must have at least one basic block", func.name),
                func.span,
            ));
        }

        let mut block_params: HashMap<&str, &[(String, Type)]> = HashMap::new();
        for block in &func.blocks {
            if block_params
                .insert(block.label.as_str(), block.params.as_slice())
                .is_some()
            {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_BLOCK",
                    format!(
                        "Duplicate block label '{}' in function '{}'",
                        block.label, func.name
                    ),
                    block.span,
                ));
            }
        }

        let entry = &func.blocks[0];
        if !entry.params.is_empty() {
            return Err(Diagnostic::error(
                "ERR_ENTRY_BLOCK_PARAMS",
                format!(
                    "Entry block '{}' of function '{}' must not declare parameters; function parameters are in scope implicitly",
                    entry.label, func.name
                ),
                entry.span,
            )
            .with_context(serde_json::json!({
                "function": func.name,
                "block": entry.label,
            })));
        }

        // Branch targets must exist, and nothing may branch back to the entry block.
        for block in &func.blocks {
            let targets: Vec<(&String, Span)> = match &block.terminator {
                Terminator::Jmp { target, span, .. } => vec![(target, *span)],
                Terminator::Br {
                    then_block,
                    else_block,
                    span,
                    ..
                } => vec![(then_block, *span), (else_block, *span)],
                Terminator::Ret { .. } => vec![],
            };
            for (target, span) in targets {
                if !block_params.contains_key(target.as_str()) {
                    return Err(Diagnostic::error(
                        "ERR_UNDEFINED_BLOCK",
                        format!("Branch target block '{target}' does not exist"),
                        span,
                    )
                    .with_instruction_index(block.instructions.len())
                    .with_context(serde_json::json!({
                        "function": func.name,
                        "block": block.label,
                        "target": target,
                        "available_blocks": func.blocks.iter().map(|b| b.label.clone()).collect::<Vec<_>>(),
                    })));
                }
                if *target == entry.label {
                    return Err(Diagnostic::error(
                        "ERR_BRANCH_TO_ENTRY",
                        format!(
                            "Block '{}' branches to the entry block '{target}'; the entry block cannot be a branch target (use a separate loop header block)",
                            block.label
                        ),
                        span,
                    )
                    .with_instruction_index(block.instructions.len())
                    .with_context(serde_json::json!({
                        "function": func.name,
                        "block": block.label,
                        "target": target,
                    })));
                }
            }
        }

        // Pre-pass: where is each register defined?
        let mut def_blocks: HashMap<&str, &str> = HashMap::new();
        for block in &func.blocks {
            for (name, _) in &block.params {
                def_blocks.entry(name.as_str()).or_insert(&block.label);
            }
            for inst in &block.instructions {
                if let Some(dst) = inst_dst(inst) {
                    def_blocks.entry(dst).or_insert(&block.label);
                }
            }
        }

        let ctx = FnCtx {
            func,
            block_params,
            def_blocks,
        };

        // Function parameters are visible everywhere.
        let mut defs: HashSet<String> = HashSet::new();
        let mut base_scope: Visible = HashMap::new();
        for (p_name, p_ty) in &func.params {
            if !defs.insert(p_name.clone()) {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_PARAM",
                    format!("Duplicate parameter '{p_name}' in function '{}'", func.name),
                    func.span,
                ));
            }
            base_scope.insert(p_name.clone(), *p_ty);
        }

        // Validate blocks in reverse post-order so every block starts from the final
        // scope of its immediate dominator. Unreachable blocks only see parameters.
        let cfg = Cfg::build(func);
        let mut order: Vec<usize> = cfg.rpo.clone();
        order.extend((0..func.blocks.len()).filter(|&i| !cfg.reachable[i]));

        let mut final_scopes: Vec<Option<Visible>> = vec![None; func.blocks.len()];
        for &b_idx in &order {
            let block = &func.blocks[b_idx];
            let mut scope: Visible = if b_idx != 0 && cfg.reachable[b_idx] {
                final_scopes[cfg.idom[b_idx]]
                    .clone()
                    .unwrap_or_else(|| base_scope.clone())
            } else {
                base_scope.clone()
            };

            for (p_name, p_ty) in &block.params {
                if !defs.insert(p_name.clone()) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!(
                            "Register '{p_name}' re-defined in block '{}' (violates SSA)",
                            block.label
                        ),
                        block.span,
                    )
                    .with_context(serde_json::json!({
                        "function": func.name,
                        "block": block.label,
                        "target": p_name,
                    })));
                }
                scope.insert(p_name.clone(), *p_ty);
            }

            for (i_idx, inst) in block.instructions.iter().enumerate() {
                self.validate_instruction(&ctx, inst, &mut scope, &mut defs)
                    .map_err(|d| annotate(d, func, block, i_idx))?;
            }

            self.validate_terminator(&ctx, &block.terminator, &scope)
                .map_err(|d| annotate(d, func, block, block.instructions.len()))?;

            final_scopes[b_idx] = Some(scope);
        }

        Ok(())
    }

    fn define(
        scope: &mut Visible,
        defs: &mut HashSet<String>,
        dst: &str,
        ty: Type,
        span: Span,
    ) -> Result<(), Diagnostic> {
        if !defs.insert(dst.to_string()) {
            return Err(Diagnostic::error(
                "ERR_SSA_REDEFINITION",
                format!("Register '{dst}' assigned multiple times (violates SSA)"),
                span,
            )
            .with_context(serde_json::json!({ "target": dst })));
        }
        scope.insert(dst.to_string(), ty);
        Ok(())
    }

    fn validate_instruction(
        &self,
        ctx: &FnCtx,
        inst: &Instruction,
        scope: &mut Visible,
        defs: &mut HashSet<String>,
    ) -> Result<(), Diagnostic> {
        match inst {
            Instruction::AssignConst { dst, val, ty, span } => {
                match (val, ty) {
                    (Constant::Int(n), Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Ptr) => {
                        if !ty.int_literal_fits(*n) {
                            return Err(Diagnostic::error(
                                "ERR_CONSTANT_OUT_OF_RANGE",
                                format!("Constant {n} does not fit in type '{ty}'"),
                                *span,
                            )
                            .with_context(serde_json::json!({ "value": n, "type": ty.as_str() })));
                        }
                    }
                    (Constant::Float(_), Type::F32 | Type::F64) => {}
                    _ => {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Constant value does not match specified type '{ty}'"),
                            *span,
                        ));
                    }
                }
                Self::define(scope, defs, dst, *ty, *span)?;
            }
            Instruction::Binary {
                op,
                dst,
                lhs,
                rhs,
                span,
            } => {
                let lhs_ty = self.check_reg(ctx, lhs, scope, *span)?;
                let rhs_ty = self.check_reg(ctx, rhs, scope, *span)?;
                let out_ty = binary_result_type(*op, lhs_ty, rhs_ty)
                    .map_err(|(code, msg)| Diagnostic::error(code, msg, *span))?;
                Self::define(scope, defs, dst, out_ty, *span)?;
            }
            Instruction::Load { dst, ptr, ty, span } => {
                let ptr_ty = self.check_reg(ctx, ptr, scope, *span)?;
                if ptr_ty != Type::Ptr {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!(
                            "Load pointer operand '{ptr}' must be of type 'ptr', found '{ptr_ty}'"
                        ),
                        *span,
                    ));
                }
                Self::define(scope, defs, dst, *ty, *span)?;
            }
            Instruction::Store { ptr, val, span } => {
                let ptr_ty = self.check_reg(ctx, ptr, scope, *span)?;
                if ptr_ty != Type::Ptr {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!(
                            "Store target pointer '{ptr}' must be of type 'ptr', found '{ptr_ty}'"
                        ),
                        *span,
                    ));
                }
                self.check_reg(ctx, val, scope, *span)?;
            }
            Instruction::Call {
                dst,
                func,
                args,
                span,
            } => {
                let (param_types, ret_type) = self.functions.get(func).ok_or_else(|| {
                    Diagnostic::error(
                        "ERR_UNDEFINED_FUNCTION",
                        format!("Call to undefined function '{func}'"),
                        *span,
                    )
                })?;

                if args.len() != param_types.len() {
                    return Err(Diagnostic::error(
                        "ERR_ARITY_MISMATCH",
                        format!(
                            "Function '{func}' expects {} arguments, received {}",
                            param_types.len(),
                            args.len()
                        ),
                        *span,
                    ));
                }

                for (arg, &expected_ty) in args.iter().zip(param_types) {
                    let actual_ty = self.check_reg(ctx, arg, scope, *span)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Argument '{arg}' passed to '{func}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                }

                if let Some(d) = dst {
                    let rty = ret_type.ok_or_else(|| {
                        Diagnostic::error(
                            "ERR_VOID_ASSIGNMENT",
                            format!("Function '{func}' does not return a value"),
                            *span,
                        )
                    })?;
                    Self::define(scope, defs, d, rty, *span)?;
                }
            }
            Instruction::Splat { dst, src, span } => {
                let src_ty = self.check_reg(ctx, src, scope, *span)?;
                if src_ty.lane_count().is_none() {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Splat operand '{src}' must be numeric scalar, found '{src_ty}'"),
                        *span,
                    ));
                }
                Self::define(scope, defs, dst, Type::V128, *span)?;
            }
            Instruction::ExtractLane {
                dst,
                vec,
                lane,
                ty,
                span,
            } => {
                let vec_ty = self.check_reg(ctx, vec, scope, *span)?;
                if vec_ty != Type::V128 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!(
                            "ExtractLane requires vector operand of type 'v128', found '{vec_ty}'"
                        ),
                        *span,
                    ));
                }
                let Some(lanes) = ty.lane_count() else {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("ExtractLane result type must be a numeric scalar, found '{ty}'"),
                        *span,
                    ));
                };
                if *lane >= lanes {
                    return Err(Diagnostic::error(
                        "ERR_OUT_OF_BOUNDS_LANE",
                        format!(
                            "ExtractLane index {lane} out of bounds: a 128-bit vector has {lanes} lanes of type '{ty}'"
                        ),
                        *span,
                    )
                    .with_context(serde_json::json!({ "lane": lane, "lanes": lanes, "type": ty.as_str() })));
                }
                Self::define(scope, defs, dst, *ty, *span)?;
            }
            Instruction::Alloc { dst, size, span } => {
                let size_ty = self.check_reg(ctx, size, scope, *span)?;
                if size_ty != Type::I64 && size_ty != Type::I32 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Alloc size must be i64 or i32, found '{size_ty}'"),
                        *span,
                    ));
                }
                Self::define(scope, defs, dst, Type::Ptr, *span)?;
            }
            Instruction::Free { ptr, span } => {
                let ptr_ty = self.check_reg(ctx, ptr, scope, *span)?;
                if ptr_ty != Type::Ptr {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Free target must be of type 'ptr', found '{ptr_ty}'"),
                        *span,
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_reg(
        &self,
        ctx: &FnCtx,
        name: &str,
        scope: &Visible,
        span: Span,
    ) -> Result<Type, Diagnostic> {
        if let Some(ty) = scope.get(name) {
            return Ok(*ty);
        }
        let mut available: Vec<&String> = scope.keys().collect();
        available.sort();

        if let Some(def_block) = ctx.def_blocks.get(name) {
            return Err(Diagnostic::error(
                "ERR_NON_DOMINATING_USE",
                format!(
                    "Register '{name}' is defined in block '{def_block}', which does not dominate this use; pass it through block parameters instead"
                ),
                span,
            )
            .with_context(serde_json::json!({
                "target": name,
                "def_block": def_block,
                "available_registers": available,
            })));
        }

        Err(Diagnostic::error(
            "ERR_UNDEFINED_REG",
            format!("Register '{name}' is used before definition"),
            span,
        )
        .with_context(serde_json::json!({
            "target": name,
            "available_registers": available,
        })))
    }

    fn check_target_args(
        &self,
        ctx: &FnCtx,
        target: &str,
        args: &[String],
        scope: &Visible,
        span: Span,
        what: &str,
    ) -> Result<(), Diagnostic> {
        let expected_params = ctx.block_params.get(target).ok_or_else(|| {
            Diagnostic::error(
                "ERR_UNDEFINED_BLOCK",
                format!("{what} target block '{target}' does not exist"),
                span,
            )
        })?;

        if args.len() != expected_params.len() {
            return Err(Diagnostic::error(
                "ERR_ARITY_MISMATCH",
                format!(
                    "Block '{target}' expects {} arguments, received {}",
                    expected_params.len(),
                    args.len()
                ),
                span,
            ));
        }

        for (arg, (_, expected_ty)) in args.iter().zip(expected_params.iter()) {
            let actual_ty = self.check_reg(ctx, arg, scope, span)?;
            if actual_ty != *expected_ty {
                return Err(Diagnostic::error(
                    "ERR_TYPE_MISMATCH",
                    format!("{what} argument '{arg}' has type '{actual_ty}', expected '{expected_ty}'"),
                    span,
                ));
            }
        }
        Ok(())
    }

    fn validate_terminator(
        &self,
        ctx: &FnCtx,
        term: &Terminator,
        scope: &Visible,
    ) -> Result<(), Diagnostic> {
        match term {
            Terminator::Jmp { target, args, span } => {
                self.check_target_args(ctx, target, args, scope, *span, "Jump")?;
            }
            Terminator::Br {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                span,
            } => {
                let cond_ty = self.check_reg(ctx, cond, scope, *span)?;
                if cond_ty != Type::I32 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!(
                            "Branch condition '{cond}' must be i32 (boolean), found '{cond_ty}'"
                        ),
                        *span,
                    ));
                }
                self.check_target_args(ctx, then_block, then_args, scope, *span, "Branch")?;
                self.check_target_args(ctx, else_block, else_args, scope, *span, "Branch")?;
            }
            Terminator::Ret { val, span } => match (val, ctx.func.ret_type) {
                (None, None) => {}
                (Some(v), Some(expected_ty)) => {
                    let actual_ty = self.check_reg(ctx, v, scope, *span)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Return value '{v}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                }
                (Some(v), None) => {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Function has void return type, but returns value '{v}'"),
                        *span,
                    ));
                }
                (None, Some(expected_ty)) => {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Function expects return type '{expected_ty}', but returned void"),
                        *span,
                    ));
                }
            },
        }
        Ok(())
    }
}

fn inst_dst(inst: &Instruction) -> Option<&str> {
    match inst {
        Instruction::AssignConst { dst, .. }
        | Instruction::Binary { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Splat { dst, .. }
        | Instruction::ExtractLane { dst, .. }
        | Instruction::Alloc { dst, .. } => Some(dst),
        Instruction::Call { dst, .. } => dst.as_deref(),
        Instruction::Store { .. } | Instruction::Free { .. } => None,
    }
}

/// Adds the instruction index plus `function`/`block` context to a diagnostic.
fn annotate(mut diag: Diagnostic, func: &Function, block: &Block, inst_idx: usize) -> Diagnostic {
    if diag.instruction_index.is_none() {
        diag.instruction_index = Some(inst_idx);
    }
    let mut ctx = match diag.context.take() {
        Some(serde_json::Value::Object(m)) => m,
        Some(other) => {
            let mut m = serde_json::Map::new();
            m.insert("detail".to_string(), other);
            m
        }
        None => serde_json::Map::new(),
    };
    ctx.entry("function")
        .or_insert_with(|| serde_json::Value::String(func.name.clone()));
    ctx.entry("block")
        .or_insert_with(|| serde_json::Value::String(block.label.clone()));
    diag.context = Some(serde_json::Value::Object(ctx));
    diag
}
