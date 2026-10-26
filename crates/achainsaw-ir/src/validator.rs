use crate::ast::*;
use crate::cfg::Cfg;
use crate::diag::{Diagnostic, Span};
use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// Prefix reserved for compiler runtime symbols; user functions may not use it.
pub const RESERVED_PREFIX: &str = "__achainsaw";

pub struct Validator {
    // Function table: name -> (param types, return type)
    /// Parameter and result types of every function.
    functions: HashMap<String, (Vec<Type>, Vec<Type>)>,
    /// Names declared with `extfn`, which `par` may not run.
    externs: HashSet<String>,
    /// Names of the module's `data`.
    data: HashSet<String>,
}

impl Validator {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            externs: HashSet::new(),
            data: HashSet::new(),
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

    if lhs.is_half() || rhs.is_half() {
        return Err((
            "ERR_INVALID_OP_FOR_TYPE",
            format!("Operation '{op:?}' is not defined for '{lhs}' and '{rhs}'; f16/bf16 are storage types, convert with 'fext x:f32' first"),
        ));
    }
    if lhs.is_vector() || rhs.is_vector() {
        return mismatch(format!(
            "Scalar op '{op:?}' cannot be applied to vectors (found '{lhs}' and '{rhs}'); use a lane-typed vector op such as 'vadd a, b:f32'"
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
            Eq | Ne | Lt | Gt | Le | Ge | Ult | Ugt | Ule | Uge => Ok(Type::I32),
            Sub => Ok(Type::I64), // pointer difference in bytes
            Umin | Umax => Ok(Type::Ptr),
            _ => Err((
                "ERR_INVALID_PTR_OP",
                format!("Invalid pointer operation '{op:?}' on two 'ptr' values"),
            )),
        },
        Type::F32 | Type::F64 => match op {
            Add | Sub | Mul | Div | Min | Max | Copysign => Ok(ty),
            Eq | Ne | Lt | Gt | Le | Ge => Ok(Type::I32),
            _ => Err((
                "ERR_INVALID_OP_FOR_TYPE",
                format!("Operation '{op:?}' is not defined for floating-point type '{ty}'"),
            )),
        },
        _ if op == Copysign => Err((
            "ERR_INVALID_OP_FOR_TYPE",
            format!("'copysign' is defined for f32 and f64, not '{ty}'"),
        )),
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

/// Lane types each vector binary op supports. The same table applies to every backend,
/// so a module that validates compiles everywhere.
pub fn vbin_lane_types(op: VBinOp) -> &'static [Type] {
    use Type::*;
    match op {
        VBinOp::Add | VBinOp::Sub | VBinOp::And | VBinOp::Or | VBinOp::Xor => {
            &[I8, I16, I32, I64, F32, F64]
        }
        VBinOp::Mul => &[I16, I32, I64, F32, F64],
        VBinOp::Div => &[F32, F64],
        VBinOp::Min | VBinOp::Max => &[I8, I16, I32, F32, F64],
        VBinOp::Umin | VBinOp::Umax => &[I8, I16, I32],
        VBinOp::Copysign => &[F32, F64],
        VBinOp::Tbl => &[I8],
    }
}

const ALL_LANES: &[Type] = &[
    Type::I8,
    Type::I16,
    Type::I32,
    Type::I64,
    Type::F32,
    Type::F64,
];

/// Integer lane types (unsigned comparisons, shifts).
const INT_LANES: &[Type] = &[Type::I8, Type::I16, Type::I32, Type::I64];

/// Lane types each vector comparison supports: every lane type, but only integers for the
/// unsigned ones.
pub fn vcmp_lane_types(op: VCmpOp) -> &'static [Type] {
    if op.is_unsigned() {
        INT_LANES
    } else {
        ALL_LANES
    }
}

/// Lane types of `vl`, `ldm` and `stm`, which only count lanes: also the 16-bit floats.
const COUNT_LANES: &[Type] = &[
    Type::I8,
    Type::I16,
    Type::I32,
    Type::I64,
    Type::F32,
    Type::F64,
    Type::F16,
    Type::BF16,
];

/// Element types supported by `mm`.
pub const MM_DTYPES: &[Type] = &[Type::BF16, Type::F16, Type::I8, Type::F32];

/// Lane types supported by `vfma`.
pub const VFMA_LANE_TYPES: &[Type] = &[Type::F32, Type::F64];

/// Result lane types each one-operand vector op supports.
/// Element types `vgather` reads: each with indices of its own width (i32 for 32-bit
/// elements, i64 for 64-bit ones).
pub const VGATHER_LANE_TYPES: &[Type] = &[Type::I32, Type::I64, Type::F32, Type::F64];

pub fn vunary_lane_types(op: VUnaryOp) -> &'static [Type] {
    match op {
        VUnaryOp::Itof | VUnaryOp::Uitof => &[Type::F32, Type::F64],
        VUnaryOp::Ftoi | VUnaryOp::Ftoui => &[Type::I32, Type::I64],
        VUnaryOp::WidenLo | VUnaryOp::WidenHi | VUnaryOp::UWidenLo | VUnaryOp::UWidenHi => {
            &[Type::I16, Type::I32, Type::I64]
        }
        VUnaryOp::Exp | VUnaryOp::Log | VUnaryOp::Tanh | VUnaryOp::Sin | VUnaryOp::Cos => {
            &[Type::F32]
        }
        VUnaryOp::FWidenLo | VUnaryOp::FWidenHi => {
            &[Type::F16, Type::BF16, Type::F32, Type::I16, Type::I32]
        }
        VUnaryOp::Abs | VUnaryOp::Neg => ALL_LANES,
        VUnaryOp::Sqrt | VUnaryOp::Rsqrt => &[Type::F32, Type::F64],
        VUnaryOp::Rev => VPERM_LANE_TYPES,
        VUnaryOp::Floor
        | VUnaryOp::Ceil
        | VUnaryOp::Round
        | VUnaryOp::RoundEven
        | VUnaryOp::RoundZ => &[Type::F32, Type::F64],
    }
}

/// Operand lane types of `vdot` (four per i32 lane of the accumulator).
pub const VDOT_LANE_TYPES: &[Type] = &[Type::I8];

/// Lane types of the permutations (`vrev`, `vziplo` and the other `VZipOp`s, `vdup`), which
/// only move lanes: every lane width, including the 16-bit floats.
pub const VPERM_LANE_TYPES: &[Type] = COUNT_LANES;

/// Result lane types of `vnarrow`: i8 and i16 from i16 and i32 lanes (saturating), f16 and
/// bf16 from f32 lanes and f32 from f64 lanes (rounding to nearest-even, as `ftrunc`).
pub const VNARROW_LANE_TYPES: &[Type] = &[Type::I8, Type::I16, Type::F16, Type::BF16, Type::F32];

/// Lane type `vnarrow` narrows from, for its result lane type.
pub fn vnarrow_source_lane(lane: Type) -> Type {
    match lane {
        Type::I8 => Type::I16,
        Type::I16 => Type::I32,
        Type::F32 => Type::F64,
        _ => Type::F32,
    }
}

/// Lane types of `vshl`, `vshr` and `vushr`.
pub const VSHIFT_LANE_TYPES: &[Type] = INT_LANES;

fn lane_list(types: &[Type]) -> String {
    types
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Rejects types a signature may not use. `vx` has no fixed size, so it cannot cross
/// function boundaries; C ABIs pass 256/512-bit vectors inconsistently, so `extfn`
/// declarations are limited to scalars and `v128`.
fn check_signature_type(
    ty: Type,
    func: &str,
    is_extern: bool,
    span: Span,
) -> Result<(), Diagnostic> {
    if ty.is_half() {
        return Err(Diagnostic::error(
            "ERR_HALF_IN_SIGNATURE",
            format!("Function '{func}' uses '{ty}' in its signature; f16/bf16 are storage types, pass f32 or a pointer instead"),
            span,
        )
        .with_context(serde_json::json!({ "function": func })));
    }
    // AIR functions may take and return `vx` (only AIR code can call functions with vector
    // signatures); a C function cannot, since `vx`'s width depends on the backend and CPU.
    if is_extern && ty == Type::Vx {
        return Err(Diagnostic::error(
            "ERR_SCALABLE_IN_SIGNATURE",
            format!("External function '{func}' uses 'vx' in its signature; its width depends on the backend and CPU, so C functions cannot take or return it, pass a pointer instead"),
            span,
        )
        .with_context(serde_json::json!({ "function": func })));
    }
    if is_extern && matches!(ty, Type::V256 | Type::V512) {
        return Err(Diagnostic::error(
            "ERR_WIDE_VECTOR_IN_EXTERN",
            format!("External function '{func}' uses '{ty}'; extfn signatures support scalars and v128 only"),
            span,
        )
        .with_context(serde_json::json!({ "function": func })));
    }
    Ok(())
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
            for ty in ext_fn.params.iter().map(|(_, t)| *t).chain(ext_fn.ret_type) {
                check_signature_type(ty, &ext_fn.name, true, ext_fn.span)?;
            }
            let param_types = ext_fn.params.iter().map(|(_, ty)| *ty).collect();
            self.functions.insert(
                ext_fn.name.clone(),
                (param_types, ext_fn.ret_type.into_iter().collect()),
            );
            self.externs.insert(ext_fn.name.clone());
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
            for ty in func.params.iter().map(|(_, t)| *t).chain(func.rets.clone()) {
                check_signature_type(ty, &func.name, false, func.span)?;
            }
            let param_types = func.params.iter().map(|(_, ty)| *ty).collect();
            self.functions
                .insert(func.name.clone(), (param_types, func.rets.clone()));
        }

        for d in &module.data {
            self.validate_data(d)?;
        }

        // Validate each function
        for func in &module.functions {
            self.validate_function(func)?;
        }
        check_inline_recursion(module)
    }

    fn validate_data(&mut self, d: &Data) -> Result<(), Diagnostic> {
        if d.name.starts_with(RESERVED_PREFIX) {
            return Err(Diagnostic::error(
                "ERR_RESERVED_NAME",
                format!(
                    "Data name '{}' uses the reserved prefix '{RESERVED_PREFIX}'",
                    d.name
                ),
                d.span,
            ));
        }
        if !self.data.insert(d.name.clone()) {
            return Err(Diagnostic::error(
                "ERR_DUPLICATE_DATA",
                format!("Duplicate data name '{}'", d.name),
                d.span,
            )
            .with_context(serde_json::json!({ "target": d.name })));
        }
        if !matches!(
            d.ty,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F32 | Type::F64
        ) {
            return Err(Diagnostic::error(
                "ERR_INVALID_DATA_TYPE",
                format!(
                    "data '{}' has element type '{}'; use i8, i16, i32, i64, f32 or f64",
                    d.name, d.ty
                ),
                d.span,
            ));
        }
        if d.bytes.is_empty() || !d.bytes.len().is_multiple_of(d.ty.byte_size()) {
            return Err(Diagnostic::error(
                "ERR_INVALID_DATA_TYPE",
                format!(
                    "data '{}' holds {} bytes, which is not a positive number of '{}' values",
                    d.name,
                    d.bytes.len(),
                    d.ty
                ),
                d.span,
            ));
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
                format!(
                    "Function '{}' must have at least one basic block",
                    func.name
                ),
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
                if let Some(dst) = inst.dst() {
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
        let mut fn_params: HashSet<&str> = HashSet::new();
        let mut base_scope: Visible = HashMap::new();
        for (p_name, p_ty) in &func.params {
            if !fn_params.insert(p_name.as_str()) {
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

            // A block parameter may shadow a register of a dominating block (see
            // `crate::names`), but a block's parameters have distinct names.
            let mut own: HashSet<&str> = HashSet::new();
            for (p_name, p_ty) in &block.params {
                if !own.insert(p_name.as_str()) {
                    return Err(Diagnostic::error(
                        "ERR_DUPLICATE_PARAM",
                        format!(
                            "Block '{}' declares parameter '{p_name}' twice",
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
                self.validate_instruction(&ctx, inst, &mut scope)
                    .map_err(|d| annotate(d, func, block, i_idx))?;
            }

            self.validate_terminator(&ctx, &block.terminator, &scope)
                .map_err(|d| annotate(d, func, block, block.instructions.len()))?;

            final_scopes[b_idx] = Some(scope);
        }

        check_reaching_definitions(func, &cfg)
    }

    /// A later definition shadows an earlier one in the code it dominates; uses that could
    /// see both are `ERR_AMBIGUOUS_REGISTER` (`check_reaching_definitions`).
    fn define(scope: &mut Visible, dst: &str, ty: Type) {
        scope.insert(dst.to_string(), ty);
    }

    fn validate_instruction(
        &self,
        ctx: &FnCtx,
        inst: &Instruction,
        scope: &mut Visible,
    ) -> Result<(), Diagnostic> {
        match inst {
            Instruction::AssignConst { dst, val, ty, span } => {
                match (val, ty) {
                    (
                        Constant::Int(n),
                        Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Ptr,
                    ) => {
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
                Self::define(scope, dst, *ty);
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
                Self::define(scope, dst, out_ty);
            }
            Instruction::Load {
                dst,
                ptr,
                index,
                ty,
                span,
            } => {
                self.check_index(ctx, index.as_ref(), *ty, "ld", scope, *span)?;
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
                Self::define(scope, dst, *ty);
            }
            Instruction::Store {
                ptr,
                index,
                val,
                span,
            } => {
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
                let val_ty = self.check_reg(ctx, val, scope, *span)?;
                self.check_index(ctx, index.as_ref(), val_ty, "st", scope, *span)?;
            }
            Instruction::AtomicRmw {
                op,
                dst,
                ptr,
                index,
                val,
                span,
            } => {
                let op = format!("atomic {}", op.as_str());
                let ty = self.check_reg(ctx, val, scope, *span)?;
                self.check_atomic(ctx, &op, ptr, index.as_ref(), val, ty, scope, *span)?;
                Self::define(scope, dst, ty);
            }
            Instruction::AtomicCas {
                dst,
                ptr,
                index,
                expected,
                new,
                span,
            } => {
                let ty = self.check_reg(ctx, expected, scope, *span)?;
                self.check_atomic(
                    ctx,
                    "atomic cas",
                    ptr,
                    index.as_ref(),
                    expected,
                    ty,
                    scope,
                    *span,
                )?;
                self.check_typed(ctx, new, ty, "atomic cas new value", scope, *span)?;
                Self::define(scope, dst, ty);
            }
            Instruction::AtomicLoad {
                dst,
                ptr,
                index,
                ty,
                span,
            } => {
                self.check_atomic(
                    ctx,
                    "atomic ld",
                    ptr,
                    index.as_ref(),
                    dst,
                    *ty,
                    scope,
                    *span,
                )?;
                Self::define(scope, dst, *ty);
            }
            Instruction::AtomicStore {
                ptr,
                index,
                val,
                span,
            } => {
                let ty = self.check_reg(ctx, val, scope, *span)?;
                self.check_atomic(ctx, "atomic st", ptr, index.as_ref(), val, ty, scope, *span)?;
            }
            Instruction::Call {
                dsts,
                func,
                args,
                span,
            } => {
                let (param_types, rets) = self.functions.get(func).ok_or_else(|| {
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

                if !dsts.is_empty() && rets.is_empty() {
                    return Err(Diagnostic::error(
                        "ERR_VOID_ASSIGNMENT",
                        format!("Function '{func}' does not return a value"),
                        *span,
                    ));
                }
                if !dsts.is_empty() && dsts.len() != rets.len() {
                    return Err(Diagnostic::error(
                        "ERR_RESULT_COUNT",
                        format!(
                            "Function '{func}' returns {} values, but the call assigns {} ({})",
                            rets.len(),
                            dsts.len(),
                            dsts.join(", ")
                        ),
                        *span,
                    )
                    .with_context(serde_json::json!({ "function": func, "returns": rets.iter().map(|t| t.as_str()).collect::<Vec<_>>() })));
                }
                for (d, rty) in dsts.iter().zip(rets.clone()) {
                    Self::define(scope, d, rty);
                }
            }
            Instruction::Par {
                count,
                func,
                args,
                span,
            } => {
                let count_ty = self.check_reg(ctx, count, scope, *span)?;
                if count_ty != Type::I64 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("par count '{count}' must be of type 'i64', found '{count_ty}'"),
                        *span,
                    ));
                }
                let (param_types, rets) = self.functions.get(func).ok_or_else(|| {
                    Diagnostic::error(
                        "ERR_UNDEFINED_FUNCTION",
                        format!("par runs undefined function '{func}'"),
                        *span,
                    )
                })?;
                let mut expected: Vec<String> = vec!["i64".into()];
                expected.extend(
                    args.iter()
                        .map(|a| scope.get(a).map_or("?".into(), |t| t.to_string())),
                );
                let expected = format!("fn {func}({})", expected.join(", "));
                let signature_error = |message: String| {
                    Diagnostic::error("ERR_PAR_SIGNATURE", message, *span).with_context(
                        serde_json::json!({ "function": func, "expected_signature": expected }),
                    )
                };
                if self.externs.contains(func) {
                    return Err(signature_error(format!(
                        "par cannot run external function '{func}'; wrap the call in an AIR function"
                    )));
                }
                if param_types.first() != Some(&Type::I64) || !rets.is_empty() {
                    return Err(signature_error(format!(
                        "par body '{func}' must take the index (i64) as its first parameter and return nothing"
                    )));
                }
                if args.len() + 1 != param_types.len() {
                    return Err(Diagnostic::error(
                        "ERR_ARITY_MISMATCH",
                        format!(
                            "par body '{func}' takes {} arguments after the index, received {}",
                            param_types.len() - 1,
                            args.len()
                        ),
                        *span,
                    ));
                }
                for (arg, &expected_ty) in args.iter().zip(&param_types[1..]) {
                    let actual_ty = self.check_reg(ctx, arg, scope, *span)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Argument '{arg}' passed to '{func}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                    if actual_ty.is_vector() {
                        return Err(signature_error(format!(
                            "par cannot pass vector '{arg}' to '{func}'; pass vectors through memory (ptr)"
                        )));
                    }
                }
            }
            Instruction::Splat { dst, src, ty, span } => {
                let src_ty = self.check_reg(ctx, src, scope, *span)?;
                if !src_ty.is_lane() {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Splat operand '{src}' must be numeric scalar, found '{src_ty}'"),
                        *span,
                    ));
                }
                if !ty.is_vector() {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Splat width must be a vector type (v128, v256, v512, vx), found '{ty}'"),
                        *span,
                    ));
                }
                Self::define(scope, dst, *ty);
            }
            Instruction::ExtractLane {
                dst,
                vec,
                lane,
                ty,
                span,
            } => {
                let vec_ty = self.check_vector(ctx, vec, scope, *span)?;
                if ty.lanes_in(vec_ty).is_none() {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("ExtractLane result type must be a numeric scalar, found '{ty}'"),
                        *span,
                    ));
                }
                Self::check_lane_index("ExtractLane", vec_ty, *lane, *ty, *span)?;
                Self::define(scope, dst, *ty);
            }
            Instruction::InsertLane {
                dst,
                vec,
                val,
                lane,
                ty,
                span,
            } => {
                Self::check_lane("inslane", *ty, ALL_LANES, *span)?;
                let vec_ty = self.check_vector(ctx, vec, scope, *span)?;
                self.check_typed(ctx, val, *ty, "inslane value", scope, *span)?;
                Self::check_lane_index("inslane", vec_ty, *lane, *ty, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VDup {
                dst,
                vec,
                lane,
                ty,
                span,
            } => {
                Self::check_lane("vdup", *ty, VPERM_LANE_TYPES, *span)?;
                let vec_ty = self.check_vector(ctx, vec, scope, *span)?;
                Self::check_lane_index("vdup", vec_ty, *lane, *ty, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VZip {
                op,
                dst,
                lhs,
                rhs,
                lane,
                span,
            } => {
                Self::check_lane(op.as_str(), *lane, VPERM_LANE_TYPES, *span)?;
                let vec_ty =
                    self.check_same_vectors(ctx, op.as_str(), &[lhs, rhs], scope, *span)?;
                Self::define(scope, dst, vec_ty);
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
                Self::define(scope, dst, Type::Ptr);
            }
            Instruction::Stack { dst, size, span } => {
                if !(1..=MAX_STACK_BYTES).contains(size) {
                    return Err(Diagnostic::error(
                        "ERR_INVALID_STACK_SIZE",
                        format!(
                            "stack size {size} is outside 1..={MAX_STACK_BYTES} bytes; use alloc for larger buffers"
                        ),
                        *span,
                    ));
                }
                Self::define(scope, dst, Type::Ptr);
            }
            Instruction::Addr { dst, name, span } => {
                if !self.data.contains(name) {
                    let mut names: Vec<&String> = self.data.iter().collect();
                    names.sort();
                    return Err(Diagnostic::error(
                        "ERR_UNDEFINED_DATA",
                        format!("addr names '{name}', which no `data` declaration defines"),
                        *span,
                    )
                    .with_context(serde_json::json!({ "target": name, "available_data": names })));
                }
                Self::define(scope, dst, Type::Ptr);
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
            Instruction::Select {
                dst,
                cond,
                then_val,
                else_val,
                span,
            } => {
                let cond_ty = self.check_reg(ctx, cond, scope, *span)?;
                if !matches!(cond_ty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Select condition must be an integer, found '{cond_ty}'"),
                        *span,
                    ));
                }
                let then_ty = self.check_reg(ctx, then_val, scope, *span)?;
                let else_ty = self.check_reg(ctx, else_val, scope, *span)?;
                if then_ty != else_ty {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!(
                            "Select alternatives must have matching types, found '{then_ty}' and '{else_ty}'"
                        ),
                        *span,
                    ));
                }
                Self::define(scope, dst, then_ty);
            }
            Instruction::Unary { op, dst, src, span } => {
                let src_ty = self.check_reg(ctx, src, scope, *span)?;
                let out_ty = match op {
                    UnaryOp::Sqrt => {
                        if !matches!(src_ty, Type::F32 | Type::F64) {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "Sqrt operand must be float (f32 or f64), found '{src_ty}'"
                                ),
                                *span,
                            ));
                        }
                        src_ty
                    }
                    _ if op.is_rounding() && !src_ty.is_float() => {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!(
                                "'{}' operand must be float (f32 or f64), found '{src_ty}'",
                                op.as_str()
                            ),
                            *span,
                        ));
                    }
                    _ if op.is_math() && src_ty != Type::F32 => {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!(
                                "'{0}' operand must be f32, found '{src_ty}' (the math functions are f32 only: convert with 'fext'/'ftrunc'; 'v{0}' takes f32 lanes)",
                                op.as_str()
                            ),
                            *span,
                        ));
                    }
                    _ if op.is_bit_count() && !src_ty.is_int() => {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!(
                                "'{}' operand must be an integer (i8 to i64), found '{src_ty}'",
                                op.as_str()
                            ),
                            *span,
                        ));
                    }
                    UnaryOp::Neg | UnaryOp::Abs => {
                        if src_ty.is_vector() || src_ty.is_half() || src_ty == Type::Ptr {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!("Operation '{op:?}' is not supported for '{src_ty}'"),
                                *span,
                            ));
                        }
                        src_ty
                    }
                    _ => src_ty,
                };
                Self::define(scope, dst, out_ty);
            }
            Instruction::Cast {
                op,
                dst,
                src,
                ty,
                span,
            } => {
                let src_ty = self.check_reg(ctx, src, scope, *span)?;
                match op {
                    CastOp::Itof => {
                        if !matches!(src_ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
                            || !matches!(ty, Type::F32 | Type::F64)
                        {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "itof requires integer source and float destination, found '{src_ty}' -> '{ty}'"
                                ),
                                *span,
                            ));
                        }
                    }
                    CastOp::Uitof => {
                        if !src_ty.is_int() || !ty.is_float() {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "uitof requires integer source and float destination, found '{src_ty}' -> '{ty}'"
                                ),
                                *span,
                            ));
                        }
                    }
                    CastOp::Ftoui => {
                        if !src_ty.is_float() || !matches!(ty, Type::I32 | Type::I64) {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "ftoui requires float source and i32 or i64 destination, found '{src_ty}' -> '{ty}'"
                                ),
                                *span,
                            ));
                        }
                    }
                    CastOp::Ftoi => {
                        if !matches!(src_ty, Type::F32 | Type::F64)
                            || !matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
                        {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "ftoi requires float source and integer destination, found '{src_ty}' -> '{ty}'"
                                ),
                                *span,
                            ));
                        }
                    }
                    CastOp::Sext | CastOp::Zext => {
                        let src_bits = src_ty.bit_width().unwrap_or(0);
                        let dst_bits = ty.bit_width().unwrap_or(0);
                        if !matches!(src_ty, Type::I8 | Type::I16 | Type::I32)
                            || !matches!(ty, Type::I16 | Type::I32 | Type::I64)
                            || dst_bits <= src_bits
                        {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "Extension requires widening integer conversion, found '{src_ty}' -> '{ty}'"
                                ),
                                *span,
                            ));
                        }
                    }
                    CastOp::Trunc => {
                        let src_bits = src_ty.bit_width().unwrap_or(0);
                        let dst_bits = ty.bit_width().unwrap_or(0);
                        if !matches!(src_ty, Type::I16 | Type::I32 | Type::I64)
                            || !matches!(ty, Type::I8 | Type::I16 | Type::I32)
                            || dst_bits >= src_bits
                        {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "Truncation requires narrowing integer conversion, found '{src_ty}' -> '{ty}'"
                                ),
                                *span,
                            ));
                        }
                    }
                    CastOp::Fext => {
                        let ok = matches!(
                            (src_ty, *ty),
                            (Type::F32, Type::F64) | (Type::F16 | Type::BF16, Type::F32)
                        );
                        if !ok {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!("fext requires f32 -> f64 or f16/bf16 -> f32, found '{src_ty}' -> '{ty}'"),
                                *span,
                            ));
                        }
                    }
                    CastOp::Ftrunc => {
                        let ok = matches!(
                            (src_ty, *ty),
                            (Type::F64, Type::F32) | (Type::F32, Type::F16 | Type::BF16)
                        );
                        if !ok {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!("ftrunc requires f64 -> f32 or f32 -> f16/bf16, found '{src_ty}' -> '{ty}'"),
                                *span,
                            ));
                        }
                    }
                    CastOp::Bitcast => {
                        if (src_ty.is_vector() || ty.is_vector()) && src_ty != *ty {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!("bitcast cannot change vector types ('{src_ty}' -> '{ty}'); vectors are untyped, so pick the lane type on each vector op instead"),
                                *span,
                            ));
                        }
                        if src_ty.byte_size() != ty.byte_size() {
                            return Err(Diagnostic::error(
                                "ERR_TYPE_MISMATCH",
                                format!(
                                    "bitcast requires source and target of identical size, found '{src_ty}' ({}B) -> '{ty}' ({}B)",
                                    src_ty.byte_size(),
                                    ty.byte_size()
                                ),
                                *span,
                            ));
                        }
                    }
                }
                Self::define(scope, dst, *ty);
            }
            Instruction::VectorReduce {
                op,
                dst,
                src,
                ty,
                span,
            } => {
                self.check_vector(ctx, src, scope, *span)?;
                Self::check_lane(op.as_str(), *ty, ALL_LANES, *span)?;
                Self::define(scope, dst, *ty);
            }
            Instruction::VBinary {
                op,
                dst,
                lhs,
                rhs,
                lane,
                span,
            } => {
                Self::check_lane(op.as_str(), *lane, vbin_lane_types(*op), *span)?;
                let vec_ty =
                    self.check_same_vectors(ctx, op.as_str(), &[lhs, rhs], scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VFma {
                dst,
                a,
                b,
                c,
                lane,
                span,
            } => {
                Self::check_lane("vfma", *lane, VFMA_LANE_TYPES, *span)?;
                let vec_ty = self.check_same_vectors(ctx, "vfma", &[a, b, c], scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::Fma { dst, a, b, c, span } => {
                let ty = self.check_reg(ctx, a, scope, *span)?;
                for r in [b, c] {
                    let t = self.check_reg(ctx, r, scope, *span)?;
                    if t != ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("fma operands must have one type, found '{ty}' and '{t}'"),
                            *span,
                        ));
                    }
                }
                if !ty.is_float() {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("fma operands must be f32 or f64, found '{ty}'"),
                        *span,
                    ));
                }
                Self::define(scope, dst, ty);
            }
            Instruction::VDot {
                dst,
                acc,
                a,
                b,
                lane,
                unsigned,
                span,
            } => {
                let op = if *unsigned { "vdotu" } else { "vdot" };
                Self::check_lane(op, *lane, VDOT_LANE_TYPES, *span)?;
                let vec_ty = self.check_same_vectors(ctx, op, &[acc, a, b], scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VCmp {
                op,
                dst,
                lhs,
                rhs,
                lane,
                span,
            } => {
                Self::check_lane(op.as_str(), *lane, vcmp_lane_types(*op), *span)?;
                let vec_ty =
                    self.check_same_vectors(ctx, op.as_str(), &[lhs, rhs], scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VSelect {
                dst,
                mask,
                then_val,
                else_val,
                span,
            } => {
                let vec_ty = self.check_same_vectors(
                    ctx,
                    "vsel",
                    &[mask, then_val, else_val],
                    scope,
                    *span,
                )?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::MaskedLoad {
                dst,
                ptr,
                index,
                count,
                ty,
                lane,
                span,
            } => {
                self.check_index(ctx, index.as_ref(), *lane, "ldm", scope, *span)?;
                self.check_typed(ctx, ptr, Type::Ptr, "ldm pointer", scope, *span)?;
                self.check_typed(ctx, count, Type::I64, "ldm lane count", scope, *span)?;
                if !ty.is_vector() {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("ldm loads a vector (v128, v256, v512, vx), found '{ty}'"),
                        *span,
                    ));
                }
                Self::check_lane("ldm", *lane, COUNT_LANES, *span)?;
                Self::define(scope, dst, *ty);
            }
            Instruction::VGather {
                dst,
                ptr,
                idx,
                count,
                lane,
                span,
            } => {
                Self::check_lane("vgather", *lane, VGATHER_LANE_TYPES, *span)?;
                self.check_typed(ctx, ptr, Type::Ptr, "vgather pointer", scope, *span)?;
                if let Some(count) = count {
                    self.check_typed(ctx, count, Type::I64, "vgather lane count", scope, *span)?;
                }
                let vec_ty = self.check_vector(ctx, idx, scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::MaskedStore {
                ptr,
                index,
                val,
                count,
                lane,
                span,
            } => {
                self.check_index(ctx, index.as_ref(), *lane, "stm", scope, *span)?;
                self.check_typed(ctx, ptr, Type::Ptr, "stm pointer", scope, *span)?;
                self.check_vector(ctx, val, scope, *span)?;
                self.check_typed(ctx, count, Type::I64, "stm lane count", scope, *span)?;
                Self::check_lane("stm", *lane, COUNT_LANES, *span)?;
            }
            Instruction::MatMul {
                pc,
                pa,
                pb,
                m,
                n,
                k,
                dtype,
                span,
            } => {
                for (reg, what) in [(pc, "C"), (pa, "A"), (pb, "B")] {
                    self.check_typed(
                        ctx,
                        reg,
                        Type::Ptr,
                        &format!("mm {what} pointer"),
                        scope,
                        *span,
                    )?;
                }
                for (reg, what) in [(m, "m"), (n, "n"), (k, "k")] {
                    self.check_typed(
                        ctx,
                        reg,
                        Type::I64,
                        &format!("mm dimension {what}"),
                        scope,
                        *span,
                    )?;
                }
                Self::check_lane("mm", *dtype, MM_DTYPES, *span)?;
            }
            Instruction::VUnary {
                op,
                dst,
                src,
                lane,
                span,
            } => {
                Self::check_lane(op.as_str(), *lane, vunary_lane_types(*op), *span)?;
                let vec_ty = self.check_vector(ctx, src, scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VNarrow {
                dst,
                lo,
                hi,
                lane,
                span,
            } => {
                Self::check_lane("vnarrow", *lane, VNARROW_LANE_TYPES, *span)?;
                let vec_ty = self.check_same_vectors(ctx, "vnarrow", &[lo, hi], scope, *span)?;
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VShift {
                op,
                dst,
                src,
                amount,
                lane,
                span,
            } => {
                Self::check_lane(op.as_str(), *lane, VSHIFT_LANE_TYPES, *span)?;
                let vec_ty = self.check_vector(ctx, src, scope, *span)?;
                let amount_ty = self.check_reg(ctx, amount, scope, *span)?;
                if !amount_ty.is_int() && amount_ty != vec_ty {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!(
                            "'{}' shift amount '{amount}' must be a scalar integer or a '{vec_ty}' of per-lane amounts, found '{amount_ty}'",
                            op.as_str()
                        ),
                        *span,
                    )
                    .with_context(serde_json::json!({ "target": amount, "found": amount_ty.as_str() })));
                }
                Self::define(scope, dst, vec_ty);
            }
            Instruction::VLen { dst, lane, span } => {
                Self::check_lane("vl", *lane, COUNT_LANES, *span)?;
                Self::define(scope, dst, Type::I64);
            }
        }
        Ok(())
    }

    /// Checks the index of a memory access (`p[i]`): an i64 register, scaled by a type with a
    /// fixed size. Without a unit, the access's element type `implied` scales it, which must
    /// not be a vector (`ld p[i]:v256` would be ambiguous).
    /// Checks an atomic access of an integer of type `ty` (the type of `what`, or the type
    /// loaded) at `ptr` or `ptr[index]`.
    #[allow(clippy::too_many_arguments)]
    fn check_atomic(
        &self,
        ctx: &FnCtx,
        op: &str,
        ptr: &str,
        index: Option<&Index>,
        what: &str,
        ty: Type,
        scope: &Visible,
        span: Span,
    ) -> Result<(), Diagnostic> {
        self.check_typed(ctx, ptr, Type::Ptr, &format!("{op} pointer"), scope, span)?;
        self.check_index(ctx, index, ty, op, scope, span)?;
        if !ty.is_int() {
            return Err(Diagnostic::error(
                "ERR_TYPE_MISMATCH",
                format!(
                    "'{op}' works on integers (i8, i16, i32, i64), found '{what}' of type '{ty}'; \
                     for floats, bitcast to an integer of the same size and loop on 'atomic cas'"
                ),
                span,
            )
            .with_context(serde_json::json!({ "op": op, "target": what, "found": ty.as_str() })));
        }
        Ok(())
    }

    fn check_index(
        &self,
        ctx: &FnCtx,
        index: Option<&Index>,
        implied: Type,
        op: &str,
        scope: &Visible,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let Some(ix) = index else {
            return Ok(());
        };
        self.check_typed(ctx, &ix.reg, Type::I64, &format!("{op} index"), scope, span)?;
        let unit = ix.unit_or(implied);
        if unit == Type::Vx || (ix.unit.is_none() && unit.is_vector()) {
            let message = if ix.unit.is_some() {
                format!("'{op}' index unit 'vx' has no fixed size; use a lane type, e.g. p[i:f32]")
            } else {
                format!("'{op}' of a vector needs the index unit, e.g. p[i:f32] (i counts f32 values) or p[i:v256]")
            };
            return Err(Diagnostic::error("ERR_INDEX_UNIT", message, span)
                .with_context(serde_json::json!({ "op": op, "unit": unit.as_str() })));
        }
        Ok(())
    }

    /// Checks a constant lane index (`extlane`, `vdup`) against the lanes of `ty` in `vec_ty`;
    /// `vx` only guarantees `MIN_VX_BITS`.
    fn check_lane_index(
        op: &str,
        vec_ty: Type,
        lane: u32,
        ty: Type,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let vec_bits = vec_ty.bit_width().unwrap_or(crate::types::MIN_VX_BITS);
        let lanes = vec_bits / ty.bit_width().expect("lane type has a width");
        if lane < lanes {
            return Ok(());
        }
        let what = if vec_ty == Type::Vx {
            format!("'vx' is only guaranteed {} bits, i.e. {lanes} lanes of type '{ty}'; store the vector to memory to read higher lanes", crate::types::MIN_VX_BITS)
        } else {
            format!("a '{vec_ty}' vector has {lanes} lanes of type '{ty}'")
        };
        Err(Diagnostic::error(
            "ERR_OUT_OF_BOUNDS_LANE",
            format!("{op} index {lane} out of bounds: {what}"),
            span,
        )
        .with_context(serde_json::json!({ "lane": lane, "lanes": lanes, "type": ty.as_str(), "vector": vec_ty.as_str() })))
    }

    fn check_lane(op: &str, lane: Type, allowed: &[Type], span: Span) -> Result<(), Diagnostic> {
        if !allowed.contains(&lane) {
            return Err(Diagnostic::error(
                "ERR_INVALID_LANE_TYPE",
                format!("'{op}' does not support lane type '{lane}'; supported: {}", lane_list(allowed)),
                span,
            )
            .with_context(serde_json::json!({ "op": op, "lane": lane.as_str(), "supported": allowed.iter().map(|t| t.as_str()).collect::<Vec<_>>() })));
        }
        Ok(())
    }

    /// Checks that `name` is a register of type `expected`.
    fn check_typed(
        &self,
        ctx: &FnCtx,
        name: &str,
        expected: Type,
        what: &str,
        scope: &Visible,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let ty = self.check_reg(ctx, name, scope, span)?;
        if ty != expected {
            return Err(Diagnostic::error(
                "ERR_TYPE_MISMATCH",
                format!("{what} '{name}' must be '{expected}', found '{ty}'"),
                span,
            )
            .with_context(serde_json::json!({ "target": name, "expected": expected.as_str(), "found": ty.as_str() })));
        }
        Ok(())
    }

    /// Checks that `name` is a vector register and returns its type.
    fn check_vector(
        &self,
        ctx: &FnCtx,
        name: &str,
        scope: &Visible,
        span: Span,
    ) -> Result<Type, Diagnostic> {
        let ty = self.check_reg(ctx, name, scope, span)?;
        if !ty.is_vector() {
            return Err(Diagnostic::error(
                "ERR_TYPE_MISMATCH",
                format!("Operand '{name}' must be a vector (v128, v256, v512, vx), found '{ty}'"),
                span,
            )
            .with_context(serde_json::json!({ "target": name, "found": ty.as_str() })));
        }
        Ok(ty)
    }

    /// Checks that all `regs` are vectors of one type and returns it.
    fn check_same_vectors(
        &self,
        ctx: &FnCtx,
        op: &str,
        regs: &[&String],
        scope: &Visible,
        span: Span,
    ) -> Result<Type, Diagnostic> {
        let first = self.check_vector(ctx, regs[0], scope, span)?;
        for r in &regs[1..] {
            let ty = self.check_vector(ctx, r, scope, span)?;
            if ty != first {
                return Err(Diagnostic::error(
                    "ERR_TYPE_MISMATCH",
                    format!("'{op}' operands must be vectors of the same width, found '{first}' and '{ty}'"),
                    span,
                ));
            }
        }
        Ok(first)
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
        // Registers the parser made for inline immediates are not the agent's to use
        let mut available: Vec<&String> =
            scope.keys().filter(|r| !r.starts_with("__imm_")).collect();
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
                    format!(
                        "{what} argument '{arg}' has type '{actual_ty}', expected '{expected_ty}'"
                    ),
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
            Terminator::Ret { vals, span } => {
                let rets = &ctx.func.rets;
                if vals.len() != rets.len() {
                    let message = match (vals.len(), rets.len()) {
                        (_, 0) => format!(
                            "Function has void return type, but returns {}",
                            vals.join(", ")
                        ),
                        (0, 1) => format!(
                            "Function expects return type '{}', but returned void",
                            rets[0]
                        ),
                        (n, m) => format!("Function returns {m} values, but 'ret' gives {n}"),
                    };
                    return Err(Diagnostic::error("ERR_TYPE_MISMATCH", message, *span));
                }
                for (v, &expected_ty) in vals.iter().zip(rets) {
                    let actual_ty = self.check_reg(ctx, v, scope, *span)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Return value '{v}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Adds the instruction index plus `function`/`block` context to a diagnostic.
/// Rejects `inline fn`s that reach themselves through calls to inline functions, which
/// inlining could never finish expanding.
fn check_inline_recursion(module: &Module) -> Result<(), Diagnostic> {
    let inline: HashMap<&str, &Function> = module
        .functions
        .iter()
        .filter(|f| f.inline)
        .map(|f| (f.name.as_str(), f))
        .collect();
    let callees = |f: &Function| -> Vec<String> {
        f.blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| match i {
                Instruction::Call { func, .. } if inline.contains_key(func.as_str()) => {
                    Some(func.clone())
                }
                _ => None,
            })
            .collect()
    };
    for (&name, &f) in &inline {
        // Depth-first from f's inline callees, looking for f.
        let mut stack = callees(f);
        let mut seen = std::collections::HashSet::new();
        while let Some(c) = stack.pop() {
            if c == name {
                return Err(Diagnostic::error(
                    "ERR_RECURSIVE_INLINE",
                    format!("inline fn '{name}' calls itself through inline functions; remove 'inline' from one of them"),
                    f.span,
                ));
            }
            if seen.insert(c.clone()) {
                stack.extend(callees(inline[c.as_str()]));
            }
        }
    }
    Ok(())
}

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

/// Where a register is defined: a block, and the instruction index there (`PARAM` for the
/// block's parameters, `FN_PARAM` for the function's, in the entry block).
type DefSite = (usize, i64);
const PARAM: i64 = -1;
const FN_PARAM: i64 = -2;

/// With shadowing, a name can have several definitions. Every use must be reached by one of
/// them only (the one that dominates it): a use that another definition can reach without
/// passing through that one, such as a name defined again in one arm of a branch and used after
/// the join, is `ERR_AMBIGUOUS_REGISTER`.
fn check_reaching_definitions(func: &Function, cfg: &Cfg) -> Result<(), Diagnostic> {
    let mut sites: HashMap<&str, Vec<DefSite>> = HashMap::new();
    for (p, _) in &func.params {
        sites.entry(p.as_str()).or_default().push((0, FN_PARAM));
    }
    for (b, block) in func.blocks.iter().enumerate() {
        for (p, _) in &block.params {
            sites.entry(p.as_str()).or_default().push((b, PARAM));
        }
        for (i, inst) in block.instructions.iter().enumerate() {
            for d in inst.dsts() {
                sites.entry(d).or_default().push((b, i as i64));
            }
        }
    }
    for (name, defs) in &sites {
        if defs.len() > 1 {
            check_name(func, cfg, name, defs)?;
        }
    }
    Ok(())
}

fn check_name(func: &Function, cfg: &Cfg, name: &str, defs: &[DefSite]) -> Result<(), Diagnostic> {
    let n = func.blocks.len();
    // The last definition in each block, which is what leaves it.
    let mut last: Vec<Option<DefSite>> = vec![None; n];
    for &d in defs {
        if last[d.0].is_none_or(|l| l.1 < d.1) {
            last[d.0] = Some(d);
        }
    }
    let entry_in: HashSet<DefSite> = defs.iter().copied().filter(|d| d.1 == FN_PARAM).collect();
    let mut ins: Vec<HashSet<DefSite>> = vec![HashSet::new(); n];
    ins[0] = entry_in.clone();
    let out = |b: usize, ins: &Vec<HashSet<DefSite>>| -> HashSet<DefSite> {
        match last[b] {
            Some(d) if d.1 != FN_PARAM => HashSet::from([d]),
            _ => ins[b].clone(),
        }
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let mut set: HashSet<DefSite> = if b == 0 {
                entry_in.clone()
            } else {
                HashSet::new()
            };
            for &p in &cfg.preds[b] {
                if cfg.reachable[p] {
                    set.extend(out(p, &ins));
                }
            }
            if set != ins[b] {
                ins[b] = set;
                changed = true;
            }
        }
    }
    for &b in &cfg.rpo {
        let block = &func.blocks[b];
        let reaching = |pos: i64| -> HashSet<DefSite> {
            match defs
                .iter()
                .filter(|d| d.0 == b && d.1 < pos && d.1 != FN_PARAM)
                .max_by_key(|d| d.1)
            {
                Some(&d) => HashSet::from([d]),
                None => ins[b].clone(),
            }
        };
        let uses = block
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, inst)| inst.operands().iter().any(|r| r.as_str() == name))
            .map(|(i, inst)| (i, inst.span()))
            .chain(
                terminator_reads(&block.terminator, name)
                    .then(|| (block.instructions.len(), terminator_span(&block.terminator))),
            );
        for (i, span) in uses {
            let seen = reaching(i as i64);
            if seen.len() > 1 {
                let mut blocks: Vec<&str> = seen
                    .iter()
                    .map(|d| func.blocks[d.0].label.as_str())
                    .collect();
                blocks.sort();
                blocks.dedup();
                return Err(Diagnostic::error(
                    "ERR_AMBIGUOUS_REGISTER",
                    format!(
                        "Register '{name}' here could hold the value defined in block {}, depending on the path taken; give the definitions different names and pass the one you mean to block '{}' as a block parameter",
                        blocks.iter().map(|b| format!("'{b}'")).collect::<Vec<_>>().join(" or "),
                        block.label
                    ),
                    span,
                )
                .with_instruction_index(i)
                .with_context(serde_json::json!({
                    "function": func.name,
                    "block": block.label,
                    "target": name,
                    "defined_in": blocks,
                })));
            }
        }
    }
    Ok(())
}

fn terminator_reads(t: &Terminator, name: &str) -> bool {
    match t {
        Terminator::Jmp { args, .. } => args.iter().any(|a| a == name),
        Terminator::Br {
            cond,
            then_args,
            else_args,
            ..
        } => cond == name || then_args.iter().chain(else_args).any(|a| a == name),
        Terminator::Ret { vals, .. } => vals.iter().any(|a| a == name),
    }
}

fn terminator_span(t: &Terminator) -> Span {
    match t {
        Terminator::Jmp { span, .. }
        | Terminator::Br { span, .. }
        | Terminator::Ret { span, .. } => *span,
    }
}
