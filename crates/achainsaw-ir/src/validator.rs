use crate::ast::*;
use crate::diag::Diagnostic;
use crate::types::Type;
use std::collections::{HashMap, HashSet};

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

    pub fn validate_module(&mut self, module: &Module) -> Result<(), Diagnostic> {
        // Collect function signatures
        for func in &module.functions {
            if self.functions.contains_key(&func.name) {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_FUNCTION",
                    format!("Duplicate function name '{}'", func.name),
                    func.span,
                ));
            }
            let param_types = func.params.iter().map(|(_, ty)| *ty).collect();
            self.functions.insert(func.name.clone(), (param_types, func.ret_type));
        }

        // Validate each function
        for func in &module.functions {
            self.validate_function(func)?;
        }

        Ok(())
    }

    fn validate_function(&self, func: &Function) -> Result<(), Diagnostic> {
        let mut block_params: HashMap<String, Vec<Type>> = HashMap::new();
        let mut block_names: HashSet<String> = HashSet::new();

        for block in &func.blocks {
            if !block_names.insert(block.label.clone()) {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_BLOCK",
                    format!("Duplicate block label '{}' in function '{}'", block.label, func.name),
                    block.span,
                ));
            }
            let p_types = block.params.iter().map(|(_, ty)| *ty).collect();
            block_params.insert(block.label.clone(), p_types);
        }

        // Register type map: reg -> type
        let mut reg_types: HashMap<String, Type> = HashMap::new();

        // Seed function parameters
        for (p_name, p_ty) in &func.params {
            if reg_types.insert(p_name.clone(), *p_ty).is_some() {
                return Err(Diagnostic::error(
                    "ERR_DUPLICATE_PARAM",
                    format!("Duplicate parameter '{p_name}' in function '{}'", func.name),
                    func.span,
                ));
            }
        }

        for (b_idx, block) in func.blocks.iter().enumerate() {
            // Block parameters
            for (p_name, p_ty) in &block.params {
                if reg_types.contains_key(p_name) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{p_name}' re-defined in block '{}' (violates SSA)", block.label),
                        block.span,
                    ));
                }
                reg_types.insert(p_name.clone(), *p_ty);
            }

            // Instructions
            for (i_idx, inst) in block.instructions.iter().enumerate() {
                self.validate_instruction(inst, &mut reg_types, i_idx)?;
            }

            // Terminator
            self.validate_terminator(
                &block.terminator,
                &reg_types,
                &block_params,
                func.ret_type,
                b_idx,
            )?;
        }

        Ok(())
    }

    fn validate_instruction(
        &self,
        inst: &Instruction,
        reg_types: &mut HashMap<String, Type>,
        inst_idx: usize,
    ) -> Result<(), Diagnostic> {
        match inst {
            Instruction::AssignConst { dst, val, ty, span } => {
                if reg_types.contains_key(dst) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{dst}' assigned multiple times (violates SSA)"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                match (val, ty) {
                    (Constant::Int(_), Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Ptr) => {}
                    (Constant::Float(_), Type::F32 | Type::F64) => {}
                    _ => {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Constant value does not match specified type '{ty}'"),
                            *span,
                        ).with_instruction_index(inst_idx));
                    }
                }
                reg_types.insert(dst.clone(), *ty);
            }
            Instruction::Binary { op, dst, lhs, rhs, span } => {
                if reg_types.contains_key(dst) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{dst}' assigned multiple times (violates SSA)"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }

                let lhs_ty = self.check_reg(lhs, reg_types, *span, inst_idx)?;
                let rhs_ty = self.check_reg(rhs, reg_types, *span, inst_idx)?;

                let out_ty = if op.is_vector() {
                    if lhs_ty != Type::V128 || rhs_ty != Type::V128 {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Vector op '{op:?}' requires operands of type 'v128', found '{lhs_ty}' and '{rhs_ty}'"),
                            *span,
                        ).with_instruction_index(inst_idx));
                    }
                    Type::V128
                } else if (lhs_ty == Type::Ptr && rhs_ty == Type::I64)
                    || (lhs_ty == Type::I64 && rhs_ty == Type::Ptr)
                {
                    if *op != BinaryOp::Add && *op != BinaryOp::Sub {
                        return Err(Diagnostic::error(
                            "ERR_INVALID_PTR_OP",
                            format!("Invalid pointer operation '{op:?}'. Only add and sub are allowed with ptr and i64"),
                            *span,
                        ).with_instruction_index(inst_idx));
                    }
                    Type::Ptr
                } else if lhs_ty != rhs_ty {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Binary op '{op:?}' operands must have matching types, found '{lhs_ty}' and '{rhs_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                } else if op.is_comparison() {
                    Type::I32 // Boolean / condition as i32
                } else {
                    lhs_ty
                };

                reg_types.insert(dst.clone(), out_ty);
            }
            Instruction::Load { dst, ptr, ty, span } => {
                if reg_types.contains_key(dst) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{dst}' assigned multiple times (violates SSA)"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                let ptr_ty = self.check_reg(ptr, reg_types, *span, inst_idx)?;
                if ptr_ty != Type::Ptr {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Load pointer operand '{ptr}' must be of type 'ptr', found '{ptr_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                reg_types.insert(dst.clone(), *ty);
            }
            Instruction::Store { ptr, val, span } => {
                let ptr_ty = self.check_reg(ptr, reg_types, *span, inst_idx)?;
                if ptr_ty != Type::Ptr {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Store target pointer '{ptr}' must be of type 'ptr', found '{ptr_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                self.check_reg(val, reg_types, *span, inst_idx)?;
            }
            Instruction::Call { dst, func, args, span } => {
                let (param_types, ret_type) = self.functions.get(func).ok_or_else(|| {
                    Diagnostic::error(
                        "ERR_UNDEFINED_FUNCTION",
                        format!("Call to undefined function '{func}'"),
                        *span,
                    ).with_instruction_index(inst_idx)
                })?;

                if args.len() != param_types.len() {
                    return Err(Diagnostic::error(
                        "ERR_ARITY_MISMATCH",
                        format!("Function '{func}' expects {} arguments, received {}", param_types.len(), args.len()),
                        *span,
                    ).with_instruction_index(inst_idx));
                }

                for (arg, &expected_ty) in args.iter().zip(param_types) {
                    let actual_ty = self.check_reg(arg, reg_types, *span, inst_idx)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Argument '{arg}' passed to '{func}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ).with_instruction_index(inst_idx));
                    }
                }

                if let Some(d) = dst {
                    if reg_types.contains_key(d) {
                        return Err(Diagnostic::error(
                            "ERR_SSA_REDEFINITION",
                            format!("Register '{d}' assigned multiple times (violates SSA)"),
                            *span,
                        ).with_instruction_index(inst_idx));
                    }
                    let rty = ret_type.ok_or_else(|| {
                        Diagnostic::error(
                            "ERR_VOID_ASSIGNMENT",
                            format!("Function '{func}' does not return a value"),
                            *span,
                        ).with_instruction_index(inst_idx)
                    })?;
                    reg_types.insert(d.clone(), rty);
                }
            }
            Instruction::Splat { dst, src, span } => {
                if reg_types.contains_key(dst) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{dst}' assigned multiple times (violates SSA)"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                let src_ty = self.check_reg(src, reg_types, *span, inst_idx)?;
                if !matches!(src_ty, Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::I16 | Type::I8) {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Splat operand '{src}' must be numeric scalar, found '{src_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                reg_types.insert(dst.clone(), Type::V128);
            }
            Instruction::ExtractLane { dst, vec, lane, ty, span } => {
                if reg_types.contains_key(dst) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{dst}' assigned multiple times (violates SSA)"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                let vec_ty = self.check_reg(vec, reg_types, *span, inst_idx)?;
                if vec_ty != Type::V128 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("ExtractLane requires vector operand of type 'v128', found '{vec_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                if *lane >= 16 {
                    return Err(Diagnostic::error(
                        "ERR_OUT_OF_BOUNDS_LANE",
                        format!("ExtractLane index {lane} out of bounds for 128-bit vector"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                reg_types.insert(dst.clone(), *ty);
            }
            Instruction::Alloc { dst, size, span } => {
                if reg_types.contains_key(dst) {
                    return Err(Diagnostic::error(
                        "ERR_SSA_REDEFINITION",
                        format!("Register '{dst}' assigned multiple times (violates SSA)"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                let size_ty = self.check_reg(size, reg_types, *span, inst_idx)?;
                if size_ty != Type::I64 && size_ty != Type::I32 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Alloc size must be i64 or i32, found '{size_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
                reg_types.insert(dst.clone(), Type::Ptr);
            }
            Instruction::Free { ptr, span } => {
                let ptr_ty = self.check_reg(ptr, reg_types, *span, inst_idx)?;
                if ptr_ty != Type::Ptr {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Free target must be of type 'ptr', found '{ptr_ty}'"),
                        *span,
                    ).with_instruction_index(inst_idx));
                }
            }
        }
        Ok(())
    }

    fn check_reg(
        &self,
        name: &str,
        reg_types: &HashMap<String, Type>,
        span: crate::diag::Span,
        inst_idx: usize,
    ) -> Result<Type, Diagnostic> {
        if let Some(ty) = reg_types.get(name) {
            Ok(*ty)
        } else {
            let available: Vec<String> = reg_types.keys().cloned().collect();
            Err(Diagnostic::error(
                "ERR_UNDEFINED_REG",
                format!("Register '{name}' is used before definition"),
                span,
            )
            .with_instruction_index(inst_idx)
            .with_context(serde_json::json!({
                "target": name,
                "available_registers": available,
            })))
        }
    }

    fn validate_terminator(
        &self,
        term: &Terminator,
        reg_types: &HashMap<String, Type>,
        block_params: &HashMap<String, Vec<Type>>,
        func_ret_ty: Option<Type>,
        _block_idx: usize,
    ) -> Result<(), Diagnostic> {
        match term {
            Terminator::Jmp { target, args, span } => {
                let expected_params = block_params.get(target).ok_or_else(|| {
                    Diagnostic::error(
                        "ERR_UNDEFINED_BLOCK",
                        format!("Jump target block '{target}' does not exist"),
                        *span,
                    )
                })?;

                if args.len() != expected_params.len() {
                    return Err(Diagnostic::error(
                        "ERR_ARITY_MISMATCH",
                        format!("Block '{target}' expects {} arguments, received {}", expected_params.len(), args.len()),
                        *span,
                    ));
                }

                for (arg, &expected_ty) in args.iter().zip(expected_params) {
                    let actual_ty = self.check_reg(arg, reg_types, *span, 0)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Jump argument '{arg}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                }
            }
            Terminator::Br {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                span,
            } => {
                let cond_ty = self.check_reg(cond, reg_types, *span, 0)?;
                if cond_ty != Type::I32 {
                    return Err(Diagnostic::error(
                        "ERR_TYPE_MISMATCH",
                        format!("Branch condition '{cond}' must be i32 (boolean), found '{cond_ty}'"),
                        *span,
                    ));
                }

                let then_expected = block_params.get(then_block).ok_or_else(|| {
                    Diagnostic::error(
                        "ERR_UNDEFINED_BLOCK",
                        format!("Branch target block '{then_block}' does not exist"),
                        *span,
                    )
                })?;
                if then_args.len() != then_expected.len() {
                    return Err(Diagnostic::error(
                        "ERR_ARITY_MISMATCH",
                        format!("Block '{then_block}' expects {} arguments, received {}", then_expected.len(), then_args.len()),
                        *span,
                    ));
                }
                for (arg, &expected_ty) in then_args.iter().zip(then_expected) {
                    let actual_ty = self.check_reg(arg, reg_types, *span, 0)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Branch argument '{arg}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                }

                let else_expected = block_params.get(else_block).ok_or_else(|| {
                    Diagnostic::error(
                        "ERR_UNDEFINED_BLOCK",
                        format!("Branch target block '{else_block}' does not exist"),
                        *span,
                    )
                })?;
                if else_args.len() != else_expected.len() {
                    return Err(Diagnostic::error(
                        "ERR_ARITY_MISMATCH",
                        format!("Block '{else_block}' expects {} arguments, received {}", else_expected.len(), else_args.len()),
                        *span,
                    ));
                }
                for (arg, &expected_ty) in else_args.iter().zip(else_expected) {
                    let actual_ty = self.check_reg(arg, reg_types, *span, 0)?;
                    if actual_ty != expected_ty {
                        return Err(Diagnostic::error(
                            "ERR_TYPE_MISMATCH",
                            format!("Branch argument '{arg}' has type '{actual_ty}', expected '{expected_ty}'"),
                            *span,
                        ));
                    }
                }
            }
            Terminator::Ret { val, span } => match (val, func_ret_ty) {
                (None, None) => {}
                (Some(v), Some(expected_ty)) => {
                    let actual_ty = self.check_reg(v, reg_types, *span, 0)?;
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
