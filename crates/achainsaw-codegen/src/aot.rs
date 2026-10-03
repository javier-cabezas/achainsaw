use achainsaw_ir::ast::{BinaryOp, Constant, Instruction, Module, Terminator};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module as ClifModule};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use target_lexicon::Triple;

use crate::jit::to_clif_type;

pub struct AotCompiler {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: ObjectModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
}

impl AotCompiler {
    pub fn new(target_triple: Option<&str>) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("is_pic", "true")?;
        flag_builder.set("opt_level", "speed")?;

        let isa = if let Some(triple_str) = target_triple {
            let triple =
                Triple::from_str(triple_str).map_err(|e| anyhow!("Invalid target triple: {e}"))?;
            cranelift_codegen::isa::lookup(triple)?.finish(settings::Flags::new(flag_builder))?
        } else {
            let isa_builder = cranelift_native::builder()
                .map_err(|msg| anyhow!("Host machine not supported by Cranelift: {msg}"))?;
            isa_builder.finish(settings::Flags::new(flag_builder))?
        };

        let builder = ObjectBuilder::new(isa, "achainsaw_aot", default_libcall_names())?;
        let mut module = ObjectModule::new(builder);

        let mut alloc_sig = module.make_signature();
        alloc_sig.params.push(AbiParam::new(types::I64));
        alloc_sig.returns.push(AbiParam::new(types::I64));
        let rt_malloc_id = module.declare_function("malloc", Linkage::Import, &alloc_sig)?;

        let mut free_sig = module.make_signature();
        free_sig.params.push(AbiParam::new(types::I64));
        let rt_free_id = module.declare_function("free", Linkage::Import, &free_sig)?;

        let ctx = module.make_context();

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            module,
            rt_malloc_id,
            rt_free_id,
        })
    }

    pub fn compile_module(&mut self, ir_mod: &Module) -> Result<()> {
        let mut func_ids = HashMap::new();
        let mut func_returns = HashMap::new();

        // 1. Declare external functions (Linkage::Import)
        for ext_fn in &ir_mod.extern_functions {
            let mut sig = self.module.make_signature();
            for (_, p_ty) in &ext_fn.params {
                sig.params.push(AbiParam::new(to_clif_type(*p_ty)));
            }
            if let Some(r_ty) = ext_fn.ret_type {
                sig.returns.push(AbiParam::new(to_clif_type(r_ty)));
            }
            func_returns.insert(ext_fn.name.clone(), ext_fn.ret_type);

            let func_id = self
                .module
                .declare_function(&ext_fn.name, Linkage::Import, &sig)?;
            func_ids.insert(ext_fn.name.clone(), func_id);
        }

        // 2. Declare internal functions (Linkage::Export)
        for func in &ir_mod.functions {
            let mut sig = self.module.make_signature();
            for (_, p_ty) in &func.params {
                sig.params.push(AbiParam::new(to_clif_type(*p_ty)));
            }
            if let Some(r_ty) = func.ret_type {
                sig.returns.push(AbiParam::new(to_clif_type(r_ty)));
            }
            func_returns.insert(func.name.clone(), func.ret_type);

            let func_id = self
                .module
                .declare_function(&func.name, Linkage::Export, &sig)?;
            func_ids.insert(func.name.clone(), func_id);
        }

        // 3. Define each function
        for func in &ir_mod.functions {
            self.compile_function(func, &func_ids, &func_returns)?;
        }

        Ok(())
    }

    fn compile_function(
        &mut self,
        func: &achainsaw_ir::ast::Function,
        func_ids: &HashMap<String, FuncId>,
        func_returns: &HashMap<String, Option<Type>>,
    ) -> Result<()> {
        let func_id = *func_ids.get(&func.name).unwrap();

        self.ctx
            .func
            .signature
            .clear(self.module.target_config().default_call_conv);
        for (_, p_ty) in &func.params {
            self.ctx
                .func
                .signature
                .params
                .push(AbiParam::new(to_clif_type(*p_ty)));
        }
        if let Some(r_ty) = func.ret_type {
            self.ctx
                .func
                .signature
                .returns
                .push(AbiParam::new(to_clif_type(r_ty)));
        }

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        let mut clif_blocks = HashMap::new();
        for block in &func.blocks {
            let clif_block = builder.create_block();
            clif_blocks.insert(block.label.clone(), clif_block);

            for (_, p_ty) in &block.params {
                builder.append_block_param(clif_block, to_clif_type(*p_ty));
            }
        }

        let entry_block = *clif_blocks.get(&func.blocks[0].label).unwrap();
        builder.append_block_params_for_function_params(entry_block);

        let mut values: HashMap<String, (Value, Type)> = HashMap::new();

        for (i, (p_name, p_ty)) in func.params.iter().enumerate() {
            let val = builder.block_params(entry_block)[i];
            values.insert(p_name.clone(), (val, *p_ty));
        }

        for block in &func.blocks {
            let clif_block = *clif_blocks.get(&block.label).unwrap();
            builder.switch_to_block(clif_block);

            if block.label != func.blocks[0].label {
                for (i, (p_name, p_ty)) in block.params.iter().enumerate() {
                    let val = builder.block_params(clif_block)[i];
                    values.insert(p_name.clone(), (val, *p_ty));
                }
            }

            for inst in &block.instructions {
                match inst {
                    Instruction::AssignConst { dst, val, ty, .. } => {
                        let clif_ty = to_clif_type(*ty);
                        let v = match val {
                            Constant::Int(n) => builder.ins().iconst(clif_ty, *n),
                            Constant::Float(f) => match ty {
                                Type::F32 => builder.ins().f32const(*f as f32),
                                Type::F64 => builder.ins().f64const(*f),
                                _ => builder.ins().f32const(*f as f32),
                            },
                        };
                        values.insert(dst.clone(), (v, *ty));
                    }
                    Instruction::Binary {
                        op, dst, lhs, rhs, ..
                    } => {
                        let (lhs_val, lhs_ty) = *values.get(lhs).unwrap();
                        let (rhs_val, _) = *values.get(rhs).unwrap();

                        let (res_val, res_ty) = match op {
                            BinaryOp::Add => (builder.ins().iadd(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Sub => (builder.ins().isub(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Mul => (builder.ins().imul(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Div => match lhs_ty {
                                Type::F32 => (builder.ins().fdiv(lhs_val, rhs_val), lhs_ty),
                                Type::F64 => (builder.ins().fdiv(lhs_val, rhs_val), lhs_ty),
                                _ => (builder.ins().sdiv(lhs_val, rhs_val), lhs_ty),
                            },
                            BinaryOp::Rem => (builder.ins().srem(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::And => (builder.ins().band(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Or => (builder.ins().bor(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Xor => (builder.ins().bxor(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Shl => (builder.ins().ishl(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Shr => (builder.ins().sshr(lhs_val, rhs_val), lhs_ty),
                            BinaryOp::Eq => {
                                let c = if lhs_ty == Type::F32 || lhs_ty == Type::F64 {
                                    builder.ins().fcmp(FloatCC::Equal, lhs_val, rhs_val)
                                } else {
                                    builder.ins().icmp(IntCC::Equal, lhs_val, rhs_val)
                                };
                                (builder.ins().uextend(types::I32, c), Type::I32)
                            }
                            BinaryOp::Ne => {
                                let c = if lhs_ty == Type::F32 || lhs_ty == Type::F64 {
                                    builder.ins().fcmp(FloatCC::NotEqual, lhs_val, rhs_val)
                                } else {
                                    builder.ins().icmp(IntCC::NotEqual, lhs_val, rhs_val)
                                };
                                (builder.ins().uextend(types::I32, c), Type::I32)
                            }
                            BinaryOp::Lt => {
                                let c = if lhs_ty == Type::F32 || lhs_ty == Type::F64 {
                                    builder.ins().fcmp(FloatCC::LessThan, lhs_val, rhs_val)
                                } else {
                                    builder.ins().icmp(IntCC::SignedLessThan, lhs_val, rhs_val)
                                };
                                (builder.ins().uextend(types::I32, c), Type::I32)
                            }
                            BinaryOp::Gt => {
                                let c = if lhs_ty == Type::F32 || lhs_ty == Type::F64 {
                                    builder.ins().fcmp(FloatCC::GreaterThan, lhs_val, rhs_val)
                                } else {
                                    builder
                                        .ins()
                                        .icmp(IntCC::SignedGreaterThan, lhs_val, rhs_val)
                                };
                                (builder.ins().uextend(types::I32, c), Type::I32)
                            }
                            BinaryOp::Le => {
                                let c = if lhs_ty == Type::F32 || lhs_ty == Type::F64 {
                                    builder
                                        .ins()
                                        .fcmp(FloatCC::LessThanOrEqual, lhs_val, rhs_val)
                                } else {
                                    builder.ins().icmp(
                                        IntCC::SignedLessThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    )
                                };
                                (builder.ins().uextend(types::I32, c), Type::I32)
                            }
                            BinaryOp::Ge => {
                                let c = if lhs_ty == Type::F32 || lhs_ty == Type::F64 {
                                    builder.ins().fcmp(
                                        FloatCC::GreaterThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    )
                                } else {
                                    builder.ins().icmp(
                                        IntCC::SignedGreaterThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    )
                                };
                                (builder.ins().uextend(types::I32, c), Type::I32)
                            }
                            BinaryOp::VfAdd => (builder.ins().fadd(lhs_val, rhs_val), Type::V128),
                            BinaryOp::VfSub => (builder.ins().fsub(lhs_val, rhs_val), Type::V128),
                            BinaryOp::VfMul => (builder.ins().fmul(lhs_val, rhs_val), Type::V128),
                            BinaryOp::VfDiv => (builder.ins().fdiv(lhs_val, rhs_val), Type::V128),
                            BinaryOp::ViAdd => (builder.ins().iadd(lhs_val, rhs_val), Type::V128),
                            BinaryOp::ViSub => (builder.ins().isub(lhs_val, rhs_val), Type::V128),
                            BinaryOp::ViMul => (builder.ins().imul(lhs_val, rhs_val), Type::V128),
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
                        dst, func, args, ..
                    } => {
                        let target_func_id = *func_ids.get(func).unwrap();
                        let callee = self
                            .module
                            .declare_func_in_func(target_func_id, builder.func);
                        let arg_vals: Vec<Value> =
                            args.iter().map(|a| values.get(a).unwrap().0).collect();
                        let call_inst = builder.ins().call(callee, &arg_vals);
                        if let Some(d) = dst {
                            let results = builder.inst_results(call_inst);
                            let res_val = results[0];
                            let ret_ir_ty = func_returns
                                .get(func)
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
                            Type::I32 => builder.ins().splat(types::I32X4, src_val),
                            Type::I64 => builder.ins().splat(types::I64X2, src_val),
                            _ => builder.ins().splat(types::F32X4, src_val),
                        };
                        values.insert(dst.clone(), (vec_val, Type::V128));
                    }
                    Instruction::ExtractLane {
                        dst, vec, lane, ty, ..
                    } => {
                        let (vec_val, _) = *values.get(vec).unwrap();
                        let scalar_val = builder.ins().extractlane(vec_val, *lane as u8);
                        values.insert(dst.clone(), (scalar_val, *ty));
                    }
                    Instruction::Alloc { dst, size, .. } => {
                        let (size_val, _) = *values.get(size).unwrap();
                        let callee = self
                            .module
                            .declare_func_in_func(self.rt_malloc_id, builder.func);
                        let call_inst = builder.ins().call(callee, &[size_val]);
                        let ptr_val = builder.inst_results(call_inst)[0];
                        values.insert(dst.clone(), (ptr_val, Type::Ptr));
                    }
                    Instruction::Free { ptr, .. } => {
                        let (ptr_val, _) = *values.get(ptr).unwrap();
                        let callee = self
                            .module
                            .declare_func_in_func(self.rt_free_id, builder.func);
                        builder.ins().call(callee, &[ptr_val]);
                    }
                }
            }

            match &block.terminator {
                Terminator::Jmp { target, args, .. } => {
                    let target_block = *clif_blocks.get(target).unwrap();
                    let arg_vals: Vec<BlockArg> = args
                        .iter()
                        .map(|a| BlockArg::Value(values.get(a).unwrap().0))
                        .collect();
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
                    builder
                        .ins()
                        .brif(cond_val, then_target, &then_vals, else_target, &else_vals);
                }
                Terminator::Ret { val, .. } => {
                    if let Some(v_name) = val {
                        let (ret_val, _) = *values.get(v_name).unwrap();
                        builder.ins().return_(&[ret_val]);
                    } else {
                        builder.ins().return_(&[]);
                    }
                }
            }
        }

        builder.seal_all_blocks();
        builder.finalize(self.module.target_config());

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        let product = self.module.finish();
        let bytes = product.emit()?;
        Ok(bytes)
    }

    pub fn emit_to_file(self, path: &Path) -> Result<usize> {
        let bytes = self.finish()?;
        std::fs::write(path, &bytes)?;
        Ok(bytes.len())
    }
}

pub fn link_shared_library(object_path: &Path, shared_path: &Path) -> Result<()> {
    // Try gcc first
    let gcc_res = Command::new("gcc")
        .arg("-shared")
        .arg("-o")
        .arg(shared_path)
        .arg(object_path)
        .output();

    if let Ok(out) = gcc_res {
        if out.status.success() {
            return Ok(());
        }
    }

    // Try clang as fallback
    let clang_res = Command::new("clang")
        .arg("-shared")
        .arg("-o")
        .arg(shared_path)
        .arg(object_path)
        .output();

    if let Ok(out) = clang_res {
        if out.status.success() {
            return Ok(());
        }
    }

    // Try cc as fallback
    let cc_res = Command::new("cc")
        .arg("-shared")
        .arg("-o")
        .arg(shared_path)
        .arg(object_path)
        .output();

    if let Ok(out) = cc_res {
        if out.status.success() {
            return Ok(());
        }
    }

    Err(anyhow!(
        "Failed to link shared library: neither gcc, clang, nor cc succeeded"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use achainsaw_ir::parse_and_validate;
    use object::{Object, ObjectSymbol};

    #[test]
    fn test_aot_compile_native_object() {
        let code = r#"
        fn add(a:i32, b:i32)->i32
          b0:
            res = add a, b
            ret res
        "#;
        let module = parse_and_validate(code).unwrap();
        let mut compiler = AotCompiler::new(None).unwrap();
        compiler.compile_module(&module).unwrap();

        let object_bytes = compiler.finish().unwrap();
        assert!(!object_bytes.is_empty());

        // Parse with object crate and inspect exported symbols
        let obj = object::File::parse(&*object_bytes).expect("Valid native object format");
        let symbols: Vec<String> = obj
            .symbols()
            .filter_map(|s| s.name().ok().map(|n| n.to_string()))
            .collect();

        assert!(
            symbols.iter().any(|s| s.contains("add")),
            "Expected 'add' symbol in object file, found: {:?}",
            symbols
        );
    }
}
