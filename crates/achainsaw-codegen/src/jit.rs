use achainsaw_ir::ast::{BinaryOp, Constant, Instruction, Module, Terminator};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module as ClifModule};
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

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

unsafe extern "C" fn rt_malloc(size: usize) -> *mut u8 {
    malloc(size)
}

unsafe extern "C" fn rt_free(ptr: *mut u8) {
    free(ptr)
}

pub struct JitEngine {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: JITModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
}

impl JitEngine {
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false")?;
        flag_builder.set("is_pic", "false")?;
        flag_builder.set("opt_level", "speed")?;

        let isa_builder = cranelift_native::builder()
            .map_err(|msg| anyhow!("Host machine not supported by Cranelift: {msg}"))?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("rt_malloc", rt_malloc as *const u8);
        jit_builder.symbol("rt_free", rt_free as *const u8);

        let mut module = JITModule::new(jit_builder);

        let mut alloc_sig = module.make_signature();
        alloc_sig.params.push(AbiParam::new(types::I64));
        alloc_sig.returns.push(AbiParam::new(types::I64));
        let rt_malloc_id = module.declare_function("rt_malloc", Linkage::Import, &alloc_sig)?;

        let mut free_sig = module.make_signature();
        free_sig.params.push(AbiParam::new(types::I64));
        let rt_free_id = module.declare_function("rt_free", Linkage::Import, &free_sig)?;

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

        // 1. Declare all functions
        for func in &ir_mod.functions {
            let mut sig = self.module.make_signature();
            for (_, p_ty) in &func.params {
                sig.params.push(AbiParam::new(to_clif_type(*p_ty)));
            }
            if let Some(r_ty) = func.ret_type {
                sig.returns.push(AbiParam::new(to_clif_type(r_ty)));
            }

            let func_id = self
                .module
                .declare_function(&func.name, Linkage::Export, &sig)?;
            func_ids.insert(func.name.clone(), func_id);
        }

        // 2. Define each function
        for func in &ir_mod.functions {
            self.compile_function(func, &func_ids)?;
        }

        // 3. Finalize all JIT definitions
        self.module.finalize_definitions()?;
        Ok(())
    }

    fn compile_function(
        &mut self,
        func: &achainsaw_ir::ast::Function,
        func_ids: &HashMap<String, FuncId>,
    ) -> Result<()> {
        let func_id = *func_ids.get(&func.name).unwrap();

        // Clear and setup function signature in context
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

        // Create all Cranelift blocks first
        let mut clif_blocks = HashMap::new();
        for block in &func.blocks {
            let clif_block = builder.create_block();
            clif_blocks.insert(block.label.clone(), clif_block);

            // Add block parameters
            for (_, p_ty) in &block.params {
                builder.append_block_param(clif_block, to_clif_type(*p_ty));
            }
        }

        // Entry block parameters are the function parameters
        let entry_block = *clif_blocks.get(&func.blocks[0].label).unwrap();
        builder.append_block_params_for_function_params(entry_block);

        // Values table mapping register name to Cranelift Value
        let mut values: HashMap<String, (Value, Type)> = HashMap::new();

        // Register function parameters in entry block
        for (i, (p_name, p_ty)) in func.params.iter().enumerate() {
            let val = builder.block_params(entry_block)[i];
            values.insert(p_name.clone(), (val, *p_ty));
        }

        // Translate each block
        for block in &func.blocks {
            let clif_block = *clif_blocks.get(&block.label).unwrap();
            builder.switch_to_block(clif_block);

            // Map block parameters (skip entry block as it already received function params)
            if block.label != func.blocks[0].label {
                for (i, (p_name, p_ty)) in block.params.iter().enumerate() {
                    let val = builder.block_params(clif_block)[i];
                    values.insert(p_name.clone(), (val, *p_ty));
                }
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
                    Instruction::Binary { op, dst, lhs, rhs, .. } => {
                        let (lhs_val, lhs_ty) = *values.get(lhs).unwrap();
                        let (rhs_val, _) = *values.get(rhs).unwrap();

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
                                    let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Ge => {
                                    let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs_val, rhs_val);
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
                                BinaryOp::ViAdd => (builder.ins().iadd(lhs_val, rhs_val), Type::V128),
                                BinaryOp::ViSub => (builder.ins().isub(lhs_val, rhs_val), Type::V128),
                                BinaryOp::ViMul => (builder.ins().imul(lhs_val, rhs_val), Type::V128),
                                _ => return Err(anyhow!("Unsupported vector op {:?}", op)),
                            },
                            _ => {
                                let (rhs_val, rhs_ty) = *values.get(rhs).unwrap();
                                let default_res_ty = if lhs_ty == Type::Ptr || rhs_ty == Type::Ptr {
                                    Type::Ptr
                                } else {
                                    lhs_ty
                                };
                                match op {
                                    BinaryOp::Add => (builder.ins().iadd(lhs_val, rhs_val), default_res_ty),
                                    BinaryOp::Sub => (builder.ins().isub(lhs_val, rhs_val), default_res_ty),
                                BinaryOp::Mul => (builder.ins().imul(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Div => (builder.ins().sdiv(lhs_val, rhs_val), lhs_ty),
                                BinaryOp::Rem => (builder.ins().srem(lhs_val, rhs_val), lhs_ty),
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
                                    let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Gt => {
                                    let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Le => {
                                    let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                    BinaryOp::Ge => {
                                        let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs_val, rhs_val);
                                        let ext = builder.ins().uextend(types::I32, cmp);
                                        (ext, Type::I32)
                                    }
                                    _ => return Err(anyhow!("Invalid scalar integer op {:?}", op)),
                                }
                            },
                        };
                        values.insert(dst.clone(), (res_val, res_ty));
                    }
                    Instruction::Load { dst, ptr, ty, .. } => {
                        let (ptr_val, _) = *values.get(ptr).unwrap();
                        let clif_ty = to_clif_type(*ty);
                        let val = builder.ins().load(clif_ty, MemFlagsData::trusted(), ptr_val, 0);
                        values.insert(dst.clone(), (val, *ty));
                    }
                    Instruction::Store { ptr, val, .. } => {
                        let (ptr_val, _) = *values.get(ptr).unwrap();
                        let (val_val, _) = *values.get(val).unwrap();
                        builder.ins().store(MemFlagsData::trusted(), val_val, ptr_val, 0);
                    }
                    Instruction::Call { dst, func, args, .. } => {
                        let target_func_id = *func_ids.get(func).unwrap();
                        let callee = self.module.declare_func_in_func(target_func_id, &mut builder.func);
                        let arg_vals: Vec<Value> = args.iter().map(|a| values.get(a).unwrap().0).collect();
                        let call_inst = builder.ins().call(callee, &arg_vals);
                        if let Some(d) = dst {
                            let results = builder.inst_results(call_inst);
                            let res_val = results[0];
                            let func_sig = &self.module.declarations().get_function_decl(target_func_id).signature;
                            let ret_clif_ty = func_sig.returns[0].value_type;
                            let ret_ir_ty = match ret_clif_ty {
                                types::I8 => Type::I8,
                                types::I16 => Type::I16,
                                types::I32 => Type::I32,
                                types::I64 => Type::I64,
                                types::F32 => Type::F32,
                                types::F64 => Type::F64,
                                _ => Type::I32,
                            };
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
                    Instruction::ExtractLane { dst, vec, lane, ty, .. } => {
                        let (vec_val, _) = *values.get(vec).unwrap();
                        let scalar_val = builder.ins().extractlane(vec_val, *lane as u8);
                        values.insert(dst.clone(), (scalar_val, *ty));
                    }
                    Instruction::Alloc { dst, size, .. } => {
                        let (size_val, _) = *values.get(size).unwrap();
                        let callee = self.module.declare_func_in_func(self.rt_malloc_id, &mut builder.func);
                        let call_inst = builder.ins().call(callee, &[size_val]);
                        let ptr_val = builder.inst_results(call_inst)[0];
                        values.insert(dst.clone(), (ptr_val, Type::Ptr));
                    }
                    Instruction::Free { ptr, .. } => {
                        let (ptr_val, _) = *values.get(ptr).unwrap();
                        let callee = self.module.declare_func_in_func(self.rt_free_id, &mut builder.func);
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
                    if let Some(v) = val {
                        let (ret_val, _) = *values.get(v).unwrap();
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

    pub fn get_fn_ptr(&self, name: &str) -> Option<*const u8> {
        let func_id = self.module.get_name(name)?;
        match func_id {
            cranelift_module::FuncOrDataId::Func(fid) => Some(self.module.get_finalized_function(fid)),
            _ => None,
        }
    }

    // High level runner helpers for agents
    pub unsafe fn run_i32_to_i32(&self, name: &str, arg: i32) -> Result<i32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        let func: extern "C" fn(i32) -> i32 = std::mem::transmute(ptr);
        Ok(func(arg))
    }

    pub unsafe fn run_i32_2_to_i32(&self, name: &str, a: i32, b: i32) -> Result<i32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        let func: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(ptr);
        Ok(func(a, b))
    }

    pub unsafe fn run_ptr_ptr_i32_to_f32(
        &self,
        name: &str,
        p0: *const f32,
        p1: *const f32,
        n: i32,
    ) -> Result<f32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        let func: extern "C" fn(*const f32, *const f32, i32) -> f32 = std::mem::transmute(ptr);
        Ok(func(p0, p1, n))
    }

    pub unsafe fn run_ptr_ptr_i64_to_f32(
        &self,
        name: &str,
        p0: *const f32,
        p1: *const f32,
        n: i64,
    ) -> Result<f32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        let func: extern "C" fn(*const f32, *const f32, i64) -> f32 = std::mem::transmute(ptr);
        Ok(func(p0, p1, n))
    }
}
