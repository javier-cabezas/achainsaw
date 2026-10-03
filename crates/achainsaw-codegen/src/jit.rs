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
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

static USER_SYMBOLS: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());
static USER_LIBRARIES: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn register_global_symbol(name: impl Into<String>, ptr: *const u8) {
    USER_SYMBOLS
        .lock()
        .unwrap()
        .push((name.into(), ptr as usize));
}

pub fn load_global_library(path: &str) -> Result<()> {
    unsafe {
        let _ = libloading::Library::new(path)?;
    }
    USER_LIBRARIES.lock().unwrap().push(path.to_string());
    Ok(())
}

pub fn get_global_symbol_address(name: &str) -> Option<*const u8> {
    if let Ok(syms) = USER_SYMBOLS.lock() {
        for (n, addr) in syms.iter().rev() {
            if n == name {
                return Some(*addr as *const u8);
            }
        }
    }
    if let Ok(libs) = USER_LIBRARIES.lock() {
        for lib_path in libs.iter().rev() {
            unsafe {
                if let Ok(lib) = libloading::Library::new(lib_path) {
                    if let Ok(sym) = lib.get::<*const u8>(name.as_bytes()) {
                        return Some(*sym);
                    }
                }
            }
        }
    }
    let default_reg = SymbolRegistry::new();
    default_reg.lookup(name)
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Ok,
    OutOfFuel,
    OutOfMemory { requested: usize, limit: usize },
}

thread_local! {
    static CURRENT_STATUS: Cell<ExecutionStatus> = const { Cell::new(ExecutionStatus::Ok) };
    static FUEL_REMAINING: Cell<i64> = const { Cell::new(-1) };
    static MEMORY_ALLOCATED: Cell<usize> = const { Cell::new(0) };
    static MEMORY_QUOTA: Cell<usize> = const { Cell::new(0) };
}

pub fn set_execution_fuel(fuel: Option<u64>) {
    FUEL_REMAINING.with(|f| match fuel {
        Some(val) => f.set(val as i64),
        None => f.set(-1),
    });
}

pub fn get_remaining_fuel() -> Option<u64> {
    FUEL_REMAINING.with(|f| {
        let val = f.get();
        if val < 0 {
            None
        } else {
            Some(val as u64)
        }
    })
}

pub fn set_memory_quota(quota_bytes: usize) {
    MEMORY_QUOTA.with(|q| q.set(quota_bytes));
}

pub fn get_allocated_memory() -> usize {
    MEMORY_ALLOCATED.with(|m| m.get())
}

pub fn get_execution_status() -> ExecutionStatus {
    CURRENT_STATUS.with(|s| s.get())
}

pub fn reset_execution_status() {
    CURRENT_STATUS.with(|s| s.set(ExecutionStatus::Ok));
}

pub fn check_execution_status() -> Result<()> {
    match get_execution_status() {
        ExecutionStatus::Ok => Ok(()),
        ExecutionStatus::OutOfFuel => Err(anyhow!(
            "[ERR_OUT_OF_FUEL] Execution halted: loop fuel budget exhausted"
        )),
        ExecutionStatus::OutOfMemory { requested, limit } => Err(anyhow!(
            "[ERR_OUT_OF_MEMORY] Allocation of {requested} bytes exceeded memory quota of {limit} bytes"
        )),
    }
}

pub extern "C" fn rt_check_fuel() -> i32 {
    FUEL_REMAINING.with(|f| {
        let fuel = f.get();
        if fuel < 0 {
            return 0;
        }
        if fuel <= 0 {
            CURRENT_STATUS.with(|s| s.set(ExecutionStatus::OutOfFuel));
            return 1;
        }
        f.set(fuel - 1);
        if fuel - 1 <= 0 {
            CURRENT_STATUS.with(|s| s.set(ExecutionStatus::OutOfFuel));
            return 1;
        }
        0
    })
}

unsafe extern "C" fn rt_malloc(size: usize) -> *mut u8 {
    let quota = MEMORY_QUOTA.with(|q| q.get());
    let current = MEMORY_ALLOCATED.with(|m| m.get());
    if quota > 0 && current.saturating_add(size) > quota {
        CURRENT_STATUS.with(|s| {
            s.set(ExecutionStatus::OutOfMemory {
                requested: size,
                limit: quota,
            })
        });
        return std::ptr::null_mut();
    }

    let total = size + 16;
    let raw = malloc(total);
    if raw.is_null() {
        CURRENT_STATUS.with(|s| {
            s.set(ExecutionStatus::OutOfMemory {
                requested: size,
                limit: quota,
            })
        });
        return std::ptr::null_mut();
    }

    *(raw as *mut usize) = size;
    MEMORY_ALLOCATED.with(|m| m.set(current + size));
    raw.add(16)
}

unsafe extern "C" fn rt_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let raw = ptr.sub(16);
    let size = *(raw as *const usize);
    MEMORY_ALLOCATED.with(|m| {
        let cur = m.get();
        m.set(cur.saturating_sub(size));
    });
    free(raw);
}

pub struct SymbolRegistry {
    custom: HashMap<String, *const u8>,
    libraries: Vec<libloading::Library>,
}

unsafe impl Send for SymbolRegistry {}
unsafe impl Sync for SymbolRegistry {}

impl Default for SymbolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            custom: HashMap::new(),
            libraries: Vec::new(),
        };
        reg.register_default_math();
        if let Ok(libs) = USER_LIBRARIES.lock() {
            for lib_path in libs.iter() {
                let _ = reg.load_library(lib_path);
            }
        }
        if let Ok(syms) = USER_SYMBOLS.lock() {
            for (name, addr) in syms.iter() {
                reg.register(name.clone(), *addr as *const u8);
            }
        }
        reg
    }

    pub fn register(&mut self, name: impl Into<String>, ptr: *const u8) {
        self.custom.insert(name.into(), ptr);
    }

    pub fn load_library(&mut self, path: &str) -> Result<()> {
        let lib = unsafe { libloading::Library::new(path)? };
        self.libraries.push(lib);
        Ok(())
    }

    pub fn lookup(&self, name: &str) -> Option<*const u8> {
        if let Some(&ptr) = self.custom.get(name) {
            return Some(ptr);
        }
        for lib in self.libraries.iter().rev() {
            unsafe {
                if let Ok(sym) = lib.get::<*const u8>(name.as_bytes()) {
                    return Some(*sym);
                }
            }
        }
        None
    }

    fn register_default_math(&mut self) {
        unsafe extern "C" fn m_sinf(x: f32) -> f32 {
            x.sin()
        }
        unsafe extern "C" fn m_cosf(x: f32) -> f32 {
            x.cos()
        }
        unsafe extern "C" fn m_tanf(x: f32) -> f32 {
            x.tan()
        }
        unsafe extern "C" fn m_sqrtf(x: f32) -> f32 {
            x.sqrt()
        }
        unsafe extern "C" fn m_expf(x: f32) -> f32 {
            x.exp()
        }
        unsafe extern "C" fn m_logf(x: f32) -> f32 {
            x.ln()
        }
        unsafe extern "C" fn m_powf(x: f32, y: f32) -> f32 {
            x.powf(y)
        }
        unsafe extern "C" fn m_fabsf(x: f32) -> f32 {
            x.abs()
        }
        unsafe extern "C" fn m_floorf(x: f32) -> f32 {
            x.floor()
        }
        unsafe extern "C" fn m_ceilf(x: f32) -> f32 {
            x.ceil()
        }
        unsafe extern "C" fn m_roundf(x: f32) -> f32 {
            x.round()
        }

        unsafe extern "C" fn m_sin(x: f64) -> f64 {
            x.sin()
        }
        unsafe extern "C" fn m_cos(x: f64) -> f64 {
            x.cos()
        }
        unsafe extern "C" fn m_tan(x: f64) -> f64 {
            x.tan()
        }
        unsafe extern "C" fn m_sqrt(x: f64) -> f64 {
            x.sqrt()
        }
        unsafe extern "C" fn m_exp(x: f64) -> f64 {
            x.exp()
        }
        unsafe extern "C" fn m_log(x: f64) -> f64 {
            x.ln()
        }
        unsafe extern "C" fn m_pow(x: f64, y: f64) -> f64 {
            x.powf(y)
        }
        unsafe extern "C" fn m_fabs(x: f64) -> f64 {
            x.abs()
        }
        unsafe extern "C" fn m_floor(x: f64) -> f64 {
            x.floor()
        }
        unsafe extern "C" fn m_ceil(x: f64) -> f64 {
            x.ceil()
        }
        unsafe extern "C" fn m_round(x: f64) -> f64 {
            x.round()
        }

        self.register("sinf", m_sinf as *const u8);
        self.register("cosf", m_cosf as *const u8);
        self.register("tanf", m_tanf as *const u8);
        self.register("sqrtf", m_sqrtf as *const u8);
        self.register("expf", m_expf as *const u8);
        self.register("logf", m_logf as *const u8);
        self.register("powf", m_powf as *const u8);
        self.register("fabsf", m_fabsf as *const u8);
        self.register("floorf", m_floorf as *const u8);
        self.register("ceilf", m_ceilf as *const u8);
        self.register("roundf", m_roundf as *const u8);

        self.register("sin", m_sin as *const u8);
        self.register("cos", m_cos as *const u8);
        self.register("tan", m_tan as *const u8);
        self.register("sqrt", m_sqrt as *const u8);
        self.register("exp", m_exp as *const u8);
        self.register("log", m_log as *const u8);
        self.register("pow", m_pow as *const u8);
        self.register("fabs", m_fabs as *const u8);
        self.register("floor", m_floor as *const u8);
        self.register("ceil", m_ceil as *const u8);
        self.register("round", m_round as *const u8);
    }
}

pub struct JitEngine {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: JITModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
    rt_check_fuel_id: FuncId,
    pub registry: Arc<RwLock<SymbolRegistry>>,
    pub fuel_enabled: bool,
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

        let registry = Arc::new(RwLock::new(SymbolRegistry::new()));
        let reg_lookup = Arc::clone(&registry);

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("rt_malloc", rt_malloc as *const u8);
        jit_builder.symbol("rt_free", rt_free as *const u8);
        jit_builder.symbol("rt_check_fuel", rt_check_fuel as *const u8);
        jit_builder.symbol_lookup_fn(Box::new(move |name: &str| {
            reg_lookup.read().unwrap().lookup(name)
        }));

        let mut module = JITModule::new(jit_builder);

        let mut alloc_sig = module.make_signature();
        alloc_sig.params.push(AbiParam::new(types::I64));
        alloc_sig.returns.push(AbiParam::new(types::I64));
        let rt_malloc_id = module.declare_function("rt_malloc", Linkage::Import, &alloc_sig)?;

        let mut free_sig = module.make_signature();
        free_sig.params.push(AbiParam::new(types::I64));
        let rt_free_id = module.declare_function("rt_free", Linkage::Import, &free_sig)?;

        let mut fuel_sig = module.make_signature();
        fuel_sig.returns.push(AbiParam::new(types::I32));
        let rt_check_fuel_id =
            module.declare_function("rt_check_fuel", Linkage::Import, &fuel_sig)?;

        let ctx = module.make_context();

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            module,
            rt_malloc_id,
            rt_free_id,
            rt_check_fuel_id,
            registry,
            fuel_enabled: true,
        })
    }

    pub fn set_fuel(&mut self, fuel: Option<u64>) {
        set_execution_fuel(fuel);
    }

    pub fn set_fuel_enabled(&mut self, enabled: bool) {
        self.fuel_enabled = enabled;
    }

    pub fn is_fuel_enabled(&self) -> bool {
        self.fuel_enabled
    }

    pub fn set_memory_quota(&self, bytes: usize) {
        set_memory_quota(bytes);
    }

    pub fn get_allocated_memory(&self) -> usize {
        get_allocated_memory()
    }

    pub fn register_symbol(&self, name: impl Into<String>, ptr: *const u8) {
        self.registry.write().unwrap().register(name, ptr);
    }

    pub fn load_library(&self, path: &str) -> Result<()> {
        self.registry.write().unwrap().load_library(path)
    }

    pub fn lookup_symbol(&self, name: &str) -> Option<*const u8> {
        self.registry.read().unwrap().lookup(name)
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

        // 4. Finalize all JIT definitions
        self.module.finalize_definitions()?;
        Ok(())
    }

    fn compile_function(
        &mut self,
        func: &achainsaw_ir::ast::Function,
        func_ids: &HashMap<String, FuncId>,
        func_returns: &HashMap<String, Option<Type>>,
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

        let fuel_trap_block = if self.fuel_enabled {
            Some(builder.create_block())
        } else {
            None
        };

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
                    Instruction::Binary {
                        op, dst, lhs, rhs, ..
                    } => {
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
                                    let cmp =
                                        builder.ins().fcmp(FloatCC::NotEqual, lhs_val, rhs_val);
                                    let ext = builder.ins().uextend(types::I32, cmp);
                                    (ext, Type::I32)
                                }
                                BinaryOp::Lt => {
                                    let cmp =
                                        builder.ins().fcmp(FloatCC::LessThan, lhs_val, rhs_val);
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
                                    let cmp = builder.ins().fcmp(
                                        FloatCC::LessThanOrEqual,
                                        lhs_val,
                                        rhs_val,
                                    );
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
                                BinaryOp::VfAdd => {
                                    (builder.ins().fadd(lhs_val, rhs_val), Type::V128)
                                }
                                BinaryOp::VfSub => {
                                    (builder.ins().fsub(lhs_val, rhs_val), Type::V128)
                                }
                                BinaryOp::VfMul => {
                                    (builder.ins().fmul(lhs_val, rhs_val), Type::V128)
                                }
                                BinaryOp::VfDiv => {
                                    (builder.ins().fdiv(lhs_val, rhs_val), Type::V128)
                                }
                                BinaryOp::ViAdd => {
                                    (builder.ins().iadd(lhs_val, rhs_val), Type::V128)
                                }
                                BinaryOp::ViSub => {
                                    (builder.ins().isub(lhs_val, rhs_val), Type::V128)
                                }
                                BinaryOp::ViMul => {
                                    (builder.ins().imul(lhs_val, rhs_val), Type::V128)
                                }
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
                                    BinaryOp::Add => {
                                        (builder.ins().iadd(lhs_val, rhs_val), default_res_ty)
                                    }
                                    BinaryOp::Sub => {
                                        (builder.ins().isub(lhs_val, rhs_val), default_res_ty)
                                    }
                                    BinaryOp::Mul => (builder.ins().imul(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Div => (builder.ins().sdiv(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Rem => (builder.ins().srem(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::And => (builder.ins().band(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Or => (builder.ins().bor(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Xor => (builder.ins().bxor(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Shl => (builder.ins().ishl(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Shr => (builder.ins().sshr(lhs_val, rhs_val), lhs_ty),
                                    BinaryOp::Eq => {
                                        let cmp =
                                            builder.ins().icmp(IntCC::Equal, lhs_val, rhs_val);
                                        let ext = builder.ins().uextend(types::I32, cmp);
                                        (ext, Type::I32)
                                    }
                                    BinaryOp::Ne => {
                                        let cmp =
                                            builder.ins().icmp(IntCC::NotEqual, lhs_val, rhs_val);
                                        let ext = builder.ins().uextend(types::I32, cmp);
                                        (ext, Type::I32)
                                    }
                                    BinaryOp::Lt => {
                                        let cmp = builder.ins().icmp(
                                            IntCC::SignedLessThan,
                                            lhs_val,
                                            rhs_val,
                                        );
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

            // Translate terminators
            match &block.terminator {
                Terminator::Jmp { target, args, .. } => {
                    let target_block = *clif_blocks.get(target).unwrap();
                    let arg_vals: Vec<BlockArg> = args
                        .iter()
                        .map(|a| BlockArg::Value(values.get(a).unwrap().0))
                        .collect();

                    if let Some(trap_block) = fuel_trap_block {
                        let callee = self
                            .module
                            .declare_func_in_func(self.rt_check_fuel_id, builder.func);
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

                    if let Some(trap_block) = fuel_trap_block {
                        let callee = self
                            .module
                            .declare_func_in_func(self.rt_check_fuel_id, builder.func);
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
        builder.finalize(self.module.target_config());

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        Ok(())
    }

    pub fn get_fn_ptr(&self, name: &str) -> Option<*const u8> {
        let func_id = self.module.get_name(name)?;
        match func_id {
            cranelift_module::FuncOrDataId::Func(fid) => {
                Some(self.module.get_finalized_function(fid))
            }
            _ => None,
        }
    }

    // High level runner helpers for agents
    pub unsafe fn run_i32_to_i32(&self, name: &str, arg: i32) -> Result<i32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        reset_execution_status();
        let func: extern "C" fn(i32) -> i32 = std::mem::transmute(ptr);
        let res = func(arg);
        check_execution_status()?;
        Ok(res)
    }

    pub unsafe fn run_i32_2_to_i32(&self, name: &str, a: i32, b: i32) -> Result<i32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        reset_execution_status();
        let func: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(ptr);
        let res = func(a, b);
        check_execution_status()?;
        Ok(res)
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
        reset_execution_status();
        let func: extern "C" fn(*const f32, *const f32, i32) -> f32 = std::mem::transmute(ptr);
        let res = func(p0, p1, n);
        check_execution_status()?;
        Ok(res)
    }

    pub unsafe fn run_f32_to_f32(&self, name: &str, arg: f32) -> Result<f32> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        reset_execution_status();
        let func: extern "C" fn(f32) -> f32 = std::mem::transmute(ptr);
        let res = func(arg);
        check_execution_status()?;
        Ok(res)
    }

    pub unsafe fn run_f64_to_f64(&self, name: &str, arg: f64) -> Result<f64> {
        let ptr = self
            .get_fn_ptr(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;
        reset_execution_status();
        let func: extern "C" fn(f64) -> f64 = std::mem::transmute(ptr);
        let res = func(arg);
        check_execution_status()?;
        Ok(res)
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
        reset_execution_status();
        let func: extern "C" fn(*const f32, *const f32, i64) -> f32 = std::mem::transmute(ptr);
        let res = func(p0, p1, n);
        check_execution_status()?;
        Ok(res)
    }
}
