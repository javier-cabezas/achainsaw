use achainsaw_ir::ast::Module;
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module as ClifModule};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::lower::push_abi_params;
pub use crate::lower::{to_clif_type, LowerConfig, RtValue};

static USER_SYMBOLS: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());
static USER_LIBRARIES: Mutex<Vec<Arc<libloading::Library>>> = Mutex::new(Vec::new());

pub fn register_global_symbol(name: impl Into<String>, ptr: *const u8) {
    USER_SYMBOLS
        .lock()
        .unwrap()
        .push((name.into(), ptr as usize));
}

pub fn load_global_library(path: &str) -> Result<()> {
    let lib = unsafe { libloading::Library::new(path)? };
    USER_LIBRARIES.lock().unwrap().push(Arc::new(lib));
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
        for lib in libs.iter().rev() {
            unsafe {
                if let Ok(sym) = lib.get::<*const u8>(name.as_bytes()) {
                    return Some(*sym);
                }
            }
        }
    }
    let default_reg = SymbolRegistry::new();
    default_reg.lookup(name)
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

/// Charges `units` of fuel at once for bulk operations (`mm`). Same contract as
/// `rt_check_fuel`: returns 1 and records `OutOfFuel` when the budget runs out.
pub extern "C" fn rt_consume_fuel(units: i64) -> i32 {
    FUEL_REMAINING.with(|f| {
        let fuel = f.get();
        if fuel < 0 {
            return 0;
        }
        let remaining = fuel.saturating_sub(units.max(0));
        if remaining <= 0 {
            f.set(0);
            CURRENT_STATUS.with(|s| s.set(ExecutionStatus::OutOfFuel));
            return 1;
        }
        f.set(remaining);
        0
    })
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
    libraries: Vec<Arc<libloading::Library>>,
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
            for lib in libs.iter() {
                reg.libraries.push(Arc::clone(lib));
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
        let arc_lib = Arc::new(lib);
        self.libraries.push(Arc::clone(&arc_lib));
        USER_LIBRARIES.lock().unwrap().push(arc_lib);
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
    rt_consume_fuel_id: FuncId,
    pub registry: Arc<RwLock<SymbolRegistry>>,
    pub fuel_enabled: bool,
    pub signatures: HashMap<String, (Vec<Type>, Option<Type>)>,
    function_ptrs: HashMap<String, usize>,
    trampoline_ptrs: HashMap<String, usize>,
}

unsafe impl Send for JitEngine {}

impl JitEngine {
    /// JIT engine for the host CPU, honoring the ISA cap (`ACHAINSAW_MAX_ISA`).
    pub fn new() -> Result<Self> {
        Self::with_features(&crate::cpu::CpuFeatures::effective()?)
    }

    /// JIT engine restricted to `features`, which must be a subset of the host's.
    pub fn with_features(features: &crate::cpu::CpuFeatures) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false")?;
        flag_builder.set("is_pic", "false")?;
        flag_builder.set("opt_level", "speed")?;
        // v256/v512 values span 2/4 vector registers, more than ABIs return in registers;
        // spill extra return values through an implicit struct-return pointer.
        flag_builder.set("enable_multi_ret_implicit_sret", "true")?;

        let isa =
            crate::cpu::native_isa_builder(features)?.finish(settings::Flags::new(flag_builder))?;

        let registry = Arc::new(RwLock::new(SymbolRegistry::new()));
        let reg_lookup = Arc::clone(&registry);

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("rt_malloc", rt_malloc as *const u8);
        jit_builder.symbol("rt_free", rt_free as *const u8);
        jit_builder.symbol("rt_check_fuel", rt_check_fuel as *const u8);
        jit_builder.symbol("rt_consume_fuel", rt_consume_fuel as *const u8);
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

        let mut consume_sig = module.make_signature();
        consume_sig.params.push(AbiParam::new(types::I64));
        consume_sig.returns.push(AbiParam::new(types::I32));
        let rt_consume_fuel_id =
            module.declare_function("rt_consume_fuel", Linkage::Import, &consume_sig)?;

        let ctx = module.make_context();

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            module,
            rt_malloc_id,
            rt_free_id,
            rt_check_fuel_id,
            rt_consume_fuel_id,
            registry,
            fuel_enabled: true,
            signatures: HashMap::new(),
            function_ptrs: HashMap::new(),
            trampoline_ptrs: HashMap::new(),
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
                push_abi_params(&mut sig.params, *p_ty);
            }
            if let Some(r_ty) = ext_fn.ret_type {
                push_abi_params(&mut sig.returns, r_ty);
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
                push_abi_params(&mut sig.params, *p_ty);
            }
            if let Some(r_ty) = func.ret_type {
                push_abi_params(&mut sig.returns, r_ty);
            }
            func_returns.insert(func.name.clone(), func.ret_type);

            let func_id = self
                .module
                .declare_function(&func.name, Linkage::Export, &sig)?;
            func_ids.insert(func.name.clone(), func_id);

            let param_tys: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
            self.signatures
                .insert(func.name.clone(), (param_tys, func.ret_type));
        }

        // 3. Lower each function and its dynamic invocation trampoline
        let config = LowerConfig {
            fuel_check_func_id: if self.fuel_enabled {
                Some(self.rt_check_fuel_id)
            } else {
                None
            },
            fuel_consume_func_id: self.fuel_enabled.then_some(self.rt_consume_fuel_id),
            rt_malloc_id: self.rt_malloc_id,
            rt_free_id: self.rt_free_id,
        };

        for func in &ir_mod.functions {
            let func_id = func_ids[&func.name];
            crate::lower::lower_function(
                &mut self.module,
                &mut self.ctx,
                &mut self.builder_context,
                func,
                &func_ids,
                &func_returns,
                &config,
            )?;
            self.module.define_function(func_id, &mut self.ctx)?;
            self.module.clear_context(&mut self.ctx);

            // Host calls go through a scalar trampoline; functions that take or return
            // vectors are only callable from AIR code.
            let (param_tys, ret_ty) = &self.signatures[&func.name];
            if param_tys.iter().chain(ret_ty.iter()).any(|t| t.is_vector()) {
                continue;
            }
            crate::lower::lower_trampoline(
                &mut self.module,
                &mut self.ctx,
                &mut self.builder_context,
                &func.name,
                func_id,
                param_tys,
                *ret_ty,
            )?;
        }

        // 4. Finalize all JIT definitions
        self.module.finalize_definitions()?;

        // 5. Cache pointers
        for func in &ir_mod.functions {
            let func_id = func_ids[&func.name];
            let ptr = self.module.get_finalized_function(func_id);
            self.function_ptrs.insert(func.name.clone(), ptr as usize);

            let tramp_name = format!("__achainsaw_trampoline_{}", func.name);
            if let Some(cranelift_module::FuncOrDataId::Func(tid)) =
                self.module.get_name(&tramp_name)
            {
                let tptr = self.module.get_finalized_function(tid);
                self.trampoline_ptrs
                    .insert(func.name.clone(), tptr as usize);
            }
        }

        Ok(())
    }

    pub fn get_fn_ptr(&self, name: &str) -> Option<*const u8> {
        if let Some(&ptr) = self.function_ptrs.get(name) {
            return Some(ptr as *const u8);
        }
        let func_id = self.module.get_name(name)?;
        match func_id {
            cranelift_module::FuncOrDataId::Func(fid) => {
                Some(self.module.get_finalized_function(fid))
            }
            _ => None,
        }
    }

    /// Generic typed function invoker supporting any signature and calling convention
    pub unsafe fn call_typed(&self, name: &str, args: &[RtValue]) -> Result<Option<RtValue>> {
        let (param_types, ret_type) = self
            .signatures
            .get(name)
            .ok_or_else(|| anyhow!("Function '{name}' not found"))?;

        let tramp_ptr = self
            .trampoline_ptrs
            .get(name)
            .copied()
            .ok_or_else(|| {
                anyhow!("Function '{name}' takes or returns vectors, so it can only be called from AIR code; pass vectors through memory (ptr) instead")
            })?;

        if args.len() != param_types.len() {
            return Err(anyhow!(
                "Function '{name}' expects {} arguments, received {}",
                param_types.len(),
                args.len()
            ));
        }

        let mut raw_args = Vec::with_capacity(args.len());
        for (i, (&arg, &ty)) in args.iter().zip(param_types).enumerate() {
            let raw = arg.to_u64(ty).map_err(|e| anyhow!("Argument {i}: {e}"))?;
            raw_args.push(raw);
        }

        let mut raw_ret: u64 = 0;
        reset_execution_status();

        let tramp_fn: extern "C" fn(*const u64, *mut u64) = std::mem::transmute(tramp_ptr);
        tramp_fn(raw_args.as_ptr(), &mut raw_ret);

        check_execution_status()?;

        if let Some(r_ty) = *ret_type {
            Ok(Some(RtValue::from_u64(raw_ret, r_ty)))
        } else {
            Ok(None)
        }
    }

    // High-level runner helpers for convenience and backwards-compatibility
    pub unsafe fn run_i32_to_i32(&self, name: &str, arg: i32) -> Result<i32> {
        match self.call_typed(name, &[RtValue::I32(arg)])? {
            Some(RtValue::I32(v)) => Ok(v),
            Some(RtValue::I64(v)) => Ok(v as i32),
            other => Err(anyhow!("Unexpected return value: {:?}", other)),
        }
    }

    pub unsafe fn run_i32_2_to_i32(&self, name: &str, a: i32, b: i32) -> Result<i32> {
        match self.call_typed(name, &[RtValue::I32(a), RtValue::I32(b)])? {
            Some(RtValue::I32(v)) => Ok(v),
            Some(RtValue::I64(v)) => Ok(v as i32),
            other => Err(anyhow!("Unexpected return value: {:?}", other)),
        }
    }

    pub unsafe fn run_ptr_ptr_i32_to_f32(
        &self,
        name: &str,
        p0: *const f32,
        p1: *const f32,
        n: i32,
    ) -> Result<f32> {
        match self.call_typed(
            name,
            &[
                RtValue::Ptr(p0 as usize),
                RtValue::Ptr(p1 as usize),
                RtValue::I32(n),
            ],
        )? {
            Some(RtValue::F32(v)) => Ok(v),
            other => Err(anyhow!("Unexpected return value: {:?}", other)),
        }
    }

    pub unsafe fn run_f32_to_f32(&self, name: &str, arg: f32) -> Result<f32> {
        match self.call_typed(name, &[RtValue::F32(arg)])? {
            Some(RtValue::F32(v)) => Ok(v),
            other => Err(anyhow!("Unexpected return value: {:?}", other)),
        }
    }

    pub unsafe fn run_f64_to_f64(&self, name: &str, arg: f64) -> Result<f64> {
        match self.call_typed(name, &[RtValue::F64(arg)])? {
            Some(RtValue::F64(v)) => Ok(v),
            other => Err(anyhow!("Unexpected return value: {:?}", other)),
        }
    }

    pub unsafe fn run_ptr_ptr_i64_to_f32(
        &self,
        name: &str,
        p0: *const f32,
        p1: *const f32,
        n: i64,
    ) -> Result<f32> {
        match self.call_typed(
            name,
            &[
                RtValue::Ptr(p0 as usize),
                RtValue::Ptr(p1 as usize),
                RtValue::I64(n),
            ],
        )? {
            Some(RtValue::F32(v)) => Ok(v),
            other => Err(anyhow!("Unexpected return value: {:?}", other)),
        }
    }
}
