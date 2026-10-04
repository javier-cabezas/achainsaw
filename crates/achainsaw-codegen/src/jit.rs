use achainsaw_ir::ast::Module;
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module as ClifModule};
use std::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::backend::Backend;
use crate::lower::push_abi_params;
pub use crate::lower::{to_clif_type, LowerConfig, RtValue, SandboxConfig};

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
    OutOfMemory {
        requested: usize,
        limit: usize,
    },
    /// Sandboxed code accessed `size` bytes at `addr` outside its arena, or freed a
    /// pointer that is not a live allocation (`size` 0).
    MemoryViolation {
        addr: usize,
        size: usize,
    },
    /// Sandboxed code exceeded `SANDBOX_STACK_BYTES` of call stack.
    StackOverflow,
}

/// Call stack sandboxed code may use before it halts with `ERR_STACK_OVERFLOW`.
pub const SANDBOX_STACK_BYTES: usize = 1 << 20;
/// Smallest sandbox arena; masked accesses assume a vector always fits.
pub const MIN_SANDBOX_ARENA_BYTES: usize = 4096;

/// Memory owned by a sandboxed engine. `alloc` bump-allocates from it, and every access
/// that sandboxed code makes is bounds-checked against `[base, base + len)`. Live
/// allocations are tracked here, outside the arena, so sandboxed code cannot forge them.
struct Arena {
    base: *mut u8,
    layout: Layout,
    top: usize,
    /// `(offset, size)` of live allocations, in increasing offset order.
    live: Vec<(usize, usize)>,
}

impl Arena {
    fn new(len: usize) -> Result<Self> {
        let len = len.max(MIN_SANDBOX_ARENA_BYTES);
        // Alignment 16 lets the allocator use calloc, so untouched pages stay uncommitted.
        let layout = Layout::from_size_align(len, 16)?;
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        if base.is_null() {
            return Err(anyhow!(
                "[ERR_OUT_OF_MEMORY] Cannot reserve a {len}-byte sandbox arena"
            ));
        }
        Ok(Self {
            base,
            layout,
            top: 0,
            live: Vec::new(),
        })
    }

    fn len(&self) -> usize {
        self.layout.size()
    }

    fn reset(&mut self) {
        self.top = 0;
        self.live.clear();
    }

    /// True when `[addr, addr + size)` lies inside the arena.
    fn contains(&self, addr: usize, size: usize) -> bool {
        let off = addr.wrapping_sub(self.base as usize);
        size <= self.len() && off <= self.len() - size
    }

    fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        let start = self.top.checked_add(15)? & !15;
        let end = start.checked_add(size)?;
        if end > self.len() {
            return None;
        }
        self.top = end;
        self.live.push((start, size));
        Some(unsafe { self.base.add(start) })
    }

    /// Releases a live allocation and returns its size. Freeing the most recent
    /// allocation gives its space back, so alloc/free loops do not exhaust the arena.
    fn free(&mut self, ptr: *mut u8) -> Option<usize> {
        let off = (ptr as usize).wrapping_sub(self.base as usize);
        let idx = self.live.iter().rposition(|&(o, _)| o == off)?;
        let (_, size) = self.live.remove(idx);
        self.top = self.live.last().map_or(0, |&(o, s)| o + s);
        Some(size)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.base, self.layout) };
    }
}

thread_local! {
    static CURRENT_STATUS: Cell<ExecutionStatus> = const { Cell::new(ExecutionStatus::Ok) };
    static FUEL_REMAINING: Cell<i64> = const { Cell::new(-1) };
    static MEMORY_ALLOCATED: Cell<usize> = const { Cell::new(0) };
    static MEMORY_QUOTA: Cell<usize> = const { Cell::new(0) };
    /// Arena of the sandboxed engine currently executing on this thread, or null.
    static ACTIVE_ARENA: Cell<*const RefCell<Arena>> = const { Cell::new(std::ptr::null()) };
    /// Lowest stack address sandboxed code may reach; 0 disables the check.
    static STACK_LIMIT: Cell<usize> = const { Cell::new(0) };
}

/// Records `status` unless an earlier failure is already recorded, so the first cause
/// (e.g. an allocation failure) is reported rather than its consequences.
fn raise(status: ExecutionStatus) {
    CURRENT_STATUS.with(|s| {
        if s.get() == ExecutionStatus::Ok {
            s.set(status);
        }
    });
}

fn halted() -> bool {
    CURRENT_STATUS.with(|s| s.get()) != ExecutionStatus::Ok
}

fn with_active_arena<R>(f: impl FnOnce(&RefCell<Arena>) -> R) -> Option<R> {
    let arena = ACTIVE_ARENA.with(|a| a.get());
    // SAFETY: set only by `call_typed` for the duration of a call on its own engine.
    (!arena.is_null()).then(|| f(unsafe { &*arena }))
}

/// Approximate current stack pointer.
#[inline(never)]
fn stack_address() -> usize {
    let marker = 0u8;
    std::hint::black_box(&marker) as *const u8 as usize
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
        ExecutionStatus::MemoryViolation { addr, size: 0 } => Err(anyhow!(
            "[ERR_MEMORY_VIOLATION] free of {addr:#x}, which is not a live allocation"
        )),
        ExecutionStatus::MemoryViolation { addr, size } => Err(anyhow!(
            "[ERR_MEMORY_VIOLATION] Access of {size} bytes at {addr:#x} is outside memory obtained from alloc"
        )),
        ExecutionStatus::StackOverflow => Err(anyhow!(
            "[ERR_STACK_OVERFLOW] Execution halted: call stack exceeded {SANDBOX_STACK_BYTES} bytes (unbounded recursion?)"
        )),
    }
}

/// Sandbox bounds-check failure: records the violation; the caller then unwinds.
extern "C" fn rt_sandbox_fault(addr: usize, size: usize) {
    raise(ExecutionStatus::MemoryViolation { addr, size });
}

/// Called on entry to every sandboxed function; returns 1 (and records
/// `StackOverflow`) once the stack grows past the limit set by `call_typed`.
extern "C" fn rt_stack_check() -> i32 {
    let limit = STACK_LIMIT.with(|l| l.get());
    if limit != 0 && stack_address() < limit {
        raise(ExecutionStatus::StackOverflow);
        return 1;
    }
    0
}

/// Bounds-checks the three operands of `mm` before it runs (positive dimensions only).
/// Returns 1 and records a violation if any matrix leaves the arena, including when
/// its byte size overflows.
extern "C" fn rt_sandbox_check_mm(
    pc: usize,
    pa: usize,
    pb: usize,
    m: i64,
    n: i64,
    k: i64,
    esize: i64,
) -> i32 {
    let bytes = |rows: i64, cols: i64, es: i64| {
        (rows as usize)
            .checked_mul(cols as usize)
            .and_then(|e| e.checked_mul(es as usize))
            .unwrap_or(usize::MAX)
    };
    let operands = [
        (pc, bytes(m, n, 4)),
        (pa, bytes(m, k, esize)),
        (pb, bytes(k, n, esize)),
    ];
    let bad = with_active_arena(|arena| {
        let arena = arena.borrow();
        operands
            .into_iter()
            .find(|&(p, size)| !arena.contains(p, size))
    });
    match bad {
        Some(Some((addr, size))) => {
            raise(ExecutionStatus::MemoryViolation { addr, size });
            1
        }
        Some(None) => 0,
        // Not sandboxed: nothing to check.
        None => 0,
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
            raise(ExecutionStatus::OutOfFuel);
            return 1;
        }
        f.set(remaining);
        0
    })
}

/// Charged on every branch. Also returns 1 once any failure has been recorded, so code
/// that keeps running after a callee unwound (e.g. on a memory violation) stops at its
/// next branch.
pub extern "C" fn rt_check_fuel() -> i32 {
    if halted() {
        return 1;
    }
    FUEL_REMAINING.with(|f| {
        let fuel = f.get();
        if fuel < 0 {
            return 0;
        }
        if fuel <= 0 {
            raise(ExecutionStatus::OutOfFuel);
            return 1;
        }
        f.set(fuel - 1);
        if fuel - 1 <= 0 {
            raise(ExecutionStatus::OutOfFuel);
            return 1;
        }
        0
    })
}

unsafe extern "C" fn rt_malloc(size: usize) -> *mut u8 {
    if let Some(ptr) = with_active_arena(|arena| {
        let mut arena = arena.borrow_mut();
        arena.alloc(size).unwrap_or_else(|| {
            raise(ExecutionStatus::OutOfMemory {
                requested: size,
                limit: arena.len(),
            });
            std::ptr::null_mut()
        })
    }) {
        if !ptr.is_null() {
            MEMORY_ALLOCATED.with(|m| m.set(m.get() + size));
        }
        return ptr;
    }

    let quota = MEMORY_QUOTA.with(|q| q.get());
    let current = MEMORY_ALLOCATED.with(|m| m.get());
    if quota > 0 && current.saturating_add(size) > quota {
        raise(ExecutionStatus::OutOfMemory {
            requested: size,
            limit: quota,
        });
        return std::ptr::null_mut();
    }

    let total = size + 16;
    let raw = malloc(total);
    if raw.is_null() {
        raise(ExecutionStatus::OutOfMemory {
            requested: size,
            limit: quota,
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
    if let Some(freed) = with_active_arena(|arena| arena.borrow_mut().free(ptr)) {
        match freed {
            Some(size) => MEMORY_ALLOCATED.with(|m| m.set(m.get().saturating_sub(size))),
            None => raise(ExecutionStatus::MemoryViolation {
                addr: ptr as usize,
                size: 0,
            }),
        }
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

/// Points the runtime hooks at a sandboxed engine's arena and stack limit for one call,
/// restoring the previous state when dropped (calls may nest through host callbacks).
struct SandboxActivation {
    prev_arena: *const RefCell<Arena>,
    prev_limit: usize,
}

impl SandboxActivation {
    fn new(arena: &RefCell<Arena>) -> Self {
        arena.borrow_mut().reset();
        let prev_arena = ACTIVE_ARENA.with(|a| a.replace(arena));
        let limit = stack_address().saturating_sub(SANDBOX_STACK_BYTES);
        let prev_limit = STACK_LIMIT.with(|l| l.replace(limit));
        Self {
            prev_arena,
            prev_limit,
        }
    }
}

impl Drop for SandboxActivation {
    fn drop(&mut self) {
        ACTIVE_ARENA.with(|a| a.set(self.prev_arena));
        STACK_LIMIT.with(|l| l.set(self.prev_limit));
    }
}

/// Cranelift JIT module plus the ids of the runtime hooks it imports.
struct CraneliftJit {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: JITModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
    rt_check_fuel_id: FuncId,
    rt_consume_fuel_id: FuncId,
    rt_sandbox_fault_id: FuncId,
    rt_stack_check_id: FuncId,
    rt_sandbox_check_mm_id: FuncId,
}

impl CraneliftJit {
    fn new(
        features: &crate::cpu::CpuFeatures,
        registry: &Arc<RwLock<SymbolRegistry>>,
    ) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false")?;
        flag_builder.set("is_pic", "false")?;
        flag_builder.set("opt_level", "speed")?;
        // v256/v512 values span 2/4 vector registers, more than ABIs return in registers;
        // spill extra return values through an implicit struct-return pointer.
        flag_builder.set("enable_multi_ret_implicit_sret", "true")?;
        // Probe large stack frames page by page so they cannot skip over the guard page.
        flag_builder.set("enable_probestack", "true")?;
        flag_builder.set("probestack_strategy", "inline")?;

        let isa =
            crate::cpu::native_isa_builder(features)?.finish(settings::Flags::new(flag_builder))?;

        let reg_lookup = Arc::clone(registry);

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("rt_malloc", rt_malloc as *const u8);
        jit_builder.symbol("rt_free", rt_free as *const u8);
        jit_builder.symbol("rt_check_fuel", rt_check_fuel as *const u8);
        jit_builder.symbol("rt_consume_fuel", rt_consume_fuel as *const u8);
        jit_builder.symbol("rt_sandbox_fault", rt_sandbox_fault as *const u8);
        jit_builder.symbol("rt_stack_check", rt_stack_check as *const u8);
        jit_builder.symbol("rt_sandbox_check_mm", rt_sandbox_check_mm as *const u8);
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

        let mut fault_sig = module.make_signature();
        fault_sig.params.push(AbiParam::new(types::I64));
        fault_sig.params.push(AbiParam::new(types::I64));
        let rt_sandbox_fault_id =
            module.declare_function("rt_sandbox_fault", Linkage::Import, &fault_sig)?;

        let rt_stack_check_id =
            module.declare_function("rt_stack_check", Linkage::Import, &fuel_sig)?;

        let mut mm_check_sig = module.make_signature();
        for _ in 0..7 {
            mm_check_sig.params.push(AbiParam::new(types::I64));
        }
        mm_check_sig.returns.push(AbiParam::new(types::I32));
        let rt_sandbox_check_mm_id =
            module.declare_function("rt_sandbox_check_mm", Linkage::Import, &mm_check_sig)?;

        let ctx = module.make_context();

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            module,
            rt_malloc_id,
            rt_free_id,
            rt_check_fuel_id,
            rt_consume_fuel_id,
            rt_sandbox_fault_id,
            rt_stack_check_id,
            rt_sandbox_check_mm_id,
        })
    }

    /// Compiles `ir_mod` and returns `(name, function, trampoline)` addresses.
    fn compile(
        &mut self,
        ir_mod: &Module,
        fuel_enabled: bool,
        sandbox: Option<(i64, i64)>,
    ) -> Result<Vec<(String, usize, Option<usize>)>> {
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
        }

        // 3. Lower each function and its dynamic invocation trampoline
        let sandbox = sandbox.map(|(arena_base, arena_len)| SandboxConfig {
            arena_base,
            arena_len,
            fault_id: self.rt_sandbox_fault_id,
            stack_check_id: self.rt_stack_check_id,
            mm_check_id: self.rt_sandbox_check_mm_id,
        });
        let config = LowerConfig {
            // Sandboxed code always checks at branches so it stops after a violation.
            fuel_check_func_id: (fuel_enabled || sandbox.is_some())
                .then_some(self.rt_check_fuel_id),
            fuel_consume_func_id: fuel_enabled.then_some(self.rt_consume_fuel_id),
            rt_malloc_id: self.rt_malloc_id,
            rt_free_id: self.rt_free_id,
            sandbox,
        };

        let mut tramp_ids = HashMap::new();
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
            let param_tys: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
            if param_tys
                .iter()
                .chain(func.ret_type.iter())
                .any(|t| t.is_vector())
            {
                continue;
            }
            let tid = crate::lower::lower_trampoline(
                &mut self.module,
                &mut self.ctx,
                &mut self.builder_context,
                &func.name,
                func_id,
                &param_tys,
                func.ret_type,
            )?;
            tramp_ids.insert(func.name.clone(), tid);
        }

        // 4. Finalize all JIT definitions
        self.module.finalize_definitions()?;

        // 5. Collect pointers
        Ok(ir_mod
            .functions
            .iter()
            .map(|func| {
                let ptr = self.module.get_finalized_function(func_ids[&func.name]) as usize;
                let tramp = tramp_ids
                    .get(&func.name)
                    .map(|tid| self.module.get_finalized_function(*tid) as usize);
                (func.name.clone(), ptr, tramp)
            })
            .collect())
    }
}

/// Code generator behind a `JitEngine`.
enum Codegen {
    Cranelift(Box<CraneliftJit>),
    #[cfg(feature = "llvm")]
    Llvm {
        target: achainsaw_llvm::TargetSpec,
        vx: achainsaw_llvm::VxShape,
        vector_width: Option<u32>,
        vx_bits: u32,
        /// One LLJIT per `compile_module` call; kept alive so earlier code stays valid.
        modules: Vec<achainsaw_llvm::LlvmJit>,
    },
}

pub struct JitEngine {
    backend: Backend,
    codegen: Codegen,
    /// Set by `enable_sandbox`; boxed so its address stays fixed while code runs.
    sandbox: Option<Box<RefCell<Arena>>>,
    pub registry: Arc<RwLock<SymbolRegistry>>,
    pub fuel_enabled: bool,
    pub signatures: HashMap<String, (Vec<Type>, Option<Type>)>,
    function_ptrs: HashMap<String, usize>,
    trampoline_ptrs: HashMap<String, usize>,
}

unsafe impl Send for JitEngine {}

impl JitEngine {
    /// JIT engine for the host CPU, honoring the ISA cap (`ACHAINSAW_MAX_ISA`). Uses the
    /// backend named by `ACHAINSAW_BACKEND`, or Cranelift.
    pub fn new() -> Result<Self> {
        Self::with_features(&crate::cpu::CpuFeatures::effective()?)
    }

    /// JIT engine restricted to `features`, which must be a subset of the host's. Uses the
    /// backend named by `ACHAINSAW_BACKEND`, or Cranelift.
    pub fn with_features(features: &crate::cpu::CpuFeatures) -> Result<Self> {
        Self::with_backend(Backend::resolve(None)?, features)
    }

    /// JIT engine for the (capped) host using a backend chosen by name: `cranelift`,
    /// `llvm`, or `auto`/`None` (see [`Backend::resolve`]).
    pub fn for_backend(choice: Option<&str>) -> Result<Self> {
        Self::with_backend(
            Backend::resolve(choice)?,
            &crate::cpu::CpuFeatures::effective()?,
        )
    }

    /// JIT engine for the (capped) host to compile `module`, choosing the backend with
    /// [`Backend::resolve_for`] (so `auto` uses LLVM for wide-vector modules when built in).
    pub fn for_module(choice: Option<&str>, module: &Module) -> Result<Self> {
        Self::with_backend(
            Backend::resolve_for(choice, module)?,
            &crate::cpu::CpuFeatures::effective()?,
        )
    }

    /// JIT engine using `backend` for the host CPU restricted to `features`.
    pub fn with_backend(backend: Backend, features: &crate::cpu::CpuFeatures) -> Result<Self> {
        let registry = Arc::new(RwLock::new(SymbolRegistry::new()));
        let codegen = match backend {
            Backend::Cranelift => {
                Codegen::Cranelift(Box::new(CraneliftJit::new(features, &registry)?))
            }
            #[cfg(feature = "llvm")]
            Backend::Llvm => {
                // Same host-subset rule as Cranelift: JIT code must run on this CPU.
                crate::cpu::native_isa_builder(features)?;
                let (vx, vector_width) = features.llvm_vector_shape(true);
                let vx_bits = features.native_vector_bits();
                let (cpu, features) = features.llvm_target();
                Codegen::Llvm {
                    target: achainsaw_llvm::TargetSpec {
                        triple: None,
                        cpu,
                        features,
                    },
                    vx,
                    vector_width,
                    vx_bits,
                    modules: Vec::new(),
                }
            }
            #[cfg(not(feature = "llvm"))]
            Backend::Llvm => {
                // `Backend::resolve` already rejects this; keep the error for direct callers.
                return Err(anyhow!(
                    "[ERR_BACKEND_UNAVAILABLE] The llvm backend is not in this build"
                ));
            }
        };
        Ok(Self {
            backend,
            codegen,
            sandbox: None,
            registry,
            fuel_enabled: true,
            signatures: HashMap::new(),
            function_ptrs: HashMap::new(),
            trampoline_ptrs: HashMap::new(),
        })
    }

    /// Backend generating this engine's code.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Width of `vx` in bits for this engine's code (what `vl` reports times the lane
    /// width): 128 on Cranelift; on LLVM 512/256/128 by AVX-512F/AVX2, or the host's SVE
    /// vector length.
    pub fn vx_bits(&self) -> u32 {
        match &self.codegen {
            Codegen::Cranelift(_) => crate::cpu::CRANELIFT_VECTOR_BITS,
            #[cfg(feature = "llvm")]
            Codegen::Llvm { vx_bits, .. } => *vx_bits,
        }
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

    /// Runs untrusted code safely: `alloc` draws from a private `arena_bytes` arena
    /// (at least `MIN_SANDBOX_ARENA_BYTES`), every `ld`/`st`/`ldm`/`stm`/`mm` access and
    /// `free` is checked against it, and recursion is capped at `SANDBOX_STACK_BYTES`.
    /// Violations halt execution with `ERR_MEMORY_VIOLATION` or `ERR_STACK_OVERFLOW`
    /// instead of crashing the process. Must be called before `compile_module`, and calls
    /// need a thread with comfortably more than `SANDBOX_STACK_BYTES` of free stack.
    pub fn enable_sandbox(&mut self, arena_bytes: usize) -> Result<()> {
        if !self.signatures.is_empty() {
            return Err(anyhow!(
                "enable_sandbox must be called before compile_module"
            ));
        }
        self.sandbox = Some(Box::new(RefCell::new(Arena::new(arena_bytes)?)));
        Ok(())
    }

    pub fn is_sandboxed(&self) -> bool {
        self.sandbox.is_some()
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
        let sandbox = self.sandbox.as_ref().map(|arena| {
            let arena = arena.borrow();
            (arena.base as i64, arena.len() as i64)
        });
        let compiled = match &mut self.codegen {
            Codegen::Cranelift(clif) => clif.compile(ir_mod, self.fuel_enabled, sandbox)?,
            #[cfg(feature = "llvm")]
            Codegen::Llvm {
                target,
                vx,
                vector_width,
                modules,
                ..
            } => {
                let hooks = achainsaw_llvm::RuntimeHooks {
                    malloc: rt_malloc as *const () as usize,
                    free: rt_free as *const () as usize,
                    check_fuel: rt_check_fuel as *const () as usize,
                    consume_fuel: rt_consume_fuel as *const () as usize,
                    sandbox_fault: rt_sandbox_fault as *const () as usize,
                    stack_check: rt_stack_check as *const () as usize,
                    sandbox_check_mm: rt_sandbox_check_mm as *const () as usize,
                };
                let opts = achainsaw_llvm::LowerOptions {
                    fuel: self.fuel_enabled,
                    sandbox: sandbox.map(|(base, len)| achainsaw_llvm::SandboxBounds {
                        base: base as u64,
                        len: len as u64,
                    }),
                    vx: *vx,
                    vector_width: *vector_width,
                    ..Default::default()
                };
                // Functions from earlier modules first, then registered symbols.
                let earlier = &self.function_ptrs;
                let registry = &self.registry;
                let resolve = |name: &str| {
                    earlier
                        .get(name)
                        .copied()
                        .or_else(|| registry.read().unwrap().lookup(name).map(|p| p as usize))
                };
                let jit =
                    achainsaw_llvm::LlvmJit::compile(ir_mod, target, &opts, &hooks, &resolve)?;
                let out = ir_mod
                    .functions
                    .iter()
                    .map(|f| {
                        (
                            f.name.clone(),
                            jit.function(&f.name).expect("compiled function"),
                            jit.trampoline(&f.name),
                        )
                    })
                    .collect();
                modules.push(jit);
                out
            }
        };

        for func in &ir_mod.functions {
            let param_tys: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
            self.signatures
                .insert(func.name.clone(), (param_tys, func.ret_type));
        }
        for (name, ptr, tramp) in compiled {
            self.function_ptrs.insert(name.clone(), ptr);
            if let Some(t) = tramp {
                self.trampoline_ptrs.insert(name, t);
            }
        }
        Ok(())
    }

    pub fn get_fn_ptr(&self, name: &str) -> Option<*const u8> {
        if let Some(&ptr) = self.function_ptrs.get(name) {
            return Some(ptr as *const u8);
        }
        match &self.codegen {
            Codegen::Cranelift(clif) => match clif.module.get_name(name)? {
                cranelift_module::FuncOrDataId::Func(fid) => {
                    Some(clif.module.get_finalized_function(fid))
                }
                _ => None,
            },
            #[cfg(feature = "llvm")]
            Codegen::Llvm { .. } => None,
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

        let _sandbox = self.sandbox.as_deref().map(SandboxActivation::new);
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
