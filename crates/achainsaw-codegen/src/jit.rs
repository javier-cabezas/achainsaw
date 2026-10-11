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
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::backend::Backend;
use crate::lower::push_abi_params;
pub use crate::lower::{to_clif_type, LowerConfig, ParConfig, RtValue, SandboxConfig};

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

// The arena owns its memory; `par` workers share it behind a `Mutex`.
unsafe impl Send for Arena {}

/// Locks an arena. A panic while it was held cannot leave it inconsistent (every update is
/// a single assignment or `Vec` operation), so poisoning is ignored.
fn lock(arena: &Mutex<Arena>) -> std::sync::MutexGuard<'_, Arena> {
    arena.lock().unwrap_or_else(|e| e.into_inner())
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
    static ACTIVE_ARENA: Cell<*const Mutex<Arena>> = const { Cell::new(std::ptr::null()) };
    /// Lowest stack address sandboxed code may reach; 0 disables the check.
    static STACK_LIMIT: Cell<usize> = const { Cell::new(0) };
    /// Fuel counter of the engine whose code is running on this thread (see
    /// `FuelActivation`), or null.
    static ACTIVE_FUEL: Cell<*const Cell<i64>> = const { Cell::new(std::ptr::null()) };
    /// `par` region this thread is a worker of, or null.
    static PAR_REGION: Cell<*const ParRegion> = const { Cell::new(std::ptr::null()) };
    /// Most threads a `par` started on this thread may use; 0 means the whole pool.
    static PAR_THREADS: Cell<usize> = const { Cell::new(0) };
}

/// Engine fuel counter value meaning "no budget": never reaches zero in practice.
const UNLIMITED_FUEL: i64 = i64::MAX;

/// Records `status` unless an earlier failure is already recorded, so the first cause
/// (e.g. an allocation failure) is reported rather than its consequences. Also empties the
/// running engine's fuel counter, so its code stops at the next branch: JIT code only
/// consults the runtime when the inline counter runs out.
fn raise(status: ExecutionStatus) {
    CURRENT_STATUS.with(|s| {
        if s.get() == ExecutionStatus::Ok {
            s.set(status);
        }
    });
    with_active_fuel(|counter| counter.set(0));
}

fn with_active_fuel<R>(f: impl FnOnce(&Cell<i64>) -> R) -> Option<R> {
    let counter = ACTIVE_FUEL.with(|a| a.get());
    // SAFETY: set only by `call_typed` for the duration of a call on its own engine.
    (!counter.is_null()).then(|| f(unsafe { &*counter }))
}

fn halted() -> bool {
    CURRENT_STATUS.with(|s| s.get()) != ExecutionStatus::Ok
}

fn with_active_arena<R>(f: impl FnOnce(&Mutex<Arena>) -> R) -> Option<R> {
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
        let arena = lock(arena);
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

/// Charges `units` of fuel at once for bulk operations (`mm`): returns 1 and records
/// `OutOfFuel` when the budget runs out, 0 otherwise.
pub extern "C" fn rt_consume_fuel(units: i64) -> i32 {
    if let Some(r) = with_active_fuel(|counter| {
        if charge(counter, units.max(0)) {
            0
        } else {
            raise(ExecutionStatus::OutOfFuel);
            1
        }
    }) {
        return r;
    }
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

/// Takes `units` from an inline fuel counter, topping it up from the shared budget when
/// this thread is a `par` worker. False when the budget cannot cover them.
fn charge(counter: &Cell<i64>, units: i64) -> bool {
    let mut left = counter.get().saturating_sub(units);
    if left <= 0 {
        if let Some(region) = active_region() {
            left = left.saturating_add(region.draw(PAR_FUEL_CHUNK.saturating_sub(left)));
        }
    }
    counter.set(left.max(0));
    left > 0
}

/// Slow path of the inline fuel check that JIT code runs on every branch (decrement the
/// engine's counter, continue while it is positive). Reached when the counter hits zero:
/// either the budget is spent, or `raise` emptied it after a failure. Returns 1 (unwind)
/// in both cases, recording `OutOfFuel` for the former. A `par` worker first tries to
/// refill its counter from the shared budget.
pub extern "C" fn rt_fuel_exhausted() -> i32 {
    if halted() {
        return 1;
    }
    if with_active_fuel(|counter| charge(counter, 0)) == Some(true) {
        return 0;
    }
    raise(ExecutionStatus::OutOfFuel);
    1
}

/// `rt_fuel_exhausted` for Cranelift code, in the `PreserveAll` convention: saves every
/// register a C call may clobber, so the register allocator keeps loop values in registers
/// across the (cold) call instead of spilling them on the hot path. It returns nothing:
/// the caller reloads the fuel counter, which is positive exactly when execution continues.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
extern "C" fn rt_fuel_exhausted_preserve_all() {
    // Saves the System V and Windows caller-saved GPRs and xmm0-15 (the vector registers
    // Cranelift allocates), keeps the stack 16-byte aligned, and leaves the 32 bytes of
    // shadow space a Windows callee may use.
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "push rax",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push r8",
        "push r9",
        "push r10",
        "push r11",
        "sub rsp, 296",
        "movdqu [rsp + 32], xmm0",
        "movdqu [rsp + 48], xmm1",
        "movdqu [rsp + 64], xmm2",
        "movdqu [rsp + 80], xmm3",
        "movdqu [rsp + 96], xmm4",
        "movdqu [rsp + 112], xmm5",
        "movdqu [rsp + 128], xmm6",
        "movdqu [rsp + 144], xmm7",
        "movdqu [rsp + 160], xmm8",
        "movdqu [rsp + 176], xmm9",
        "movdqu [rsp + 192], xmm10",
        "movdqu [rsp + 208], xmm11",
        "movdqu [rsp + 224], xmm12",
        "movdqu [rsp + 240], xmm13",
        "movdqu [rsp + 256], xmm14",
        "movdqu [rsp + 272], xmm15",
        "call {hook}",
        "movdqu xmm0, [rsp + 32]",
        "movdqu xmm1, [rsp + 48]",
        "movdqu xmm2, [rsp + 64]",
        "movdqu xmm3, [rsp + 80]",
        "movdqu xmm4, [rsp + 96]",
        "movdqu xmm5, [rsp + 112]",
        "movdqu xmm6, [rsp + 128]",
        "movdqu xmm7, [rsp + 144]",
        "movdqu xmm8, [rsp + 160]",
        "movdqu xmm9, [rsp + 176]",
        "movdqu xmm10, [rsp + 192]",
        "movdqu xmm11, [rsp + 208]",
        "movdqu xmm12, [rsp + 224]",
        "movdqu xmm13, [rsp + 240]",
        "movdqu xmm14, [rsp + 256]",
        "movdqu xmm15, [rsp + 272]",
        "add rsp, 296",
        "pop r11",
        "pop r10",
        "pop r9",
        "pop r8",
        "pop rdi",
        "pop rsi",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "pop rbp",
        "ret",
        hook = sym rt_fuel_exhausted,
    );
}

/// AArch64 version of the `PreserveAll` stub: saves x0-x18 and the full q0-q31.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
extern "C" fn rt_fuel_exhausted_preserve_all() {
    core::arch::naked_asm!(
        "stp x29, x30, [sp, #-16]!",
        "mov x29, sp",
        "sub sp, sp, #672",
        "stp x0, x1, [sp, #0]",
        "stp x2, x3, [sp, #16]",
        "stp x4, x5, [sp, #32]",
        "stp x6, x7, [sp, #48]",
        "stp x8, x9, [sp, #64]",
        "stp x10, x11, [sp, #80]",
        "stp x12, x13, [sp, #96]",
        "stp x14, x15, [sp, #112]",
        "stp x16, x17, [sp, #128]",
        "str x18, [sp, #144]",
        "stp q0, q1, [sp, #160]",
        "stp q2, q3, [sp, #192]",
        "stp q4, q5, [sp, #224]",
        "stp q6, q7, [sp, #256]",
        "stp q8, q9, [sp, #288]",
        "stp q10, q11, [sp, #320]",
        "stp q12, q13, [sp, #352]",
        "stp q14, q15, [sp, #384]",
        "stp q16, q17, [sp, #416]",
        "stp q18, q19, [sp, #448]",
        "stp q20, q21, [sp, #480]",
        "stp q22, q23, [sp, #512]",
        "stp q24, q25, [sp, #544]",
        "stp q26, q27, [sp, #576]",
        "stp q28, q29, [sp, #608]",
        "stp q30, q31, [sp, #640]",
        "bl {hook}",
        "ldp q0, q1, [sp, #160]",
        "ldp q2, q3, [sp, #192]",
        "ldp q4, q5, [sp, #224]",
        "ldp q6, q7, [sp, #256]",
        "ldp q8, q9, [sp, #288]",
        "ldp q10, q11, [sp, #320]",
        "ldp q12, q13, [sp, #352]",
        "ldp q14, q15, [sp, #384]",
        "ldp q16, q17, [sp, #416]",
        "ldp q18, q19, [sp, #448]",
        "ldp q20, q21, [sp, #480]",
        "ldp q22, q23, [sp, #512]",
        "ldp q24, q25, [sp, #544]",
        "ldp q26, q27, [sp, #576]",
        "ldp q28, q29, [sp, #608]",
        "ldp q30, q31, [sp, #640]",
        "ldp x0, x1, [sp, #0]",
        "ldp x2, x3, [sp, #16]",
        "ldp x4, x5, [sp, #32]",
        "ldp x6, x7, [sp, #48]",
        "ldp x8, x9, [sp, #64]",
        "ldp x10, x11, [sp, #80]",
        "ldp x12, x13, [sp, #96]",
        "ldp x14, x15, [sp, #112]",
        "ldp x16, x17, [sp, #128]",
        "ldr x18, [sp, #144]",
        "add sp, sp, #672",
        "ldp x29, x30, [sp], #16",
        "ret",
        hook = sym rt_fuel_exhausted,
    );
}

/// `rt_fuel_exhausted` for LLVM code, which may keep values in any register (`ymm`/`zmm`,
/// AVX-512 mask registers): saves the caller-saved general-purpose registers and, with
/// `xsave`, the x87, SSE, AVX and AVX-512 state (not AMX tiles, which no fuel-checked code
/// holds) in a 64-byte aligned area, so it may be entered at any stack alignment. Like the
/// Cranelift stub it returns nothing and the caller reloads the fuel counter.
#[cfg(all(target_arch = "x86_64", feature = "llvm"))]
#[unsafe(naked)]
extern "C" fn rt_fuel_exhausted_xsave() {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "push rax",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push r8",
        "push r9",
        "push r10",
        "push r11",
        // x87, SSE, AVX, opmask, ZMM_Hi256 and Hi16_ZMM end at byte 2688 of the standard
        // layout. XSAVE writes only the XSTATE_BV bits it saves, and XRSTOR faults unless the
        // rest of the 64-byte header is zero.
        "sub rsp, 2688",
        "and rsp, -64",
        "xor eax, eax",
        "mov [rsp + 512], rax",
        "mov [rsp + 520], rax",
        "mov [rsp + 528], rax",
        "mov [rsp + 536], rax",
        "mov [rsp + 544], rax",
        "mov [rsp + 552], rax",
        "mov [rsp + 560], rax",
        "mov [rsp + 568], rax",
        "mov eax, 0xe7",
        "xor edx, edx",
        "xsave [rsp]",
        "sub rsp, 32",
        "call {hook}",
        "add rsp, 32",
        "mov eax, 0xe7",
        "xor edx, edx",
        "xrstor [rsp]",
        "lea rsp, [rbp - 72]",
        "pop r11",
        "pop r10",
        "pop r9",
        "pop r8",
        "pop rdi",
        "pop rsi",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "pop rbp",
        "ret",
        hook = sym rt_fuel_exhausted,
    );
}

/// Stub the LLVM backend's fuel slow path calls with every register preserved
/// (`LowerOptions::fuel_stub`), or 0 to call `rt_fuel_exhausted` directly: the `xsave` stub
/// on x86-64. AArch64 calls the hook (SVE code would need z and p registers saved too).
#[cfg(feature = "llvm")]
fn llvm_fuel_stub() -> u64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("xsave") {
        return rt_fuel_exhausted_xsave as *const () as u64;
    }
    0
}

/// Fuel counter of the code running on this thread, or null when no engine call is active
/// (code reached through `get_fn_ptr`, which then uses its engine's counter). Called on
/// entry by functions of modules that use `par`, whose workers each count their own fuel.
extern "C" fn rt_fuel_counter() -> *const Cell<i64> {
    ACTIVE_FUEL.with(|a| a.get())
}

/// Fuel a `par` worker takes from the shared budget at a time. Bounds how long a worker
/// keeps running after another one fails.
const PAR_FUEL_CHUNK: i64 = 1 << 16;

/// Stack size of threads that run AIR code: `SANDBOX_STACK_BYTES` of recursion budget plus
/// headroom for the runtime and host frames.
pub const EXECUTION_STACK_BYTES: usize = 8 << 20;

/// A value on cache lines of its own, so writes to it do not slow down threads reading its
/// neighbours (128 bytes: x86 prefetches lines in pairs).
#[repr(align(128))]
struct CachePadded<T>(T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Body arguments at most this many slots long are copied to the stack of each worker.
const PAR_STACK_ARGS: usize = 32;

/// State shared by the threads running one `par`, on the stack of its caller. Helpers reach
/// it only after joining through `ParPool::joined`, and the caller returns only once every
/// helper that joined has left.
struct ParRegion {
    /// Scalar trampoline of the body, `void(u64 *args, u64 *ret)`.
    tramp: extern "C" fn(*const u64, *mut u64),
    /// Body arguments (`slots` values, the caller's; slot 0 is replaced by the index).
    args: *const u64,
    slots: usize,
    count: i64,
    /// Set by the first failure; workers stop taking indices and fuel.
    abort: AtomicBool,
    failure: Mutex<Option<ExecutionStatus>>,
    arena: *const Mutex<Arena>,
    sandboxed: bool,
    quota: usize,
    /// Next index to run. Written by every worker, so it does not share a cache line with
    /// the fields above, which every worker reads.
    next: CachePadded<AtomicI64>,
    /// Fuel budget left for workers to draw from (`UNLIMITED_FUEL` when there is none).
    fuel: CachePadded<AtomicI64>,
    /// No budget to share (`fuel` cannot run out): workers still refill their counters a
    /// chunk at a time, so they notice `abort`, but leave `fuel` alone.
    unlimited: bool,
    /// Heap accounting for non-sandboxed code under a quota (see `rt_malloc`).
    allocated: CachePadded<AtomicUsize>,
}

// `args` and `arena` are the caller's, alive while the caller waits in `rt_par_for`, which
// outlasts every worker.
unsafe impl Send for ParRegion {}
unsafe impl Sync for ParRegion {}

impl ParRegion {
    /// Takes up to `want` fuel from the shared budget; 0 once it is spent or aborted.
    fn draw(&self, want: i64) -> i64 {
        if want <= 0 || self.abort.load(Ordering::Relaxed) {
            return 0;
        }
        if self.unlimited {
            return want;
        }
        let mut left = self.fuel.load(Ordering::Acquire);
        loop {
            let got = left.min(want).max(0);
            if got == 0 {
                return 0;
            }
            match self.fuel.compare_exchange_weak(
                left,
                left - got,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return got,
                Err(now) => left = now,
            }
        }
    }

    /// Records the first failure and stops every worker.
    fn fail(&self, status: ExecutionStatus) {
        let mut failure = self.failure.lock().unwrap_or_else(|e| e.into_inner());
        if failure.is_none() {
            *failure = Some(status);
        }
        self.abort.store(true, Ordering::Release);
    }

    /// Runs indices until they run out or a worker fails, and returns whether it ran any.
    /// Unused fuel goes back to the shared budget before it returns.
    fn participate(&self, is_caller: bool) -> bool {
        let counter = Cell::new(0);
        let prev_fuel = ACTIVE_FUEL.with(|a| a.replace(&counter));
        let prev_region = PAR_REGION.with(|r| r.replace(self));
        let prev_arena = ACTIVE_ARENA.with(|a| a.replace(self.arena));
        // The caller keeps the stack limit of its call; helpers start a fresh budget.
        let prev_limit = (!is_caller).then(|| {
            let limit = if self.sandboxed {
                stack_address().saturating_sub(SANDBOX_STACK_BYTES)
            } else {
                0
            };
            STACK_LIMIT.with(|l| l.replace(limit))
        });
        reset_execution_status();

        // This thread's copy of the arguments, on the stack unless they are long.
        let mut stack_args = [0u64; PAR_STACK_ARGS];
        let mut heap_args = Vec::new();
        // SAFETY: the caller's `slots` values, alive while it waits for the workers.
        let src = unsafe { std::slice::from_raw_parts(self.args, self.slots) };
        let args: &mut [u64] = if self.slots <= PAR_STACK_ARGS {
            &mut stack_args[..self.slots]
        } else {
            heap_args.extend_from_slice(src);
            &mut heap_args
        };
        args.copy_from_slice(src);
        let mut ret = 0u64;
        let mut worked = false;
        while !self.abort.load(Ordering::Relaxed) {
            // Only claims the index: results reach the caller through `ParPool::joined`.
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.count {
                break;
            }
            worked = true;
            // Each index costs one unit, so even bodies without branches are bounded.
            if !charge(&counter, 1) {
                raise(ExecutionStatus::OutOfFuel);
            } else {
                args[0] = i as u64;
                (self.tramp)(args.as_ptr(), &mut ret);
            }
            if halted() {
                self.fail(get_execution_status());
                break;
            }
        }

        if !self.unlimited {
            self.fuel.fetch_add(counter.get().max(0), Ordering::AcqRel);
        }
        reset_execution_status();
        ACTIVE_FUEL.with(|a| a.set(prev_fuel));
        PAR_REGION.with(|r| r.set(prev_region));
        ACTIVE_ARENA.with(|a| a.set(prev_arena));
        if let Some(limit) = prev_limit {
            STACK_LIMIT.with(|l| l.set(limit));
        }
        worked
    }
}

fn active_region() -> Option<&'static ParRegion> {
    let region = PAR_REGION.with(|r| r.get());
    // SAFETY: set only by `participate` while the region is alive.
    (!region.is_null()).then(|| unsafe { &*region })
}

/// True on a thread running `par` iterations. Such a thread must not block waiting for an
/// engine: the call that started the `par` may hold it.
pub fn in_par_worker() -> bool {
    active_region().is_some()
}

/// How long a helper that just ran iterations spins before parking, so back-to-back `par`
/// loops do not pay a thread wake-up each (tens to hundreds of microseconds on some hosts).
/// Kept short because spinning helpers take cores from other runtimes' threads: alternating
/// with NumPy's OpenBLAS pool cost 8.4 ms per pair of calls with a 2 ms spin and 4.4 ms
/// with 200 us (2.7 ms run apart).
const PAR_SPIN: std::time::Duration = std::time::Duration::from_micros(200);

/// `ParPool::announce`: the `par` number above these bits, the helpers it uses in them.
const ANNOUNCE_USED_BITS: u32 = 16;
/// `ParPool::joined`: the `par` number above these bits, then the closed bit, then the
/// number of helpers in the region.
const JOINED_COUNT_BITS: u32 = 16;
const JOINED_CLOSED: u64 = 1 << JOINED_COUNT_BITS;
const JOINED_COUNT_MASK: u64 = JOINED_CLOSED - 1;
const JOINED_EPOCH_SHIFT: u32 = JOINED_COUNT_BITS + 1;

/// Helper threads that run `par` iterations next to the calling thread. A `par` publishes
/// its region and bumps `announce`; each helper it uses joins by incrementing the count in
/// `joined` while the region is open, runs indices, and leaves by decrementing it. The
/// caller closes the region once it has run out of indices and waits for the count to
/// reach zero, so a helper that arrives late (still waking up) finds it closed, never
/// touches it, and costs the caller nothing. No locks or allocations on this path.
struct ParPool {
    helpers: Vec<Helper>,
    /// The current `par`'s number and helper count (see `ANNOUNCE_USED_BITS`); helpers
    /// spin on it.
    announce: CachePadded<AtomicU64>,
    /// The current `par`'s number, closed bit and joined helpers (see `JOINED_COUNT_BITS`).
    joined: CachePadded<AtomicU64>,
    /// The current `par`'s region, valid for a helper that has joined it.
    region: AtomicPtr<ParRegion>,
    /// The caller is parked waiting for helpers to leave; the last one to leave wakes it.
    waiting: AtomicBool,
    waiter: Mutex<Option<std::thread::Thread>>,
    /// Held by the thread running a `par` on the pool; others run theirs serially.
    busy: Mutex<()>,
}

struct Helper {
    thread: std::thread::Thread,
    parked: AtomicBool,
}

/// Number of threads `par` runs on: `ACHAINSAW_THREADS`, or one per core.
pub fn par_pool_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("ACHAINSAW_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            // Helper counts fit the bit fields of `ParPool::announce` and `joined`.
            .min(JOINED_COUNT_MASK as usize)
    })
}

/// Process-wide `par` pool: the calling thread plus `par_pool_threads() - 1` helpers,
/// started on first use.
fn par_pool() -> &'static ParPool {
    static POOL: OnceLock<ParPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let helpers = (1..par_pool_threads())
            .map(|index| {
                let handle = std::thread::Builder::new()
                    .name(format!("achainsaw-par-{index}"))
                    .stack_size(EXECUTION_STACK_BYTES)
                    .spawn(move || helper_loop(index - 1))
                    .expect("cannot start a par helper thread");
                Helper {
                    thread: handle.thread().clone(),
                    parked: AtomicBool::new(false),
                }
            })
            .collect();
        ParPool {
            helpers,
            announce: CachePadded(AtomicU64::new(0)),
            joined: CachePadded(AtomicU64::new(JOINED_CLOSED)),
            region: AtomicPtr::new(std::ptr::null_mut()),
            waiting: AtomicBool::new(false),
            waiter: Mutex::new(None),
            busy: Mutex::new(()),
        }
    })
}

impl ParPool {
    /// Joins `par` number `epoch` and runs its indices, unless its caller has closed it.
    fn join(&self, epoch: u64) {
        let seen = self.joined.fetch_add(1, Ordering::Acquire);
        if seen >> JOINED_EPOCH_SHIFT != epoch || seen & JOINED_CLOSED != 0 {
            // Closed, or already another `par`: take back the increment, whose caller waits
            // for it like for a helper leaving.
            self.leave();
            return;
        }
        // SAFETY: published before `announce`, and alive until this helper leaves.
        let region = unsafe { &*self.region.load(Ordering::Acquire) };
        region.participate(false);
        self.leave();
    }

    /// Leaves the current region (the caller may return as soon as this runs), waking the
    /// caller if it is parked.
    fn leave(&self) {
        self.joined.fetch_sub(1, Ordering::SeqCst);
        // Pairs with the caller's `waiting` store then count load.
        if self.waiting.load(Ordering::SeqCst) {
            if let Some(caller) = &*self.waiter.lock().unwrap_or_else(|e| e.into_inner()) {
                caller.unpark();
            }
        }
    }

    /// Closes the current region to new helpers and waits until those inside have left.
    fn close(&self) {
        self.joined.fetch_or(JOINED_CLOSED, Ordering::AcqRel);
        let inside = || self.joined.load(Ordering::SeqCst) & JOINED_COUNT_MASK != 0;
        let mut spins = 0u32;
        while inside() {
            if spins < 1 << 14 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            *self.waiter.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::thread::current());
            self.waiting.store(true, Ordering::SeqCst);
            while inside() {
                std::thread::park();
            }
            self.waiting.store(false, Ordering::SeqCst);
        }
    }

    /// Opens the region of `par` number `epoch`. Waits out late helpers still taking back
    /// their increment of an earlier `par`, so none is lost.
    fn open(&self, epoch: u64) {
        let open = epoch << JOINED_EPOCH_SHIFT;
        let mut current = self.joined.load(Ordering::Relaxed);
        loop {
            if current & JOINED_COUNT_MASK != 0 {
                std::hint::spin_loop();
                current = self.joined.load(Ordering::Relaxed);
                continue;
            }
            match self.joined.compare_exchange_weak(
                current,
                open,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(now) => current = now,
            }
        }
    }
}

/// Waits for each new `par` (spinning, then parked) and joins it unless it is capped below
/// this helper's index.
fn helper_loop(index: usize) {
    let pool = par_pool();
    let me = &pool.helpers[index];
    let epoch_of = |announce: u64| announce >> ANNOUNCE_USED_BITS;
    let mut seen = 0u64;
    let mut spin = PAR_SPIN;
    loop {
        let idle_since = std::time::Instant::now();
        let mut spins = 0u32;
        while epoch_of(pool.announce.load(Ordering::SeqCst)) == seen {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) && idle_since.elapsed() >= spin {
                // Pairs with the dispatcher's `announce` store then `parked` load.
                me.parked.store(true, Ordering::SeqCst);
                if epoch_of(pool.announce.load(Ordering::SeqCst)) == seen {
                    std::thread::park();
                }
                me.parked.store(false, Ordering::SeqCst);
            } else {
                std::hint::spin_loop();
            }
        }
        let announce = pool.announce.load(Ordering::SeqCst);
        seen = epoch_of(announce);
        // Helpers a `par` does not use (it has fewer tasks than threads, or a lower thread
        // cap) park at once, so they neither slow down busy threads sharing their core nor
        // need waking for loops of the same size; the others wait for the next `par`.
        let used = index < (announce & ((1 << ANNOUNCE_USED_BITS) - 1)) as usize;
        if used {
            pool.join(seen);
        }
        spin = if used {
            PAR_SPIN
        } else {
            std::time::Duration::ZERO
        };
    }
}

/// Runs `par n, f(args)`: calls the trampoline `tramp` of `f` with `[i, args...]` (`slots`
/// u64 values at `args`, slot 0 reserved for `i`) for every `i` in `[0, n)`, on this thread
/// and the pool's helpers, and returns once all calls finish. The caller's fuel budget is
/// shared by all of them. Returns 1 after recording the first failure on the calling
/// thread, 0 otherwise. Runs serially inside another `par` (nested), when limited to one
/// thread, or while another thread's `par` has the pool.
extern "C" fn rt_par_for(tramp: usize, args: *const u64, slots: i64, n: i64) -> i32 {
    if n <= 0 || halted() {
        return halted() as i32;
    }
    // SAFETY: the JIT passes a trampoline address and `slots` initialized u64 values.
    let tramp: extern "C" fn(*const u64, *mut u64) = unsafe { std::mem::transmute(tramp) };
    let slots = slots as usize;
    let threads = match PAR_THREADS.with(|t| t.get()) {
        0 => par_pool_threads(),
        cap => cap.min(par_pool_threads()),
    };
    let pool = (active_region().is_none() && threads > 1 && n > 1).then(par_pool);
    let busy = pool.and_then(|p| p.busy.try_lock().ok());

    let (Some(pool), Some(_busy)) = (pool, busy) else {
        // SAFETY: as above.
        let mut args = unsafe { std::slice::from_raw_parts(args, slots) }.to_vec();
        let mut ret = 0u64;
        for i in 0..n {
            let ok = with_active_fuel(|counter| charge(counter, 1)).unwrap_or(true);
            if !ok {
                raise(ExecutionStatus::OutOfFuel);
                return 1;
            }
            args[0] = i as u64;
            tramp(args.as_ptr(), &mut ret);
            if halted() {
                return 1;
            }
        }
        return 0;
    };

    let caller_fuel = ACTIVE_FUEL.with(|a| a.get());
    // SAFETY: set by `call_typed` for the duration of the call that reached this hook.
    let caller_fuel = (!caller_fuel.is_null()).then(|| unsafe { &*caller_fuel });
    let fuel = caller_fuel.map_or(UNLIMITED_FUEL, |c| c.get());
    let region = ParRegion {
        tramp,
        args,
        slots,
        count: n,
        abort: AtomicBool::new(false),
        failure: Mutex::new(None),
        arena: ACTIVE_ARENA.with(|a| a.get()),
        sandboxed: STACK_LIMIT.with(|l| l.get()) != 0,
        quota: MEMORY_QUOTA.with(|q| q.get()),
        next: CachePadded(AtomicI64::new(0)),
        fuel: CachePadded(AtomicI64::new(fuel)),
        // A limited budget is at most `i64::MAX` units, as is `UNLIMITED_FUEL`, but code
        // that ran through half of that is not finishing anyway.
        unlimited: fuel > UNLIMITED_FUEL / 2,
        allocated: CachePadded(AtomicUsize::new(MEMORY_ALLOCATED.with(|m| m.get()))),
    };
    // At most one thread per index.
    let used = threads.min(n as usize) - 1;
    let epoch = (pool.announce.load(Ordering::Relaxed) >> ANNOUNCE_USED_BITS) + 1;
    pool.region.store(
        &region as *const ParRegion as *mut ParRegion,
        Ordering::Release,
    );
    pool.open(epoch);
    pool.announce.store(
        (epoch << ANNOUNCE_USED_BITS) | used as u64,
        Ordering::SeqCst,
    );
    for helper in &pool.helpers[..used] {
        if helper.parked.load(Ordering::SeqCst) {
            helper.thread.unpark();
        }
    }

    let _ = region.participate(true);
    pool.close();

    if let Some(counter) = caller_fuel {
        counter.set(region.fuel.load(Ordering::Acquire));
    }
    MEMORY_ALLOCATED.with(|m| m.set(region.allocated.load(Ordering::Acquire)));
    let failure = *region.failure.lock().unwrap_or_else(|e| e.into_inner());
    match failure {
        Some(status) => {
            raise(status);
            1
        }
        None => 0,
    }
}

/// Adds `size` bytes to the heap accounting of this thread, or of its `par` region;
/// false (nothing added) when that would exceed a non-zero `quota`.
fn reserve_allocated(size: usize, quota: usize) -> bool {
    let fits = |cur: usize| quota == 0 || cur.saturating_add(size) <= quota;
    if let Some(region) = active_region() {
        return update_usize(&region.allocated, |cur| fits(cur).then(|| cur + size));
    }
    MEMORY_ALLOCATED.with(|m| {
        let cur = m.get();
        let ok = fits(cur);
        if ok {
            m.set(cur + size);
        }
        ok
    })
}

fn release_allocated(size: usize) {
    if let Some(region) = active_region() {
        update_usize(&region.allocated, |cur| Some(cur.saturating_sub(size)));
        return;
    }
    MEMORY_ALLOCATED.with(|m| m.set(m.get().saturating_sub(size)));
}

/// Replaces `a` with `f(a)` atomically; false (and no change) when `f` returns `None`.
fn update_usize(a: &AtomicUsize, f: impl Fn(usize) -> Option<usize>) -> bool {
    let mut cur = a.load(Ordering::Acquire);
    loop {
        let Some(new) = f(cur) else {
            return false;
        };
        match a.compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => cur = now,
        }
    }
}

fn memory_quota() -> usize {
    active_region().map_or_else(|| MEMORY_QUOTA.with(|q| q.get()), |r| r.quota)
}

unsafe extern "C" fn rt_malloc(size: usize) -> *mut u8 {
    if let Some(ptr) = with_active_arena(|arena| {
        let mut arena = lock(arena);
        arena.alloc(size).unwrap_or_else(|| {
            raise(ExecutionStatus::OutOfMemory {
                requested: size,
                limit: arena.len(),
            });
            std::ptr::null_mut()
        })
    }) {
        if !ptr.is_null() {
            reserve_allocated(size, 0);
        }
        return ptr;
    }

    let quota = memory_quota();
    if !reserve_allocated(size, quota) {
        raise(ExecutionStatus::OutOfMemory {
            requested: size,
            limit: quota,
        });
        return std::ptr::null_mut();
    }

    let total = size + 16;
    let raw = malloc(total);
    if raw.is_null() {
        release_allocated(size);
        raise(ExecutionStatus::OutOfMemory {
            requested: size,
            limit: quota,
        });
        return std::ptr::null_mut();
    }

    *(raw as *mut usize) = size;
    raw.add(16)
}

unsafe extern "C" fn rt_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    if let Some(freed) = with_active_arena(|arena| lock(arena).free(ptr)) {
        match freed {
            Some(size) => release_allocated(size),
            None => raise(ExecutionStatus::MemoryViolation {
                addr: ptr as usize,
                size: 0,
            }),
        }
        return;
    }
    let raw = ptr.sub(16);
    let size = *(raw as *const usize);
    release_allocated(size);
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

/// Loads the thread's fuel budget (`set_execution_fuel`) into an engine's inline counter for
/// one call and makes it the active counter; on drop, writes the remaining budget back and
/// restores the previous counter (calls may nest through host callbacks).
struct FuelActivation<'a> {
    counter: &'a Cell<i64>,
    limited: bool,
    prev: *const Cell<i64>,
}

impl<'a> FuelActivation<'a> {
    fn new(counter: &'a Cell<i64>) -> Self {
        let budget = FUEL_REMAINING.with(|f| f.get());
        let limited = budget >= 0;
        counter.set(if limited { budget } else { UNLIMITED_FUEL });
        let prev = ACTIVE_FUEL.with(|a| a.replace(counter));
        Self {
            counter,
            limited,
            prev,
        }
    }
}

impl Drop for FuelActivation<'_> {
    fn drop(&mut self) {
        if self.limited {
            let left = self.counter.get().max(0);
            FUEL_REMAINING.with(|f| f.set(left));
        }
        ACTIVE_FUEL.with(|a| a.set(self.prev));
    }
}

/// Points the runtime hooks at a sandboxed engine's arena and stack limit for one call,
/// restoring the previous state when dropped (calls may nest through host callbacks).
struct SandboxActivation {
    prev_arena: *const Mutex<Arena>,
    prev_limit: usize,
}

impl SandboxActivation {
    fn new(arena: &Mutex<Arena>) -> Self {
        lock(arena).reset();
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

/// Caps the threads `par` may use for one call, restoring the previous cap when dropped.
struct ThreadsActivation(usize);

impl ThreadsActivation {
    fn new(threads: usize) -> Self {
        Self(PAR_THREADS.with(|t| t.replace(threads)))
    }
}

impl Drop for ThreadsActivation {
    fn drop(&mut self) {
        PAR_THREADS.with(|t| t.set(self.0));
    }
}

/// Cranelift JIT module plus the ids of the runtime hooks it imports.
struct CraneliftJit {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: JITModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
    rt_fuel_exhausted_id: FuncId,
    rt_consume_fuel_id: FuncId,
    rt_sandbox_fault_id: FuncId,
    rt_stack_check_id: FuncId,
    rt_sandbox_check_mm_id: FuncId,
    rt_par_for_id: FuncId,
    rt_fuel_counter_id: FuncId,
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
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        jit_builder.symbol(
            "rt_fuel_exhausted",
            rt_fuel_exhausted_preserve_all as *const u8,
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        jit_builder.symbol("rt_fuel_exhausted", rt_fuel_exhausted as *const u8);
        jit_builder.symbol("rt_consume_fuel", rt_consume_fuel as *const u8);
        jit_builder.symbol("rt_sandbox_fault", rt_sandbox_fault as *const u8);
        jit_builder.symbol("rt_stack_check", rt_stack_check as *const u8);
        jit_builder.symbol("rt_sandbox_check_mm", rt_sandbox_check_mm as *const u8);
        jit_builder.symbol("rt_par_for", rt_par_for as *const u8);
        jit_builder.symbol("rt_fuel_counter", rt_fuel_counter as *const u8);
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

        // No result: the caller reads the fuel counter back (see `emit_fuel_check`).
        let mut exhausted_sig = module.make_signature();
        if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
            exhausted_sig.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
        }
        let rt_fuel_exhausted_id =
            module.declare_function("rt_fuel_exhausted", Linkage::Import, &exhausted_sig)?;

        let mut fuel_sig = module.make_signature();
        fuel_sig.returns.push(AbiParam::new(types::I32));

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

        let mut par_sig = module.make_signature();
        for _ in 0..4 {
            par_sig.params.push(AbiParam::new(types::I64));
        }
        par_sig.returns.push(AbiParam::new(types::I32));
        let rt_par_for_id = module.declare_function("rt_par_for", Linkage::Import, &par_sig)?;

        let mut counter_sig = module.make_signature();
        counter_sig.returns.push(AbiParam::new(types::I64));
        let rt_fuel_counter_id =
            module.declare_function("rt_fuel_counter", Linkage::Import, &counter_sig)?;

        let ctx = module.make_context();

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            module,
            rt_malloc_id,
            rt_free_id,
            rt_fuel_exhausted_id,
            rt_consume_fuel_id,
            rt_sandbox_fault_id,
            rt_stack_check_id,
            rt_sandbox_check_mm_id,
            rt_par_for_id,
            rt_fuel_counter_id,
        })
    }

    /// Compiles `ir_mod` and returns `(name, function, trampoline)` addresses.
    fn compile(
        &mut self,
        ir_mod: &Module,
        fuel_enabled: bool,
        fuel_counter: i64,
        sandbox: Option<(i64, i64)>,
        fast_math: bool,
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
            func_returns.insert(ext_fn.name.clone(), ext_fn.ret_type.into_iter().collect());

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
            for r_ty in &func.rets {
                push_abi_params(&mut sig.returns, *r_ty);
            }
            func_returns.insert(func.name.clone(), func.rets.clone());

            let func_id = self
                .module
                .declare_function(&func.name, Linkage::Export, &sig)?;
            func_ids.insert(func.name.clone(), func_id);
        }

        // Host calls (and `par`) go through a scalar trampoline; functions that take or
        // return vectors are only callable from AIR code. Declared up front so `par` can
        // take their addresses.
        let mut tramp_ids = HashMap::new();
        for func in &ir_mod.functions {
            if !func.has_trampoline() {
                continue;
            }
            let tid = crate::lower::declare_trampoline(&mut self.module, &func.name)?;
            tramp_ids.insert(func.name.clone(), tid);
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
            fuel_check: (fuel_enabled || sandbox.is_some())
                .then_some((fuel_counter, self.rt_fuel_exhausted_id)),
            fuel_consume_func_id: fuel_enabled.then_some(self.rt_consume_fuel_id),
            rt_malloc_id: self.rt_malloc_id,
            rt_free_id: self.rt_free_id,
            sandbox,
            par: Some(ParConfig {
                par_for_id: self.rt_par_for_id,
                trampolines: tramp_ids.clone(),
                // `par` workers count fuel in their own counters.
                fuel_counter_id: ir_mod.uses_par().then_some(self.rt_fuel_counter_id),
            }),
            fast_math,
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

            if tramp_ids.contains_key(&func.name) {
                let param_tys: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
                crate::lower::lower_trampoline(
                    &mut self.module,
                    &mut self.ctx,
                    &mut self.builder_context,
                    &func.name,
                    func_id,
                    &param_tys,
                    func.single_ret().ok().flatten(),
                )?;
            }
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
        matrix: achainsaw_llvm::MatrixUnits,
        native_f16: bool,
        int8_dot: achainsaw_llvm::Int8Dot,
        vx_bits: u32,
        /// One LLJIT per `compile_module` call; kept alive so earlier code stays valid.
        modules: Vec<achainsaw_llvm::LlvmJit>,
    },
}

pub struct JitEngine {
    backend: Backend,
    codegen: Codegen,
    /// Set by `enable_sandbox`; boxed so its address stays fixed while code runs.
    sandbox: Option<Box<Mutex<Arena>>>,
    pub registry: Arc<RwLock<SymbolRegistry>>,
    pub fuel_enabled: bool,
    /// Fuel counter JIT code decrements inline at every branch; boxed so its address stays
    /// fixed. `call_typed` loads it from the thread's budget for each call.
    fuel: Box<Cell<i64>>,
    /// Most threads `par` may use (0: the whole pool).
    threads: usize,
    /// See `CodegenOptions::fast_math`.
    fast_math: bool,
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
                let matrix = features.llvm_matrix_units();
                let native_f16 = features.llvm_native_f16();
                let int8_dot = features.llvm_int8_dot();
                let (cpu, features) = features.llvm_target();
                Codegen::Llvm {
                    target: achainsaw_llvm::TargetSpec {
                        triple: None,
                        cpu,
                        features,
                    },
                    vx,
                    vector_width,
                    matrix,
                    native_f16,
                    int8_dot,
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
            fuel: Box::new(Cell::new(UNLIMITED_FUEL)),
            threads: 0,
            fast_math: false,
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
        // Also applies to code called directly through `get_fn_ptr`.
        self.fuel
            .set(fuel.map_or(UNLIMITED_FUEL, |f| f.min(i64::MAX as u64) as i64));
    }

    /// Limits `par` to `threads` threads (at most the pool size, see [`par_pool_threads`]);
    /// `None` uses the whole pool. `Some(1)` runs `par` loops serially.
    pub fn set_threads(&mut self, threads: Option<usize>) {
        self.threads = threads.unwrap_or(0);
    }

    /// Threads `par` runs on for this engine.
    pub fn threads(&self) -> usize {
        match self.threads {
            0 => par_pool_threads(),
            n => n.min(par_pool_threads()),
        }
    }

    /// Generates float min/max as compare and select (see `CodegenOptions::fast_math`) in
    /// modules compiled from now on.
    pub fn set_fast_math(&mut self, on: bool) {
        self.fast_math = on;
    }

    pub fn fast_math(&self) -> bool {
        self.fast_math
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
        self.sandbox = Some(Box::new(Mutex::new(Arena::new(arena_bytes)?)));
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
        // `inline fn` calls are expanded before lowering, for both backends.
        let inlined = achainsaw_ir::inline::inline_module(ir_mod);
        let ir_mod = &*inlined;
        let sandbox = self.sandbox.as_ref().map(|arena| {
            let arena = lock(arena);
            (arena.base as i64, arena.len() as i64)
        });
        let fuel_counter = self.fuel.as_ptr() as i64;
        let compiled = match &mut self.codegen {
            Codegen::Cranelift(clif) => clif.compile(
                ir_mod,
                self.fuel_enabled,
                fuel_counter,
                sandbox,
                self.fast_math,
            )?,
            #[cfg(feature = "llvm")]
            Codegen::Llvm {
                target,
                vx,
                vector_width,
                matrix,
                native_f16,
                int8_dot,
                modules,
                ..
            } => {
                let hooks = achainsaw_llvm::RuntimeHooks {
                    malloc: rt_malloc as *const () as usize,
                    free: rt_free as *const () as usize,
                    fuel_exhausted: rt_fuel_exhausted as *const () as usize,
                    consume_fuel: rt_consume_fuel as *const () as usize,
                    sandbox_fault: rt_sandbox_fault as *const () as usize,
                    stack_check: rt_stack_check as *const () as usize,
                    sandbox_check_mm: rt_sandbox_check_mm as *const () as usize,
                    par_for: rt_par_for as *const () as usize,
                    fuel_counter: rt_fuel_counter as *const () as usize,
                };
                let opts = achainsaw_llvm::LowerOptions {
                    fuel: self.fuel_enabled,
                    fuel_counter: fuel_counter as u64,
                    fuel_stub: llvm_fuel_stub(),
                    sandbox: sandbox.map(|(base, len)| achainsaw_llvm::SandboxBounds {
                        base: base as u64,
                        len: len as u64,
                    }),
                    vx: *vx,
                    vector_width: *vector_width,
                    matrix: *matrix,
                    native_f16: *native_f16,
                    int8_dot: *int8_dot,
                    fast_math: self.fast_math,
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
            self.signatures.insert(
                func.name.clone(),
                (param_tys, func.single_ret().ok().flatten()),
            );
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

        let _fuel = FuelActivation::new(&self.fuel);
        let _sandbox = self.sandbox.as_deref().map(SandboxActivation::new);
        let _threads = ThreadsActivation::new(self.threads);
        let tramp_fn: extern "C" fn(*const u64, *mut u64) = std::mem::transmute(tramp_ptr);
        tramp_fn(raw_args.as_ptr(), &mut raw_ret);

        check_execution_status()?;

        if let Some(r_ty) = *ret_type {
            Ok(Some(RtValue::from_u64(raw_ret, r_ty)))
        } else {
            Ok(None)
        }
    }

    // Typed convenience wrappers around `call_typed`
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
