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
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
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

/// State shared by the threads running one `par`.
struct ParRegion {
    /// Scalar trampoline of the body, `void(u64 *args, u64 *ret)`.
    tramp: extern "C" fn(*const u64, *mut u64),
    /// Body arguments; slot 0 is replaced by the index.
    args: Vec<u64>,
    count: i64,
    next: AtomicI64,
    /// Helper threads that may join (the calling thread always works too).
    helpers: usize,
    /// Threads still between joining and leaving; the caller returns once it is 0.
    inflight: AtomicUsize,
    caller: std::thread::Thread,
    /// Fuel budget left for workers to draw from (`UNLIMITED_FUEL` when there is none).
    fuel: AtomicI64,
    /// Set by the first failure; workers stop taking indices and fuel.
    abort: AtomicBool,
    failure: Mutex<Option<ExecutionStatus>>,
    arena: *const Mutex<Arena>,
    sandboxed: bool,
    /// Heap accounting for non-sandboxed code under a quota (see `rt_malloc`).
    allocated: AtomicUsize,
    quota: usize,
}

// `arena` is the caller's, alive while `inflight` is non-zero; threads that join later
// find no index left and never use it.
unsafe impl Send for ParRegion {}
unsafe impl Sync for ParRegion {}

impl ParRegion {
    /// Takes up to `want` fuel from the shared budget; 0 once it is spent or aborted.
    fn draw(&self, want: i64) -> i64 {
        if want <= 0 || self.abort.load(Ordering::Relaxed) {
            return 0;
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

    /// Runs indices until they run out or a worker fails. A thread counts as in flight from
    /// before its first index to after it has returned its unused fuel, so the caller, which
    /// waits for none to be in flight, sees every result.
    fn participate(&self, is_caller: bool) {
        self.inflight.fetch_add(1, Ordering::SeqCst);
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

        let mut args = self.args.clone();
        let mut ret = 0u64;
        while !self.abort.load(Ordering::Relaxed) {
            let i = self.next.fetch_add(1, Ordering::SeqCst);
            if i >= self.count {
                break;
            }
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

        // Unused fuel goes back to the budget the caller resumes with.
        self.fuel.fetch_add(counter.get().max(0), Ordering::AcqRel);
        reset_execution_status();
        ACTIVE_FUEL.with(|a| a.set(prev_fuel));
        PAR_REGION.with(|r| r.set(prev_region));
        ACTIVE_ARENA.with(|a| a.set(prev_arena));
        if let Some(limit) = prev_limit {
            STACK_LIMIT.with(|l| l.set(limit));
        }
        if self.inflight.fetch_sub(1, Ordering::SeqCst) == 1 && !is_caller {
            self.caller.unpark();
        }
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

/// How long an idle helper spins before parking, so back-to-back `par` loops do not pay a
/// thread wake-up each (tens to hundreds of microseconds on some hosts).
const PAR_SPIN: std::time::Duration = std::time::Duration::from_millis(2);

/// Helper threads that run `par` iterations next to the calling thread.
struct ParPool {
    helpers: Vec<Helper>,
    /// Bumped for each `par`, together with `job`.
    epoch: AtomicU64,
    job: Mutex<(u64, Option<Arc<ParRegion>>)>,
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
            epoch: AtomicU64::new(0),
            job: Mutex::new((0, None)),
            busy: Mutex::new(()),
        }
    })
}

/// Waits for each new `par` (spinning, then parked) and joins it unless it is capped below
/// this helper's index.
fn helper_loop(index: usize) {
    let pool = par_pool();
    let me = &pool.helpers[index];
    let mut seen = 0u64;
    loop {
        let idle_since = std::time::Instant::now();
        let mut spins = 0u32;
        while pool.epoch.load(Ordering::SeqCst) == seen {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) && idle_since.elapsed() > PAR_SPIN {
                // Pairs with the dispatcher's epoch store then `parked` load.
                me.parked.store(true, Ordering::SeqCst);
                if pool.epoch.load(Ordering::SeqCst) == seen {
                    std::thread::park();
                }
                me.parked.store(false, Ordering::SeqCst);
            } else {
                std::hint::spin_loop();
            }
        }
        let (epoch, job) = {
            let slot = pool.job.lock().unwrap_or_else(|e| e.into_inner());
            (slot.0, slot.1.clone())
        };
        seen = epoch;
        if let Some(region) = job {
            if index < region.helpers {
                region.participate(false);
            }
        }
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
    let mut args = unsafe { std::slice::from_raw_parts(args, slots as usize) }.to_vec();
    let threads = match PAR_THREADS.with(|t| t.get()) {
        0 => par_pool_threads(),
        cap => cap.min(par_pool_threads()),
    };
    let pool = (active_region().is_none() && threads > 1 && n > 1).then(par_pool);
    let busy = pool.and_then(|p| p.busy.try_lock().ok());

    let (Some(pool), Some(_busy)) = (pool, busy) else {
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
    let region = Arc::new(ParRegion {
        tramp,
        args,
        count: n,
        next: AtomicI64::new(0),
        helpers: threads - 1,
        inflight: AtomicUsize::new(0),
        caller: std::thread::current(),
        fuel: AtomicI64::new(caller_fuel.map_or(UNLIMITED_FUEL, |c| c.get())),
        abort: AtomicBool::new(false),
        failure: Mutex::new(None),
        arena: ACTIVE_ARENA.with(|a| a.get()),
        sandboxed: STACK_LIMIT.with(|l| l.get()) != 0,
        allocated: AtomicUsize::new(MEMORY_ALLOCATED.with(|m| m.get())),
        quota: MEMORY_QUOTA.with(|q| q.get()),
    });
    {
        let mut slot = pool.job.lock().unwrap_or_else(|e| e.into_inner());
        slot.0 += 1;
        slot.1 = Some(region.clone());
        pool.epoch.store(slot.0, Ordering::SeqCst);
    }
    for helper in &pool.helpers[..region.helpers] {
        if helper.parked.load(Ordering::SeqCst) {
            helper.thread.unpark();
        }
    }

    region.participate(true);
    // Helpers still finishing an index unpark this thread when the last one leaves.
    let mut spins = 0u32;
    while region.inflight.load(Ordering::SeqCst) != 0 {
        if spins < 1 << 14 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::park();
        }
    }
    pool.job.lock().unwrap_or_else(|e| e.into_inner()).1 = None;

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

        let mut fuel_sig = module.make_signature();
        fuel_sig.returns.push(AbiParam::new(types::I32));
        let rt_fuel_exhausted_id =
            module.declare_function("rt_fuel_exhausted", Linkage::Import, &fuel_sig)?;

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

        // Host calls (and `par`) go through a scalar trampoline; functions that take or
        // return vectors are only callable from AIR code. Declared up front so `par` can
        // take their addresses.
        let mut tramp_ids = HashMap::new();
        for func in &ir_mod.functions {
            if func
                .params
                .iter()
                .map(|(_, ty)| ty)
                .chain(func.ret_type.iter())
                .any(|t| t.is_vector())
            {
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
                    func.ret_type,
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
        let sandbox = self.sandbox.as_ref().map(|arena| {
            let arena = lock(arena);
            (arena.base as i64, arena.len() as i64)
        });
        let fuel_counter = self.fuel.as_ptr() as i64;
        let compiled = match &mut self.codegen {
            Codegen::Cranelift(clif) => {
                clif.compile(ir_mod, self.fuel_enabled, fuel_counter, sandbox)?
            }
            #[cfg(feature = "llvm")]
            Codegen::Llvm {
                target,
                vx,
                vector_width,
                matrix,
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
                    sandbox: sandbox.map(|(base, len)| achainsaw_llvm::SandboxBounds {
                        base: base as u64,
                        len: len as u64,
                    }),
                    vx: *vx,
                    vector_width: *vector_width,
                    matrix: *matrix,
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
