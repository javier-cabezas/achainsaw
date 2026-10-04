//! `par n, f(args)`: fork-join loops on the runtime's thread pool. Results must match a
//! serial run, failures in any worker must stop the whole loop, and the fuel budget is
//! shared so parallel code uses exactly the fuel serial code does.

use achainsaw_codegen::{
    get_execution_status, get_remaining_fuel, set_execution_fuel, ExecutionStatus, JitEngine,
    RtValue,
};
use achainsaw_ir::parse_and_validate;
use std::collections::HashSet;

fn engine(src: &str, sandbox: Option<usize>, threads: Option<usize>) -> JitEngine {
    let module = parse_and_validate(src).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    let mut engine = JitEngine::new().unwrap();
    if let Some(bytes) = sandbox {
        engine.enable_sandbox(bytes).unwrap();
    }
    engine.set_threads(threads);
    engine.compile_module(&module).unwrap();
    engine
}

fn call(engine: &JitEngine, name: &str, args: &[RtValue]) -> Result<Option<RtValue>, String> {
    unsafe { engine.call_typed(name, args) }.map_err(|e| e.to_string())
}

fn call_i64(engine: &JitEngine, name: &str, args: &[i64]) -> Result<i64, String> {
    let args: Vec<RtValue> = args.iter().map(|&a| RtValue::I64(a)).collect();
    match call(engine, name, &args)? {
        Some(RtValue::I64(v)) => Ok(v),
        other => panic!("{name}: unexpected {other:?}"),
    }
}

/// `squares(n)` fills a buffer with `i*i` in parallel, then sums it serially.
const SQUARES: &str = r#"
fn fill(i:i64, p:ptr)
  b0:
    off = mul i, 8:i64
    q = add p, off
    v = mul i, i
    st q, v
    ret

fn squares(n:i64)->i64
  b0:
    bytes = mul n, 8:i64
    p = alloc bytes
    par n, fill(p)
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    off = mul i, 8:i64
    q = add p, off
    v = ld q:i64
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    free p
    ret acc
"#;

fn sum_squares(n: i64) -> i64 {
    (0..n).map(|i| i * i).sum()
}

#[test]
fn par_matches_serial_sum() {
    for threads in [None, Some(1), Some(3)] {
        let e = engine(SQUARES, None, threads);
        for n in [0, 1, 7, 1000, 100_000] {
            assert_eq!(
                call_i64(&e, "squares", &[n]).unwrap(),
                sum_squares(n),
                "n={n} threads={threads:?}"
            );
        }
    }
}

#[test]
fn par_in_sandbox() {
    let e = engine(SQUARES, Some(1 << 20), None);
    assert_eq!(
        call_i64(&e, "squares", &[10_000]).unwrap(),
        sum_squares(10_000)
    );
}

/// Every scalar type survives the trip through the runtime's argument slots.
#[test]
fn par_passes_every_scalar_type() {
    let src = r#"
fn put(i:i64, p:ptr, a:i8, b:i16, c:i32, d:f32, e:f64)
  b0:
    a2 = sext a:i64
    b2 = sext b:i64
    c2 = sext c:i64
    s = add a2, b2
    s2 = add s, c2
    s3 = add s2, i
    f = itof s3:f64
    d2 = fext d:f64
    g = add f, d2
    h = add g, e
    off = mul i, 8:i64
    q = add p, off
    st q, h
    ret
"#;
    let e = engine(src, None, None);
    let mut out = vec![0f64; 64];
    let module = parse_and_validate(&format!(
        "{src}\nfn run(p:ptr)\n  b0:\n    par 64:i64, put(p, -3:i8, -300:i16, -70000:i32, 0.5:f32, -0.25:f64)\n    ret\n"
    ))
    .unwrap();
    let mut e2 = JitEngine::new().unwrap();
    e2.compile_module(&module).unwrap();
    drop(e);
    call(&e2, "run", &[RtValue::Ptr(out.as_mut_ptr() as usize)]).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, (i as f64 - 70303.0) + 0.5 - 0.25, "i={i}");
    }
}

/// Row-parallel matrix-vector product with host buffers, against a scalar reference.
#[test]
fn par_gemv_rows() {
    let src = r#"
fn row(i:i64, a:ptr, x:ptr, y:ptr, k:i64)
  b0:
    row_bytes = mul k, 4:i64
    off = mul i, row_bytes
    pa = add a, off
    z = cst 0.0:f32
    jmp loop(0:i64, z)
  loop(j:i64, acc:f32):
    more = lt j, k
    br more, body, done
  body:
    o = mul j, 4:i64
    qa = add pa, o
    qx = add x, o
    va = ld qa:f32
    vx = ld qx:f32
    p = mul va, vx
    acc2 = add acc, p
    j2 = add j, 1:i64
    jmp loop(j2, acc2)
  done:
    yo = mul i, 4:i64
    qy = add y, yo
    st qy, acc
    ret

fn gemv(a:ptr, x:ptr, y:ptr, m:i64, k:i64)
  b0:
    par m, row(a, x, y, k)
    ret
"#;
    let (m, k) = (257usize, 129usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
    let x: Vec<f32> = (0..k).map(|i| (i % 5) as f32 * 0.5).collect();
    let expected: Vec<f32> = (0..m)
        .map(|r| (0..k).fold(0.0, |acc, j| acc + a[r * k + j] * x[j]))
        .collect();
    for threads in [None, Some(1)] {
        let e = engine(src, None, threads);
        let mut y = vec![0f32; m];
        call(
            &e,
            "gemv",
            &[
                RtValue::Ptr(a.as_ptr() as usize),
                RtValue::Ptr(x.as_ptr() as usize),
                RtValue::Ptr(y.as_mut_ptr() as usize),
                RtValue::I64(m as i64),
                RtValue::I64(k as i64),
            ],
        )
        .unwrap();
        assert_eq!(y, expected, "threads={threads:?}");
    }
}

/// A `par` inside a `par` body runs serially in its worker.
#[test]
fn nested_par() {
    let src = r#"
fn cell(j:i64, p:ptr, i:i64, n:i64)
  b0:
    r = mul i, n
    idx = add r, j
    off = mul idx, 8:i64
    q = add p, off
    st q, idx
    ret

fn line(i:i64, p:ptr, n:i64)
  b0:
    par n, cell(p, i, n)
    ret

fn grid(n:i64)->i64
  b0:
    nn = mul n, n
    bytes = mul nn, 8:i64
    p = alloc bytes
    par n, line(p, n)
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, nn
    br more, body, done
  body:
    off = mul i, 8:i64
    q = add p, off
    v = ld q:i64
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    free p
    ret acc
"#;
    let e = engine(src, Some(1 << 20), None);
    let nn = 50 * 50;
    assert_eq!(call_i64(&e, "grid", &[50]).unwrap(), nn * (nn - 1) / 2);
}

thread_local! {
    static WORKER_ID: u64 = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut h);
        h.finish() >> 1
    };
}

extern "C" fn worker_id() -> i64 {
    WORKER_ID.with(|w| *w as i64)
}

/// Iterations really run on several threads (and on one with `threads = 1`).
#[test]
fn par_uses_the_pool() {
    let src = r#"
extfn worker_id()->i64

fn spin(i:i64, p:ptr)
  b0:
    jmp loop(0:i64)
  loop(j:i64):
    more = lt j, 200000:i64
    br more, body, done
  body:
    j2 = add j, 1:i64
    jmp loop(j2)
  done:
    id = call worker_id()
    off = mul i, 8:i64
    q = add p, off
    st q, id
    ret

fn run(p:ptr, n:i64)
  b0:
    par n, spin(p)
    ret
"#;
    let pool = achainsaw_codegen::par_pool_threads();
    let distinct_ids = |threads: Option<usize>| {
        let module = parse_and_validate(src).unwrap();
        let mut e = JitEngine::new().unwrap();
        e.register_symbol("worker_id", worker_id as *const u8);
        e.set_threads(threads);
        e.compile_module(&module).unwrap();
        let n = 64;
        let mut ids = vec![0i64; n];
        let args = [
            RtValue::Ptr(ids.as_mut_ptr() as usize),
            RtValue::I64(n as i64),
        ];
        call(&e, "run", &args).unwrap();
        let distinct: HashSet<i64> = ids.iter().copied().collect();
        assert!(!distinct.contains(&0));
        distinct
    };

    let serial = distinct_ids(Some(1));
    assert_eq!(serial.len(), 1, "{serial:?}");
    assert_eq!(serial.into_iter().next(), Some(worker_id()));
    if pool > 1 {
        // A `par` runs serially while another test's `par` has the pool, so retry.
        let spread = (0..100).any(|_| distinct_ids(None).len() > 1);
        assert!(spread, "pool of {pool} always used one thread");
    }
}

/// Workers share the caller's budget: a parallel run uses exactly the fuel of a serial one.
#[test]
fn par_fuel_accounting_matches_serial() {
    let src = r#"
fn work(i:i64, p:ptr)
  b0:
    jmp loop(0:i64)
  loop(j:i64):
    more = lt j, i
    br more, body, done
  body:
    j2 = add j, 1:i64
    jmp loop(j2)
  done:
    ret

fn run(n:i64)->i64
  b0:
    p = alloc 8:i64
    par n, work(p)
    free p
    ret n
"#;
    let budget = 10_000_000;
    let mut used = Vec::new();
    for threads in [Some(1), None, Some(2)] {
        let mut e = engine(src, None, threads);
        e.set_fuel(Some(budget));
        assert_eq!(call_i64(&e, "run", &[2000]).unwrap(), 2000);
        used.push(budget - get_remaining_fuel().unwrap());
        set_execution_fuel(None);
    }
    assert!(used[0] > 2000 * 1000, "{used:?}");
    assert!(used.iter().all(|&u| u == used[0]), "{used:?}");
}

#[test]
fn par_out_of_fuel_stops_all_workers() {
    // Index 3 never terminates; every other index finishes quickly.
    let src = r#"
fn work(i:i64)
  b0:
    jmp loop(0:i64)
  loop(j:i64):
    stuck = eq i, 3:i64
    more = lt j, 100:i64
    go = or stuck, more
    br go, body, done
  body:
    j2 = add j, 1:i64
    jmp loop(j2)
  done:
    ret

fn run(n:i64)->i64
  b0:
    par n, work()
    ret 1:i64
"#;
    for sandbox in [None, Some(4096)] {
        let mut e = engine(src, sandbox, None);
        e.set_fuel(Some(5_000_000));
        let err = call_i64(&e, "run", &[64]).unwrap_err();
        assert!(err.contains("ERR_OUT_OF_FUEL"), "{err}");
        assert_eq!(get_execution_status(), ExecutionStatus::OutOfFuel);
        assert_eq!(get_remaining_fuel(), Some(0));
        set_execution_fuel(None);
    }
}

#[test]
fn par_budget_bounds_bodies_without_branches() {
    let src = r#"
fn nop(i:i64)
  b0:
    ret

fn run(n:i64)->i64
  b0:
    par n, nop()
    ret 1:i64
"#;
    let mut e = engine(src, None, None);
    // A negative count runs nothing.
    assert_eq!(call_i64(&e, "run", &[-5]).unwrap(), 1);
    e.set_fuel(Some(1000));
    let err = call_i64(&e, "run", &[1 << 40]).unwrap_err();
    assert!(err.contains("ERR_OUT_OF_FUEL"), "{err}");
    set_execution_fuel(None);
}

#[test]
fn par_memory_violation_in_worker() {
    let src = r#"
fn poke(i:i64, p:ptr)
  b0:
    bad = eq i, 37:i64
    far = select bad, 1000000:i64, 0:i64
    off = mul i, 8:i64
    off2 = add off, far
    q = add p, off2
    st q, i
    ret

fn run(n:i64)->i64
  b0:
    bytes = mul n, 8:i64
    p = alloc bytes
    par n, poke(p)
    ret 1:i64
"#;
    let e = engine(src, Some(4096), None);
    let err = call_i64(&e, "run", &[64]).unwrap_err();
    assert!(err.contains("ERR_MEMORY_VIOLATION"), "{err}");
    assert!(matches!(
        get_execution_status(),
        ExecutionStatus::MemoryViolation { .. }
    ));
    // The engine still works afterwards.
    let ok = engine(SQUARES, Some(1 << 16), None);
    assert_eq!(call_i64(&ok, "squares", &[100]).unwrap(), sum_squares(100));
}

#[test]
fn par_stack_overflow_in_worker() {
    let src = r#"
fn down(x:i64)->i64
  b0:
    y = call down(x)
    ret y

fn deep(i:i64)
  b0:
    big = eq i, 5:i64
    br big, rec, done
  rec:
    r = call down(i)
    jmp done
  done:
    ret

fn run(n:i64)->i64
  b0:
    par n, deep()
    ret 1:i64
"#;
    let e = engine(src, Some(4096), None);
    let err = call_i64(&e, "run", &[16]).unwrap_err();
    assert!(err.contains("ERR_STACK_OVERFLOW"), "{err}");
}

/// Sandboxed bodies allocate and free scratch memory concurrently from the shared arena.
#[test]
fn par_alloc_free_in_sandbox() {
    let src = r#"
fn scratch(i:i64, out:ptr)
  b0:
    s = alloc 256:i64
    st s, i
    v = ld s:i64
    w = mul v, 3:i64
    free s
    off = mul i, 8:i64
    q = add out, off
    st q, w
    ret

fn run(n:i64)->i64
  b0:
    bytes = mul n, 8:i64
    out = alloc bytes
    par n, scratch(out)
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    off = mul i, 8:i64
    q = add out, off
    v = ld q:i64
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc
"#;
    let e = engine(src, Some(1 << 20), None);
    let n = 5000;
    assert_eq!(call_i64(&e, "run", &[n]).unwrap(), 3 * n * (n - 1) / 2);
}

#[test]
fn par_out_of_arena_memory() {
    let src = r#"
fn hog(i:i64)
  b0:
    s = alloc 1024:i64
    ret

fn run(n:i64)->i64
  b0:
    par n, hog()
    ret 1:i64
"#;
    let e = engine(src, Some(4096), None);
    let err = call_i64(&e, "run", &[64]).unwrap_err();
    assert!(err.contains("ERR_OUT_OF_MEMORY"), "{err}");
}

/// Code reached through `get_fn_ptr` (no `call_typed` activation) still runs `par`.
#[test]
fn par_through_raw_function_pointer() {
    let e = engine(SQUARES, None, None);
    let f: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(e.get_fn_ptr("squares").unwrap()) };
    assert_eq!(f(20_000), sum_squares(20_000));
}

/// AOT objects have no runtime, so `par` runs as a serial loop with the same results.
#[test]
fn par_in_aot_shared_library() {
    use achainsaw_codegen::{compile_object, link_shared_library, AotTarget, Backend};
    let module = parse_and_validate(SQUARES).unwrap();
    let dir = std::env::temp_dir().join(format!("achainsaw-par-aot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut backends = vec![Backend::Cranelift];
    if cfg!(feature = "llvm") {
        backends.push(Backend::Llvm);
    }
    for backend in backends {
        let obj = compile_object(&module, &AotTarget::default(), backend).unwrap();
        let obj_path = dir.join(format!("squares_{backend}.o"));
        let lib_path = dir.join(format!("libsquares_{backend}.so"));
        std::fs::write(&obj_path, &obj.bytes).unwrap();
        if let Err(e) = link_shared_library(&obj_path, &lib_path) {
            eprintln!("skipping AOT link check: {e}");
            return;
        }
        unsafe {
            let lib = libloading::Library::new(&lib_path).unwrap();
            let f: libloading::Symbol<extern "C" fn(i64) -> i64> = lib.get(b"squares").unwrap();
            assert_eq!(f(1000), sum_squares(1000), "{backend}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
