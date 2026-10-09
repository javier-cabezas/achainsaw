//! `JitEngine::enable_sandbox`: memory accesses confined to the arena, validated `free`,
//! and capped recursion, all reported as execution errors instead of crashing.

use achainsaw_codegen::{get_execution_status, ExecutionStatus, JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

const ARENA: usize = 4096;

fn sandboxed(src: &str) -> JitEngine {
    let module = parse_and_validate(src).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    let mut engine = JitEngine::new().unwrap();
    engine.enable_sandbox(ARENA).unwrap();
    engine.compile_module(&module).unwrap();
    engine
}

fn call(engine: &JitEngine, name: &str, args: &[RtValue]) -> Result<Option<RtValue>, String> {
    unsafe { engine.call_typed(name, args) }.map_err(|e| e.to_string())
}

fn i64s(args: &[i64]) -> Vec<RtValue> {
    args.iter().map(|&a| RtValue::I64(a)).collect()
}

/// Asserts the call failed with a memory violation and returns its message.
fn expect_violation(engine: &JitEngine, name: &str, args: &[RtValue]) -> String {
    let err = call(engine, name, args).expect_err(name);
    assert!(err.contains("ERR_MEMORY_VIOLATION"), "{name}: {err}");
    assert!(
        matches!(
            get_execution_status(),
            ExecutionStatus::MemoryViolation { .. }
        ),
        "{name}: {:?}",
        get_execution_status()
    );
    err
}

/// Each function allocates the whole arena, then accesses it at byte offset `off`.
const ACCESSES: &str = r#"
fn ld_f32(off:i64)->f32
  b0:
    p = alloc 4096:i64
    q = add p, off
    v = ld q:f32
    ret v

fn st_f32(off:i64)->i32
  b0:
    p = alloc 4096:i64
    q = add p, off
    x = cst 2.5:f32
    st q, x
    y = ld q:f32
    r = ftoi y:i32
    ret r

fn ld_v512(off:i64)->f32
  b0:
    p = alloc 4096:i64
    q = add p, off
    v = ld q:v512
    s = vsum v:f32
    ret s

fn st_v512(off:i64)->i32
  b0:
    p = alloc 4096:i64
    q = add p, off
    one = cst 1.0:f32
    v = splat one:v512
    st q, v
    ret 0:i32

fn ldm_v512(off:i64, n:i64)->f32
  b0:
    p = alloc 4096:i64
    q = add p, off
    v = ldm q:v512, n:f32
    s = vsum v:f32
    ret s

fn stm_v512(off:i64, n:i64)->i32
  b0:
    p = alloc 4096:i64
    q = add p, off
    one = cst 1.0:f32
    v = splat one:v512
    stm q, v, n:f32
    ret 0:i32

fn mm_at(c_off:i64, m:i64, n:i64, k:i64)->f32
  b0:
    p = alloc 4096:i64
    c = add p, c_off
    mm c, p, p, m, n, k:f32
    v = ld p:f32
    ret v
"#;

#[test]
fn accesses_reaching_the_arena_end_succeed() {
    let e = sandboxed(ACCESSES);
    let end = ARENA as i64;
    assert_eq!(
        call(&e, "st_f32", &i64s(&[end - 4])).unwrap(),
        Some(RtValue::I32(2))
    );
    assert!(call(&e, "ld_f32", &i64s(&[end - 4])).is_ok());
    assert!(call(&e, "ld_v512", &i64s(&[end - 64])).is_ok());
    assert!(call(&e, "st_v512", &i64s(&[end - 64])).is_ok());
    // A masked access may start where a full vector would overrun, as long as its
    // first n lanes fit.
    assert!(call(&e, "ldm_v512", &i64s(&[end - 8, 2])).is_ok());
    assert!(call(&e, "stm_v512", &i64s(&[end - 8, 2])).is_ok());
    // Zero lanes touch nothing, so any address is fine.
    assert!(call(&e, "ldm_v512", &i64s(&[1 << 40, 0])).is_ok());
    assert!(call(&e, "stm_v512", &i64s(&[-(1 << 40), -3])).is_ok());
    // 16x16 f32 C right at the end; A and B alias the start.
    assert!(call(&e, "mm_at", &i64s(&[end - 16 * 16 * 4, 16, 16, 16])).is_ok());
    // Non-positive dimensions are a no-op even with C far outside the arena.
    assert!(call(&e, "mm_at", &i64s(&[1 << 40, 0, 16, 16])).is_ok());
}

#[test]
fn accesses_leaving_the_arena_are_violations() {
    let e = sandboxed(ACCESSES);
    let end = ARENA as i64;
    for off in [end - 3, end, -1, -4096, 1 << 40, i64::MIN, i64::MAX] {
        expect_violation(&e, "ld_f32", &i64s(&[off]));
        expect_violation(&e, "st_f32", &i64s(&[off]));
    }
    for off in [end - 63, end, -16] {
        expect_violation(&e, "ld_v512", &i64s(&[off]));
        expect_violation(&e, "st_v512", &i64s(&[off]));
    }
    // Three f32 lanes from 8 bytes before the end overrun by 4.
    expect_violation(&e, "ldm_v512", &i64s(&[end - 8, 3]));
    expect_violation(&e, "stm_v512", &i64s(&[end - 8, 3]));
    expect_violation(&e, "ldm_v512", &i64s(&[-4, 1]));
    // C one row too low, or A/B (16x17 / 17x16 = 1088 bytes) still fine but C huge.
    expect_violation(&e, "mm_at", &i64s(&[end - 16 * 16 * 4 + 4, 16, 16, 16]));
    expect_violation(&e, "mm_at", &i64s(&[0, 64, 64, 64]));
    // Byte sizes that overflow usize must not wrap into a passing check.
    expect_violation(&e, "mm_at", &i64s(&[0, 1 << 40, 1 << 40, 1 << 40]));
}

#[test]
fn violations_report_address_and_size() {
    let e = sandboxed(ACCESSES);
    let msg = expect_violation(&e, "ld_v512", &i64s(&[ARENA as i64]));
    assert!(msg.contains("64 bytes"), "{msg}");
}

#[test]
fn host_memory_is_unreachable_through_pointer_arguments() {
    let src = r#"
fn read(p:ptr)->i32
  b0:
    v = ld p:i32
    ret v

fn write(p:ptr)
  b0:
    x = cst 7:i32
    st p, x
    ret

fn write_masked(p:ptr)
  b0:
    one = cst 1.0:f32
    v = splat one:v128
    stm p, v, 4:f32
    ret
"#;
    let e = sandboxed(src);
    let mut host = vec![0i32; 16];
    let p = RtValue::Ptr(host.as_mut_ptr() as usize);
    expect_violation(&e, "read", &[p]);
    expect_violation(&e, "write", &[p]);
    expect_violation(&e, "write_masked", &[p]);
    assert!(
        host.iter().all(|&x| x == 0),
        "host buffer modified: {host:?}"
    );

    // The same code without a sandbox does write through the pointer.
    let module = parse_and_validate(src).unwrap();
    let mut plain = JitEngine::new().unwrap();
    plain.compile_module(&module).unwrap();
    call(&plain, "write", &[p]).unwrap();
    assert_eq!(host[0], 7);
}

#[test]
fn free_only_accepts_live_allocations() {
    let src = r#"
fn free_at(off:i64)
  b0:
    p = alloc 64:i64
    q = add p, off
    free q
    ret

fn double_free()
  b0:
    p = alloc 64:i64
    free p
    free p
    ret

fn churn(iters:i64)->i64
  b0:
    jmp b1(0:i64)
  b1(i:i64):
    c = lt i, iters
    br c, b2, b3
  b2:
    p = alloc 1024:i64
    st p, i
    free p
    i2 = add i, 1:i64
    jmp b1(i2)
  b3:
    ret i
"#;
    let e = sandboxed(src);
    assert!(call(&e, "free_at", &i64s(&[0])).is_ok());
    for off in [16, -16, 1 << 40] {
        let msg = expect_violation(&e, "free_at", &i64s(&[off]));
        assert!(msg.contains("not a live allocation"), "{msg}");
    }
    expect_violation(&e, "double_free", &[]);
    // Freed space is reused, so a loop far exceeding the arena still succeeds.
    assert_eq!(
        call(&e, "churn", &i64s(&[10_000])).unwrap(),
        Some(RtValue::I64(10_000))
    );
}

#[test]
fn exhausting_the_arena_reports_out_of_memory() {
    let src = r#"
fn too_big()->i32
  b0:
    p = alloc 8192:i64
    v = ld p:i32
    ret v

fn reset_between_calls()->i32
  b0:
    p = alloc 3000:i64
    ret 1:i32
"#;
    let e = sandboxed(src);
    let err = call(&e, "too_big", &[]).unwrap_err();
    // The null dereference that follows is caught too, but the first cause is reported.
    assert!(err.contains("ERR_OUT_OF_MEMORY"), "{err}");
    // Each call starts with an empty arena.
    for _ in 0..3 {
        assert!(call(&e, "reset_between_calls", &[]).is_ok());
    }
}

#[test]
fn unbounded_recursion_halts_with_stack_overflow() {
    let src = r#"
fn down(n:i64)->i64
  b0:
    n2 = add n, 1:i64
    r = call down(n2)
    ret r

fn depth(n:i64)->i64
  b0:
    c = le n, 0:i64
    br c, b1, b2
  b1:
    ret 0:i64
  b2:
    n2 = sub n, 1:i64
    r = call depth(n2)
    r2 = add r, 1:i64
    ret r2
"#;
    let e = sandboxed(src);
    let err = call(&e, "down", &i64s(&[0])).unwrap_err();
    assert!(err.contains("ERR_STACK_OVERFLOW"), "{err}");
    assert_eq!(get_execution_status(), ExecutionStatus::StackOverflow);
    // Reasonable recursion still works, and the limit is reset per call.
    assert_eq!(
        call(&e, "depth", &i64s(&[1000])).unwrap(),
        Some(RtValue::I64(1000))
    );
}

#[test]
fn caller_stops_after_a_callee_violation_without_fuel() {
    // Fuel is off on this thread, so only the halt check ends the loop.
    let src = r#"
fn bad()->i32
  b0:
    p = cst 0:ptr
    v = ld p:i32
    ret v

fn spin()->i32
  b0:
    jmp b1
  b1:
    x = call bad()
    jmp b1
"#;
    let mut e = sandboxed(src);
    e.set_fuel(None);
    expect_violation(&e, "spin", &[]);
}

#[test]
fn caller_stops_after_a_violation_in_a_par_worker() {
    // Workers share the arena; one of them leaves it, and the spinning caller must stop.
    let src = r#"
fn poke(i:i64, p:ptr)
  b0:
    off = mul i, 4096:i64
    q = add p, off
    st q, i
    ret

fn spin()->i32
  b0:
    p = alloc 64:i64
    jmp b1
  b1:
    par 16:i64, poke(p)
    jmp b1
"#;
    let mut e = sandboxed(src);
    e.set_fuel(None);
    expect_violation(&e, "spin", &[]);
}

#[test]
fn sandbox_must_be_enabled_before_compiling() {
    let module = parse_and_validate("fn f()\n  b0:\n    ret\n").unwrap();
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    assert!(e.enable_sandbox(ARENA).is_err());
}

/// Indexed accesses `p[i]` are checked at the address they compute: element `i` of the
/// arena, scaled by the element size (or the index's unit).
#[test]
fn indexed_accesses_are_checked_at_their_scaled_address() {
    let engine = sandboxed(
        r#"
fn ld_idx(i:i64)->f32
  b0:
    p = alloc 4096:i64
    v = ld p[i]:f32
    ret v

fn st_idx(i:i64)->i32
  b0:
    p = alloc 4096:i64
    st p[i], 1.5:f64
    ret 0:i32

fn ldm_idx(i:i64, n:i64)->f32
  b0:
    p = alloc 4096:i64
    v = ldm p[i]:v128, n:f32
    s = vsum v:f32
    ret s

fn stm_idx(i:i64, n:i64)->i32
  b0:
    p = alloc 4096:i64
    z = splat 0.0:f32:v128
    stm p[i:v128], z, n:f32
    ret 0:i32
"#,
    );
    // The last element of each kind fits; one more does not.
    assert!(call(&engine, "ld_idx", &i64s(&[1023])).is_ok());
    expect_violation(&engine, "ld_idx", &i64s(&[1024]));
    expect_violation(&engine, "ld_idx", &i64s(&[-1]));
    assert!(call(&engine, "st_idx", &i64s(&[511])).is_ok());
    expect_violation(&engine, "st_idx", &i64s(&[512]));
    assert!(call(&engine, "ldm_idx", &i64s(&[1021, 3])).is_ok());
    expect_violation(&engine, "ldm_idx", &i64s(&[1021, 4]));
    assert!(call(&engine, "stm_idx", &i64s(&[255, 4])).is_ok());
    expect_violation(&engine, "stm_idx", &i64s(&[256, 1]));
}
