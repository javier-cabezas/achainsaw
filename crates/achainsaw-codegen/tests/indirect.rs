//! Function handles (`fnref`), indirect calls (`call_indirect`) and `par` through a handle on
//! the backend selected by `ACHAINSAW_BACKEND`: results of every scalar type, several results and vectors, handles
//! kept in memory (a dispatch table built at run time), calls through handles that fail
//! their checks (out of range, another signature) giving zeros and `ERR_BAD_INDIRECT_CALL`,
//! fuel, the sandbox (including recursion through handles), devirtualization, validation,
//! AIRB, and object code for every target.

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::ast::Instruction;
use achainsaw_ir::parse_and_validate;

const PROGRAM: &str = r#"
fn sq(x:i64)->i64
  b0:
    r = mul x, x
    ret r

fn neg(x:i64)->i64
  b0:
    r = sub 0:i64, x
    ret r

fn twice(x:i64)->i64
  b0:
    r = add x, x
    ret r

fn half(x:f32)->f32
  b0:
    r = mul x, 0.5:f32
    ret r

fn narrow(x:i32, y:i8)->i16
  b0:
    a = trunc x:i16
    b = sext y:i16
    r = add a, b
    ret r

fn divmod(a:i64, b:i64)->(i64, i64)
  b0:
    q = div a, b
    m = rem a, b
    ret q, m

fn vdouble(v:vx)->vx
  b0:
    r = vadd v, v:f32
    ret r

fn vsum4(v:v128)->f64
  b0:
    s = vsum v:f32
    w = fext s:f64
    ret w

fn put(p:ptr, x:i64)
  b0:
    st p, x
    ret

# Handles leave the function that takes them, so the calls below stay indirect.
fn handle_of(k:i64)->i64
  b0:
    switch k, none, 0: a, 1: b, 2: c, 3: d, 4: e, 5: f, 6: g, 7: h, 8: i, 9: j
  a:
    r0 = fnref sq
    ret r0
  b:
    r1 = fnref neg
    ret r1
  c:
    r2 = fnref twice
    ret r2
  d:
    r3 = fnref half
    ret r3
  e:
    r4 = fnref narrow
    ret r4
  f:
    r5 = fnref divmod
    ret r5
  g:
    r6 = fnref vdouble
    ret r6
  h:
    r7 = fnref vsum4
    ret r7
  i:
    r8 = fnref put
    ret r8
  j:
    r9 = fnref fib
    ret r9
  none:
    ret 0:i64

fn call_i64(k:i64, x:i64)->i64
  b0:
    h = call handle_of(k)
    r = call_indirect h(x)->i64
    ret r

fn call_f32(k:i64, x:f32)->f32
  b0:
    h = call handle_of(k)
    r = call_indirect h(x)->f32
    ret r

fn call_narrow(k:i64, x:i32, y:i32)->i32
  b0:
    h = call handle_of(k)
    yb = trunc y:i8
    r = call_indirect h(x, yb)->i16
    w = sext r:i32
    ret w

fn call_divmod(k:i64, a:i64, b:i64)->i64
  b0:
    h = call handle_of(k)
    q, m = call_indirect h(a, b)->(i64, i64)
    t = mul q, 1000:i64
    r = add t, m
    ret r

# x, x + 1, x + 2, ... in the lanes of vx, doubled through a handle and summed.
fn call_vx(k:i64, x:f32)->f32
  b0:
    h = call handle_of(k)
    w = vl f32
    jmp fill(0:i64, x)
  fill(i:i64, a:f32):
    more = lt i, 4:i64
    br more, fill_body, go
  fill_body:
    i2 = add i, 1:i64
    a2 = add a, 1.0:f32
    jmp fill(i2, a2)
  go:
    base = splat x:vx
    d = call_indirect h(base)->vx
    s = vsum d:f32
    ret s

fn call_v128(k:i64, x:f32)->f64
  b0:
    h = call handle_of(k)
    v = splat x:v128
    r = call_indirect h(v)->f64
    ret r

# Results declared but not assigned (one, then two): the calls still run, then put stores.
fn call_discard(x:i64)->i64
  b0:
    h = call handle_of(0:i64)
    d = call handle_of(5:i64)
    g = call handle_of(8:i64)
    p = alloc 8:i64
    st p, 0:i64
    call_indirect h(x)->i64
    call_indirect d(x, 3:i64)->(i64, i64)
    call_indirect g(p, x)
    v = ld p:i64
    free p
    ret v

fn call_void(k:i64, x:i64)->i64
  b0:
    h = call handle_of(k)
    p = alloc 8:i64
    st p, 0:i64
    call_indirect h(p, x)
    r = ld p:i64
    free p
    ret r

# A dispatch table in memory, filled at run time, then a loop calling entry i mod 3.
fn table_loop(n:i64)->i64
  b0:
    t = alloc 24:i64
    h0 = call handle_of(0:i64)
    h1 = call handle_of(1:i64)
    h2 = call handle_of(2:i64)
    st t[0], h0
    st t[1], h1
    st t[2], h2
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    k = urem i, 3:i64
    h = ld t[k]:i64
    v = call_indirect h(i)->i64
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    free t
    ret acc

# Recursion through a handle.
fn fib(n:i64)->i64
  b0:
    small = lt n, 2:i64
    br small, base, rec
  base:
    ret n
  rec:
    h = call handle_of(9:i64)
    a = sub n, 1:i64
    b = sub n, 2:i64
    x = call_indirect h(a)->i64
    y = call_indirect h(b)->i64
    r = add x, y
    ret r

# A handle used in the function that takes it: a direct call before code generation.
fn local(x:i64)->i64
  b0:
    h = fnref sq
    r = call_indirect h(x)->i64
    ret r

fn handle_value(k:i64)->i64
  b0:
    h = call handle_of(k)
    ret h

# Calls that fail their checks: give zeros, which the next steps see.
fn raw(h:i64, x:i64)->i64
  b0:
    r = call_indirect h(x)->i64
    s = add r, 7:i64
    ret s

fn wrong_args(k:i64, x:i64)->i64
  b0:
    h = call handle_of(k)
    r = call_indirect h(x, x)->i64
    ret r

fn wrong_ret(k:i64, x:i64)->i64
  b0:
    h = call handle_of(k)
    r = call_indirect h(x)->i32
    w = sext r:i64
    ret w

# A par body: out[i] = i * i + k.
fn sqk(i:i64, out:ptr, k:i64)
  b0:
    a = mul i, i
    v = add a, k
    st out[i], v
    ret

fn sqk_handle()->i64
  b0:
    h = fnref sqk
    ret h

# sum of out[0 .. n) after a par over sqk, through its handle (h) or directly (h = 0).
fn par_sum(h:i64, n:i64, k:i64)->i64
  b0:
    b = mul n, 8:i64
    out = alloc b
    jmp zero(0:i64)
  zero(z:i64):
    zm = lt z, n
    br zm, zero_body, run
  zero_body:
    st out[z], 0:i64
    z2 = add z, 1:i64
    jmp zero(z2)
  run:
    direct = eq h, 0:i64
    br direct, run_direct, run_handle
  run_direct:
    par n, sqk(out, k)
    jmp sum(0:i64, 0:i64)
  run_handle:
    par n, call_indirect h(out, k)
    jmp sum(0:i64, 0:i64)
  sum(i:i64, s:i64):
    more = lt i, n
    br more, sum_body, done
  sum_body:
    v = ld out[i]:i64
    s2 = add s, v
    i2 = add i, 1:i64
    jmp sum(i2, s2)
  done:
    free out
    ret s

# The same body through a known handle: a direct par before code generation.
fn par_local(n:i64)->i64
  b0:
    b = mul n, 8:i64
    out = alloc b
    h = fnref sqk
    par n, call_indirect h(out, 1:i64)
    last = sub n, 1:i64
    v = ld out[last]:i64
    free out
    ret v

# Handles named like mnemonics, in every position.
fn par_mnemonic(n:i64)->i64
  b0:
    b = mul n, 8:i64
    out = alloc b
    min = call sqk_handle()
    par n, call_indirect min(out, 2:i64)
    last = sub n, 1:i64
    v = ld out[last]:i64
    free out
    ret v

# par through a handle of another signature (sq: no index-first, a result): runs nothing.
fn par_wrong(n:i64)->i64
  b0:
    h = call handle_of(0:i64)
    b = mul n, 8:i64
    out = alloc b
    st out, 42:i64
    par n, call_indirect h(out, 1:i64)
    v = ld out:i64
    free out
    ret v
"#;

fn engine(sandbox: bool) -> JitEngine {
    let module = parse_and_validate(PROGRAM).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    if sandbox {
        e.enable_sandbox(1 << 20).unwrap();
    }
    e.compile_module(&module).unwrap();
    e
}

fn call(e: &JitEngine, name: &str, args: &[RtValue]) -> Result<Option<RtValue>, String> {
    unsafe { e.call_typed(name, args) }.map_err(|err| err.to_string())
}

fn int(e: &JitEngine, name: &str, args: &[i64]) -> i64 {
    let args: Vec<RtValue> = args.iter().map(|&a| RtValue::I64(a)).collect();
    match call(e, name, &args).unwrap_or_else(|err| panic!("{name}: {err}")) {
        Some(RtValue::I64(v)) => v,
        other => panic!("{name}: {other:?}"),
    }
}

fn fib(n: i64) -> i64 {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn check(e: &JitEngine) {
    for x in [-5i64, 0, 3, 1 << 20] {
        assert_eq!(int(e, "call_i64", &[0, x]), x.wrapping_mul(x), "sq({x})");
        assert_eq!(int(e, "call_i64", &[1, x]), -x, "neg({x})");
        assert_eq!(int(e, "call_i64", &[2, x]), 2 * x, "twice({x})");
        assert_eq!(int(e, "local", &[x]), x.wrapping_mul(x), "local({x})");
        assert_eq!(int(e, "call_void", &[8, x]), x, "put({x})");
        assert_eq!(int(e, "call_discard", &[x]), x, "call_discard({x})");
    }
    match call(e, "call_f32", &[RtValue::I64(3), RtValue::F32(5.0)]) {
        Ok(Some(RtValue::F32(v))) => assert_eq!(v, 2.5),
        other => panic!("half: {other:?}"),
    }
    match call(
        e,
        "call_narrow",
        &[RtValue::I64(4), RtValue::I32(70000), RtValue::I32(-3)],
    ) {
        Ok(Some(RtValue::I32(v))) => assert_eq!(v, (70000i32 as i16).wrapping_add(-3) as i32),
        other => panic!("narrow: {other:?}"),
    }
    assert_eq!(int(e, "call_divmod", &[5, 47, 5]), 9 * 1000 + 2);
    let lanes = e.vx_bits() as usize / 32;
    match call(e, "call_vx", &[RtValue::I64(6), RtValue::F32(1.5)]) {
        Ok(Some(RtValue::F32(v))) => assert_eq!(v, 3.0 * lanes as f32),
        other => panic!("vdouble: {other:?}"),
    }
    match call(e, "call_v128", &[RtValue::I64(7), RtValue::F32(1.25)]) {
        Ok(Some(RtValue::F64(v))) => assert_eq!(v, 5.0),
        other => panic!("vsum4: {other:?}"),
    }
    for n in [0i64, 1, 2, 10, 1000] {
        let want: i64 = (0..n)
            .map(|i| match i % 3 {
                0 => i * i,
                1 => -i,
                _ => 2 * i,
            })
            .sum();
        assert_eq!(int(e, "table_loop", &[n]), want, "table_loop({n})");
    }
    assert_eq!(int(e, "fib", &[15]), fib(15));
    let h = int(e, "sqk_handle", &[]);
    for n in [0i64, 1, 7, 1000] {
        let want: i64 = (0..n).map(|i| i * i + 3).sum();
        assert_eq!(
            int(e, "par_sum", &[h, n, 3]),
            want,
            "par through a handle, n = {n}"
        );
        assert_eq!(int(e, "par_sum", &[0, n, 3]), want, "direct par, n = {n}");
    }
    assert_eq!(int(e, "par_local", &[10]), 82);
    assert_eq!(int(e, "par_mnemonic", &[10]), 83);
    // Handles are 1 + the function's index in the module.
    assert_eq!(int(e, "handle_value", &[0]), 1);
    assert_eq!(int(e, "handle_value", &[9]), 20);
}

#[test]
fn indirect_calls_match_direct_ones() {
    check(&engine(false));
}

#[test]
fn indirect_calls_run_under_fuel_and_in_the_sandbox() {
    let mut e = engine(false);
    e.set_fuel(Some(1 << 40));
    check(&e);
    check(&engine(true));
    // Recursion through handles meets the sandbox's stack limit like any other.
    let src = "fn f(n:i64)->i64\n  b0:\n    h = call g()\n    m = add n, 1:i64\n    r = call_indirect h(m)->i64\n    ret r\n\nfn g()->i64\n  b0:\n    h = fnref f\n    ret h\n";
    let module = parse_and_validate(src).unwrap();
    let mut e = JitEngine::new().unwrap();
    e.enable_sandbox(1 << 20).unwrap();
    e.compile_module(&module).unwrap();
    let err = call(&e, "f", &[RtValue::I64(0)]).unwrap_err();
    assert!(err.contains("ERR_STACK_OVERFLOW"), "{err}");
}

#[test]
fn bad_indirect_calls_give_zeros_and_report() {
    for sandbox in [false, true] {
        let e = engine(sandbox);
        // Not a handle: 0, negative, past the table; then valid handles of another
        // signature (arguments, results).
        for h in [0i64, -1, 26, 1 << 40, i64::MIN] {
            let err = call(&e, "raw", &[RtValue::I64(h), RtValue::I64(3)]).unwrap_err();
            assert!(
                err.contains("ERR_BAD_INDIRECT_CALL") && err.contains(&h.to_string()),
                "{h}: {err}"
            );
        }
        for (name, k) in [
            ("wrong_args", 0i64),
            ("wrong_ret", 0),
            ("call_i64", 3),
            ("call_i64", 8),
        ] {
            let err = call(&e, name, &[RtValue::I64(k), RtValue::I64(3)]).unwrap_err();
            assert!(err.contains("ERR_BAD_INDIRECT_CALL"), "{name}({k}): {err}");
        }
        // par through a mismatched or bad handle runs no index.
        let err = call(&e, "par_wrong", &[RtValue::I64(5)]).unwrap_err();
        assert!(err.contains("ERR_BAD_INDIRECT_CALL"), "par_wrong: {err}");
        for h in [-1i64, 999] {
            let args = [RtValue::I64(h), RtValue::I64(4), RtValue::I64(1)];
            let err = call(&e, "par_sum", &args).unwrap_err();
            assert!(err.contains("ERR_BAD_INDIRECT_CALL"), "par_sum({h}): {err}");
        }
        // The engine keeps working.
        assert_eq!(int(&e, "call_i64", &[0, 6]), 36);
    }
    // Outside the sandbox and without a budget the run goes on with zero results: `raw`
    // returns 0 + 7 before the call reports.
    let e = engine(false);
    let p = e.get_fn_ptr("raw").unwrap();
    let raw: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(p) };
    achainsaw_codegen::reset_execution_status();
    assert_eq!(raw(0, 3), 7);
    assert!(achainsaw_codegen::check_execution_status().is_err());
    achainsaw_codegen::reset_execution_status();
}

/// A `par` through a handle uses exactly the fuel of the same `par` called directly (one
/// unit per index plus the body's branches; the handle's checks take none). With the whole
/// pool, totals under an ample budget; with one thread (the same runtime path, but
/// deterministic: which of several workers exhausts a tight budget, and how much is left
/// over, depends on scheduling), the same stopping point under budgets that run out.
#[test]
fn par_through_a_handle_uses_the_fuel_of_a_direct_par() {
    use achainsaw_codegen::{get_remaining_fuel, set_execution_fuel};
    for threads in [None, Some(1)] {
        let mut e = engine(false);
        e.set_fuel(Some(u64::MAX >> 2));
        e.set_threads(threads);
        let h = int(&e, "sqk_handle", &[]);
        let run = |h: i64, budget: u64| {
            set_execution_fuel(Some(budget));
            let r = call(
                &e,
                "par_sum",
                &[RtValue::I64(h), RtValue::I64(50), RtValue::I64(1)],
            );
            let left = get_remaining_fuel();
            set_execution_fuel(None);
            (r.map_err(|err| err.contains("ERR_OUT_OF_FUEL")), left)
        };
        let budgets: &[u64] = if threads.is_none() {
            &[1 << 30]
        } else {
            &[1 << 30, 40, 120, 200, 260]
        };
        for &budget in budgets {
            assert_eq!(
                run(h, budget),
                run(0, budget),
                "budget {budget}, threads {threads:?}"
            );
        }
    }
}

#[test]
fn known_handles_become_direct_calls() {
    let module = parse_and_validate(PROGRAM).unwrap();
    let prepared = achainsaw_ir::prepare_for_codegen(&module);
    let indirect = |name: &str| {
        prepared
            .functions
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(i, Instruction::CallIndirect { .. }))
            .count()
    };
    assert_eq!(indirect("local"), 0);
    assert_eq!(indirect("call_i64"), 1);
    let par_indirect = |name: &str| {
        prepared
            .functions
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(i, Instruction::ParIndirect { .. }))
            .count()
    };
    assert_eq!(par_indirect("par_local"), 0);
    assert_eq!(par_indirect("par_sum"), 1);
    // A known handle of another signature stays indirect, and fails at run time.
    let src = "fn f(x:i64)->i64\n  b0:\n    ret x\n\nfn g(x:i32)->i32\n  b0:\n    h = fnref f\n    r = call_indirect h(x)->i32\n    ret r\n";
    let m = parse_and_validate(src).unwrap();
    let prepared = achainsaw_ir::prepare_for_codegen(&m);
    assert!(prepared.functions[1]
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .any(|i| matches!(i, Instruction::CallIndirect { .. })));
}

fn rejected(src: &str) -> String {
    match parse_and_validate(src) {
        Ok(_) => panic!("accepted:\n{src}"),
        Err(d) => d.error_code.clone(),
    }
}

#[test]
fn fnref_and_call_indirect_are_validated() {
    let f = |body: &str| {
        format!("extfn sinf(x:f32)->f32\n\nfn k(x:i64)->i64\n  b0:\n    ret x\n\nfn t(x:i64, y:i32, p:ptr)->i64\n  b0:\n{body}    ret x\n")
    };
    assert_eq!(
        rejected(&f("    h = fnref nothing\n")),
        "ERR_UNDEFINED_FUNCTION"
    );
    assert_eq!(
        rejected(&f("    h = fnref sinf\n")),
        "ERR_UNDEFINED_FUNCTION"
    );
    assert_eq!(
        rejected(&f("    r = call_indirect y(x)->i64\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        rejected(&f("    r = call_indirect p(x)->i64\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        rejected(&f("    r = call_indirect x(x)\n")),
        "ERR_EXPECTED_RESULT_TYPE"
    );
    assert_eq!(
        rejected(&f("    a, b = call_indirect x(x)->i64\n")),
        "ERR_RESULT_COUNT"
    );
    assert_eq!(
        rejected(&f("    r = call_indirect x(x)->f16\n")),
        "ERR_HALF_IN_SIGNATURE"
    );
    assert_eq!(
        rejected(&f("    r = call_indirect x(undef)->i64\n")),
        "ERR_UNDEFINED_REG"
    );
    assert_eq!(
        rejected(&f("    par y, call_indirect x(p)\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        rejected(&f("    par x, call_indirect y(p)\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        rejected(&f("    v = splat x:v128\n    par x, call_indirect x(v)\n")),
        "ERR_PAR_SIGNATURE"
    );
    // Accepted: results ignored, several results, no results; the mnemonics as names.
    for ok in [
        f("    h = fnref k\n    call_indirect h(x)->i64\n"),
        f("    h = fnref k\n    a, b = call_indirect h(x)->(i64, f32)\n"),
        f("    fnref = add x, 1:i64\n    call_indirect = fnref k\n    call_indirect call_indirect(fnref)\n"),
    ] {
        parse_and_validate(&ok).unwrap_or_else(|d| panic!("{d:?}\n{ok}"));
    }
}

#[test]
fn indirect_calls_round_trip_through_airb() {
    use achainsaw_ir::binary::{decode_module, encode_module, to_air_text};
    let module = parse_and_validate(PROGRAM).unwrap();
    let back = decode_module(&encode_module(&module).unwrap()).unwrap();
    assert_eq!(to_air_text(&back), to_air_text(&module));
    let text = to_air_text(&module);
    assert!(
        text.contains("q, m = call_indirect h(a, b)->(i64, i64)"),
        "{text}"
    );
    assert!(text.contains("call_indirect h(p, x)\n"), "{text}");
    assert!(text.contains("par n, call_indirect h(out, k)\n"), "{text}");
    parse_and_validate(&text).unwrap_or_else(|d| panic!("{d:?}"));
}

/// x86-64 and AArch64 objects (SVE too) on Cranelift and, with the `llvm` feature, LLVM:
/// the table holds relocated function addresses.
#[test]
fn indirect_calls_compile_for_every_target() {
    use achainsaw_codegen::{compile_object, AotTarget, Backend, CodegenOptions};
    let module = parse_and_validate(PROGRAM).unwrap();
    let mut backends = vec![Backend::Cranelift];
    if cfg!(feature = "llvm") {
        backends.push(Backend::Llvm);
    }
    for (triple, cpu, features) in [
        ("x86_64-unknown-linux-gnu", "x86-64", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
        ("aarch64-unknown-linux-gnu", "generic", ""),
        ("aarch64-unknown-linux-gnu", "generic", "+sve"),
        ("aarch64-apple-darwin", "generic", ""),
    ] {
        for &backend in &backends {
            if backend == Backend::Cranelift && features == "+sve" {
                continue;
            }
            let target = AotTarget {
                triple: Some(triple.into()),
                cpu: Some(cpu.into()),
                features: (!features.is_empty()).then(|| features.into()),
            };
            let obj = compile_object(&module, &target, backend, &CodegenOptions::default())
                .unwrap_or_else(|e| panic!("{triple} {cpu} {features} {backend:?}: {e}"));
            assert!(!obj.bytes.is_empty());
        }
    }
}
