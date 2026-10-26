//! `switch` on the backend selected by `ACHAINSAW_BACKEND`: every integer width with cases
//! written signed and unsigned, dense case ranges (jump tables) and sparse ones (compare
//! trees), several cases into one block with different arguments, a switch with no cases,
//! switches in loops and in an inlined helper, under a fuel budget and in the sandbox, the
//! validator's checks, constant folding, AIRB, and object code for every target.

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

const PROGRAM: &str = r#"
# Cases at the ends of i8's signed and unsigned ranges: 200 is the i8 value -56.
fn sw_i8(x:i64)->i64
  b0:
    t = trunc x:i8
    switch t, other, -128: a, -1: b, 0: c, 127: d, 200: e
  a:
    ret 1:i64
  b:
    ret 2:i64
  c:
    ret 3:i64
  d:
    ret 4:i64
  e:
    ret 5:i64
  other:
    ret 0:i64

fn sw_i16(x:i64)->i64
  b0:
    t = trunc x:i16
    switch t, other, -32768: a, 65535: b, 1000: c
  a:
    ret 1:i64
  b:
    ret 2:i64
  c:
    ret 3:i64
  other:
    ret 0:i64

fn sw_i32(x:i64)->i64
  b0:
    t = trunc x:i32
    switch t, other, -2147483648: a, 4294967295: b, 7: c, 8: c, 9: c
  a:
    ret 1:i64
  b:
    ret 2:i64
  c:
    ret 3:i64
  other:
    ret 0:i64

fn sw_i64(x:i64)->i64
  b0:
    switch x, other, -9223372036854775808: a, 9223372036854775807: b, 1099511627776: c, -1: d
  a:
    ret 1:i64
  b:
    ret 2:i64
  c:
    ret 3:i64
  d:
    ret 4:i64
  other:
    ret 0:i64

# A dense range (a jump table), each case passing its own arguments to shared blocks.
fn dense(x:i64)->i64
  b0:
    y = mul x, 3:i64
    switch x, other(x), 0: lin(1:i64, 0:i64), 1: lin(2:i64, 5:i64), 2: sq(y), 3: lin(-1:i64, 9:i64), 4: sq(x), 5: lin(0:i64, 77:i64), 6: sq(7:i64), 7: other(100:i64), 8: lin(3:i64, x), 9: sq(y)
  lin(m:i64, c:i64):
    p = mul m, x
    r = add p, c
    ret r
  sq(s:i64):
    r2 = mul s, s
    ret r2
  other(o:i64):
    r3 = sub 0:i64, o
    ret r3

# Only a default.
fn none(x:i64)->i64
  b0:
    switch x, out(x)
  out(v:i64):
    r = add v, 1:i64
    ret r

# A loop whose body dispatches on i mod 5: a state machine over the iterations.
fn machine(n:i64)->i64
  b0:
    jmp loop(0:i64, 1:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    k = urem i, 5:i64
    switch k, s4, 0: s0, 1: s1, 2: s2, 3: s3
  s0:
    a0 = add acc, i
    jmp next(a0)
  s1:
    a1 = mul acc, 3:i64
    jmp next(a1)
  s2:
    a2 = xor acc, i
    jmp next(a2)
  s3:
    a3 = sub acc, 7:i64
    jmp next(a3)
  s4:
    a4 = shr acc, 1:i64
    jmp next(a4)
  next(an:i64):
    i2 = add i, 1:i64
    jmp loop(i2, an)
  done:
    ret acc

inline fn pick(k:i64, v:i64)->i64
  b0:
    switch k, d(v), 0: z, 1: d(10:i64), 2: two(v)
  z:
    ret 0:i64
  two(t:i64):
    r = add t, t
    ret r
  d(w:i64):
    ret w

# The inlined helper in a loop.
fn picks(n:i64)->i64
  b0:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    k = urem i, 4:i64
    p = call pick(k, i)
    acc2 = add acc, p
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc
"#;

fn sw_i8(x: i64) -> i64 {
    match x as i8 {
        -128 => 1,
        -1 => 2,
        0 => 3,
        127 => 4,
        -56 => 5,
        _ => 0,
    }
}

fn sw_i16(x: i64) -> i64 {
    match x as i16 {
        -32768 => 1,
        -1 => 2,
        1000 => 3,
        _ => 0,
    }
}

fn sw_i32(x: i64) -> i64 {
    match x as i32 {
        i32::MIN => 1,
        -1 => 2,
        7..=9 => 3,
        _ => 0,
    }
}

fn sw_i64(x: i64) -> i64 {
    match x {
        i64::MIN => 1,
        i64::MAX => 2,
        1099511627776 => 3,
        -1 => 4,
        _ => 0,
    }
}

fn dense(x: i64) -> i64 {
    let y = x.wrapping_mul(3);
    let lin = |m: i64, c: i64| m.wrapping_mul(x).wrapping_add(c);
    let sq = |s: i64| s.wrapping_mul(s);
    match x {
        0 => lin(1, 0),
        1 => lin(2, 5),
        2 => sq(y),
        3 => lin(-1, 9),
        4 => sq(x),
        5 => lin(0, 77),
        6 => sq(7),
        7 => -100,
        8 => lin(3, x),
        9 => sq(y),
        _ => x.wrapping_neg(),
    }
}

fn machine(n: i64) -> i64 {
    let mut acc: i64 = 1;
    for i in 0..n.max(0) {
        acc = match i % 5 {
            0 => acc.wrapping_add(i),
            1 => acc.wrapping_mul(3),
            2 => acc ^ i,
            3 => acc.wrapping_sub(7),
            _ => acc >> 1,
        };
    }
    acc
}

fn picks(n: i64) -> i64 {
    (0..n.max(0))
        .map(|i| match i % 4 {
            0 => 0,
            1 => 10,
            2 => 2 * i,
            _ => i,
        })
        .sum()
}

fn engine(sandbox: bool) -> JitEngine {
    let module = parse_and_validate(PROGRAM).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    if sandbox {
        e.enable_sandbox(1 << 20).unwrap();
    }
    e.compile_module(&module).unwrap();
    e
}

fn int(e: &JitEngine, name: &str, arg: i64) -> i64 {
    match unsafe { e.call_typed(name, &[RtValue::I64(arg)]) } {
        Ok(Some(RtValue::I64(v))) => v,
        other => panic!("{name}({arg}): {other:?}"),
    }
}

/// Inputs at and around every case value, at the ends of each width, and beyond.
fn inputs() -> Vec<i64> {
    let mut v: Vec<i64> = (-20..=20).collect();
    for base in [
        127i64,
        128,
        200,
        255,
        256,
        1000,
        32767,
        32768,
        65535,
        65536,
        i32::MAX as i64,
        1 << 31,
        u32::MAX as i64,
        1 << 32,
        1 << 40,
        i64::MAX,
    ] {
        for d in -2..=2 {
            v.push(base.wrapping_add(d));
            v.push(base.wrapping_neg().wrapping_add(d));
        }
    }
    v.extend([i64::MIN, i64::MIN + 1]);
    v
}

type Reference = fn(i64) -> i64;

fn check(e: &JitEngine) {
    let cases: [(&str, Reference); 5] = [
        ("sw_i8", sw_i8),
        ("sw_i16", sw_i16),
        ("sw_i32", sw_i32),
        ("sw_i64", sw_i64),
        ("dense", dense),
    ];
    for x in inputs() {
        for (name, f) in cases {
            assert_eq!(int(e, name, x), f(x), "{name}({x})");
        }
        assert_eq!(int(e, "none", x), x.wrapping_add(1), "none({x})");
    }
    for n in [0, 1, 4, 5, 6, 99, 1000] {
        assert_eq!(int(e, "machine", n), machine(n), "machine({n})");
        assert_eq!(int(e, "picks", n), picks(n), "picks({n})");
    }
}

#[test]
fn switch_matches_reference() {
    check(&engine(false));
}

#[test]
fn switch_runs_under_fuel_and_in_the_sandbox() {
    let mut e = engine(false);
    e.set_fuel(Some(1 << 40));
    check(&e);
    check(&engine(true));
    // Every switch costs fuel like a branch: a budget stops the loop.
    let mut e = engine(false);
    e.set_fuel(Some(1000));
    let err = unsafe { e.call_typed("machine", &[RtValue::I64(1_000_000)]) }.unwrap_err();
    assert!(err.to_string().contains("fuel"), "{err}");
}

fn rejected(src: &str) -> String {
    match parse_and_validate(src) {
        Ok(_) => panic!("accepted:\n{src}"),
        Err(d) => d.error_code.clone(),
    }
}

#[test]
fn switch_is_validated() {
    let f = |head: &str, sw: &str| {
        format!("fn f(x:i64, y:f32, p:ptr)->i64\n  b0:\n{head}    {sw}\n  a:\n    ret 1:i64\n  b(v:i64):\n    ret v\n")
    };
    let t8 = "    t = trunc x:i8\n";
    assert_eq!(rejected(&f("", "switch y, a")), "ERR_TYPE_MISMATCH");
    assert_eq!(rejected(&f("", "switch p, a")), "ERR_TYPE_MISMATCH");
    assert_eq!(
        rejected(&f(t8, "switch t, a, 256: a")),
        "ERR_INVALID_SWITCH_CASE"
    );
    assert_eq!(
        rejected(&f(t8, "switch t, a, -129: a")),
        "ERR_INVALID_SWITCH_CASE"
    );
    assert_eq!(
        rejected(&f(t8, "switch t, a, -1: a, 255: a")),
        "ERR_INVALID_SWITCH_CASE"
    );
    assert_eq!(
        rejected(&f("", "switch x, a, 1: a, 1: a")),
        "ERR_INVALID_SWITCH_CASE"
    );
    assert_eq!(
        rejected(&f("", "switch x, a, 1: nowhere")),
        "ERR_UNDEFINED_BLOCK"
    );
    assert_eq!(
        rejected(&f("", "switch x, a, 1: b0")),
        "ERR_BRANCH_TO_ENTRY"
    );
    assert_eq!(rejected(&f("", "switch x, a, 1: b")), "ERR_ARITY_MISMATCH");
    assert_eq!(rejected(&f("", "switch x, b(y)")), "ERR_TYPE_MISMATCH");
    // Accepted: both spellings of an i8 value, and the empty case list.
    for ok in [
        f(t8, "switch t, a, -128: a, 255: b(x)"),
        f("", "switch x, b(x)"),
    ] {
        parse_and_validate(&ok).unwrap_or_else(|d| panic!("{d:?}\n{ok}"));
    }
}

#[test]
fn switch_stays_a_register_name() {
    // As a destination, alone or the first of several, and as the switched value.
    let src = "fn pair(x:i64)->(i64, i64)\n  b0:\n    ret x, x\n\nfn f(x:i64)->i64\n  b0:\n    switch, y = call pair(x)\n    switch = add switch, 1:i64\n    switch switch, a(switch)\n  a(v:i64):\n    ret v\n";
    let m = parse_and_validate(src).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&m).unwrap();
    assert_eq!(int(&e, "f", 41), 42);
}

#[test]
fn switch_round_trips_and_folds() {
    use achainsaw_ir::ast::Terminator;
    use achainsaw_ir::binary::{decode_module as decode, encode_module as encode};
    let module = parse_and_validate(PROGRAM).unwrap();
    let back = decode(&encode(&module).unwrap()).unwrap();
    assert_eq!(
        achainsaw_ir::binary::to_air_text(&back),
        achainsaw_ir::binary::to_air_text(&module)
    );
    // A switch on a constant becomes a jump to its case, matched at the value's width.
    let src = "fn f()->i64\n  b0:\n    k = cst -1:i8\n    switch k, a, 255: b\n  a:\n    ret 1:i64\n  b:\n    ret 2:i64\n";
    let mut m = parse_and_validate(src).unwrap();
    achainsaw_ir::opt::optimize_module(&mut m);
    let f = &m.functions[0];
    assert!(
        f.blocks
            .iter()
            .all(|b| !matches!(b.terminator, Terminator::Switch { .. })),
        "{}",
        achainsaw_ir::binary::to_air_text(&m)
    );
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&m).unwrap();
    assert_eq!(
        unsafe { e.call_typed("f", &[]) }.unwrap(),
        Some(RtValue::I64(2))
    );
}

/// x86-64 and AArch64 objects (SVE too) on Cranelift and, with the `llvm` feature, LLVM.
#[test]
fn switch_compiles_for_every_target() {
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
