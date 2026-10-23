//! Shadowed register names (`achainsaw_ir::names`) on the backend selected by
//! `ACHAINSAW_BACKEND`: block parameters that reuse a dominating block's names, names defined
//! again later in a block, a shadowed function parameter, and an inline function whose names
//! match its caller's, all compute what the same program with distinct names computes. Also
//! the spellings agents write that the parser accepts: op mnemonics as register names (`rem`)
//! and integer literals of float type (`0:f32`).

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

const SRC: &str = r#"
# Sum of x[i] * s for i < n, in blocks of 4 and then one at a time.
fn sum_scaled(x:ptr, n:i64, s:f32)->f32
  b0:
    acc = cst 0:f32
    jmp blocks(0:i64, acc)
  blocks(i:i64, acc:f32):
    end = add i, 4:i64
    more = le end, n
    br more, body(i, acc), tail(i, acc)
  body(i:i64, acc:f32):
    a = ld x[i]:f32
    i = add i, 1:i64
    b = ld x[i]:f32
    i = add i, 1:i64
    c = ld x[i]:f32
    i = add i, 1:i64
    d = ld x[i]:f32
    i = add i, 1:i64
    ab = add a, b
    cd = add c, d
    abcd = add ab, cd
    acc = fma abcd, s, acc
    jmp blocks(i, acc)
  tail(i:i64, acc:f32):
    rem = sub n, i
    more = gt rem, 0:i64
    br more, one(i, acc), done(acc)
  one(i:i64, acc:f32):
    a = ld x[i]:f32
    acc = fma a, s, acc
    i = add i, 1:i64
    jmp tail(i, acc)
  done(acc:f32):
    ret acc

# Shadows its parameter.
fn plus_two(n:i64)->i64
  b0:
    n = add n, 1:i64
    n = add n, 1:i64
    ret n

# Its names (i, acc, more) are the caller's names too.
inline fn tri(n:i64)->i64
  b0:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body(i, acc), done(acc)
  body(i:i64, acc:i64):
    acc = add acc, i
    i = add i, 1:i64
    jmp loop(i, acc)
  done(acc:i64):
    ret acc

fn tri_sum(n:i64)->i64
  b0:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = le i, n
    br more, body(i, acc), done(acc)
  body(i:i64, acc:i64):
    t = call tri(i)
    acc = add acc, t
    i = add i, 1:i64
    jmp loop(i, acc)
  done(acc:i64):
    ret acc
"#;

fn engine() -> JitEngine {
    let module = parse_and_validate(SRC).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    e
}

#[test]
fn shadowed_names_compute_what_distinct_names_compute() {
    let e = engine();
    let f: extern "C" fn(*const f32, i64, f32) -> f32 =
        unsafe { std::mem::transmute(e.get_fn_ptr("sum_scaled").unwrap()) };
    for n in [0usize, 1, 3, 4, 5, 8, 11] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let want: f32 = x.iter().sum::<f32>() * 0.5;
        assert_eq!(f(x.as_ptr(), n as i64, 0.5), want, "n={n}");
    }
    let got = unsafe { e.call_typed("plus_two", &[RtValue::I64(40)]) }.unwrap();
    assert_eq!(got, Some(RtValue::I64(42)));
    for n in [0i64, 1, 5, 20] {
        let got = unsafe { e.call_typed("tri_sum", &[RtValue::I64(n)]) }.unwrap();
        let want: i64 = (0..=n).map(|k| k * (k - 1) / 2).sum();
        assert_eq!(got, Some(RtValue::I64(want)), "n={n}");
    }
}

#[test]
fn shadowed_names_compile_ahead_of_time() {
    use achainsaw_codegen::{compile_object, AotTarget, Backend};
    let module = parse_and_validate(SRC).unwrap();
    let mut backends = vec![Backend::Cranelift];
    if cfg!(feature = "llvm") {
        backends.push(Backend::Llvm);
    }
    for backend in backends {
        compile_object(&module, &AotTarget::default(), backend, &Default::default())
            .unwrap_or_else(|e| panic!("{backend}: {e}"));
    }
}

#[test]
fn optimizer_keeps_shadowed_names_apart() {
    let mut module = parse_and_validate(SRC).unwrap();
    achainsaw_ir::optimize_module(&mut module);
    let text = achainsaw_ir::to_air_text(&module);
    parse_and_validate(&text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    let got = unsafe { e.call_typed("plus_two", &[RtValue::I64(1)]) }.unwrap();
    assert_eq!(got, Some(RtValue::I64(3)));
}
