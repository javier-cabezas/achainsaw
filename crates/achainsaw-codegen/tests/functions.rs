//! Functions with several results and `inline fn`, on the backend selected by
//! `ACHAINSAW_BACKEND`: results of every kind (scalars, fixed vectors, `vx`) cross calls in
//! order, inlined bodies with loops and branches compute what calls compute, inside `par`
//! bodies too, and fuel still stops runaway loops in inlined code.

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

const SRC: &str = r#"
fn split(x:i64, v:v256, w:vx)->(i64, v256, vx, f32)
  b0:
    h = shr x, 1:i64
    d = vadd v, v:i32
    t = vmul w, w:f32
    e = extlane w, 0:f32
    ret h, d, t, e

fn use_split(x:i64, p:ptr, q:ptr)->f32
  b0:
    v = ld p:v256
    w = ld q:vx
    h, d, t, e = call split(x, v, w)
    st p, d
    st q, t
    hf = itof h:f32
    r = add hf, e
    ret r

# Sum of i * s for i < n (a loop and a branch inside an inline function).
inline fn tri(n:i64, s:i64)->(i64, i64)
  b0:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    t = mul i, s
    acc2 = add acc, t
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc, n

fn tri_twice(n:i64)->i64
  b0:
    a, k = call tri(n, 1:i64)
    b, _k2 = call tri(k, 2:i64)
    s = add a, b
    ret s

fn tri_task(i:i64, out:ptr)
  b0:
    v, _n = call tri(i, 3:i64)
    st out[i], v
    ret

fn tri_par(out:ptr, n:i64)
  b0:
    par n, tri_task(out)
    ret

fn spin_inline(n:i64)->i64
  b0:
    a, _b = call tri(n, 1:i64)
    ret a
"#;

fn engine() -> JitEngine {
    let module = parse_and_validate(SRC).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    e
}

#[test]
fn several_results_cross_calls() {
    let e = engine();
    let vx_lanes = e.vx_bits() as usize / 32;
    let mut v: Vec<i32> = (1..=8).collect();
    let mut w: Vec<f32> = (0..vx_lanes).map(|i| i as f32 + 0.5).collect();
    let f: extern "C" fn(i64, *mut i32, *mut f32) -> f32 =
        unsafe { std::mem::transmute(e.get_fn_ptr("use_split").unwrap()) };
    let r = f(41, v.as_mut_ptr(), w.as_mut_ptr());
    assert_eq!(r, 20.0 + 0.5);
    assert_eq!(v, (1..=8).map(|x| 2 * x).collect::<Vec<_>>());
    for (i, x) in w.iter().enumerate() {
        assert_eq!(*x, (i as f32 + 0.5) * (i as f32 + 0.5));
    }
    // Hosts cannot call a function with several results directly.
    let err = unsafe { e.call_typed("split", &[]) }
        .unwrap_err()
        .to_string();
    assert!(err.contains("only be called from AIR"), "{err}");
}

#[test]
fn inlined_bodies_compute_what_calls_compute() {
    let e = engine();
    for n in [0i64, 1, 5, 100] {
        let got = unsafe { e.call_typed("tri_twice", &[RtValue::I64(n)]) }.unwrap();
        let tri = |s: i64| (0..n).map(|i| i * s).sum::<i64>();
        assert_eq!(got, Some(RtValue::I64(tri(1) + tri(2))), "n={n}");
    }
    let f: extern "C" fn(*mut i64, i64) =
        unsafe { std::mem::transmute(e.get_fn_ptr("tri_par").unwrap()) };
    let mut out = vec![-1i64; 64];
    f(out.as_mut_ptr(), 64);
    for (i, v) in out.iter().enumerate() {
        let i = i as i64;
        assert_eq!(*v, 3 * i * (i - 1) / 2, "task {i}");
    }
}

#[test]
fn fuel_stops_loops_in_inlined_code() {
    let mut e = engine();
    e.set_fuel(Some(10_000));
    let err = unsafe { e.call_typed("spin_inline", &[RtValue::I64(i64::MAX)]) }
        .unwrap_err()
        .to_string();
    assert!(err.contains("ERR_OUT_OF_FUEL"), "{err}");
    e.set_fuel(None);
}

/// Several results of mixed kinds (a scalable `vx` with scalars on SVE) compile for every
/// target, on both backends.
#[test]
fn several_results_compile_for_every_target() {
    use achainsaw_codegen::{compile_object, AotTarget, Backend};
    let module = parse_and_validate(SRC).unwrap();
    let mut targets = vec![
        ("x86_64-unknown-linux-gnu", "x86-64", "", Backend::Cranelift),
        (
            "aarch64-unknown-linux-gnu",
            "generic",
            "",
            Backend::Cranelift,
        ),
    ];
    if cfg!(feature = "llvm") {
        targets.extend([
            ("x86_64-unknown-linux-gnu", "x86-64-v4", "", Backend::Llvm),
            ("aarch64-unknown-linux-gnu", "generic", "", Backend::Llvm),
            (
                "aarch64-unknown-linux-gnu",
                "generic",
                "+sve",
                Backend::Llvm,
            ),
            (
                "aarch64-unknown-linux-gnu",
                "generic",
                "+sve2",
                Backend::Llvm,
            ),
        ]);
    }
    for (triple, cpu, features, backend) in targets {
        let target = AotTarget {
            triple: Some(triple.into()),
            cpu: Some(cpu.into()),
            features: (!features.is_empty()).then(|| features.into()),
        };
        compile_object(&module, &target, backend, &Default::default())
            .unwrap_or_else(|e| panic!("{backend} {triple} {cpu} {features}: {e}"));
    }
}
