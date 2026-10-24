//! Atomic accesses on the backend selected by `ACHAINSAW_BACKEND` (Cranelift by default):
//! every read-modify-write op, `cas`, `ld` and `st` at every integer width, against a
//! scalar reference. `parallel.rs` covers them under contention and `sandbox.rs` their
//! bounds and alignment checks.

use achainsaw_codegen::JitEngine;
use achainsaw_ir::ast::AtomicOp;
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;

const TYPES: [Type; 4] = [Type::I8, Type::I16, Type::I32, Type::I64];

/// `fn f(p:ptr, out:ptr, a:i64, b:i64)` operating on the integer at `p`.
type Kernel = extern "C" fn(*mut u8, *mut i64, i64, i64);

/// AIR that narrows `a` and `b` to `ty` as `x` and `y` and stores `x` at `p`.
fn prologue(ty: Type) -> String {
    if ty == Type::I64 {
        "    x = add a, 0:i64\n    y = add b, 0:i64\n    st p, x\n".into()
    } else {
        format!("    x = trunc a:{ty}\n    y = trunc b:{ty}\n    st p, x\n")
    }
}

/// AIR that stores `r` (of type `ty`), sign-extended, at `out[k]`.
fn put(ty: Type, r: &str, k: usize) -> String {
    let wide = if ty == Type::I64 {
        format!("    w{k} = add {r}, 0:i64\n")
    } else {
        format!("    w{k} = sext {r}:i64\n")
    };
    format!("{wide}    st out[{k}:i64], w{k}\n")
}

/// Each op as a function that leaves the old value in `out[0]` and the new one in `out[1]`;
/// `cas` swaps in `b` when the value is `a` (always, here) and when it is `b` (only if
/// `a == b`), `ld` and `st` read and write `y`.
fn source() -> String {
    let mut src = String::new();
    for ty in TYPES {
        let head = |name: &str| format!("fn {name}_{ty}(p:ptr, out:ptr, a:i64, b:i64)\n  b0:\n");
        for op in AtomicOp::ALL {
            src += &head(op.as_str());
            src += &prologue(ty);
            src += &format!("    o = atomic {} p, y\n    n = ld p:{ty}\n", op.as_str());
            src += &put(ty, "o", 0);
            src += &put(ty, "n", 1);
            src += "    ret\n\n";
        }
        src += &head("cas");
        src += &prologue(ty);
        src += "    o = atomic cas p, x, y\n    n = ld p:";
        src += &format!("{ty}\n    o2 = atomic cas p, x, x\n    n2 = atomic ld p:{ty}\n");
        for (k, r) in ["o", "n", "o2", "n2"].iter().enumerate() {
            src += &put(ty, r, k);
        }
        src += "    ret\n\n";
        src += &head("ldst");
        src += &prologue(ty);
        src += &format!("    atomic st p, y\n    v = atomic ld p:{ty}\n");
        src += &put(ty, "v", 0);
        src += "    ret\n\n";
    }
    src
}

/// `v` wrapped to the width of `ty` and sign-extended.
fn wrap(ty: Type, v: i64) -> i64 {
    ty.wrap_int(v)
}

fn unsigned(ty: Type, v: i64) -> u64 {
    let bits = ty.bit_width().unwrap();
    if bits == 64 {
        v as u64
    } else {
        v as u64 & ((1u64 << bits) - 1)
    }
}

fn reference(op: AtomicOp, ty: Type, x: i64, y: i64) -> i64 {
    let r = match op {
        AtomicOp::Add => x.wrapping_add(y),
        AtomicOp::Sub => x.wrapping_sub(y),
        AtomicOp::And => x & y,
        AtomicOp::Or => x | y,
        AtomicOp::Xor => x ^ y,
        AtomicOp::Xchg => y,
        AtomicOp::Min => x.min(y),
        AtomicOp::Max => x.max(y),
        AtomicOp::Umin => unsigned(ty, x).min(unsigned(ty, y)) as i64,
        AtomicOp::Umax => unsigned(ty, x).max(unsigned(ty, y)) as i64,
    };
    wrap(ty, r)
}

fn values() -> Vec<i64> {
    let mut v = vec![
        0,
        1,
        -1,
        2,
        -2,
        0x7f,
        0x80,
        0x7fff,
        -0x8000,
        i32::MAX as i64,
    ];
    v.extend([i32::MIN as i64, i64::MAX, i64::MIN, 0x1234_5678_9abc_def0]);
    v
}

#[test]
fn atomic_ops_match_reference_at_every_width() {
    let module = parse_and_validate(&source()).unwrap_or_else(|d| panic!("{d:?}"));
    let mut engine = JitEngine::new().unwrap();
    engine.compile_module(&module).unwrap();
    let kernel =
        |name: &str| -> Kernel { unsafe { std::mem::transmute(engine.get_fn_ptr(name).unwrap()) } };
    for ty in TYPES {
        let ops: Vec<(AtomicOp, Kernel)> = AtomicOp::ALL
            .iter()
            .map(|&op| (op, kernel(&format!("{}_{ty}", op.as_str()))))
            .collect();
        let (cas, ldst) = (kernel(&format!("cas_{ty}")), kernel(&format!("ldst_{ty}")));
        for &a in &values() {
            for &b in &values() {
                let (x, y) = (wrap(ty, a), wrap(ty, b));
                // 8-byte aligned storage for the integer.
                let mut cell = [0u64; 2];
                let p = cell.as_mut_ptr() as *mut u8;
                let mut out = [0i64; 4];
                for &(op, f) in &ops {
                    f(p, out.as_mut_ptr(), a, b);
                    let want = [x, reference(op, ty, x, y)];
                    assert_eq!(out[..2], want, "atomic {} {ty}: {x}, {y}", op.as_str());
                }
                cas(p, out.as_mut_ptr(), a, b);
                // The first cas finds x and stores y; the second expects x, so it swaps
                // (storing x) only when y == x.
                assert_eq!(
                    out,
                    [x, y, y, if x == y { x } else { y }],
                    "cas {ty}: {x}, {y}"
                );
                ldst(p, out.as_mut_ptr(), a, b);
                assert_eq!(out[0], y, "ld/st {ty}: {y}");
            }
        }
    }
}

#[test]
fn atomics_take_indexed_operands() {
    let src = r#"
fn bump(p:ptr, i:i64)->i32
  b0:
    o = atomic add p[i], 3:i32
    s = atomic sub p[i], 1:i32
    c = atomic cas p[i], 0:i32, 7:i32
    atomic st p[i:i8], 1:i8
    v = atomic ld p[i]:i32
    ret v
"#;
    let module = parse_and_validate(src).unwrap();
    let mut engine = JitEngine::new().unwrap();
    engine.compile_module(&module).unwrap();
    let f: extern "C" fn(*mut i32, i64) -> i32 =
        unsafe { std::mem::transmute(engine.get_fn_ptr("bump").unwrap()) };
    let mut cells = [10i32, 20, 30, 40];
    // p[2] = 30 + 3 - 1 = 32; cas fails (not 0); then byte 2 (in p[0]) is set to 1.
    assert_eq!(f(cells.as_mut_ptr(), 2), 32);
    assert_eq!(cells, [10 | (1 << 16), 20, 32, 40]);
}

/// x86-64 and AArch64 objects, with and without LSE's single-instruction atomics, on
/// Cranelift and (with the `llvm` feature) LLVM.
#[test]
fn atomics_compile_for_every_target() {
    use achainsaw_codegen::{compile_object, AotTarget, Backend, CodegenOptions};
    let module = parse_and_validate(&source()).unwrap();
    let mut backends = vec![Backend::Cranelift];
    if cfg!(feature = "llvm") {
        backends.push(Backend::Llvm);
    }
    for (triple, cpu, features) in [
        ("x86_64-unknown-linux-gnu", "x86-64", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
        ("aarch64-unknown-linux-gnu", "generic", ""),
        ("aarch64-unknown-linux-gnu", "generic", "+lse"),
    ] {
        let target = AotTarget {
            triple: Some(triple.into()),
            cpu: Some(cpu.into()),
            features: (!features.is_empty()).then(|| features.into()),
        };
        for &backend in &backends {
            let obj = compile_object(&module, &target, backend, &CodegenOptions::default())
                .unwrap_or_else(|e| panic!("{triple} {cpu} {features} {backend:?}: {e}"));
            assert!(!obj.bytes.is_empty());
        }
    }
}
