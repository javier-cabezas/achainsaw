//! AIR v2 vector ops on the backend selected by `ACHAINSAW_BACKEND` (Cranelift by default).
//!
//! Every op x lane type x width combination the validator accepts must:
//! - compile for x86_64 (each ISA level this host reaches via the JIT, and every x86-64
//!   micro-architecture level via AOT) and for aarch64 (NEON, and SVE on LLVM), and
//! - produce exactly the results of a scalar reference model, lane by lane, at the
//!   engine's `vx` width.
//!
//! With the `llvm` feature, fixed-width results must also match bit for bit between the
//! two backends.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::{AotCompiler, AotTarget, JitEngine};
use achainsaw_ir::ast::{VBinOp, VCmpOp, VectorReduceOp};
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;
use achainsaw_ir::validator::{vbin_lane_types, VFMA_LANE_TYPES};

const WIDTHS: [Type; 4] = [Type::V128, Type::V256, Type::V512, Type::Vx];
const LANES: [Type; 6] = [
    Type::I8,
    Type::I16,
    Type::I32,
    Type::I64,
    Type::F32,
    Type::F64,
];
const REDUCE_OPS: [VectorReduceOp; 3] = [
    VectorReduceOp::Sum,
    VectorReduceOp::Max,
    VectorReduceOp::Min,
];
/// Input/output buffers are sized for the widest vector: a 2048-bit SVE `vx`.
const BUF: usize = 256;
const TRIALS: u64 = 12;

/// Bytes a vector occupies; `vx` is `vx_bytes` wide on the engine under test.
fn width_bytes(w: Type, vx_bytes: usize) -> usize {
    w.bit_width().map_or(vx_bytes, |b| b as usize / 8)
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Bin(VBinOp, Type, Type),
    Cmp(VCmpOp, Type, Type),
    Fma(Type, Type),
    Sel(Type),
    Reduce(VectorReduceOp, Type, Type),
    Splat(Type, Type),
}

impl Kernel {
    fn name(&self) -> String {
        match self {
            Kernel::Bin(op, l, w) => format!("k_{}_{l}_{w}", op.as_str()),
            Kernel::Cmp(op, l, w) => format!("k_{}_{l}_{w}", op.as_str()),
            Kernel::Fma(l, w) => format!("k_vfma_{l}_{w}"),
            Kernel::Sel(w) => format!("k_vsel_{w}"),
            Kernel::Reduce(op, l, w) => format!("k_{}_{l}_{w}", op.as_str()),
            Kernel::Splat(l, w) => format!("k_splat_{l}_{w}"),
        }
    }

    /// `fn k(pa:ptr, pb:ptr, pc:ptr, po:ptr)`: loads operands, applies the op, stores.
    fn air(&self) -> String {
        let loads = |w: Type| format!("    a = ld pa:{w}\n    b = ld pb:{w}\n    c = ld pc:{w}\n");
        let body = match self {
            Kernel::Bin(op, l, w) => format!("{}    r = {} a, b:{l}\n", loads(*w), op.as_str()),
            Kernel::Cmp(op, l, w) => format!("{}    r = {} a, b:{l}\n", loads(*w), op.as_str()),
            Kernel::Fma(l, w) => format!("{}    r = vfma a, b, c:{l}\n", loads(*w)),
            Kernel::Sel(w) => format!("{}    r = vsel c, a, b\n", loads(*w)),
            Kernel::Reduce(op, l, w) => format!("{}    r = {} a:{l}\n", loads(*w), op.as_str()),
            Kernel::Splat(l, w) => format!("    s = ld pa:{l}\n    r = splat s:{w}\n"),
        };
        format!(
            "fn {}(pa:ptr, pb:ptr, pc:ptr, po:ptr)\n  b0:\n{body}    st po, r\n    ret\n",
            self.name()
        )
    }

    fn lane(&self) -> Type {
        match self {
            Kernel::Bin(_, l, _)
            | Kernel::Cmp(_, l, _)
            | Kernel::Fma(l, _)
            | Kernel::Reduce(_, l, _)
            | Kernel::Splat(l, _) => *l,
            Kernel::Sel(_) => Type::I8,
        }
    }

    fn width(&self) -> Type {
        match self {
            Kernel::Bin(_, _, w)
            | Kernel::Cmp(_, _, w)
            | Kernel::Fma(_, w)
            | Kernel::Sel(w)
            | Kernel::Reduce(_, _, w)
            | Kernel::Splat(_, w) => *w,
        }
    }

    /// Bytes of output to compare.
    fn out_bytes(&self, vx_bytes: usize) -> usize {
        match self {
            Kernel::Reduce(_, l, _) => l.byte_size(),
            _ => width_bytes(self.width(), vx_bytes),
        }
    }

    /// Whether output lanes are floats (compared NaN-tolerantly).
    fn float_output(&self) -> bool {
        match self {
            Kernel::Bin(op, l, _) => l.is_float() && !op.is_bitwise(),
            Kernel::Fma(..) | Kernel::Reduce(..) | Kernel::Splat(..) => self.lane().is_float(),
            Kernel::Cmp(..) | Kernel::Sel(..) => false,
        }
    }
}

fn all_kernels() -> Vec<Kernel> {
    let mut ks = Vec::new();
    for w in WIDTHS {
        for op in VBinOp::ALL {
            for &l in vbin_lane_types(op) {
                ks.push(Kernel::Bin(op, l, w));
            }
        }
        for op in VCmpOp::ALL {
            for l in LANES {
                ks.push(Kernel::Cmp(op, l, w));
            }
        }
        for &l in VFMA_LANE_TYPES {
            ks.push(Kernel::Fma(l, w));
        }
        ks.push(Kernel::Sel(w));
        for op in REDUCE_OPS {
            for l in LANES {
                ks.push(Kernel::Reduce(op, l, w));
            }
        }
        for l in LANES {
            ks.push(Kernel::Splat(l, w));
        }
    }
    ks
}

fn module_source(kernels: &[Kernel]) -> String {
    kernels
        .iter()
        .map(Kernel::air)
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------------------
// Scalar reference model
// ---------------------------------------------------------------------------------------

fn get_int(buf: &[u8], lane: Type, i: usize) -> i64 {
    let n = lane.byte_size();
    let b = &buf[i * n..i * n + n];
    match n {
        1 => b[0] as i8 as i64,
        2 => i16::from_le_bytes([b[0], b[1]]) as i64,
        4 => i32::from_le_bytes(b.try_into().unwrap()) as i64,
        _ => i64::from_le_bytes(b.try_into().unwrap()),
    }
}

fn put_int(buf: &mut [u8], lane: Type, i: usize, v: i64) {
    let n = lane.byte_size();
    buf[i * n..i * n + n].copy_from_slice(&v.to_le_bytes()[..n]);
}

fn get_f32(buf: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap())
}

fn get_f64(buf: &[u8], i: usize) -> f64 {
    f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap())
}

/// Wasm/Cranelift `fmin`/`fmax`: NaN if either input is NaN; -0.0 orders below +0.0.
macro_rules! wasm_minmax {
    ($name:ident, $t:ty, $min:expr) => {
        fn $name(a: $t, b: $t) -> $t {
            if a.is_nan() || b.is_nan() {
                <$t>::NAN
            } else if a == b {
                if a.is_sign_negative() == $min {
                    a
                } else {
                    b
                }
            } else if (a < b) == $min {
                a
            } else {
                b
            }
        }
    };
}
wasm_minmax!(fmin32, f32, true);
wasm_minmax!(fmax32, f32, false);
wasm_minmax!(fmin64, f64, true);
wasm_minmax!(fmax64, f64, false);

fn int_bin(op: VBinOp, lane: Type, a: i64, b: i64) -> i64 {
    let r = match op {
        VBinOp::Add => a.wrapping_add(b),
        VBinOp::Sub => a.wrapping_sub(b),
        VBinOp::Mul => a.wrapping_mul(b),
        VBinOp::Min => a.min(b),
        VBinOp::Max => a.max(b),
        VBinOp::And => a & b,
        VBinOp::Or => a | b,
        VBinOp::Xor => a ^ b,
        VBinOp::Div => unreachable!("integer vdiv is rejected by the validator"),
    };
    lane.wrap_int(r)
}

macro_rules! float_bin {
    ($name:ident, $t:ty, $min:ident, $max:ident) => {
        fn $name(op: VBinOp, a: $t, b: $t) -> $t {
            match op {
                VBinOp::Add => a + b,
                VBinOp::Sub => a - b,
                VBinOp::Mul => a * b,
                VBinOp::Div => a / b,
                VBinOp::Min => $min(a, b),
                VBinOp::Max => $max(a, b),
                _ => unreachable!(),
            }
        }
    };
}
float_bin!(f32_bin, f32, fmin32, fmax32);
float_bin!(f64_bin, f64, fmin64, fmax64);

fn cmp<T: PartialOrd>(op: VCmpOp, a: T, b: T) -> bool {
    match op {
        VCmpOp::Eq => a == b,
        VCmpOp::Ne => a != b,
        VCmpOp::Lt => a < b,
        VCmpOp::Gt => a > b,
        VCmpOp::Le => a <= b,
        VCmpOp::Ge => a >= b,
    }
}

/// Recursive-halves reduction: `reduce(v) = op(reduce(lo), reduce(hi))`.
fn tree<T: Copy>(vals: &[T], f: &impl Fn(T, T) -> T) -> T {
    if vals.len() == 1 {
        return vals[0];
    }
    let (lo, hi) = vals.split_at(vals.len() / 2);
    f(tree(lo, f), tree(hi, f))
}

fn reference(k: Kernel, a: &[u8], b: &[u8], c: &[u8], vx_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; BUF];
    let w = width_bytes(k.width(), vx_bytes);
    let lane = k.lane();
    let n = w / lane.byte_size();
    match k {
        Kernel::Bin(op, Type::F32, _) if !op.is_bitwise() => {
            for i in 0..n {
                let r = f32_bin(op, get_f32(a, i), get_f32(b, i));
                out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
            }
        }
        Kernel::Bin(op, Type::F64, _) if !op.is_bitwise() => {
            for i in 0..n {
                let r = f64_bin(op, get_f64(a, i), get_f64(b, i));
                out[i * 8..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
            }
        }
        Kernel::Bin(op, l, _) => {
            for i in 0..n {
                put_int(
                    &mut out,
                    l,
                    i,
                    int_bin(op, l, get_int(a, l, i), get_int(b, l, i)),
                );
            }
        }
        Kernel::Cmp(op, l, _) => {
            for i in 0..n {
                let t = match l {
                    Type::F32 => cmp(op, get_f32(a, i), get_f32(b, i)),
                    Type::F64 => cmp(op, get_f64(a, i), get_f64(b, i)),
                    _ => cmp(op, get_int(a, l, i), get_int(b, l, i)),
                };
                put_int(
                    &mut out,
                    if l.is_float() { int_of(l) } else { l },
                    i,
                    -(t as i64),
                );
            }
        }
        Kernel::Fma(Type::F32, _) => {
            for i in 0..n {
                let r = get_f32(a, i).mul_add(get_f32(b, i), get_f32(c, i));
                out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
            }
        }
        Kernel::Fma(_, _) => {
            for i in 0..n {
                let r = get_f64(a, i).mul_add(get_f64(b, i), get_f64(c, i));
                out[i * 8..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
            }
        }
        Kernel::Sel(_) => {
            for i in 0..w {
                out[i] = (c[i] & a[i]) | (!c[i] & b[i]);
            }
        }
        Kernel::Reduce(op, Type::F32, _) => {
            let vals: Vec<f32> = (0..n).map(|i| get_f32(a, i)).collect();
            let r = match op {
                VectorReduceOp::Sum => tree(&vals, &|x, y| x + y),
                VectorReduceOp::Max => tree(&vals, &fmax32),
                VectorReduceOp::Min => tree(&vals, &fmin32),
            };
            out[..4].copy_from_slice(&r.to_le_bytes());
        }
        Kernel::Reduce(op, Type::F64, _) => {
            let vals: Vec<f64> = (0..n).map(|i| get_f64(a, i)).collect();
            let r = match op {
                VectorReduceOp::Sum => tree(&vals, &|x, y| x + y),
                VectorReduceOp::Max => tree(&vals, &fmax64),
                VectorReduceOp::Min => tree(&vals, &fmin64),
            };
            out[..8].copy_from_slice(&r.to_le_bytes());
        }
        Kernel::Reduce(op, l, _) => {
            let vals: Vec<i64> = (0..n).map(|i| get_int(a, l, i)).collect();
            let r = match op {
                VectorReduceOp::Sum => tree(&vals, &|x, y| l.wrap_int(x.wrapping_add(y))),
                VectorReduceOp::Max => tree(&vals, &|x: i64, y| x.max(y)),
                VectorReduceOp::Min => tree(&vals, &|x: i64, y| x.min(y)),
            };
            put_int(&mut out, l, 0, r);
        }
        Kernel::Splat(l, _) => {
            let s = l.byte_size();
            for i in 0..n {
                out[i * s..i * s + s].copy_from_slice(&a[..s]);
            }
        }
    }
    out
}

/// Integer type with the same width as a float lane (comparison mask lanes).
fn int_of(lane: Type) -> Type {
    if lane == Type::F32 {
        Type::I32
    } else {
        Type::I64
    }
}

// ---------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

const F32_SPECIAL: [f32; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE,
    1.0e-45, // smallest denormal
    f32::MAX,
    3.25,
    -7.0e-3,
];
const F64_SPECIAL: [f64; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::MIN_POSITIVE,
    5.0e-324,
    f64::MAX,
    3.25,
    -7.0e-3,
];

/// Random float with a random sign and exponent in [-20, 20] and full mantissa, so sums
/// round differently depending on evaluation order.
fn varied(rng: &mut Rng) -> f64 {
    let m = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    let e = (rng.next() % 41) as i32 - 20;
    let v = (1.0 + m) * 2f64.powi(e);
    if rng.next() & 1 == 0 {
        v
    } else {
        -v
    }
}

/// Fills a buffer with lanes of `lane` type: special values in trial 0, a catastrophic
/// cancellation pattern in trial 2 (any change in reduction order changes the result),
/// then a mix of special values, varied-magnitude values, and random bit patterns.
fn fill(buf: &mut [u8], lane: Type, rng: &mut Rng, trial: u64) {
    let s = lane.byte_size();
    if trial == 2 && lane.is_float() {
        // [big, 1, -big, 1, ...] with big = 2^(mantissa bits + 2), so big + 1 == big.
        let big = if lane == Type::F32 {
            33554432.0
        } else {
            18014398509481984.0
        };
        for i in 0..BUF / s {
            let v: f64 = match i % 4 {
                0 => big,
                2 => -big,
                _ => 1.0,
            };
            if lane == Type::F32 {
                buf[i * 4..i * 4 + 4].copy_from_slice(&(v as f32).to_le_bytes());
            } else {
                buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
        }
        return;
    }
    for i in 0..BUF / s {
        let pick = rng.next();
        let special = trial == 0 || pick.is_multiple_of(3);
        let idx = (pick >> 8) as usize;
        match lane {
            Type::F32 => {
                let v = if special {
                    F32_SPECIAL[(i + idx) % F32_SPECIAL.len()]
                } else if pick % 3 == 1 || trial % 2 == 1 {
                    varied(rng) as f32
                } else {
                    f32::from_bits((pick >> 32) as u32)
                };
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            Type::F64 => {
                let v = if special {
                    F64_SPECIAL[(i + idx) % F64_SPECIAL.len()]
                } else if pick % 3 == 1 || trial % 2 == 1 {
                    varied(rng)
                } else {
                    f64::from_bits(rng.next())
                };
                buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            _ => {
                let bits = lane.bit_width().unwrap();
                let min = i64::MIN >> (64 - bits);
                let max = !min;
                let specials = [0, 1, -1, min, max, 2, -2];
                let v = if special {
                    specials[(i + idx) % specials.len()]
                } else {
                    lane.wrap_int(rng.next() as i64)
                };
                put_int(buf, lane, i, v);
            }
        }
    }
}

fn check_output(k: Kernel, expected: &[u8], actual: &[u8], vx_bytes: usize, ctx: &str) {
    let n = k.out_bytes(vx_bytes);
    if k.float_output() {
        let s = k.lane().byte_size();
        for i in 0..n / s {
            let (e, a) = if s == 4 {
                (get_f32(expected, i) as f64, get_f32(actual, i) as f64)
            } else {
                (get_f64(expected, i), get_f64(actual, i))
            };
            let eb = &expected[i * s..i * s + s];
            let ab = &actual[i * s..i * s + s];
            let ok = if e.is_nan() { a.is_nan() } else { eb == ab };
            assert!(
                ok,
                "{} lane {i}: expected {e:?}, got {a:?} ({ctx})",
                k.name()
            );
        }
    } else {
        assert_eq!(&expected[..n], &actual[..n], "{} ({ctx})", k.name());
    }
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

type KernelFn = extern "C" fn(*const u8, *const u8, *const u8, *mut u8);

fn host_levels() -> Vec<(IsaLevel, CpuFeatures)> {
    let host = CpuFeatures::host();
    IsaLevel::names()
        .iter()
        .map(|n| n.parse::<IsaLevel>().unwrap())
        .filter(|l| l.arch() == host.arch)
        .filter_map(|l| {
            let f = host.capped(l).ok()?;
            (f.max_level() == Some(l)).then_some((l, f))
        })
        .collect()
}

#[test]
fn vector_ops_match_reference_at_every_isa_level() {
    let kernels = all_kernels();
    let module = parse_and_validate(&module_source(&kernels)).expect("kernels validate");
    let levels = host_levels();
    assert!(!levels.is_empty());

    for (level, features) in levels {
        let mut engine = JitEngine::with_features(&features).expect("JIT init");
        engine
            .compile_module(&module)
            .unwrap_or_else(|e| panic!("compile at {level}: {e}"));
        let vx_bytes = engine.vx_bits() as usize / 8;
        let ctx = |trial| {
            format!(
                "ISA {level}, {} vx={vx_bytes}B, trial {trial}",
                engine.backend()
            )
        };

        for k in &kernels {
            let f: KernelFn = unsafe { std::mem::transmute(engine.get_fn_ptr(&k.name()).unwrap()) };
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ k.name().len() as u64);
            for trial in 0..TRIALS {
                let (mut a, mut b, mut c) = ([0u8; BUF], [0u8; BUF], [0u8; BUF]);
                fill(&mut a, k.lane(), &mut rng, trial);
                fill(&mut b, k.lane(), &mut rng, trial + 1);
                fill(&mut c, k.lane(), &mut rng, trial + 2);
                if trial == 1 {
                    b = a; // exercise equal operands (min/max of equal values, x - x)
                }
                let mut out = [0u8; BUF];
                f(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr());
                let expected = reference(*k, &a, &b, &c, vx_bytes);
                check_output(*k, &expected, &out, vx_bytes, &ctx(trial));
            }
        }
    }
}

#[test]
fn vector_ops_compile_for_every_target() {
    let kernels = all_kernels();
    let module = parse_and_validate(&module_source(&kernels)).expect("kernels validate");
    let targets = [
        ("x86_64-unknown-linux-gnu", "x86-64"),
        ("x86_64-unknown-linux-gnu", "x86-64-v2"),
        ("x86_64-unknown-linux-gnu", "x86-64-v3"),
        ("x86_64-unknown-linux-gnu", "x86-64-v4"),
        ("aarch64-unknown-linux-gnu", "generic"),
    ];
    for (triple, cpu) in targets {
        let mut compiler = AotCompiler::with_target(&AotTarget {
            triple: Some(triple.into()),
            cpu: Some(cpu.into()),
            features: None,
        })
        .unwrap();
        compiler
            .compile_module(&module)
            .unwrap_or_else(|e| panic!("{triple} {cpu}: {e}"));
        assert!(!compiler.finish().unwrap().is_empty());
    }
}

#[test]
fn extract_lane_reads_every_lane_of_wide_vectors() {
    // Store each lane of a v512 separately and compare with the input bytes.
    for lane in LANES {
        let n = 64 / lane.byte_size();
        let mut src = "fn k(pa:ptr, po:ptr)\n  b0:\n    a = ld pa:v512\n".to_string();
        for i in 0..n {
            src += &format!(
                "    e{i} = extlane a, {i}:{lane}\n    o{i} = add po, {}:i64\n    st o{i}, e{i}\n",
                i * lane.byte_size()
            );
        }
        src += "    ret\n";
        let module = parse_and_validate(&src).expect("valid");
        let mut engine = JitEngine::new().unwrap();
        engine.compile_module(&module).unwrap();
        let f: extern "C" fn(*const u8, *mut u8) =
            unsafe { std::mem::transmute(engine.get_fn_ptr("k").unwrap()) };
        let input: Vec<u8> = (0..64u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut out = [0u8; 64];
        f(input.as_ptr(), out.as_mut_ptr());
        assert_eq!(&input[..], &out[..], "lane type {lane}");
    }
}

#[test]
fn vl_and_wide_values_across_blocks_and_calls() {
    // v256 values flow through block parameters, a loop, and an internal call that takes
    // and returns a v512; vl reports the engine's vx lane count.
    let src = r#"
fn twice(v:v512)->v512
  b0:
    r = vadd v, v:f32
    ret r

fn sum_rows(p:ptr, rows:i64)->f32
  b0:
    z = cst 0.0:f32
    acc0 = splat z:v256
    jmp b1(0:i64, acc0)
  b1(i:i64, acc:v256):
    c = lt i, rows
    br c, b2, b3
  b2:
    off = mul i, 32:i64
    q = add p, off
    row = ld q:v256
    acc2 = vadd acc, row:f32
    i2 = add i, 1:i64
    jmp b1(i2, acc2)
  b3:
    s = vsum acc:f32
    wide = ld p:v512
    w2 = call twice(wide)
    t = vsum w2:f32
    r = add s, t
    ret r

fn lanes()->i64
  b0:
    a = vl f32
    b = vl i8
    r = add a, b
    ret r
"#;
    let module = parse_and_validate(src).expect("valid");
    let mut engine = JitEngine::new().unwrap();
    engine.compile_module(&module).unwrap();
    unsafe {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let f: extern "C" fn(*const f32, i64) -> f32 =
            std::mem::transmute(engine.get_fn_ptr("sum_rows").unwrap());
        // 4 rows of 8: sum(0..32) = 496; twice(first 16) = 2 * sum(0..16) = 240
        assert_eq!(f(data.as_ptr(), 4), 496.0 + 240.0);
        let g: extern "C" fn() -> i64 = std::mem::transmute(engine.get_fn_ptr("lanes").unwrap());
        let vx_bytes = engine.vx_bits() as i64 / 8;
        assert_eq!(g(), vx_bytes / 4 + vx_bytes);
    }
    // Functions with vector signatures have no host trampoline, with a clear error.
    let err = unsafe { engine.call_typed("twice", &[]) }
        .unwrap_err()
        .to_string();
    assert!(err.contains("only be called from AIR"), "{err}");
}

#[test]
fn unaligned_vector_loads_and_stores() {
    let src = r#"
fn shift(pa:ptr, po:ptr)
  b0:
    v = ld pa:v512
    w = vadd v, v:i32
    st po, w
    ret
"#;
    let module = parse_and_validate(src).unwrap();
    let mut engine = JitEngine::new().unwrap();
    engine.compile_module(&module).unwrap();
    let f: extern "C" fn(*const u8, *mut u8) =
        unsafe { std::mem::transmute(engine.get_fn_ptr("shift").unwrap()) };
    let input: Vec<u8> = (0..80u8).collect();
    for misalign in [1usize, 3, 4, 8, 12] {
        let mut out = [0u8; 80];
        f(input[misalign..].as_ptr(), out[misalign..].as_mut_ptr());
        for i in 0..16 {
            let o = misalign + i * 4;
            let x = i32::from_le_bytes(input[o..o + 4].try_into().unwrap());
            let y = i32::from_le_bytes(out[o..o + 4].try_into().unwrap());
            assert_eq!(y, x.wrapping_add(x), "misalign {misalign}, lane {i}");
        }
    }
}

/// Targets every LLVM build must handle, including scalable `vx` (SVE).
#[cfg(feature = "llvm")]
pub const LLVM_TARGETS: &[(&str, &str, &str)] = &[
    ("x86_64-unknown-linux-gnu", "x86-64", ""),
    ("x86_64-unknown-linux-gnu", "x86-64-v3", ""),
    ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
    ("aarch64-unknown-linux-gnu", "generic", ""),
    ("aarch64-unknown-linux-gnu", "generic", "+sve"),
    ("aarch64-unknown-linux-gnu", "generic", "+sve2"),
];

#[cfg(feature = "llvm")]
#[test]
fn vector_ops_compile_for_every_llvm_target() {
    use achainsaw_codegen::{compile_object, Backend};
    let kernels = all_kernels();
    let module = parse_and_validate(&module_source(&kernels)).expect("kernels validate");
    for (triple, cpu, features) in LLVM_TARGETS {
        let target = AotTarget {
            triple: Some((*triple).into()),
            cpu: Some((*cpu).into()),
            features: (!features.is_empty()).then(|| (*features).into()),
        };
        let obj = compile_object(&module, &target, Backend::Llvm)
            .unwrap_or_else(|e| panic!("{triple} {cpu} {features}: {e}"));
        assert!(!obj.bytes.is_empty());
    }
}

/// Fixed-width vectors must give bit-identical results on both backends (NaN payloads
/// aside), at every ISA level.
#[cfg(feature = "llvm")]
#[test]
fn fixed_width_vector_ops_agree_across_backends() {
    use achainsaw_codegen::Backend;
    let kernels: Vec<Kernel> = all_kernels()
        .into_iter()
        .filter(|k| k.width() != Type::Vx)
        .collect();
    let module = parse_and_validate(&module_source(&kernels)).expect("kernels validate");
    for (level, features) in host_levels() {
        let engines: Vec<JitEngine> = [Backend::Cranelift, Backend::Llvm]
            .into_iter()
            .map(|b| {
                let mut e = JitEngine::with_backend(b, &features).unwrap();
                e.compile_module(&module).unwrap();
                e
            })
            .collect();
        for k in &kernels {
            let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ k.name().len() as u64);
            for trial in 0..TRIALS {
                let (mut a, mut b, mut c) = ([0u8; BUF], [0u8; BUF], [0u8; BUF]);
                fill(&mut a, k.lane(), &mut rng, trial);
                fill(&mut b, k.lane(), &mut rng, trial + 1);
                fill(&mut c, k.lane(), &mut rng, trial + 2);
                let outs: Vec<[u8; BUF]> = engines
                    .iter()
                    .map(|e| {
                        let f: KernelFn =
                            unsafe { std::mem::transmute(e.get_fn_ptr(&k.name()).unwrap()) };
                        let mut out = [0u8; BUF];
                        f(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr());
                        out
                    })
                    .collect();
                check_output(
                    *k,
                    &outs[0],
                    &outs[1],
                    16,
                    &format!("cranelift vs llvm, ISA {level}, trial {trial}"),
                );
            }
        }
    }
}
