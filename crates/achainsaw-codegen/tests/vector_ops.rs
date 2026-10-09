//! Vector ops on the backend selected by `ACHAINSAW_BACKEND` (Cranelift by default).
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
use achainsaw_ir::ast::{vexp, VBinOp, VCmpOp, VShiftOp, VUnaryOp, VZipOp, VectorReduceOp};
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;
use achainsaw_ir::validator::{
    vbin_lane_types, vnarrow_source_lane, vunary_lane_types, VFMA_LANE_TYPES, VNARROW_LANE_TYPES,
    VPERM_LANE_TYPES, VSHIFT_LANE_TYPES,
};

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
    /// Result lane type, width.
    Unary(VUnaryOp, Type, Type),
    Narrow(Type, Type),
    /// Shift amount: the i32 at `pc`.
    Shift(VShiftOp, Type, Type),
    /// `vmin`/`vmax` compiled with `fast_math` (compare and select).
    FastBin(VBinOp, Type, Type),
    /// `vminr`/`vmaxr` compiled with `fast_math`.
    FastReduce(VectorReduceOp, Type, Type),
    Zip(VZipOp, Type, Type),
    /// Lane type, width, lane index.
    Dup(Type, Type, u32),
    /// `vdot c, a, b:i8`.
    Dot(Type),
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
            Kernel::Unary(op, l, w) => format!("k_{}_{l}_{w}", op.as_str()),
            Kernel::Narrow(l, w) => format!("k_vnarrow_{l}_{w}"),
            Kernel::Shift(op, l, w) => format!("k_{}_{l}_{w}", op.as_str()),
            Kernel::FastBin(op, l, w) => format!("k_fast_{}_{l}_{w}", op.as_str()),
            Kernel::FastReduce(op, l, w) => format!("k_fast_{}_{l}_{w}", op.as_str()),
            Kernel::Zip(op, l, w) => format!("k_{}_{l}_{w}", op.as_str()),
            Kernel::Dup(l, w, i) => format!("k_vdup{i}_{l}_{w}"),
            Kernel::Dot(w) => format!("k_vdot_{w}"),
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
            Kernel::Unary(op, l, w) => format!("{}    r = {} a:{l}\n", loads(*w), op.as_str()),
            Kernel::Narrow(l, w) => format!("{}    r = vnarrow a, b:{l}\n", loads(*w)),
            Kernel::Shift(op, l, w) => format!(
                "    a = ld pa:{w}\n    n = ld pc:i32\n    r = {} a, n:{l}\n",
                op.as_str()
            ),
            Kernel::FastBin(op, l, w) => format!("{}    r = {} a, b:{l}\n", loads(*w), op.as_str()),
            Kernel::FastReduce(op, l, w) => format!("{}    r = {} a:{l}\n", loads(*w), op.as_str()),
            Kernel::Zip(op, l, w) => format!("{}    r = {} a, b:{l}\n", loads(*w), op.as_str()),
            Kernel::Dup(l, w, i) => format!("{}    r = vdup a, {i}:{l}\n", loads(*w)),
            Kernel::Dot(w) => format!("{}    r = vdot c, a, b:i8\n", loads(*w)),
        };
        format!(
            "fn {}(pa:ptr, pb:ptr, pc:ptr, po:ptr)\n  b0:\n{body}    st po, r\n    ret\n",
            self.name()
        )
    }

    /// Whether the kernel must be compiled with `CodegenOptions::fast_math`.
    fn fast_math(&self) -> bool {
        matches!(self, Kernel::FastBin(..) | Kernel::FastReduce(..))
    }

    fn lane(&self) -> Type {
        match self {
            Kernel::Bin(_, l, _)
            | Kernel::Cmp(_, l, _)
            | Kernel::Fma(l, _)
            | Kernel::Reduce(_, l, _)
            | Kernel::Splat(l, _)
            | Kernel::Unary(_, l, _)
            | Kernel::Narrow(l, _)
            | Kernel::Shift(_, l, _)
            | Kernel::FastBin(_, l, _)
            | Kernel::FastReduce(_, l, _)
            | Kernel::Zip(_, l, _)
            | Kernel::Dup(l, _, _) => *l,
            Kernel::Sel(_) | Kernel::Dot(_) => Type::I8,
        }
    }

    /// Lane type of the results: `lane()`, except for `vfwiden*` (f32).
    fn out_lane(&self) -> Type {
        match self {
            Kernel::Unary(op, l, _) => op.result_lane(*l),
            _ => self.lane(),
        }
    }

    /// Lane type of the operands (the result lane type `lane()` for most ops).
    fn input_lane(&self) -> Type {
        match self {
            Kernel::Unary(op, l, _) => op.source_lane(*l),
            Kernel::Narrow(l, _) => wide_of(*l),
            _ => self.lane(),
        }
    }

    fn width(&self) -> Type {
        match self {
            Kernel::Bin(_, _, w)
            | Kernel::Cmp(_, _, w)
            | Kernel::Fma(_, w)
            | Kernel::Sel(w)
            | Kernel::Reduce(_, _, w)
            | Kernel::Splat(_, w)
            | Kernel::Unary(_, _, w)
            | Kernel::Narrow(_, w)
            | Kernel::Shift(_, _, w)
            | Kernel::FastBin(_, _, w)
            | Kernel::FastReduce(_, _, w)
            | Kernel::Zip(_, _, w)
            | Kernel::Dup(_, w, _)
            | Kernel::Dot(w) => *w,
        }
    }

    /// Bytes of output to compare.
    fn out_bytes(&self, vx_bytes: usize) -> usize {
        match self {
            Kernel::Reduce(_, l, _) | Kernel::FastReduce(_, l, _) => l.byte_size(),
            _ => width_bytes(self.width(), vx_bytes),
        }
    }

    /// Whether output lanes are floats (compared NaN-tolerantly). Ops that only move lanes
    /// or sign bits must keep NaNs bit for bit.
    fn float_output(&self) -> bool {
        match self {
            Kernel::Bin(op, l, _) => l.is_float() && !op.is_bitwise(),
            Kernel::Unary(VUnaryOp::Abs | VUnaryOp::Neg | VUnaryOp::Rev, ..) => false,
            Kernel::Fma(..)
            | Kernel::Reduce(..)
            | Kernel::Splat(..)
            | Kernel::Unary(..)
            | Kernel::FastBin(..)
            | Kernel::FastReduce(..) => self.out_lane().is_float(),
            Kernel::Narrow(l, _) => l.is_half(),
            Kernel::Cmp(..)
            | Kernel::Sel(..)
            | Kernel::Shift(..)
            | Kernel::Zip(..)
            | Kernel::Dup(..)
            | Kernel::Dot(..) => false,
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
        for op in VUnaryOp::ALL {
            for &l in vunary_lane_types(op) {
                ks.push(Kernel::Unary(op, l, w));
            }
        }
        for &l in VNARROW_LANE_TYPES {
            ks.push(Kernel::Narrow(l, w));
        }
        for op in VShiftOp::ALL {
            for &l in VSHIFT_LANE_TYPES {
                ks.push(Kernel::Shift(op, l, w));
            }
        }
        for &l in VPERM_LANE_TYPES {
            for op in VZipOp::ALL {
                ks.push(Kernel::Zip(op, l, w));
            }
            // The first lane, one in the middle, and the last (vx: of its guaranteed 128 bits).
            let lanes = w.bit_width().unwrap_or(128) / l.bit_width().unwrap();
            let mut indices = vec![0, lanes / 2, lanes - 1];
            indices.dedup();
            for i in indices {
                ks.push(Kernel::Dup(l, w, i));
            }
        }
        ks.push(Kernel::Dot(w));
        for l in [Type::F32, Type::F64] {
            for op in [VBinOp::Min, VBinOp::Max] {
                ks.push(Kernel::FastBin(op, l, w));
            }
            for op in [VectorReduceOp::Min, VectorReduceOp::Max] {
                ks.push(Kernel::FastReduce(op, l, w));
            }
        }
    }
    ks
}

/// The kernels as modules compiled without and with `fast_math`, skipping empty groups.
fn modules_by_mode(kernels: &[Kernel]) -> Vec<(bool, Vec<Kernel>, achainsaw_ir::Module)> {
    [false, true]
        .into_iter()
        .filter_map(|fast| {
            let group: Vec<Kernel> = kernels
                .iter()
                .copied()
                .filter(|k| k.fast_math() == fast)
                .collect();
            if group.is_empty() {
                return None;
            }
            let module = parse_and_validate(&module_source(&group)).expect("kernels validate");
            Some((fast, group, module))
        })
        .collect()
}

/// Float min/max with `fast_math`: `a > b ? a : b` / `a < b ? a : b` (so a NaN operand or
/// two zeros give `b`).
fn fast_minmax(is_max: bool, a: f64, b: f64) -> f64 {
    let pick_a = if is_max { a > b } else { a < b };
    if pick_a {
        a
    } else {
        b
    }
}

/// Lane type twice as wide as an integer lane (`vnarrow`'s operands).
fn wide_of(lane: Type) -> Type {
    vnarrow_source_lane(lane)
}

/// f16 or bf16 bits to f32, as the `half` crate converts them.
fn half_to_f32(lane: Type, bits: u16) -> f32 {
    if lane == Type::F16 {
        half::f16::from_bits(bits).to_f32()
    } else {
        half::bf16::from_bits(bits).to_f32()
    }
}

/// f32 to f16 or bf16 bits, rounding to nearest-even (the `half` crate).
fn f32_to_half(lane: Type, x: f32) -> u16 {
    if lane == Type::F16 {
        half::f16::from_f32(x).to_bits()
    } else {
        half::bf16::from_f32(x).to_bits()
    }
}

/// `vexp` as AIR defines it, step by step (see `achainsaw_ir::ast::vexp`).
fn vexp_ref(x: f32) -> f32 {
    let x = if x < vexp::CLAMP_LO {
        vexp::CLAMP_LO
    } else {
        x
    };
    let x = if x > vexp::CLAMP_HI {
        vexp::CLAMP_HI
    } else {
        x
    };
    let tm = x * vexp::LOG2E + vexp::MAGIC;
    let n = tm - vexp::MAGIC;
    let r = n.mul_add(vexp::NEG_LN2_HI, x);
    let r = n.mul_add(vexp::LN2_LO, r);
    let mut q = vexp::POLY[0];
    for &c in &vexp::POLY[1..] {
        q = q.mul_add(r, c);
    }
    let p = q.mul_add(r * r, r + 1.0);
    let e = (tm.to_bits() as i32)
        .wrapping_add(vexp::EXP_BIAS)
        .wrapping_shl(23);
    p * f32::from_bits(e as u32)
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
        VBinOp::Div | VBinOp::Copysign => unreachable!("float-only, rejected by the validator"),
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
                VBinOp::Copysign => a.copysign(b),
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
    let n = w / k.out_lane().byte_size();
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
        Kernel::Unary(op, l, _) => match op {
            VUnaryOp::Itof => {
                for i in 0..n {
                    let r = get_int(a, Type::I32, i) as f32;
                    out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                }
            }
            // Saturating, NaN to 0, exactly as Rust's `as`.
            VUnaryOp::Ftoi => {
                (0..n).for_each(|i| put_int(&mut out, l, i, get_f32(a, i) as i32 as i64))
            }
            VUnaryOp::Exp => {
                for i in 0..n {
                    let r = vexp_ref(get_f32(a, i));
                    out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                }
            }
            VUnaryOp::WidenLo | VUnaryOp::WidenHi => {
                let narrow = op.source_lane(l);
                let base = if op == VUnaryOp::WidenLo { 0 } else { n };
                for i in 0..n {
                    put_int(&mut out, l, i, get_int(a, narrow, base + i));
                }
            }
            VUnaryOp::FWidenLo | VUnaryOp::FWidenHi => {
                let base = if op.is_low_half() { 0 } else { n };
                for i in 0..n {
                    let r = half_to_f32(l, get_int(a, l, base + i) as u16);
                    out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                }
            }
            // Floats: the sign bit cleared or flipped, NaNs included.
            VUnaryOp::Abs | VUnaryOp::Neg if l.is_float() => {
                let sign = 1i64 << (l.bit_width().unwrap() - 1);
                let bits = int_of(l);
                for i in 0..n {
                    let x = get_int(a, bits, i);
                    let r = if op == VUnaryOp::Abs {
                        x & !sign
                    } else {
                        x ^ sign
                    };
                    put_int(&mut out, bits, i, r);
                }
            }
            VUnaryOp::Abs | VUnaryOp::Neg => {
                for i in 0..n {
                    let x = get_int(a, l, i);
                    let r = if op == VUnaryOp::Abs {
                        x.wrapping_abs()
                    } else {
                        x.wrapping_neg()
                    };
                    put_int(&mut out, l, i, l.wrap_int(r));
                }
            }
            VUnaryOp::Sqrt | VUnaryOp::Rsqrt if l == Type::F32 => {
                for i in 0..n {
                    let r = get_f32(a, i).sqrt();
                    let r = if op == VUnaryOp::Sqrt { r } else { 1.0 / r };
                    out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                }
            }
            VUnaryOp::Sqrt | VUnaryOp::Rsqrt => {
                for i in 0..n {
                    let r = get_f64(a, i).sqrt();
                    let r = if op == VUnaryOp::Sqrt { r } else { 1.0 / r };
                    out[i * 8..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
                }
            }
            VUnaryOp::Floor
            | VUnaryOp::Ceil
            | VUnaryOp::Round
            | VUnaryOp::RoundEven
            | VUnaryOp::RoundZ => {
                let f = |x: f64| match op {
                    VUnaryOp::Floor => x.floor(),
                    VUnaryOp::Ceil => x.ceil(),
                    VUnaryOp::Round => x.round(),
                    VUnaryOp::RoundEven => x.round_ties_even(),
                    _ => x.trunc(),
                };
                for i in 0..n {
                    if l == Type::F32 {
                        // Exact in f64 too: rounding an f32 gives an f32.
                        let r = f(get_f32(a, i) as f64) as f32;
                        out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                    } else {
                        let r = f(get_f64(a, i));
                        out[i * 8..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
                    }
                }
            }
            VUnaryOp::Rev => {
                let s = l.byte_size();
                for i in 0..n {
                    let j = n - 1 - i;
                    out[i * s..i * s + s].copy_from_slice(&a[j * s..j * s + s]);
                }
            }
        },
        Kernel::Zip(op, l, _) => {
            // Lane `pick(i)` of a followed by b.
            let s = l.byte_size();
            let ab: Vec<u8> = a[..w].iter().chain(&b[..w]).copied().collect();
            for i in 0..n {
                let j = match op {
                    VZipOp::ZipLo => i / 2 + (i % 2) * n,
                    VZipOp::ZipHi => n / 2 + i / 2 + (i % 2) * n,
                    VZipOp::UnzipLo => 2 * i,
                    VZipOp::UnzipHi => 2 * i + 1,
                };
                out[i * s..i * s + s].copy_from_slice(&ab[j * s..j * s + s]);
            }
        }
        // Four exact i8 products per i32 lane, added to c's lane with wrapping.
        Kernel::Dot(_) => {
            for i in 0..w / 4 {
                let dot: i64 = (0..4)
                    .map(|q| get_int(a, Type::I8, 4 * i + q) * get_int(b, Type::I8, 4 * i + q))
                    .sum();
                let r = (get_int(c, Type::I32, i) as i32).wrapping_add(dot as i32);
                put_int(&mut out, Type::I32, i, r as i64);
            }
        }
        Kernel::Dup(l, _, k) => {
            let (s, k) = (l.byte_size(), k as usize);
            for i in 0..n {
                out[i * s..i * s + s].copy_from_slice(&a[k * s..k * s + s]);
            }
        }
        Kernel::Narrow(l, _) if l.is_half() => {
            let half = w / 4;
            for i in 0..half {
                put_int(&mut out, l, i, f32_to_half(l, get_f32(a, i)) as i64);
                put_int(&mut out, l, half + i, f32_to_half(l, get_f32(b, i)) as i64);
            }
        }
        Kernel::Narrow(l, _) => {
            let wide = wide_of(l);
            let half = w / wide.byte_size();
            let bits = l.bit_width().unwrap();
            let (min, max) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
            for i in 0..half {
                put_int(&mut out, l, i, get_int(a, wide, i).clamp(min, max));
                put_int(&mut out, l, half + i, get_int(b, wide, i).clamp(min, max));
            }
        }
        Kernel::FastBin(op, l, _) => {
            let is_max = op == VBinOp::Max;
            for i in 0..n {
                if l == Type::F32 {
                    let (x, y) = (get_f32(a, i), get_f32(b, i));
                    let pick_x = if is_max { x > y } else { x < y };
                    let r = if pick_x { x } else { y };
                    out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                } else {
                    let r = fast_minmax(is_max, get_f64(a, i), get_f64(b, i));
                    out[i * 8..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
                }
            }
        }
        Kernel::FastReduce(op, l, _) => {
            let is_max = op == VectorReduceOp::Max;
            if l == Type::F32 {
                let vals: Vec<f32> = (0..n).map(|i| get_f32(a, i)).collect();
                let r = tree(&vals, &|x: f32, y: f32| {
                    let pick_x = if is_max { x > y } else { x < y };
                    if pick_x {
                        x
                    } else {
                        y
                    }
                });
                out[..4].copy_from_slice(&r.to_le_bytes());
            } else {
                let vals: Vec<f64> = (0..n).map(|i| get_f64(a, i)).collect();
                let r = tree(&vals, &|x, y| fast_minmax(is_max, x, y));
                out[..8].copy_from_slice(&r.to_le_bytes());
            }
        }
        Kernel::Shift(op, l, _) => {
            let bits = l.bit_width().unwrap() as u64;
            let amount =
                (i32::from_le_bytes(c[..4].try_into().unwrap()) as u64 & (bits - 1)) as u32;
            let mask = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            for i in 0..n {
                let x = get_int(a, l, i);
                let r = match op {
                    VShiftOp::Shl => x.wrapping_shl(amount),
                    VShiftOp::Shr => x >> amount,
                    VShiftOp::Ushr => ((x as u64 & mask) >> amount) as i64,
                };
                put_int(&mut out, l, i, l.wrap_int(r));
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
            Type::F16 | Type::BF16 => {
                // Zeros, ones, largest finite, infinities, smallest subnormal, NaNs (quiet and
                // signaling), then random bit patterns.
                let specials: [u16; 10] = if lane == Type::F16 {
                    [
                        0, 0x8000, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x0001, 0x7e00, 0x7c01,
                    ]
                } else {
                    [
                        0, 0x8000, 0x3f80, 0xbf80, 0x7f7f, 0x7f80, 0xff80, 0x0001, 0x7fc0, 0x7f81,
                    ]
                };
                let v = if special {
                    specials[(i + idx) % specials.len()]
                } else {
                    (pick >> 16) as u16
                };
                buf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
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
    if k.float_output() && k.out_lane().is_half() {
        // NaN payloads aside: 16-bit floats compare bit for bit unless both are NaN.
        let lane = k.out_lane();
        for i in 0..n / 2 {
            let e = u16::from_le_bytes([expected[i * 2], expected[i * 2 + 1]]);
            let a = u16::from_le_bytes([actual[i * 2], actual[i * 2 + 1]]);
            let ok = e == a || (half_to_f32(lane, e).is_nan() && half_to_f32(lane, a).is_nan());
            assert!(
                ok,
                "{} lane {i}: expected {e:#06x}, got {a:#06x} ({ctx})",
                k.name()
            );
        }
    } else if k.float_output() {
        let s = k.out_lane().byte_size();
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
    let levels = host_levels();
    assert!(!levels.is_empty());
    check_against_reference(&all_kernels(), levels);
}

/// Only the `vx` kernels, at the host's full ISA: what changes with the vector length, and
/// small enough to run under emulation (CI reruns it under QEMU at several SVE lengths).
#[test]
fn vx_ops_match_reference_at_host_vector_length() {
    let kernels: Vec<Kernel> = all_kernels()
        .into_iter()
        .filter(|k| k.width() == Type::Vx)
        .collect();
    let features = CpuFeatures::effective().unwrap();
    let level = features.max_level().expect("host ISA level");
    check_against_reference(&kernels, vec![(level, features)]);
}

fn check_against_reference(kernels: &[Kernel], levels: Vec<(IsaLevel, CpuFeatures)>) {
    let modules = modules_by_mode(kernels);
    for (level, features) in levels {
        for (fast, group, module) in &modules {
            let mut engine = JitEngine::with_features(&features).expect("JIT init");
            engine.set_fast_math(*fast);
            engine
                .compile_module(module)
                .unwrap_or_else(|e| panic!("compile at {level}: {e}"));
            let vx_bytes = engine.vx_bits() as usize / 8;
            let ctx = |trial| {
                format!(
                    "ISA {level}, {} vx={vx_bytes}B, fast_math={fast}, trial {trial}",
                    engine.backend()
                )
            };

            for k in group {
                let f: KernelFn =
                    unsafe { std::mem::transmute(engine.get_fn_ptr(&k.name()).unwrap()) };
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ k.name().len() as u64);
                for trial in 0..TRIALS {
                    let (mut a, mut b, mut c) = ([0u8; BUF], [0u8; BUF], [0u8; BUF]);
                    fill(&mut a, k.input_lane(), &mut rng, trial);
                    fill(&mut b, k.input_lane(), &mut rng, trial + 1);
                    fill(&mut c, k.input_lane(), &mut rng, trial + 2);
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
}

#[test]
fn vector_ops_compile_for_every_target() {
    let modules = modules_by_mode(&all_kernels());
    // VNNI and DotProd select other `vdot` sequences.
    let targets = [
        ("x86_64-unknown-linux-gnu", "x86-64", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v2", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v3", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v3", "+avxvnni"),
        ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v4", "+avx512vnni"),
        ("aarch64-unknown-linux-gnu", "generic", ""),
        ("aarch64-unknown-linux-gnu", "generic", "+dotprod"),
    ];
    for (triple, cpu, features) in targets {
        for (fast, _, module) in &modules {
            let mut compiler = AotCompiler::with_target(&AotTarget {
                triple: Some(triple.into()),
                cpu: Some(cpu.into()),
                features: (!features.is_empty()).then(|| features.into()),
            })
            .unwrap();
            compiler.set_fast_math(*fast);
            compiler
                .compile_module(module)
                .unwrap_or_else(|e| panic!("{triple} {cpu} {features} fast_math={fast}: {e}"));
            assert!(!compiler.finish().unwrap().is_empty());
        }
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
    ("x86_64-unknown-linux-gnu", "x86-64-v3", "+avxvnni"),
    ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
    ("x86_64-unknown-linux-gnu", "x86-64-v4", "+avx512vnni"),
    ("aarch64-unknown-linux-gnu", "generic", ""),
    ("aarch64-unknown-linux-gnu", "generic", "+dotprod"),
    ("aarch64-unknown-linux-gnu", "generic", "+sve"),
    ("aarch64-unknown-linux-gnu", "generic", "+sve2"),
];

#[cfg(feature = "llvm")]
#[test]
fn vector_ops_compile_for_every_llvm_target() {
    use achainsaw_codegen::{compile_object, Backend, CodegenOptions};
    let modules = modules_by_mode(&all_kernels());
    for (triple, cpu, features) in LLVM_TARGETS {
        let target = AotTarget {
            triple: Some((*triple).into()),
            cpu: Some((*cpu).into()),
            features: (!features.is_empty()).then(|| (*features).into()),
        };
        for (fast, _, module) in &modules {
            let options = CodegenOptions { fast_math: *fast };
            let obj = compile_object(module, &target, Backend::Llvm, &options)
                .unwrap_or_else(|e| panic!("{triple} {cpu} {features} fast_math={fast}: {e}"));
            assert!(!obj.bytes.is_empty());
        }
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
    let modules = modules_by_mode(&kernels);
    for (level, features) in host_levels() {
        for (fast, group, module) in &modules {
            let engines: Vec<JitEngine> = [Backend::Cranelift, Backend::Llvm]
                .into_iter()
                .map(|b| {
                    let mut e = JitEngine::with_backend(b, &features).unwrap();
                    e.set_fast_math(*fast);
                    e.compile_module(module).unwrap();
                    e
                })
                .collect();
            for k in group {
                let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ k.name().len() as u64);
                for trial in 0..TRIALS {
                    let (mut a, mut b, mut c) = ([0u8; BUF], [0u8; BUF], [0u8; BUF]);
                    fill(&mut a, k.input_lane(), &mut rng, trial);
                    fill(&mut b, k.input_lane(), &mut rng, trial + 1);
                    fill(&mut c, k.input_lane(), &mut rng, trial + 2);
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
}

/// `vexp`'s fixed algorithm is accurate: within 2 ulp of e^x over its whole range (the
/// reference model above is what both backends match bit for bit), with NaN propagated and
/// inputs past the clamp mapped to the endpoints.
#[test]
fn vexp_is_accurate() {
    let ulps = |got: f32, want: f64| {
        let w = want as f32;
        (got.to_bits() as i64 - w.to_bits() as i64).unsigned_abs()
    };
    let (lo, hi) = (vexp::CLAMP_LO as f64, vexp::CLAMP_HI as f64);
    let steps = 2_000_000;
    let mut worst = (0u64, 0f32);
    for i in 0..=steps {
        let x = (lo + (hi - lo) * i as f64 / steps as f64) as f32;
        let e = ulps(vexp_ref(x), (x as f64).exp());
        if e > worst.0 {
            worst = (e, x);
        }
    }
    assert!(
        worst.0 <= 2,
        "vexp is {} ulp off at x = {}",
        worst.0,
        worst.1
    );
    assert!(vexp_ref(f32::NAN).is_nan());
    assert_eq!(vexp_ref(f32::INFINITY), vexp_ref(vexp::CLAMP_HI));
    assert_eq!(vexp_ref(f32::NEG_INFINITY), vexp_ref(vexp::CLAMP_LO));
    assert_eq!(vexp_ref(0.0), 1.0);
}

/// A `vx` helper (SwiGLU's gate on full vectors) called from a loop: vx values cross calls
/// as arguments and results, on both backends, and compile for every target including SVE.
const VX_HELPER: &str = r#"
fn silu_mul(g:vx, u:vx)->vx
  b0:
    z = cst 0.0:f32
    zero = splat z:vx
    ng = vsub zero, g:f32
    e = vexp ng:f32
    one = cst 1.0:f32
    v_one = splat one:vx
    den = vadd e, v_one:f32
    gu = vmul g, u:f32
    r = vdiv gu, den:f32
    ret r

fn swiglu(gate:ptr, up:ptr, out:ptr, n:i64)
  b0:
    w = vl f32
    jmp loop(0:i64)
  loop(i:i64):
    more = lt i, n
    br more, body, done
  body:
    rest = sub n, i
    off = mul i, 4:i64
    pg = add gate, off
    pu = add up, off
    po = add out, off
    g = ldm pg:vx, rest:f32
    u = ldm pu:vx, rest:f32
    y = call silu_mul(g, u)
    stm po, y, rest:f32
    i2 = add i, w
    jmp loop(i2)
  done:
    ret
"#;

#[test]
fn vx_values_cross_calls() {
    let module = parse_and_validate(VX_HELPER).expect("valid");
    for (level, features) in host_levels() {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.compile_module(&module).unwrap();
        let f: extern "C" fn(*const f32, *const f32, *mut f32, i64) =
            unsafe { std::mem::transmute(engine.get_fn_ptr("swiglu").unwrap()) };
        for n in [1usize, 3, 4, 17, 100] {
            let g: Vec<f32> = (0..n).map(|i| i as f32 * 0.37 - 9.0).collect();
            let u: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.01).collect();
            let mut out = vec![f32::NAN; n + 1];
            f(g.as_ptr(), u.as_ptr(), out.as_mut_ptr(), n as i64);
            for i in 0..n {
                let want = g[i] * u[i] / (vexp_ref(-g[i]) + 1.0);
                assert_eq!(out[i], want, "n={n} [{i}] at {level}");
            }
            assert!(out[n].is_nan(), "wrote past n at {level}");
        }
        assert!(unsafe { engine.call_typed("silu_mul", &[]) }.is_err());
    }
}

#[cfg(feature = "llvm")]
#[test]
fn vx_signatures_compile_for_every_llvm_target() {
    use achainsaw_codegen::{compile_object, Backend};
    let module = parse_and_validate(VX_HELPER).expect("valid");
    for (triple, cpu, features) in LLVM_TARGETS {
        let target = AotTarget {
            triple: Some((*triple).into()),
            cpu: Some((*cpu).into()),
            features: (!features.is_empty()).then(|| (*features).into()),
        };
        compile_object(&module, &target, Backend::Llvm, &Default::default())
            .unwrap_or_else(|e| panic!("{triple} {cpu} {features}: {e}"));
    }
}
