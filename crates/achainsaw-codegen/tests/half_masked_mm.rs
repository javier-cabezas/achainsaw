//! f16/bf16 conversions, tail-masked `ldm`/`stm`, and `mm` on the Cranelift backend.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::{AotCompiler, AotTarget, JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;
use half::{bf16, f16};

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

fn jit(src: &str, features: &CpuFeatures) -> JitEngine {
    let module = parse_and_validate(src).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    let mut engine = JitEngine::with_features(features).unwrap();
    engine.compile_module(&module).unwrap();
    engine
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform-ish value in [-4, 4).
    fn small(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 8.0 - 4.0
    }
}

// ---------------------------------------------------------------------------------------
// f16 / bf16
// ---------------------------------------------------------------------------------------

const CONVERSIONS: &str = r#"
fn h2f(p:ptr, o:ptr)
  b0:
    h = ld p:f16
    f = fext h:f32
    st o, f
    ret

fn b2f(p:ptr, o:ptr)
  b0:
    h = ld p:bf16
    f = fext h:f32
    st o, f
    ret

fn f2h(p:ptr, o:ptr)
  b0:
    f = ld p:f32
    h = ftrunc f:f16
    st o, h
    ret

fn f2b(p:ptr, o:ptr)
  b0:
    f = ld p:f32
    h = ftrunc f:bf16
    st o, h
    ret

fn bits(p:ptr)->i16
  b0:
    h = ld p:bf16
    i = bitcast h:i16
    g = bitcast i:f16
    j = bitcast g:i16
    ret j
"#;

type Conv = extern "C" fn(*const u8, *mut u8);

fn conv(engine: &JitEngine, name: &str) -> Conv {
    unsafe { std::mem::transmute(engine.get_fn_ptr(name).unwrap()) }
}

/// f32 inputs that stress f32 -> 16-bit rounding: every representable 16-bit value,
/// the midpoints between neighbours (ties), one f32 ulp either side, and random bits.
fn rounding_inputs(to_f32: impl Fn(u16) -> f32) -> Vec<f32> {
    let mut v = Vec::new();
    for b in 0..=u16::MAX {
        let x = to_f32(b);
        let y = to_f32(b.wrapping_add(1));
        v.push(x);
        if x.is_finite() && y.is_finite() && x.is_sign_negative() == y.is_sign_negative() {
            let mid = ((x as f64 + y as f64) / 2.0) as f32;
            v.extend([
                mid,
                f32::from_bits(mid.to_bits() + 1),
                f32::from_bits(mid.to_bits() - 1),
            ]);
        }
    }
    let mut rng = Rng(42);
    v.extend((0..200_000).map(|_| f32::from_bits(rng.next() as u32)));
    v.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN_POSITIVE,
        1.0e-45,
    ]);
    v
}

fn same_bits_or_nan(expected: f32, actual: f32) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else {
        expected.to_bits() == actual.to_bits()
    }
}

/// Vector `vfwidenlo`/`vfwidenhi` and `vnarrow` to f16/bf16 over whole buffers, `vl` lanes at
/// a time (buffers are padded to a multiple of any vector length).
const VECTOR_CONVERSIONS: &str = r#"
fn vwiden_f16(src:ptr, dst:ptr, n:i64)
  b0:
    w16 = vl i16
    half = udiv w16, 2:i64
    jmp lp(0:i64)
  lp(i:i64):
    more = lt i, n
    br more, body, done
  body:
    so = mul i, 2:i64
    ps = add src, so
    v = ld ps:vx
    lo = vfwidenlo v:f16
    hi = vfwidenhi v:f16
    do = mul i, 4:i64
    pd = add dst, do
    st pd, lo
    ih = add i, half
    dho = mul ih, 4:i64
    pdh = add dst, dho
    st pdh, hi
    i2 = add i, w16
    jmp lp(i2)
  done:
    ret

fn vwiden_bf16(src:ptr, dst:ptr, n:i64)
  b0:
    w16 = vl i16
    half = udiv w16, 2:i64
    jmp lp(0:i64)
  lp(i:i64):
    more = lt i, n
    br more, body, done
  body:
    so = mul i, 2:i64
    ps = add src, so
    v = ld ps:vx
    lo = vfwidenlo v:bf16
    hi = vfwidenhi v:bf16
    do = mul i, 4:i64
    pd = add dst, do
    st pd, lo
    ih = add i, half
    dho = mul ih, 4:i64
    pdh = add dst, dho
    st pdh, hi
    i2 = add i, w16
    jmp lp(i2)
  done:
    ret

fn vnarrow_f16(src:ptr, dst:ptr, n:i64)
  b0:
    w = vl f32
    w2 = mul w, 2:i64
    jmp lp(0:i64)
  lp(i:i64):
    more = lt i, n
    br more, body, done
  body:
    so = mul i, 4:i64
    pa = add src, so
    a = ld pa:vx
    ib = add i, w
    sbo = mul ib, 4:i64
    pb = add src, sbo
    b = ld pb:vx
    r = vnarrow a, b:f16
    do = mul i, 2:i64
    pd = add dst, do
    st pd, r
    i2 = add i, w2
    jmp lp(i2)
  done:
    ret

fn vnarrow_bf16(src:ptr, dst:ptr, n:i64)
  b0:
    w = vl f32
    w2 = mul w, 2:i64
    jmp lp(0:i64)
  lp(i:i64):
    more = lt i, n
    br more, body, done
  body:
    so = mul i, 4:i64
    pa = add src, so
    a = ld pa:vx
    ib = add i, w
    sbo = mul ib, 4:i64
    pb = add src, sbo
    b = ld pb:vx
    r = vnarrow a, b:bf16
    do = mul i, 2:i64
    pd = add dst, do
    st pd, r
    i2 = add i, w2
    jmp lp(i2)
  done:
    ret
"#;

#[test]
fn vector_half_conversions_match_reference_exhaustively() {
    type Conv = extern "C" fn(*const u8, *mut u8, i64);
    let pad = |n: usize| n.div_ceil(1024) * 1024;
    for (level, features) in host_levels() {
        let engine = jit(VECTOR_CONVERSIONS, &features);
        let f = |name: &str| -> Conv {
            unsafe { std::mem::transmute(engine.get_fn_ptr(name).unwrap()) }
        };
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let src: Vec<u8> = all.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut out = vec![0u8; all.len() * 4];
        f("vwiden_f16")(src.as_ptr(), out.as_mut_ptr(), all.len() as i64);
        for (i, &b) in all.iter().enumerate() {
            let got = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
            let want = f16::from_bits(b).to_f32();
            assert!(
                same_bits_or_nan(want, got),
                "vfwiden f16 {b:#06x} -> {got:?}, want {want:?} at {level}"
            );
        }
        f("vwiden_bf16")(src.as_ptr(), out.as_mut_ptr(), all.len() as i64);
        for (i, &b) in all.iter().enumerate() {
            let got = u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
            let want = bf16::from_bits(b).to_f32();
            assert!(
                same_bits_or_nan(want, f32::from_bits(got)),
                "vfwiden bf16 {b:#06x} -> {got:#010x} at {level}"
            );
        }

        for (name, reference, to_f32) in [
            (
                "vnarrow_f16",
                (|x: f32| f16::from_f32(x).to_bits()) as fn(f32) -> u16,
                (|b| f16::from_bits(b).to_f32()) as fn(u16) -> f32,
            ),
            (
                "vnarrow_bf16",
                |x: f32| bf16::from_f32(x).to_bits(),
                |b| bf16::from_bits(b).to_f32(),
            ),
        ] {
            let inputs = rounding_inputs(to_f32);
            let n = pad(inputs.len());
            let mut src = vec![0u8; n * 4];
            for (i, x) in inputs.iter().enumerate() {
                src[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            let mut out = vec![0u8; n * 2];
            f(name)(src.as_ptr(), out.as_mut_ptr(), n as i64);
            for (i, &x) in inputs.iter().enumerate() {
                let got = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
                if x.is_nan() {
                    assert!(
                        to_f32(got).is_nan(),
                        "{name}: NaN {:#010x} -> {got:#06x} at {level}",
                        x.to_bits()
                    );
                } else {
                    assert_eq!(
                        got,
                        reference(x),
                        "{name}: {x:e} ({:#010x}) at {level}",
                        x.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn half_conversions_match_reference_exhaustively() {
    for (level, features) in host_levels() {
        let engine = jit(CONVERSIONS, &features);
        let (h2f, b2f, f2h, f2b) = (
            conv(&engine, "h2f"),
            conv(&engine, "b2f"),
            conv(&engine, "f2h"),
            conv(&engine, "f2b"),
        );

        for b in 0..=u16::MAX {
            let mut out = [0u8; 4];
            h2f(b.to_le_bytes().as_ptr(), out.as_mut_ptr());
            let got = f32::from_le_bytes(out);
            let want = f16::from_bits(b).to_f32();
            assert!(
                same_bits_or_nan(want, got),
                "f16 {b:#06x} -> {got:?}, want {want:?} at {level}"
            );

            b2f(b.to_le_bytes().as_ptr(), out.as_mut_ptr());
            // Widening bf16 is a 16-bit shift, so even NaN payloads must match.
            assert_eq!(
                u32::from_le_bytes(out),
                (b as u32) << 16,
                "bf16 {b:#06x} at {level}"
            );
        }

        for (name, f, reference) in [
            (
                "f16",
                f2h,
                (|x: f32| f16::from_f32(x).to_bits()) as fn(f32) -> u16,
            ),
            ("bf16", f2b, |x: f32| bf16::from_f32(x).to_bits()),
        ] {
            let to_f32: fn(u16) -> f32 = if name == "f16" {
                |b| f16::from_bits(b).to_f32()
            } else {
                |b| bf16::from_bits(b).to_f32()
            };
            for x in rounding_inputs(to_f32) {
                let mut out = [0u8; 2];
                f(x.to_le_bytes().as_ptr(), out.as_mut_ptr());
                let got = u16::from_le_bytes(out);
                let want = reference(x);
                if x.is_nan() {
                    assert!(
                        to_f32(got).is_nan(),
                        "{name}: NaN {:#010x} -> {got:#06x} at {level}",
                        x.to_bits()
                    );
                } else {
                    assert_eq!(
                        got,
                        want,
                        "{name}: {x:e} ({:#010x}) at {level}",
                        x.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn half_bitcasts_preserve_bits() {
    let engine = jit(CONVERSIONS, &CpuFeatures::host());
    let f: extern "C" fn(*const u8) -> i16 =
        unsafe { std::mem::transmute(engine.get_fn_ptr("bits").unwrap()) };
    for b in [0u16, 1, 0x7fff, 0x8000, 0xffff, 0x3c00, 0x7e01] {
        assert_eq!(f(b.to_le_bytes().as_ptr()) as u16, b);
    }
}

// ---------------------------------------------------------------------------------------
// ldm / stm
// ---------------------------------------------------------------------------------------

const WIDTHS: [Type; 4] = [Type::V128, Type::V256, Type::V512, Type::Vx];
const LANES: [Type; 8] = [
    Type::I8,
    Type::I16,
    Type::I32,
    Type::I64,
    Type::F32,
    Type::F64,
    Type::F16,
    Type::BF16,
];

fn masked_module() -> String {
    let mut src = String::new();
    for w in WIDTHS {
        for l in LANES {
            src += &format!(
                "fn ldm_{l}_{w}(p:ptr, n:i64, o:ptr)\n  b0:\n    v = ldm p:{w}, n:{l}\n    st o, v\n    ret\n\n\
                 fn stm_{l}_{w}(p:ptr, n:i64, o:ptr)\n  b0:\n    v = ld p:{w}\n    stm o, v, n:{l}\n    ret\n\n"
            );
        }
    }
    src
}

type Masked = extern "C" fn(*const u8, i64, *mut u8);

/// Bytes in a vector of type `w` on `engine` (`vx` depends on the backend and ISA).
fn bytes_of(w: Type, engine: &JitEngine) -> usize {
    w.bit_width().unwrap_or(engine.vx_bits()) as usize / 8
}

#[test]
fn masked_load_store_semantics() {
    check_masked_semantics(&WIDTHS, host_levels());
}

/// `ldm`/`stm` on `vx` at the host's full ISA, including the guard-page check: what changes
/// with the vector length (CI reruns it under QEMU at several SVE lengths).
#[test]
fn vx_masked_ops_at_host_vector_length() {
    let features = CpuFeatures::effective().unwrap();
    let level = features.max_level().expect("host ISA level");
    check_masked_semantics(&[Type::Vx], vec![(level, features)]);
    #[cfg(unix)]
    check_masked_tail_page(&[Type::Vx]);
}

fn check_masked_semantics(widths: &[Type], levels: Vec<(IsaLevel, CpuFeatures)>) {
    let src = masked_module();
    for (level, features) in levels {
        let engine = jit(&src, &features);
        for &w in widths {
            for l in LANES {
                let width = bytes_of(w, &engine);
                let s = l.byte_size();
                let lanes = (width / s) as i64;
                let input: Vec<u8> = (0..width)
                    .map(|i| (i as u8).wrapping_mul(29).wrapping_add(7))
                    .collect();
                let ldm: Masked = unsafe {
                    std::mem::transmute(engine.get_fn_ptr(&format!("ldm_{l}_{w}")).unwrap())
                };
                let stm: Masked = unsafe {
                    std::mem::transmute(engine.get_fn_ptr(&format!("stm_{l}_{w}")).unwrap())
                };
                for n in [
                    -3,
                    0,
                    1,
                    lanes / 2,
                    lanes - 1,
                    lanes,
                    lanes + 1,
                    1 << 40,
                    i64::MIN,
                ] {
                    let kept = n.clamp(0, lanes) as usize * s;

                    let mut out = vec![0xEEu8; width];
                    ldm(input.as_ptr(), n, out.as_mut_ptr());
                    assert_eq!(&out[..kept], &input[..kept], "ldm {l} {w} n={n} at {level}");
                    assert!(
                        out[kept..].iter().all(|&b| b == 0),
                        "ldm {l} {w} n={n} zero fill at {level}"
                    );

                    let mut dst = vec![0xAAu8; width];
                    stm(input.as_ptr(), n, dst.as_mut_ptr());
                    assert_eq!(&dst[..kept], &input[..kept], "stm {l} {w} n={n} at {level}");
                    assert!(
                        dst[kept..].iter().all(|&b| b == 0xAA),
                        "stm {l} {w} n={n} wrote past n at {level}"
                    );
                }
            }
        }
    }
}

/// Masked accesses that end exactly at an inaccessible page must not touch it.
#[cfg(unix)]
#[test]
fn masked_ops_never_touch_memory_past_the_tail() {
    check_masked_tail_page(&WIDTHS);
}

#[cfg(unix)]
fn check_masked_tail_page(widths: &[Type]) {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            2 * page,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(base, libc::MAP_FAILED);
    let base = base as *mut u8;
    let guard = unsafe { base.add(page) };
    assert_eq!(
        unsafe { libc::mprotect(guard.cast(), page, libc::PROT_NONE) },
        0
    );

    let engine = jit(&masked_module(), &CpuFeatures::effective().unwrap());
    for &w in widths {
        for l in LANES {
            let s = l.byte_size();
            let width = bytes_of(w, &engine);
            let lanes = width / s;
            let ldm: Masked =
                unsafe { std::mem::transmute(engine.get_fn_ptr(&format!("ldm_{l}_{w}")).unwrap()) };
            let stm: Masked =
                unsafe { std::mem::transmute(engine.get_fn_ptr(&format!("stm_{l}_{w}")).unwrap()) };
            for n in 0..lanes {
                // The first n lanes end exactly at the guard page.
                let tail = unsafe { guard.sub(n * s) };
                let mut out = vec![0u8; width];
                ldm(tail, n as i64, out.as_mut_ptr());
                let src = vec![0x5Au8; width];
                stm(src.as_ptr(), n as i64, tail);
                let written = unsafe { std::slice::from_raw_parts(tail, n * s) };
                assert!(written.iter().all(|&b| b == 0x5A), "stm {l} {w} n={n}");
            }
        }
    }
    unsafe { libc::munmap(base.cast(), 2 * page) };
}

#[test]
fn saxpy_example_uses_masked_tails() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/saxpy_vx.air"
    ))
    .unwrap();
    assert!(src.contains("ldm") && src.contains("stm"));
    for (level, features) in host_levels() {
        let engine = jit(&src, &features);
        let f: extern "C" fn(f32, *const f32, *mut f32, i64) =
            unsafe { std::mem::transmute(engine.get_fn_ptr("saxpy").unwrap()) };
        for n in [0usize, 1, 3, 4, 5, 11, 64, 67] {
            let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
            let mut y: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
            let mut guard = y.clone();
            guard.push(-1.0);
            f(2.0, x.as_ptr(), guard.as_mut_ptr(), n as i64);
            assert_eq!(guard[n], -1.0, "saxpy wrote past n={n} at {level}");
            y.iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v += 2.0 * (i as f32 * 0.5));
            assert_eq!(&guard[..n], &y[..], "n={n} at {level}");
        }
    }
}

// ---------------------------------------------------------------------------------------
// mm
// ---------------------------------------------------------------------------------------

const MM: &str = r#"
fn mm_bf16(c:ptr, a:ptr, b:ptr, m:i64, n:i64, k:i64)
  b0:
    mm c, a, b, m, n, k:bf16
    ret

fn mm_f16(c:ptr, a:ptr, b:ptr, m:i64, n:i64, k:i64)
  b0:
    mm c, a, b, m, n, k:f16
    ret

fn mm_f32(c:ptr, a:ptr, b:ptr, m:i64, n:i64, k:i64)
  b0:
    mm c, a, b, m, n, k:f32
    ret

fn mm_i8(c:ptr, a:ptr, b:ptr, m:i64, n:i64, k:i64)
  b0:
    mm c, a, b, m, n, k:i8
    ret

fn mm_fixed(c:ptr, a:ptr, b:ptr)
  b0:
    mm c, a, b, 2, 3:i64, 4:f32
    ret
"#;

type MmFn = extern "C" fn(*mut u8, *const u8, *const u8, i64, i64, i64);

/// A and B as raw bytes in `dtype`, plus their exact values as f64.
fn mm_operand(rng: &mut Rng, len: usize, dtype: Type) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::new();
    let mut vals = Vec::new();
    for _ in 0..len {
        match dtype {
            Type::BF16 => {
                let h = bf16::from_f32(rng.small());
                bytes.extend(h.to_bits().to_le_bytes());
                vals.push(h.to_f64());
            }
            Type::F16 => {
                let h = f16::from_f32(rng.small());
                bytes.extend(h.to_bits().to_le_bytes());
                vals.push(h.to_f64());
            }
            Type::F32 => {
                let x = rng.small();
                bytes.extend(x.to_le_bytes());
                vals.push(x as f64);
            }
            _ => {
                let x = rng.next() as i8;
                bytes.push(x as u8);
                vals.push(x as f64);
            }
        }
    }
    (bytes, vals)
}

#[test]
fn mm_matches_reference() {
    check_mm(host_levels());
}

/// `mm` at the host's full ISA only (matrix engines and widest vectors): CI reruns this under
/// QEMU at several SVE/SME vector lengths.
#[test]
fn mm_at_host_vector_length() {
    let features = CpuFeatures::effective().unwrap();
    let level = features.max_level().expect("host ISA level");
    check_mm(vec![(level, features)]);
}

fn check_mm(levels: Vec<(IsaLevel, CpuFeatures)>) {
    let shapes = [
        (1, 1, 1),
        (3, 5, 7),
        (4, 8, 16),
        (16, 16, 32),
        (7, 1, 33),
        // Strip and tile edges for 128-2048-bit vectors, 16x16 AMX tiles and SME's ZA.
        (5, 15, 9),
        (17, 17, 31),
        (3, 33, 64),
        (2, 65, 3),
        (33, 70, 67),
        (2, 9, 0),
        (0, 4, 4),
        (-1, 3, 3),
    ];
    for (level, features) in levels {
        let engine = jit(MM, &features);
        for dtype in [Type::BF16, Type::F16, Type::F32, Type::I8] {
            let f: MmFn =
                unsafe { std::mem::transmute(engine.get_fn_ptr(&format!("mm_{dtype}")).unwrap()) };
            let mut rng = Rng(7 + dtype.byte_size() as u64);
            for (m, n, k) in shapes {
                let (mu, nu, ku) = (m.max(0) as usize, n.max(0) as usize, k.max(0) as usize);
                let (a, av) = mm_operand(&mut rng, mu * ku, dtype);
                let (b, bv) = mm_operand(&mut rng, ku * nu, dtype);
                // C starts non-zero to check accumulation (+=); C is i32 for i8 inputs.
                let c0: Vec<f64> = (0..mu * nu).map(|i| (i % 7) as f64 - 3.0).collect();
                let mut c: Vec<u8> = c0
                    .iter()
                    .flat_map(|&v| {
                        if dtype == Type::I8 {
                            (v as i32).to_le_bytes()
                        } else {
                            (v as f32).to_le_bytes()
                        }
                    })
                    .collect();
                f(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m, n, k);
                for i in 0..mu {
                    for j in 0..nu {
                        let idx = i * nu + j;
                        let terms: Vec<f64> = (0..ku)
                            .map(|kk| av[i * ku + kk] * bv[kk * nu + j])
                            .collect();
                        let want = c0[idx] + terms.iter().sum::<f64>();
                        let raw: [u8; 4] = c[idx * 4..idx * 4 + 4].try_into().unwrap();
                        let ctx = format!("{dtype} {m}x{n}x{k} C[{i},{j}] at {level}");
                        if dtype == Type::I8 {
                            assert_eq!(i32::from_le_bytes(raw) as f64, want, "{ctx}");
                        } else {
                            let got = f32::from_le_bytes(raw) as f64;
                            let scale = c0[idx].abs() + terms.iter().map(|t| t.abs()).sum::<f64>();
                            let tol = (ku as f64 + 2.0) * f32::EPSILON as f64 * scale;
                            assert!(
                                (got - want).abs() <= tol,
                                "{ctx}: got {got}, want {want}, tol {tol}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Every f16 and bf16 bit pattern goes through `mm`'s widening of B (vector lanes on
/// Cranelift) exactly: C = 0 + 1.0 * B, at every ISA level.
#[test]
fn mm_widens_every_half_value_exactly() {
    for (level, features) in host_levels() {
        let engine = jit(MM, &features);
        for dtype in [Type::F16, Type::BF16] {
            let f: MmFn =
                unsafe { std::mem::transmute(engine.get_fn_ptr(&format!("mm_{dtype}")).unwrap()) };
            let one: u16 = if dtype == Type::F16 { 0x3c00 } else { 0x3f80 };
            let b: Vec<u8> = (0..=u16::MAX).flat_map(|h| h.to_le_bytes()).collect();
            let mut c = vec![0u8; 4 << 16];
            f(
                c.as_mut_ptr(),
                one.to_le_bytes().as_ptr(),
                b.as_ptr(),
                1,
                1 << 16,
                1,
            );
            for h in 0..=u16::MAX {
                let want = if dtype == Type::F16 {
                    f16::from_bits(h).to_f32()
                } else {
                    bf16::from_bits(h).to_f32()
                };
                let j = h as usize * 4;
                let got = f32::from_le_bytes(c[j..j + 4].try_into().unwrap());
                assert!(
                    got == want || (got.is_nan() && want.is_nan()),
                    "{dtype} {h:#06x} at {level}: got {got}, want {want}"
                );
            }
        }
    }
}

/// `mm` reads and writes only its three matrices: each operand in turn ends exactly at a
/// guard page, for shapes that end in every tile and tail case.
#[cfg(unix)]
#[test]
fn mm_stays_inside_its_matrices() {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let map = || unsafe {
        let base = libc::mmap(
            std::ptr::null_mut(),
            2 * page,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(base, libc::MAP_FAILED);
        let base = base as *mut u8;
        assert_eq!(
            libc::mprotect(base.add(page).cast(), page, libc::PROT_NONE),
            0
        );
        base
    };
    let (a_map, b_map, c_map) = (map(), map(), map());
    for (_, features) in host_levels() {
        let engine = jit(MM, &features);
        for dtype in [Type::BF16, Type::F16, Type::F32, Type::I8] {
            let f: MmFn =
                unsafe { std::mem::transmute(engine.get_fn_ptr(&format!("mm_{dtype}")).unwrap()) };
            let es = dtype.byte_size();
            for (m, n, k) in [(1, 1, 1), (5, 13, 3), (4, 8, 2), (3, 12, 1), (9, 7, 4)] {
                let end = |map: *mut u8, bytes: usize| unsafe { map.add(page - bytes) };
                let a = end(a_map, m * k * es);
                let b = end(b_map, k * n * es);
                let c = end(c_map, m * n * 4);
                // Zeroed pages: every value is +0, so C stays finite whatever the order.
                f(c, a, b, m as i64, n as i64, k as i64);
            }
        }
    }
    unsafe {
        for map in [a_map, b_map, c_map] {
            libc::munmap(map.cast(), 2 * page);
        }
    }
}

#[test]
fn mm_accepts_literal_dimensions() {
    let engine = jit(MM, &CpuFeatures::host());
    let f: extern "C" fn(*mut f32, *const f32, *const f32) =
        unsafe { std::mem::transmute(engine.get_fn_ptr("mm_fixed").unwrap()) };
    let a: Vec<f32> = (1..=8).map(|v| v as f32).collect(); // 2x4
    let b: Vec<f32> = (1..=12).map(|v| v as f32).collect(); // 4x3
    let mut c = vec![0.0f32; 6];
    f(c.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    assert_eq!(c, vec![70.0, 80.0, 90.0, 158.0, 184.0, 210.0]);
}

#[test]
fn mm_charges_fuel_before_running() {
    let mut engine = jit(MM, &CpuFeatures::host());
    let dim = 64usize;
    let a = vec![1.0f32; dim * dim];
    let b = vec![1.0f32; dim * dim];
    let mut c = vec![0.0f32; dim * dim];
    let args = |c: &mut Vec<f32>| {
        vec![
            RtValue::Ptr(c.as_mut_ptr() as usize),
            RtValue::Ptr(a.as_ptr() as usize),
            RtValue::Ptr(b.as_ptr() as usize),
            RtValue::I64(dim as i64),
            RtValue::I64(dim as i64),
            RtValue::I64(dim as i64),
        ]
    };
    // 64^3 multiply-adds = 257 fuel units.
    engine.set_fuel(Some(100));
    let err = unsafe { engine.call_typed("mm_f32", &args(&mut c)) }
        .unwrap_err()
        .to_string();
    assert!(err.contains("ERR_OUT_OF_FUEL"), "{err}");
    assert!(
        c.iter().all(|&v| v == 0.0),
        "C must be untouched when fuel runs out"
    );

    engine.set_fuel(Some(1000));
    unsafe { engine.call_typed("mm_f32", &args(&mut c)) }.unwrap();
    assert!(c.iter().all(|&v| v == dim as f32));
    engine.set_fuel(None);
}

#[test]
fn new_ops_compile_for_every_target() {
    let src = format!("{MM}\n{}\n{CONVERSIONS}", masked_module());
    let module = parse_and_validate(&src).unwrap();
    for (triple, cpu) in [
        ("x86_64-unknown-linux-gnu", "x86-64"),
        ("x86_64-unknown-linux-gnu", "x86-64-v4"),
        ("aarch64-unknown-linux-gnu", "generic"),
    ] {
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

#[cfg(feature = "llvm")]
#[test]
fn new_ops_compile_for_every_llvm_target() {
    use achainsaw_codegen::{compile_object, Backend};
    let src = format!("{MM}\n{}\n{CONVERSIONS}", masked_module());
    let module = parse_and_validate(&src).unwrap();
    for (triple, cpu, features) in [
        ("x86_64-unknown-linux-gnu", "x86-64", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v3", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
        ("aarch64-unknown-linux-gnu", "generic", ""),
        ("aarch64-unknown-linux-gnu", "generic", "+sve"),
        ("aarch64-unknown-linux-gnu", "generic", "+sve2"),
    ] {
        let target = AotTarget {
            triple: Some(triple.into()),
            cpu: Some(cpu.into()),
            features: (!features.is_empty()).then(|| features.into()),
        };
        let obj = compile_object(&module, &target, Backend::Llvm, &Default::default())
            .unwrap_or_else(|e| panic!("{triple} {cpu} {features}: {e}"));
        assert!(!obj.bytes.is_empty());
    }
}

/// Each target gets the `mm` kernel its hardware supports, visible in the assembly: AMX on
/// Sapphire/Granite Rapids (AMX-FP16 only on Granite Rapids), SME outer products on SME
/// targets, and vector FMAs elsewhere. AMX code cannot run on the CI hosts, so this (plus the
/// LLVM verifier) is its only check.
#[cfg(feature = "llvm")]
#[test]
fn mm_kernels_match_target_matrix_engines() {
    use achainsaw_codegen::{compile_assembly, Backend};
    let module = parse_and_validate(MM).unwrap();
    /// (triple, cpu, features, instructions expected, instructions not expected)
    type Case = (
        &'static str,
        &'static str,
        &'static str,
        &'static [&'static str],
        &'static [&'static str],
    );
    let cases: &[Case] = &[
        (
            "x86_64-unknown-linux-gnu",
            "sapphirerapids",
            "",
            &["ldtilecfg", "tdpbf16ps", "tdpbssd", "tilestored", "vfmadd"],
            &["tdpfp16ps"],
        ),
        (
            "x86_64-unknown-linux-gnu",
            "graniterapids",
            "",
            &["tdpbf16ps", "tdpbssd", "tdpfp16ps"],
            &[],
        ),
        (
            "x86_64-unknown-linux-gnu",
            "x86-64-v4",
            "",
            &["vfmadd", "zmm"],
            &["tdpbf16ps"],
        ),
        (
            "aarch64-unknown-linux-gnu",
            "generic",
            "+sve,+sme",
            &["smstart", "fmopa", "bfmopa", "smopa", "smstop"],
            &[],
        ),
        (
            "aarch64-unknown-linux-gnu",
            "neoverse-v2",
            "",
            &["whilelo"],
            &["fmopa", "smstart"],
        ),
        (
            "aarch64-unknown-linux-gnu",
            "generic",
            "",
            &["fmla"],
            &["whilelo", "fmopa"],
        ),
    ];
    for (triple, cpu, features, want, not_want) in cases {
        let target = AotTarget {
            triple: Some((*triple).into()),
            cpu: Some((*cpu).into()),
            features: (!features.is_empty()).then(|| (*features).into()),
        };
        let (asm, _) = compile_assembly(&module, &target, Backend::Llvm, &Default::default())
            .unwrap_or_else(|e| panic!("{cpu} {features}: {e}"));
        for w in *want {
            assert!(
                asm.contains(w),
                "{cpu} {features}: expected `{w}` in assembly"
            );
        }
        for w in *not_want {
            assert!(
                !asm.contains(w),
                "{cpu} {features}: unexpected `{w}` in assembly"
            );
        }
    }
}
