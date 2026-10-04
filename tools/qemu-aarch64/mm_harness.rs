//! Freestanding AArch64 test harness for AOT-compiled `mm` kernels, run under
//! `qemu-aarch64 -cpu max` so SVE and SME code can be tested on any host. See `run.sh`.
//!
//! Links against an object exporting `mm_bf16`, `mm_f16`, `mm_f32` and `mm_i8`
//! (`fn mm_<dtype>(c:ptr, a:ptr, b:ptr, m:i64, n:i64, k:i64)`), checks them against an f64
//! reference (floats within `(k + 2) * eps * sum|terms|`, i8 exact), and exits non-zero on
//! any mismatch.
#![no_std]
#![no_main]

use core::arch::asm;
use core::ffi::c_void;

type MmFn = unsafe extern "C" fn(*mut u8, *const u8, *const u8, i64, i64, i64);

extern "C" {
    fn mm_bf16(c: *mut u8, a: *const u8, b: *const u8, m: i64, n: i64, k: i64);
    fn mm_f16(c: *mut u8, a: *const u8, b: *const u8, m: i64, n: i64, k: i64);
    fn mm_f32(c: *mut u8, a: *const u8, b: *const u8, m: i64, n: i64, k: i64);
    fn mm_i8(c: *mut u8, a: *const u8, b: *const u8, m: i64, n: i64, k: i64);
}

const SHAPES: [(i64, i64, i64); 14] = [
    (1, 1, 1),
    (3, 5, 7),
    (4, 8, 16),
    (16, 16, 32),
    (7, 1, 33),
    (5, 15, 9),
    (17, 17, 31),
    (3, 33, 64),
    (2, 65, 3),
    (33, 70, 67),
    (64, 64, 64),
    (2, 9, 0),
    (0, 4, 4),
    (-1, 3, 3),
];
const MAX: usize = 70 * 70;

static mut A: [u8; MAX * 4] = [0; MAX * 4];
static mut B: [u8; MAX * 4] = [0; MAX * 4];
static mut C: [u8; MAX * 4] = [0; MAX * 4];
static mut AV: [f64; MAX] = [0.0; MAX];
static mut BV: [f64; MAX] = [0.0; MAX];
static mut C0: [f64; MAX] = [0.0; MAX];

fn write(s: &[u8]) {
    unsafe {
        asm!("svc 0", in("x8") 64, inout("x0") 1usize => _, in("x1") s.as_ptr(), in("x2") s.len());
    }
}

fn exit(code: i32) -> ! {
    unsafe { asm!("svc 0", in("x8") 93, in("x0") code, options(noreturn)) }
}

fn write_int(mut v: i64) {
    let mut buf = [0u8; 24];
    let mut i = buf.len();
    let neg = v < 0;
    if v == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while v != 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10).unsigned_abs() as u8;
        v /= 10;
    }
    if neg {
        i -= 1;
        buf[i] = b'-';
    }
    write(&buf[i..]);
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    write(b"panic\n");
    exit(101)
}

// Freestanding: provide the memory routines compiled code may call.
#[no_mangle]
unsafe extern "C" fn memcpy(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void {
    let (dp, sp) = (d as *mut u8, s as *const u8);
    for i in 0..n {
        *dp.add(i) = *sp.add(i);
    }
    d
}
#[no_mangle]
unsafe extern "C" fn memmove(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void {
    let (dp, sp) = (d as *mut u8, s as *const u8);
    if (dp as usize) < (sp as usize) {
        return memcpy(d, s, n);
    }
    for i in (0..n).rev() {
        *dp.add(i) = *sp.add(i);
    }
    d
}
#[no_mangle]
unsafe extern "C" fn memset(d: *mut c_void, c: i32, n: usize) -> *mut c_void {
    let dp = d as *mut u8;
    for i in 0..n {
        *dp.add(i) = c as u8;
    }
    d
}
#[no_mangle]
unsafe extern "C" fn memcmp(a: *const c_void, b: *const c_void, n: usize) -> i32 {
    let (ap, bp) = (a as *const u8, b as *const u8);
    for i in 0..n {
        let (x, y) = (*ap.add(i), *bp.add(i));
        if x != y {
            return x as i32 - y as i32;
        }
    }
    0
}
#[no_mangle]
unsafe extern "C" fn bcmp(a: *const c_void, b: *const c_void, n: usize) -> i32 {
    memcmp(a, b, n)
}
/// Referenced by the prebuilt `core`; never called with `panic=abort`.
#[no_mangle]
extern "C" fn rust_eh_personality() {}
/// SME ABI: save a lazily saved ZA buffer. Nothing here enables lazy saves, so TPIDR2_EL0
/// stays zero and this is never reached.
#[no_mangle]
extern "C" fn __arm_tpidr2_save() {
    write(b"unexpected __arm_tpidr2_save\n");
    exit(102)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Value in [-4, 4).
    fn small(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 8.0 - 4.0
    }
}

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_value(h: u16) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

/// Nearest f16 for |x| < 8 (no overflow or NaN in these tests).
fn f16_bits(x: f32) -> u16 {
    let v = x as f64;
    let sign = if v < 0.0 { 0x8000u16 } else { 0 };
    let a = abs(v);
    if a < 6.103515625e-05 {
        let q = a / 5.960464477539063e-08; // subnormal step 2^-24
        let r = (q + 0.5) as u16; // round-half-up is fine here (never exactly .5 in practice)
        return sign | r;
    }
    let mut e = 0i32;
    let mut m = a;
    while m >= 2.0 {
        m /= 2.0;
        e += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        e -= 1;
    }
    let frac = ((m - 1.0) * 1024.0 + 0.5) as u16;
    let (frac, e) = if frac == 1024 { (0, e + 1) } else { (frac, e) };
    sign | (((e + 15) as u16) << 10) | frac
}

fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let f = (h & 0x3ff) as f64;
    let mag = if e == 0 {
        f * 5.960464477539063e-08
    } else {
        let mut p = 1.0f64;
        for _ in 0..(e - 15).abs() {
            p *= 2.0;
        }
        let p = if e >= 15 { p } else { 1.0 / p };
        (1.0 + f / 1024.0) * p
    };
    sign * mag
}

/// Fills `bytes`/`vals` with `len` random elements of dtype `d` (0 bf16, 1 f16, 2 f32, 3 i8).
unsafe fn fill(rng: &mut Rng, d: usize, len: usize, bytes: *mut u8, vals: *mut f64) {
    for i in 0..len {
        match d {
            0 => {
                let h = bf16_bits(rng.small());
                *bytes.add(2 * i) = h as u8;
                *bytes.add(2 * i + 1) = (h >> 8) as u8;
                *vals.add(i) = bf16_value(h);
            }
            1 => {
                let h = f16_bits(rng.small());
                *bytes.add(2 * i) = h as u8;
                *bytes.add(2 * i + 1) = (h >> 8) as u8;
                *vals.add(i) = f16_value(h);
            }
            2 => {
                let x = rng.small();
                let le = x.to_le_bytes();
                for (j, byte) in le.iter().enumerate() {
                    *bytes.add(4 * i + j) = *byte;
                }
                *vals.add(i) = x as f64;
            }
            _ => {
                let x = rng.next() as i8;
                *bytes.add(i) = x as u8;
                *vals.add(i) = x as f64;
            }
        }
    }
}

unsafe fn check(d: usize, f: MmFn) -> bool {
    let a = &raw mut A as *mut u8;
    let b = &raw mut B as *mut u8;
    let c = &raw mut C as *mut u8;
    let (av, bv, c0) = (
        &raw mut AV as *mut f64,
        &raw mut BV as *mut f64,
        &raw mut C0 as *mut f64,
    );
    let mut rng = Rng(7 + d as u64);
    let mut ok = true;
    for &(m, n, k) in SHAPES.iter() {
        let (mu, nu, ku) = (m.max(0) as usize, n.max(0) as usize, k.max(0) as usize);
        fill(&mut rng, d, mu * ku, a, av);
        fill(&mut rng, d, ku * nu, b, bv);
        for i in 0..mu * nu {
            let v = (i % 7) as f64 - 3.0;
            *c0.add(i) = v;
            let le = if d == 3 {
                (v as i32).to_le_bytes()
            } else {
                (v as f32).to_le_bytes()
            };
            for (j, byte) in le.iter().enumerate() {
                *c.add(4 * i + j) = *byte;
            }
        }
        f(c, a, b, m, n, k);
        for i in 0..mu {
            for j in 0..nu {
                let idx = i * nu + j;
                let (mut want, mut scale) = (*c0.add(idx), abs(*c0.add(idx)));
                for kk in 0..ku {
                    let t = *av.add(i * ku + kk) * *bv.add(kk * nu + j);
                    want += t;
                    scale += abs(t);
                }
                let raw = [
                    *c.add(4 * idx),
                    *c.add(4 * idx + 1),
                    *c.add(4 * idx + 2),
                    *c.add(4 * idx + 3),
                ];
                let good = if d == 3 {
                    i32::from_le_bytes(raw) as f64 == want
                } else {
                    let got = f32::from_le_bytes(raw) as f64;
                    let tol = (ku as f64 + 2.0) * f32::EPSILON as f64 * scale;
                    abs(got - want) <= tol
                };
                if !good && ok {
                    write(b"  mismatch at shape ");
                    write_int(m);
                    write(b"x");
                    write_int(n);
                    write(b"x");
                    write_int(k);
                    write(b" C[");
                    write_int(i as i64);
                    write(b",");
                    write_int(j as i64);
                    write(b"]\n");
                    ok = false;
                }
            }
        }
    }
    ok
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let kernels: [(&[u8], MmFn); 4] = [
        (b"bf16", mm_bf16),
        (b"f16", mm_f16),
        (b"f32", mm_f32),
        (b"i8", mm_i8),
    ];
    let mut failures = 0;
    for (d, (name, f)) in kernels.iter().enumerate() {
        let ok = unsafe { check(d, *f) };
        write(if ok { b"PASS mm " } else { b"FAIL mm " });
        write(name);
        write(b"\n");
        if !ok {
            failures += 1;
        }
    }
    exit(failures)
}
