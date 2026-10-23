//! Loops whose masked accesses get a full-vector copy (`achainsaw_ir::loops`), on the backend
//! selected by `ACHAINSAW_BACKEND`: the vector loops models write (a masked load and store on
//! every iteration, the count `n - i` or `min(n - i, vl)`, tested in the header or the body),
//! one whose result leaves the loop, one with an inner loop, and fixed-width vectors, compute
//! what they compute unsplit, for every length, and never touch memory past their buffers.

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::{loops::split_masked_loops, parse_and_validate, to_air_text};

const SRC: &str = r#"
# Masked on every iteration, the count computed in the body.
fn clamp(x:ptr, n:i64, lo:f32, hi:f32)
  entry:
    vlo = splat lo:vx
    vhi = splat hi:vx
    w = vl f32
    jmp head(0:i64)
  head(i:i64):
    c = lt i, n
    br c, body, done
  body:
    r = sub n, i
    v = ldm x[i:f32]:vx, r:f32
    a = vmax v, vlo:f32
    b = vmin a, vhi:f32
    stm x[i:f32], b, r:f32
    j = add i, w
    jmp head(j)
  done:
    ret

# The count in the header, as `min(n - i, vl)`; the header tests it.
fn axpy(a:f32, x:ptr, y:ptr, n:i64)
  entry:
    va = splat a:vx
    w = vl f32
    jmp loop(0:i64)
  loop(i:i64):
    rem = sub n, i
    more = gt rem, 0:i64
    br more, body, done
  body:
    m = min rem, w
    xv = ldm x[i]:vx, m:f32
    yv = ldm y[i]:vx, m:f32
    r = vfma va, xv, yv:f32
    stm y[i], r, m:f32
    i2 = add i, w
    jmp loop(i2)
  done:
    ret

# The sum leaves the loop through the header's parameter.
fn sum(x:ptr, n:i64)->i32
  entry:
    w = vl i32
    z = splat 0:i32:vx
    jmp loop(0:i64, z)
  loop(i:i64, acc:vx):
    c = lt i, n
    br c, body, done
  body:
    r = sub n, i
    v = ldm x[i:i32]:vx, r:i32
    acc2 = vadd acc, v:i32
    i2 = add i, w
    jmp loop(i2, acc2)
  done:
    s = vsum acc:i32
    ret s

# An inner loop between the masked load and store (Horner's rule, integer lanes).
fn horner(c:ptr, deg:i64, x:ptr, y:ptr, n:i64)
  entry:
    w = vl i32
    jmp loop(0:i64)
  loop(i:i64):
    left = sub n, i
    g = gt left, 0:i64
    br g, body, done
  body:
    m = min left, w
    xv = ldm x[i:i32]:vx, m:i32
    top = ld c[deg]:i32
    a0 = splat top:vx
    jmp inner(a0, deg)
  inner(acc:vx, k:i64):
    more = gt k, 0:i64
    br more, step, fin
  step:
    k1 = sub k, 1:i64
    ck = ld c[k1]:i32
    cv = splat ck:vx
    t = vmul acc, xv:i32
    a2 = vadd t, cv:i32
    jmp inner(a2, k1)
  fin:
    stm y[i:i32], acc, m:i32
    i2 = add i, w
    jmp loop(i2)
  done:
    ret

# Fixed-width vectors, the result passed to the exit as a block argument.
fn sum128(x:ptr, n:i64)->i32
  entry:
    z = splat 0:i32:v128
    jmp loop(0:i64, z)
  loop(i:i64, acc:v128):
    c = lt i, n
    br c, body, done(acc)
  body:
    r = sub n, i
    v = ldm x[i:i32]:v128, r:i32
    acc2 = vadd acc, v:i32
    i2 = add i, 4:i64
    jmp loop(i2, acc2)
  done(a:v128):
    s = vsum a:i32
    ret s

# A count the guard cannot cover (a constant): left alone.
fn first3(x:ptr, y:ptr)
  entry:
    jmp loop(0:i64)
  loop(i:i64):
    c = lt i, 1:i64
    br c, body, done
  body:
    v = ldm x[i:f32]:vx, 3:f32
    stm y[i:f32], v, 3:f32
    i2 = add i, 1:i64
    jmp loop(i2)
  done:
    ret
"#;

fn engine() -> JitEngine {
    let module = parse_and_validate(SRC).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    e
}

#[test]
fn masked_loops_are_split_into_valid_air() {
    let mut module = parse_and_validate(SRC).unwrap();
    assert_eq!(split_masked_loops(&mut module), 5);
    let text = to_air_text(&module);
    let reparsed = parse_and_validate(&text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    // Each split loop's copy loads and stores whole vectors.
    for f in ["clamp", "axpy", "sum", "horner", "sum128"] {
        let body = text.split(&format!("fn {f}(")).nth(1).unwrap();
        let body = body.split("\nfn ").next().unwrap();
        assert!(body.contains("_guard"), "{f} was not split:\n{body}");
        assert!(
            body.contains(" = ld x["),
            "{f} copy has no plain load:\n{body}"
        );
    }
    // Splitting again finds nothing new to split in the copies.
    let mut again = reparsed;
    let first3 = again
        .functions
        .iter()
        .position(|f| f.name == "first3")
        .unwrap();
    let before = again.functions[first3].clone();
    split_masked_loops(&mut again);
    assert_eq!(again.functions[first3], before);
}

#[test]
fn split_loops_compute_what_masked_loops_compute() {
    let e = engine();
    let lanes = e.vx_bits() as usize / 32;
    let clamp: extern "C" fn(*mut f32, i64, f32, f32) =
        unsafe { std::mem::transmute(e.get_fn_ptr("clamp").unwrap()) };
    let axpy: extern "C" fn(f32, *const f32, *mut f32, i64) =
        unsafe { std::mem::transmute(e.get_fn_ptr("axpy").unwrap()) };
    let horner: extern "C" fn(*const i32, i64, *const i32, *mut i32, i64) =
        unsafe { std::mem::transmute(e.get_fn_ptr("horner").unwrap()) };
    let sum128: extern "C" fn(*const i32, i64) -> i32 =
        unsafe { std::mem::transmute(e.get_fn_ptr("sum128").unwrap()) };
    let coeffs = [3i32, -2, 5, 1];
    for n in (0..=4 * lanes + 3).chain([100, 1001]) {
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.37).collect();
        // One element past n checks that nothing writes there.
        let mut x = src.clone();
        x.push(9.0);
        clamp(x.as_mut_ptr(), n as i64, -1.0, 1.5);
        let want: Vec<f32> = src.iter().map(|v| v.clamp(-1.0, 1.5)).collect();
        assert_eq!(&x[..n], &want[..], "clamp n={n}");
        assert_eq!(x[n], 9.0, "clamp wrote past n={n}");

        let mut y: Vec<f32> = (0..=n).map(|i| i as f32).collect();
        axpy(2.0, src.as_ptr(), y.as_mut_ptr(), n as i64);
        for i in 0..n {
            assert_eq!(y[i], 2.0f32.mul_add(src[i], i as f32), "axpy n={n} i={i}");
        }
        assert_eq!(y[n], n as f32, "axpy wrote past n={n}");

        let xi: Vec<i32> = (0..n as i32).map(|i| i % 7 - 3).collect();
        let got = unsafe {
            e.call_typed(
                "sum",
                &[RtValue::Ptr(xi.as_ptr() as usize), RtValue::I64(n as i64)],
            )
        };
        assert_eq!(
            got.unwrap(),
            Some(RtValue::I32(xi.iter().sum())),
            "sum n={n}"
        );

        let mut yi = vec![77i32; n + 1];
        horner(coeffs.as_ptr(), 3, xi.as_ptr(), yi.as_mut_ptr(), n as i64);
        for i in 0..n {
            let t = xi[i];
            let want = coeffs
                .iter()
                .rev()
                .fold(0i32, |acc, &c| acc.wrapping_mul(t).wrapping_add(c));
            assert_eq!(yi[i], want, "horner n={n} i={i}");
        }
        assert_eq!(yi[n], 77, "horner wrote past n={n}");

        let got = sum128(xi.as_ptr(), n as i64);
        assert_eq!(got, xi.iter().sum::<i32>(), "sum128 n={n}");
    }
}

/// The copies read and write whole vectors only while `n - i` covers them: buffers that end
/// at an inaccessible page never fault.
#[cfg(unix)]
#[test]
fn split_loops_never_touch_memory_past_the_buffer() {
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
    } as *mut u8;
    assert_ne!(base.cast(), libc::MAP_FAILED);
    let guard = unsafe { base.add(page) };
    assert_eq!(
        unsafe { libc::mprotect(guard.cast(), page, libc::PROT_NONE) },
        0
    );

    let e = engine();
    let lanes = e.vx_bits() as usize / 32;
    let clamp: extern "C" fn(*mut f32, i64, f32, f32) =
        unsafe { std::mem::transmute(e.get_fn_ptr("clamp").unwrap()) };
    let sum128: extern "C" fn(*const i32, i64) -> i32 =
        unsafe { std::mem::transmute(e.get_fn_ptr("sum128").unwrap()) };
    for n in 0..=3 * lanes + 1 {
        let x = unsafe { guard.sub(n * 4) } as *mut f32;
        clamp(x, n as i64, -1.0, 1.0);
        sum128(x as *const i32, n as i64);
    }
    unsafe { libc::munmap(base.cast(), 2 * page) };
}

/// The split loops compile ahead of time for every target, scalable SVE vectors included.
#[test]
fn split_loops_compile_for_every_target() {
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
            (
                "aarch64-unknown-linux-gnu",
                "generic",
                "+sve",
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

/// Every example and kernel (the standard library's loops included) is valid AIR after
/// splitting.
#[test]
fn split_examples_stay_valid() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples");
    let mut split = 0;
    for dir in [root.to_string(), format!("{root}/kernels")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "air") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            let mut module = parse_and_validate(&src).unwrap();
            achainsaw_ir::inline::inline_in_place(&mut module);
            split += split_masked_loops(&mut module);
            let text = to_air_text(&module);
            parse_and_validate(&text)
                .unwrap_or_else(|d| panic!("{}: {d:?}\n{text}", path.display()));
        }
    }
    assert!(split > 10, "{split}");
}
