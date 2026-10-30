//! Module `data` (`addr`) and `stack` buffers on the backend selected by `ACHAINSAW_BACKEND`:
//! values of every element type, alignment, buffers that are distinct per instruction and
//! per call (recursion, `par` workers, an inlined helper in a loop), the same programs in the
//! sandbox (where `data` is read-only and `stack` buffers come from the arena), fuel, and
//! object code for every target.

use achainsaw_codegen::{get_execution_status, ExecutionStatus, JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

const PROGRAM: &str = r#"
data bytes:i8 = -1, 2, -3, 127, -128
data shorts:i16 = -2, 30000
data words:i32 = 7, -8, 2147483647
data longs:i64 = -9223372036854775808, 42
data floats:f32 = 1.5, -0.25, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
data doubles:f64 = 2.5e300, -0.125

fn get_i8(i:i64)->i64
  b0:
    p = addr bytes
    v = ld p[i]:i8
    w = sext v:i64
    ret w

fn get_i16(i:i64)->i64
  b0:
    p = addr shorts
    v = ld p[i]:i16
    w = sext v:i64
    ret w

fn get_i32(i:i64)->i64
  b0:
    p = addr words
    v = ld p[i]:i32
    w = sext v:i64
    ret w

fn get_i64(i:i64)->i64
  b0:
    p = addr longs
    v = ld p[i]:i64
    ret v

fn get_f32(i:i64)->f32
  b0:
    p = addr floats
    v = ld p[i]:f32
    ret v

fn get_f64(i:i64)->f64
  b0:
    p = addr doubles
    v = ld p[i]:f64
    ret v

# The low 6 bits of every table's address, or'd: 0 when all are 64-byte aligned.
fn misalignment()->i64
  b0:
    a = addr bytes
    b = addr shorts
    c = addr words
    d = addr longs
    e = addr floats
    f = addr doubles
    ai = bitcast a:i64
    bi = bitcast b:i64
    ci = bitcast c:i64
    di = bitcast d:i64
    ei = bitcast e:i64
    fi = bitcast f:i64
    ab = or ai, bi
    cd = or ci, di
    ef = or ei, fi
    abcd = or ab, cd
    all = or abcd, ef
    low = and all, 63:i64
    ret low

# A whole-vector load, and a loop over the first n floats.
fn vector_sum()->f32
  b0:
    p = addr floats
    v = ld p:v256
    s = vsum v:f32
    ret s

fn loop_sum(n:i64)->f32
  b0:
    p = addr floats
    z = cst 0.0:f32
    jmp loop(0:i64, z)
  loop(i:i64, acc:f32):
    more = lt i, n
    br more, body, done
  body:
    v = ld p[i]:f32
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc

# Writes to data: an error in the sandbox.
fn st_data()->i32
  b0:
    p = addr words
    st p, 1:i32
    ret 0:i32

fn atomic_data()->i32
  b0:
    p = addr words
    old = atomic add p, 1:i32
    ret old

fn stm_data()->i32
  b0:
    p = addr floats
    one = cst 1.0:f32
    v = splat one:v128
    stm p, v, 2:f32
    ret 0:i32

fn scatter_data()->i32
  b0:
    p = addr floats
    one = cst 1.0:f32
    v = splat one:v128
    ix = splat 1:i32:v128
    vscatter p, ix, v, 2:f32
    ret 0:i32

fn mm_data()->i32
  b0:
    c = addr floats
    a = alloc 64:i64
    b = alloc 64:i64
    mm c, a, b, 2:i64, 2:i64, 2:i64:f32
    ret 0:i32

fn loop_write(n:i64)->i32
  b0:
    p = addr floats
    one = cst 1.0:f32
    jmp loop(0:i64)
  loop(i:i64):
    more = lt i, n
    br more, body, done
  body:
    st p[i], one
    i2 = add i, 1:i64
    jmp loop(i2)
  done:
    ret 0:i32

# Two buffers of one frame: 2x + 1.
fn two_buffers(x:i64)->i64
  b0:
    a = stack 8
    b = stack 24
    st a, x
    y = add x, 1:i64
    two = cst 2:i64
    st b[two], y
    ra = ld a:i64
    rb = ld b[two]:i64
    s = add ra, rb
    ret s

# Each call's buffer survives the calls it makes: 1 + 2 + ... + n.
fn rec(n:i64)->i64
  b0:
    p = stack 16
    st p, n
    z = le n, 0:i64
    br z, base, more
  base:
    ret 0:i64
  more:
    m = sub n, 1:i64
    r = call rec(m)
    v = ld p:i64
    s = add r, v
    ret s

# An inlined helper's buffer in a loop: 1 + 2 + ... + n.
inline fn bump(x:i64)->i64
  b0:
    p = stack 8
    st p, x
    v = ld p:i64
    w = add v, 1:i64
    ret w

fn loop_scratch(n:i64)->i64
  b0:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    v = call bump(i)
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc

# n calls of a function with buffers: none of them stays allocated.
fn many(n:i64)->i64
  b0:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    v = call two_buffers(i)
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc

# Each `par` index writes its own frame's buffer: 3 * (0 + 1 + ... + n - 1).
fn worker(i:i64, out:ptr)
  b0:
    p = stack 256
    st p[i], i
    v = ld p[i]:i64
    w = mul v, 3:i64
    st out[i], w
    ret

fn par_total(n:i64)->i64
  b0:
    bytes = mul n, 8:i64
    out = alloc bytes
    par n, worker(out)
    jmp loop(0:i64, 0:i64)
  loop(i:i64, acc:i64):
    more = lt i, n
    br more, body, done
  body:
    v = ld out[i]:i64
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    free out
    ret acc
"#;

fn engine(sandbox: bool) -> JitEngine {
    let module = parse_and_validate(PROGRAM).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    if sandbox {
        e.enable_sandbox(1 << 16).unwrap();
    }
    e.compile_module(&module).unwrap();
    e
}

fn call(e: &JitEngine, name: &str, args: &[i64]) -> Result<Option<RtValue>, String> {
    let args: Vec<RtValue> = args.iter().map(|&a| RtValue::I64(a)).collect();
    unsafe { e.call_typed(name, &args) }.map_err(|err| err.to_string())
}

fn int(e: &JitEngine, name: &str, args: &[i64]) -> i64 {
    match call(e, name, args).unwrap_or_else(|err| panic!("{name}: {err}")) {
        Some(RtValue::I64(v)) => v,
        other => panic!("{name}: {other:?}"),
    }
}

fn float(e: &JitEngine, name: &str, args: &[i64]) -> f64 {
    match call(e, name, args).unwrap_or_else(|err| panic!("{name}: {err}")) {
        Some(RtValue::F32(v)) => v as f64,
        Some(RtValue::F64(v)) => v,
        other => panic!("{name}: {other:?}"),
    }
}

/// The reads and the `stack` programs, which behave the same in and out of the sandbox.
fn check_programs(e: &JitEngine) {
    for (i, want) in [-1, 2, -3, 127, -128].into_iter().enumerate() {
        assert_eq!(int(e, "get_i8", &[i as i64]), want);
    }
    assert_eq!(int(e, "get_i16", &[0]), -2);
    assert_eq!(int(e, "get_i16", &[1]), 30000);
    assert_eq!(int(e, "get_i32", &[1]), -8);
    assert_eq!(int(e, "get_i32", &[2]), i32::MAX as i64);
    assert_eq!(int(e, "get_i64", &[0]), i64::MIN);
    assert_eq!(int(e, "get_i64", &[1]), 42);
    assert_eq!(float(e, "get_f32", &[1]), -0.25);
    assert_eq!(float(e, "get_f32", &[15]), 16.0);
    assert_eq!(float(e, "get_f64", &[0]), 2.5e300);
    assert_eq!(float(e, "get_f64", &[1]), -0.125);
    assert_eq!(int(e, "misalignment", &[]), 0);
    assert_eq!(float(e, "vector_sum", &[]), 34.25);
    assert_eq!(float(e, "loop_sum", &[16]), 34.25 + 100.0);

    assert_eq!(int(e, "two_buffers", &[20]), 41);
    assert_eq!(int(e, "rec", &[100]), 5050);
    assert_eq!(int(e, "loop_scratch", &[10]), 55);
    assert_eq!(int(e, "many", &[10_000]), 10_000 * 10_000);
    assert_eq!(int(e, "par_total", &[32]), 3 * 31 * 32 / 2);
}

#[test]
fn data_and_stack_buffers_run() {
    let e = engine(false);
    check_programs(&e);
    // Out of the sandbox a write to data faults, so only reads run here.
}

#[test]
fn data_and_stack_buffers_run_under_a_fuel_budget() {
    let mut e = engine(false);
    e.set_fuel(Some(1 << 40));
    check_programs(&e);
}

#[test]
fn sandboxed_data_is_read_only_and_stack_buffers_come_from_the_arena() {
    let e = engine(true);
    check_programs(&e);
    // Every way of writing: plain, atomic, masked, scattered, `mm`'s result, and stores in a loop
    // whose accesses the loop's dispatch checks at once.
    for (name, args) in [
        ("st_data", vec![]),
        ("atomic_data", vec![]),
        ("stm_data", vec![]),
        ("scatter_data", vec![]),
        ("mm_data", vec![]),
        ("loop_write", vec![4]),
    ] {
        let err = call(&e, name, &args).expect_err(name);
        assert!(
            err.contains("ERR_MEMORY_VIOLATION") && err.contains("read-only"),
            "{name}: {err}"
        );
        assert!(
            matches!(
                get_execution_status(),
                ExecutionStatus::ReadOnlyWrite { .. }
            ),
            "{name}: {:?}",
            get_execution_status()
        );
    }
    // The tables are unchanged, and the engine still runs.
    assert_eq!(int(&e, "get_i32", &[0]), 7);
    assert_eq!(float(&e, "loop_sum", &[16]), 134.25);
    check_programs(&e);
}

#[test]
fn sandbox_arena_must_hold_the_data() {
    let module = parse_and_validate(&format!(
        "data big:i64 = {}\nfn k()->ptr\n  b0:\n    p = addr big\n    ret p\n",
        vec!["0"; 1024].join(", ")
    ))
    .unwrap();
    // 8 KiB of data leave less than the 4 KiB minimum of an 8 KiB arena writable.
    let mut e = JitEngine::new().unwrap();
    e.enable_sandbox(8192).unwrap();
    let err = e.compile_module(&module).unwrap_err().to_string();
    assert!(err.contains("ERR_OUT_OF_MEMORY"), "{err}");
    let mut e = JitEngine::new().unwrap();
    e.enable_sandbox(1 << 14).unwrap();
    e.compile_module(&module).unwrap();
}

/// x86-64 and AArch64 objects (SVE too) on Cranelift and, with the `llvm` feature, LLVM.
#[test]
fn data_and_stack_compile_for_every_target() {
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

/// `examples/crc32.air`, whose table is a `data` declaration, against a bitwise CRC-32.
#[test]
fn crc32_example_matches_a_bitwise_crc() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/crc32.air");
    let module = parse_and_validate(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    let crc32: extern "C" fn(*const u8, i64) -> i32 =
        unsafe { std::mem::transmute(e.get_fn_ptr("crc32").unwrap()) };
    let bitwise = |bytes: &[u8]| {
        let mut c = !0u32;
        for &b in bytes {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    };
    assert_eq!(crc32(b"123456789".as_ptr(), 9) as u32, 0xCBF4_3926);
    let mut s = 0x9e37_79b9u32;
    let bytes: Vec<u8> = (0..4096)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect();
    for n in [0, 1, 7, 100, 4096] {
        assert_eq!(
            crc32(bytes.as_ptr(), n as i64) as u32,
            bitwise(&bytes[..n]),
            "{n} bytes"
        );
    }
}

/// A buffer defined in a loop keeps its contents across iterations, also when the loop is
/// copied: a masked loop gets a full-vector copy (`split_masked_loops`) and, under a fuel
/// budget or the sandbox, a copy without checks (`hoist_loop_checks`), which must not make
/// the buffer two.
#[test]
fn stack_buffers_survive_loop_copies() {
    let src = r#"
fn count(n:i64)->i64
  b0:
    w = vl f32
    bytes = mul n, 4:i64
    x = alloc bytes
    jmp loop(0:i64, 0:i64)
  loop(i:i64, last:i64):
    more = lt i, n
    br more, body, done
  body:
    p = stack 8
    old = ld p:i64
    first = eq i, 0:i64
    base = select first, 0:i64, old
    new = add base, 1:i64
    st p, new
    r = sub n, i
    v = ldm x[i]:vx, r:f32
    stm x[i], v, r:f32
    i2 = add i, w
    jmp loop(i2, new)
  done:
    free x
    ret last
"#;
    let module = parse_and_validate(src).unwrap();
    for (sandbox, fuel) in [(false, None), (false, Some(1u64 << 40)), (true, None)] {
        let mut e = JitEngine::new().unwrap();
        if sandbox {
            e.enable_sandbox(1 << 20).unwrap();
        }
        e.set_fuel(fuel);
        e.compile_module(&module).unwrap();
        let lanes = e.vx_bits() as i64 / 32;
        for n in [1, lanes, 2 * lanes + 3, 1000] {
            let trips = (n + lanes - 1) / lanes;
            assert_eq!(
                int(&e, "count", &[n]),
                trips,
                "n = {n}, sandbox {sandbox}, fuel {fuel:?}"
            );
        }
    }
    // The pass leaves one buffer, in the entry block.
    let prepared = achainsaw_ir::prepare_for_codegen(&module);
    let f = &prepared.functions[0];
    let stacks = |b: &achainsaw_ir::ast::Block| {
        b.instructions
            .iter()
            .filter(|i| matches!(i, achainsaw_ir::ast::Instruction::Stack { .. }))
            .count()
    };
    assert_eq!(stacks(&f.blocks[0]), 1);
    assert_eq!(f.blocks.iter().map(stacks).sum::<usize>(), 1);
}
