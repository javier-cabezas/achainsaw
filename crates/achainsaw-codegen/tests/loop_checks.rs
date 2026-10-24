//! Loops that pay for their checks on entry (`achainsaw_ir::checks`), on the backend selected
//! by `ACHAINSAW_BACKEND`. Every program comes twice: as written, which the pass rewrites, and
//! with the loop bound copied in the header (`nn = or n, 0`), which it leaves alone but which
//! branches and accesses memory exactly the same. Under a fuel budget and in the sandbox, the
//! two give the same results, use the same fuel, write the same memory before running out of
//! fuel, and fail with the same reports.

use achainsaw_codegen::{get_remaining_fuel, set_execution_fuel, JitEngine, RtValue};
use achainsaw_ir::ast::{Block, Function, Module, Terminator};
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;
use std::collections::HashMap;

/// `{PRE}` defines `nn` in a loop header; `{N}` is the bound the header tests and `@` ends
/// the names of functions with such a loop.
const TEMPLATE: &str = r#"
# A scalar sum: `lt i, n`.
fn sum@(x:ptr, n:i64)->i64
  entry:
    jmp head(0:i64, 0:i64)
  head(i:i64, s:i64):
{PRE}
    c = lt i, {N}
    br c, body, done
  body:
    v = ld x[i]:i64
    s2 = add s, v
    i2 = add i, 1:i64
    jmp head(i2, s2)
  done:
    ret s

# Stores: what a loop that runs out of fuel has written shows where it stopped.
fn fill@(x:ptr, n:i64)
  entry:
    jmp head(0:i64)
  head(i:i64):
{PRE}
    c = gt {N}, i
    br c, body, done
  body:
    i1 = add i, 1:i64
    st x[i], i1
    jmp head(i1)
  done:
    ret

# A load in the header, which runs once more than the body: it reads x[n].
fn peek@(x:ptr, n:i64)->i64
  entry:
    jmp head(0:i64, 0:i64)
  head(i:i64, s:i64):
    v = ld x[i]:i64
{PRE}
    c = lt i, {N}
    br c, body, done
  body:
    s2 = add s, v
    i2 = add i, 1:i64
    jmp head(i2, s2)
  done:
    ret s

# Whole vectors while `n - i >= vl`, by `vl` lanes, the tail left out.
fn sumv@(x:ptr, n:i64)->i32
  entry:
    w = vl i32
    z = splat 0:i32:vx
    jmp head(0:i64, z)
  head(i:i64, acc:vx):
{PRE}
    r = sub {N}, i
    c = ge r, w
    br c, body, done
  body:
    v = ld x[i:i32]:vx
    a2 = vadd acc, v:i32
    i2 = add i, w
    jmp head(i2, a2)
  done:
    t = vsum acc:i32
    ret t

# A counter that wraps past i64::MAX for some n, so the loop never ends.
fn steps@(n:i64)->i64
  entry:
    jmp head(0:i64, 0:i64)
  head(i:i64, k:i64):
{PRE}
    c = lt i, {N}
    br c, body, done
  body:
    i2 = add i, 3:i64
    k2 = add k, 1:i64
    jmp head(i2, k2)
  done:
    ret k

# An inner loop, entered once per row; the outer loop keeps its checks.
fn rows@(x:ptr, m:i64, n:i64)->i64
  entry:
    jmp outer(0:i64, 0:i64)
  outer(r:i64, s:i64):
    more = lt r, m
    br more, start, done
  start:
    base = mul r, n
    jmp inner(0:i64, s)
  inner(j:i64, t:i64):
{PRE}
    c = lt j, {N}
    br c, step, next
  step:
    k = add base, j
    v = ld x[k]:i64
    t2 = add t, v
    j2 = add j, 1:i64
    jmp inner(j2, t2)
  next:
    r2 = add r, 1:i64
    jmp outer(r2, t)
  done:
    ret s

# Bottom-tested: the body runs before the test, once even when n <= 1.
fn bsum@(x:ptr, n:i64)->i64
  entry:
    jmp loop(0:i64, 0:i64)
  loop(i:i64, s:i64):
    v = ld x[i]:i64
    s2 = add s, v
    i2 = add i, 1:i64
{PRE}
    c = lt i2, {N}
    br c, loop(i2, s2), done(s2)
  done(t:i64):
    ret t

# Tested in the middle: the blocks up to the test run once more than the rest.
fn msum@(x:ptr, n:i64)->i64
  entry:
    jmp head(0:i64, 0:i64)
  head(i:i64, s:i64):
    v = ld x[i]:i64
    jmp test
  test:
    w = ld x[i]:i64
{PRE}
    c = gt {N}, i
    br c, tail, done
  tail:
    s2 = add s, v
    s3 = add s2, w
    i2 = add i, 1:i64
    jmp head(i2, s3)
  done:
    ret s

# An if/else in the body, both ways one branch long, with a load on one way only.
fn relu@(x:ptr, y:ptr, n:i64)->i64
  entry:
    jmp head(0:i64, 0:i64)
  head(i:i64, s:i64):
{PRE}
    c = lt i, {N}
    br c, body, done
  body:
    v = ld x[i]:i64
    below0 = lt v, 0:i64
    br below0, minus, plus
  minus:
    u = ld y[i]:i64
    jmp merge(u)
  plus:
    jmp merge(v)
  merge(m:i64):
    s2 = add s, m
    i2 = add i, 1:i64
    jmp head(i2, s2)
  done:
    ret s

# Ways of different lengths: the fuel of a trip depends on the data, so it is left alone.
fn skew@(x:ptr, n:i64)->i64
  entry:
    jmp head(0:i64, 0:i64)
  head(i:i64, s:i64):
{PRE}
    c = lt i, {N}
    br c, body, done
  body:
    v = ld x[i]:i64
    below0 = lt v, 0:i64
    br below0, minus, merge(v)
  minus:
    u = sub 0:i64, v
    jmp merge(u)
  merge(m:i64):
    s2 = add s, m
    i2 = add i, 1:i64
    jmp head(i2, s2)
  done:
    ret s

# A pointer that moves 8 bytes per trip, to an end pointer.
fn psum@(x:ptr, n:i64)->i64
  entry:
    b = mul n, 8:i64
    end = add x, b
    jmp loop(x, 0:i64)
  loop(p:ptr, s:i64):
{PRE:end}
    c = lt p, {B:end}
    br c, body, done
  body:
    v = ld p:i64
    s2 = add s, v
    p2 = add p, 8:i64
    jmp loop(p2, s2)
  done:
    ret s

# The bound passed around the loop as a parameter.
fn fsum@(x:ptr, n:i64)->i64
  entry:
    jmp loop(0:i64, 0:i64, n)
  loop(i:i64, s:i64, m:i64):
{PRE:m}
    c = lt i, {B:m}
    br c, body, done
  body:
    v = ld x[i]:i64
    s2 = add s, v
    i2 = add i, 1:i64
    jmp loop(i2, s2, m)
  done:
    ret s

# Sandboxed drivers: memory from `alloc`, at `off` bytes into a 64-byte allocation.
fn walk@(off:i64, n:i64)->i64
  entry:
    p = alloc 64:i64
    q = add p, off
    call fill@(q, n)
    s = call peek@(q, n)
    free p
    ret s

fn walk2@(off:i64, n:i64)->i64
  entry:
    p = alloc 64:i64
    q = add p, off
    a = call bsum@(q, n)
    b = call msum@(q, n)
    c = call relu@(q, q, n)
    d = call psum@(q, n)
    e = call fsum@(q, n)
    s1 = add a, b
    s2 = add s1, c
    s3 = add s2, d
    s4 = add s3, e
    free p
    ret s4

fn total@(n:i64)->i64
  entry:
    bytes = mul n, 8:i64
    bytes1 = add bytes, 8:i64
    p = alloc bytes1
    n1 = add n, 1:i64
    call fill@(p, n1)
    a = call sum@(p, n)
    b = call peek@(p, n)
    h = ushr n, 1:i64
    c = call rows@(p, h, 2:i64)
    s1 = add a, b
    s2 = add s1, c
    free p
    ret s2

# `par` workers, each summing its own row.
fn row_sum@(i:i64, x:ptr, out:ptr, n:i64)
  entry:
    off = mul i, n
    o8 = mul off, 8:i64
    p = add x, o8
    s = call sum@(p, n)
    st out[i], s
    ret

fn par_rows@(x:ptr, out:ptr, m:i64, n:i64)
  entry:
    par m, row_sum@(x, out, n)
    ret
"#;

/// The program as written (`twin` false) or with the bound copied in the loop headers.
fn program(twin: bool) -> String {
    let (suffix, pre, n) = if twin {
        ("_ref", "    nn = or n, 0:i64", "nn")
    } else {
        ("", "", "n")
    };
    let mut text = TEMPLATE
        .replace('@', suffix)
        .replace("{PRE}\n", &format!("{pre}\n"))
        .replace("{N}", n);
    // `{PRE:x}` and `{B:x}`: the same for a bound `x` other than n (a pointer, or a
    // parameter), copied as `x + 0`.
    for x in ["end", "m"] {
        let (pre, bound) = if twin {
            (format!("    bb_{x} = add {x}, 0:i64\n"), format!("bb_{x}"))
        } else {
            (String::new(), x.to_string())
        };
        text = text
            .replace(&format!("{{PRE:{x}}}\n"), &pre)
            .replace(&format!("{{B:{x}}}"), &bound);
    }
    text
}

fn source() -> String {
    format!("{}\n{}", program(false), program(true))
}

/// An engine compiled under a fuel budget (so its loops are rewritten), sandboxed if asked.
fn engine(sandbox: Option<usize>) -> JitEngine {
    let module = parse_and_validate(&source()).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    e.set_fuel(Some(u64::MAX >> 2));
    if let Some(bytes) = sandbox {
        e.enable_sandbox(bytes).unwrap();
    }
    e.compile_module(&module).unwrap();
    set_execution_fuel(None);
    e
}

/// Result, or the error, and the fuel left, of `name(args)` under `budget`.
fn run(
    e: &JitEngine,
    name: &str,
    args: &[RtValue],
    budget: u64,
) -> (Result<Option<RtValue>, String>, Option<u64>) {
    set_execution_fuel(Some(budget));
    let r = unsafe { e.call_typed(name, args) }.map_err(|e| e.to_string());
    let left = get_remaining_fuel();
    set_execution_fuel(None);
    (r, left)
}

fn ptr<T>(v: &[T]) -> RtValue {
    RtValue::Ptr(v.as_ptr() as usize)
}

fn ptr_mut<T>(v: &mut [T]) -> RtValue {
    RtValue::Ptr(v.as_mut_ptr() as usize)
}

/// The rewritten loops are the ones the twins differ in.
#[test]
fn loops_with_known_trip_counts_are_rewritten() {
    let module = parse_and_validate(&source()).unwrap();
    let mut module = achainsaw_ir::prepare_for_codegen(&module).into_owned();
    let checks = achainsaw_ir::checks::hoist_loop_checks(&mut module);
    for f in [
        "sum", "fill", "peek", "sumv", "steps", "rows", "bsum", "msum", "relu", "psum", "fsum",
    ] {
        assert!(checks.function(f).is_some(), "{f} was not rewritten");
        let twin = format!("{f}_ref");
        assert!(checks.function(&twin).is_none(), "{twin} was rewritten");
    }
    assert!(checks.function("skew").is_none(), "skew was rewritten");
    // The rewritten module is valid AIR.
    let text = achainsaw_ir::to_air_text(&module);
    parse_and_validate(&text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
}

#[test]
fn rewritten_loops_use_the_same_fuel() {
    let e = engine(None);
    let lanes = e.vx_bits() as usize / 32;
    for n in (0..=3 * lanes + 2).chain([100, 1001]) {
        let x: Vec<i64> = (0..=n as i64).map(|i| i * 7 - 3).collect();
        let x32: Vec<i32> = (0..n as i32).map(|i| i % 5 - 2).collect();
        let cases: Vec<(&str, Vec<RtValue>)> = vec![
            ("sum", vec![ptr(&x), RtValue::I64(n as i64)]),
            ("peek", vec![ptr(&x), RtValue::I64(n as i64)]),
            ("sumv", vec![ptr(&x32), RtValue::I64(n as i64)]),
            ("steps", vec![RtValue::I64(n as i64)]),
            ("bsum", vec![ptr(&x), RtValue::I64(n as i64)]),
            ("msum", vec![ptr(&x), RtValue::I64(n as i64)]),
            ("relu", vec![ptr(&x), ptr(&x), RtValue::I64(n as i64)]),
            ("skew", vec![ptr(&x), RtValue::I64(n as i64)]),
            ("psum", vec![ptr(&x), RtValue::I64(n as i64)]),
            ("fsum", vec![ptr(&x), RtValue::I64(n as i64)]),
            (
                "rows",
                vec![ptr(&x), RtValue::I64(3), RtValue::I64(n as i64 / 3)],
            ),
        ];
        for (f, args) in cases {
            let got = run(&e, f, &args, 1 << 40);
            let want = run(&e, &format!("{f}_ref"), &args, 1 << 40);
            assert!(got.0.is_ok(), "{f} n={n}: {:?}", got.0);
            assert_eq!(got, want, "{f} n={n}");
        }
    }
    // A counter that wraps never stops: both run out of fuel.
    let args = [RtValue::I64(i64::MAX)];
    let got = run(&e, "steps", &args, 100_000);
    assert_eq!(got, run(&e, "steps_ref", &args, 100_000));
    assert!(got.0.unwrap_err().contains("ERR_OUT_OF_FUEL"));
}

/// A budget the loop exceeds: the original loop runs, and stops at the same store.
#[test]
fn loops_that_run_out_of_fuel_stop_where_they_did() {
    let e = engine(None);
    let n = 30;
    for budget in 0..2 * n as u64 + 8 {
        let mut ours = vec![0i64; n];
        let mut theirs = vec![0i64; n];
        let got = run(
            &e,
            "fill",
            &[ptr_mut(&mut ours), RtValue::I64(n as i64)],
            budget,
        );
        let want = run(
            &e,
            "fill_ref",
            &[ptr_mut(&mut theirs), RtValue::I64(n as i64)],
            budget,
        );
        assert_eq!(got, want, "budget {budget}");
        assert_eq!(ours, theirs, "budget {budget}");
    }
}

/// Sandboxed loops that leave the arena fail as they did, at the same access, also when the
/// header's extra load is the one outside; those inside it give the same results.
#[test]
fn sandboxed_loops_fail_where_they_did() {
    let arena = 4096;
    let e = engine(Some(arena));
    for off in [0i64, 8, 4000, 4032, 4040, 4088, 4096, -8] {
        for n in [0i64, 1, 2, 3, 6, 7, 8, 500, 509, 510, 511, 512] {
            let args = [RtValue::I64(off), RtValue::I64(n)];
            for budget in [1 << 40, 40, 600] {
                for f in ["walk", "walk2"] {
                    let got = run(&e, f, &args, budget);
                    let want = run(&e, &format!("{f}_ref"), &args, budget);
                    assert_eq!(got, want, "{f} off={off} n={n} budget={budget}");
                }
            }
        }
    }
    for n in [0i64, 1, 5, 64, 300] {
        let args = [RtValue::I64(n)];
        let got = run(&e, "total", &args, 1 << 40);
        assert!(got.0.is_ok(), "n={n}: {:?}", got.0);
        assert_eq!(got, run(&e, "total_ref", &args, 1 << 40), "n={n}");
    }
}

/// `par` workers charge whole loops from their counters, drawing on the shared budget: the
/// fuel used does not depend on the threads, and matches the original loops'.
#[test]
fn par_workers_charge_whole_loops() {
    let mut e = engine(None);
    let (m, n) = (16usize, 50_000usize);
    let x: Vec<i64> = (0..(m * n) as i64).map(|i| i % 11).collect();
    let mut used = Vec::new();
    for f in ["par_rows", "par_rows_ref"] {
        for threads in [Some(1), None] {
            e.set_threads(threads);
            let mut out = vec![0i64; m];
            let args = [
                ptr(&x),
                ptr_mut(&mut out),
                RtValue::I64(m as i64),
                RtValue::I64(n as i64),
            ];
            let budget = 1 << 40;
            let (r, left) = run(&e, f, &args, budget);
            r.unwrap();
            for (i, s) in out.iter().enumerate() {
                assert_eq!(*s, x[i * n..(i + 1) * n].iter().sum::<i64>(), "{f} row {i}");
            }
            used.push(budget - left.unwrap());
        }
    }
    assert!(used.iter().all(|&u| u == used[0]), "{used:?}");
    // A budget the rows exceed stops them.
    e.set_threads(None);
    let mut out = vec![0i64; m];
    let args = [
        ptr(&x),
        ptr_mut(&mut out),
        RtValue::I64(m as i64),
        RtValue::I64(n as i64),
    ];
    let (r, _) = run(&e, "par_rows", &args, (m * n) as u64);
    assert!(r.unwrap_err().contains("ERR_OUT_OF_FUEL"));
}

/// One quantity a dispatch block computes, as a function of the loop function's parameters
/// and the dispatch block's scalar ones: the entry block's instructions and its own, then
/// `ret` of the register.
fn probe(f: &Function, block: &Block, reg: &str) -> Function {
    let mut params = f.params.clone();
    params.extend(block.params.iter().filter(|(_, t)| !t.is_vector()).cloned());
    let mut instructions = f.blocks[0].instructions.clone();
    instructions.extend(block.instructions.iter().cloned());
    Function {
        name: format!("{}_{reg}", f.name),
        params,
        rets: vec![Type::I64],
        inline: false,
        blocks: vec![Block {
            label: "entry".into(),
            params: Vec::new(),
            instructions,
            terminator: Terminator::Ret {
                vals: vec![reg.into()],
                span: block.span,
            },
            span: block.span,
        }],
        span: f.span,
    }
}

/// Each dispatch block's arithmetic, compiled on its own: the fuel and the byte spans it
/// computes, against trip counts worked out here, near the ends of the i64 range too.
#[test]
fn dispatch_blocks_count_trips() {
    let module = parse_and_validate(&program(false)).unwrap();
    let mut module = achainsaw_ir::prepare_for_codegen(&module).into_owned();
    let checks = achainsaw_ir::checks::hoist_loop_checks(&mut module);
    let mut probes = Module {
        functions: Vec::new(),
        extern_functions: Vec::new(),
    };
    // (function, units probe, first range's span probe)
    let mut names: HashMap<String, (String, String)> = HashMap::new();
    for f in &module.functions {
        let Some(fc) = checks.function(&f.name) else {
            continue;
        };
        if f.name == "rows" {
            continue;
        }
        let (label, d) = fc.dispatch.iter().next().unwrap();
        let block = f.blocks.iter().find(|b| &b.label == label).unwrap();
        let units = probe(f, block, &d.units);
        let span = d.ranges.first().map(|(_, span)| probe(f, block, span));
        let span_name = span.as_ref().map_or(String::new(), |s| s.name.clone());
        names.insert(f.name.clone(), (units.name.clone(), span_name));
        probes.functions.push(units);
        probes.functions.extend(span);
    }
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&probes).unwrap();
    let w = e.vx_bits() as i64 / 32;
    let call = |name: &str, vals: &[RtValue]| -> i64 {
        match unsafe { e.call_typed(name, vals) }.unwrap() {
            Some(RtValue::I64(v)) => v,
            other => panic!("{other:?}"),
        }
    };
    let eval = |name: &str, args: &[i64], ptrs: usize| -> i64 {
        let mut vals: Vec<RtValue> = (0..ptrs).map(|_| RtValue::Ptr(0)).collect();
        vals.extend(args.iter().map(|&a| RtValue::I64(a)));
        call(name, &vals)
    };
    // `cost` branches a trip and `last` on the iteration that leaves.
    let fuel = |t: Option<i128>, cost: i128, last: i128| match t {
        Some(t) if t <= 1 << 48 => (t * cost + last) as i64,
        _ => achainsaw_ir::checks::ENDLESS_UNITS,
    };
    let endless = achainsaw_ir::checks::ENDLESS_UNITS;
    let (min, max) = (i64::MIN, i64::MAX);
    let ends = [
        (0, 0),
        (0, 1),
        (5, 3),
        (-7, 9),
        (0, 1000),
        (min, max),
        (min, min + 10),
        (max - 10, max),
        (max - 5, max),
        (max - 7, max - 1),
        (-1, 0),
    ];
    let trips_to = |i0: i64, n: i64, step: i64| -> Option<i128> {
        // `i < n` from i0 by `step`; `None` when the counter wraps past i64::MAX first.
        if i0 >= n {
            return Some(0);
        }
        let t = (n as i128 - i0 as i128 + step as i128 - 1) / step as i128;
        let last = i0 as i128 + (t - 1) * step as i128;
        (last + step as i128 <= max as i128).then_some(t)
    };
    let units_of = |t: Option<i128>| match t {
        Some(t) if t <= 1 << 48 => (2 * t + 1) as i64,
        _ => endless,
    };
    // Bytes from the first access to the end of the last: the header's accesses run once
    // more than the body's, and more than 2^40 trips never fit.
    let span_of = |trips: Option<i128>, header: bool, stride: i128, bytes: i128| match trips {
        Some(0) if !header => 0,
        Some(t) if t <= 1 << 40 => ((t - !header as i128) * stride + bytes) as i64,
        _ => -1,
    };
    for (i0, n) in ends {
        let t = trips_to(i0, n, 1);
        let (u, s) = &names["sum"];
        assert_eq!(eval(u, &[n, i0, 0], 1), units_of(t), "sum {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0, 0], 1),
            span_of(t, false, 8, 8),
            "sum {i0} {n}"
        );
        let (u, s) = &names["fill"];
        assert_eq!(eval(u, &[n, i0], 1), units_of(t), "fill {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0], 1),
            span_of(t, false, 8, 8),
            "fill {i0} {n}"
        );
        // The header's load runs once more.
        let (u, s) = &names["peek"];
        assert_eq!(eval(u, &[n, i0, 0], 1), units_of(t), "peek {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0, 0], 1),
            span_of(t, true, 8, 8),
            "peek {i0} {n}"
        );
        let t3 = trips_to(i0, n, 3);
        let (u, _) = &names["steps"];
        assert_eq!(eval(u, &[n, i0, 0], 0), units_of(t3), "steps {i0} {n}");
        // `n - i >= w` by w (wrapping difference), whole vectors of w i32 lanes.
        let r0 = n.wrapping_sub(i0);
        let tv = if r0 >= w {
            Some(((r0 - w) / w + 1) as i128)
        } else {
            Some(0)
        };
        let (u, s) = &names["sumv"];
        assert_eq!(eval(u, &[n, i0], 1), units_of(tv), "sumv {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0], 1),
            span_of(tv, false, 4 * w as i128, 4 * w as i128),
            "sumv {i0} {n}"
        );
        // Tested at the bottom, on i + 1: the block runs once more than the tests that pass.
        let tb = trips_to(i0.wrapping_add(1), n, 1);
        let (u, s) = &names["bsum"];
        assert_eq!(eval(u, &[n, i0, 0], 1), fuel(tb, 1, 1), "bsum {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0, 0], 1),
            span_of(tb, true, 8, 8),
            "bsum {i0} {n}"
        );
        // Tested in the middle: 3 branches a trip, 2 on the last iteration.
        let (u, s) = &names["msum"];
        assert_eq!(eval(u, &[n, i0, 0], 1), fuel(t, 3, 2), "msum {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0, 0], 1),
            span_of(t, true, 8, 8),
            "msum {i0} {n}"
        );
        // The bound passed around the loop: the dispatch block's parameter.
        let (u, s) = &names["fsum"];
        assert_eq!(eval(u, &[n, i0, 0, n], 1), fuel(t, 2, 1), "fsum {i0} {n}");
        assert_eq!(
            eval(s, &[n, i0, 0, n], 1),
            span_of(t, false, 8, 8),
            "fsum {i0} {n}"
        );
        // A pointer from i0 by 8 bytes to x + 8n, x = 0.
        let end = n.wrapping_mul(8);
        let tp = trips_to(i0, end, 8);
        let (u, s) = &names["psum"];
        let args = [
            RtValue::Ptr(0),
            RtValue::I64(n),
            RtValue::Ptr(i0 as usize),
            RtValue::I64(0),
        ];
        assert_eq!(call(u, &args), fuel(tp, 2, 1), "psum {i0} {n}");
        assert_eq!(call(s, &args), span_of(tp, false, 8, 8), "psum {i0} {n}");
    }
}
