//! Indexed memory operands, functions with several results, `inline fn`, and the scalar
//! rounding / bit-count / rotation / copysign / fma / unsigned-conversion ops: syntax,
//! validation, inlining, and text and AIRB round trips.

use achainsaw_ir::inline::inline_module;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};

fn err_code(src: &str) -> String {
    match parse_and_validate(src) {
        Ok(_) => panic!("expected an error for:\n{src}"),
        Err(d) => d.error_code,
    }
}

/// Text and AIRB both reproduce the module exactly.
fn round_trips(src: &str) -> String {
    let module = parse_and_validate(src).unwrap_or_else(|d| panic!("{d:?}"));
    let text = to_air_text(&module);
    assert_eq!(to_air_text(&parse_and_validate(&text).unwrap()), text);
    let decoded = decode_module(&encode_module(&module).unwrap()).unwrap();
    assert_eq!(to_air_text(&decoded), text);
    text
}

const INDEXED: &str = r#"
fn k(x:ptr, y:ptr, n:i64)
  b0:
    a = ld x[n]:f32
    st y[n], a
    v = ld x[n:f32]:v256
    st y[3:f64], v
    w = ldm x[n]:vx, n:f32
    stm y[n:i8], w, n:i16
    b = ld x[7]:i64
    st x[0], b
    ret
"#;

#[test]
fn indexed_operands_round_trip() {
    let text = round_trips(INDEXED);
    for line in [
        "a = ld x[n]:f32",
        "st y[n], a",
        "v = ld x[n:f32]:v256",
        "st y[__imm_0:f64], v",
        "w = ldm x[n]:vx, n:f32",
        "stm y[n:i8], w, n:i16",
    ] {
        assert!(text.contains(line), "{line}\n{text}");
    }
}

#[test]
fn index_rules() {
    let one = |body: &str| format!("fn k(p:ptr, i:i64, j:i32)\n  b0:\n{body}    ret\n");
    // A vector access needs the unit; vx has no fixed size.
    assert_eq!(
        err_code(&one("    v = ld p[i]:v128\n    st p, v\n")),
        "ERR_INDEX_UNIT"
    );
    assert_eq!(
        err_code(&one("    v = ld p:v128\n    st p[i], v\n")),
        "ERR_INDEX_UNIT"
    );
    assert_eq!(
        err_code(&one("    v = ld p[i:vx]:vx\n    st p, v\n")),
        "ERR_INDEX_UNIT"
    );
    // The index is an i64.
    assert_eq!(err_code(&one("    v = ld p[j]:f32\n")), "ERR_TYPE_MISMATCH");
    assert_eq!(err_code(&one("    v = ld p[q]:f32\n")), "ERR_UNDEFINED_REG");
    assert!(parse_and_validate(&one("    v = ld p[i:v512]:v128\n    st p[i:i8], v\n")).is_ok());
}

const MULTI: &str = r#"
inline fn divmod(a:i64, b:i64)->(i64, i64)
  b0:
    q = div a, b
    m = rem a, b
    ret q, m

fn halves(v:vx)->(vx, vx, i32)
  b0:
    lo = vwidenlo v:i16
    hi = vwidenhi v:i16
    ret lo, hi, 1:i32

fn main(n:i64, v:vx)->i64
  b0:
    q, m = call divmod(n, 10:i64)
    call divmod(n, 3:i64)
    a, b, c = call halves(v)
    s = add q, m
    ret s
"#;

#[test]
fn several_results_and_inline_round_trip() {
    let text = round_trips(MULTI);
    assert!(
        text.contains("inline fn divmod(a:i64, b:i64)->(i64, i64)"),
        "{text}"
    );
    assert!(text.contains("fn halves(v:vx)->(vx, vx, i32)"));
    assert!(text.contains("    ret q, m"));
    assert!(text.contains("    q, m = call divmod(n, __imm_"));
    assert!(text.contains("    a, b, c = call halves(v)"));
}

#[test]
fn result_rules() {
    let src = |call: &str, ret: &str| {
        format!("fn two()->(i32, i64)\n  b0:\n    {ret}\n\nfn k()\n  b0:\n    {call}\n    ret\n")
    };
    assert!(parse_and_validate(&src("a, b = call two()", "ret 1:i32, 2:i64")).is_ok());
    assert!(parse_and_validate(&src("call two()", "ret 1:i32, 2:i64")).is_ok());
    assert_eq!(
        err_code(&src("a = call two()", "ret 1:i32, 2:i64")),
        "ERR_RESULT_COUNT"
    );
    assert_eq!(
        err_code(&src("a, b, c = call two()", "ret 1:i32, 2:i64")),
        "ERR_RESULT_COUNT"
    );
    assert_eq!(
        err_code(&src("a, b = call two()", "ret 1:i32")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&src("a, b = call two()", "ret 1:i64, 2:i64")),
        "ERR_TYPE_MISMATCH"
    );
    // Only calls assign several registers; externs return at most one value.
    assert_eq!(
        err_code("fn k(x:i32)->i32\n  b0:\n    a, b = add x, x\n    ret a\n"),
        "ERR_EXPECTED_RVALUE"
    );
    assert_eq!(
        err_code("extfn f(x:i32)->(i32, i32)\nfn k()\n  b0:\n    ret\n"),
        "ERR_MULTI_RETURN_EXTERN"
    );
    // A par body still returns nothing.
    assert_eq!(
        err_code("fn t(i:i64)->(i32, i32)\n  b0:\n    ret 1:i32, 2:i32\n\nfn k()\n  b0:\n    par 4:i64, t()\n    ret\n"),
        "ERR_PAR_SIGNATURE"
    );
}

#[test]
fn inline_recursion_is_rejected() {
    let src = "inline fn a(x:i32)->i32\n  b0:\n    r = call b(x)\n    ret r\n\n\
               inline fn b(x:i32)->i32\n  b0:\n    r = call a(x)\n    ret r\n";
    assert_eq!(err_code(src), "ERR_RECURSIVE_INLINE");
    // Recursion through a function that is not inline is fine.
    let ok = "inline fn a(x:i32)->i32\n  b0:\n    r = call b(x)\n    ret r\n\n\
              fn b(x:i32)->i32\n  b0:\n    r = call a(x)\n    ret r\n";
    assert!(parse_and_validate(ok).is_ok());
}

#[test]
fn inlining_expands_every_call_into_valid_air() {
    let src = r#"
inline fn clamp(x:i32, lo:i32, hi:i32)->i32
  b0:
    lt_lo = lt x, lo
    br lt_lo, low, upper
  low:
    ret lo
  upper:
    gt_hi = gt x, hi
    r = select gt_hi, hi, x
    ret r

inline fn clamp2(a:i32, b:i32)->(i32, i32)
  b0:
    x = call clamp(a, 0:i32, 9:i32)
    y = call clamp(b, 0:i32, 9:i32)
    ret x, y

fn main(a:i32, b:i32)->i32
  b0:
    jmp loop(0:i32, a)
  loop(i:i32, acc:i32):
    more = lt i, 3:i32
    br more, body, done
  body:
    x, y = call clamp2(acc, b)
    call clamp(x, 1:i32, 2:i32)
    acc2 = add x, y
    i2 = add i, 1:i32
    jmp loop(i2, acc2)
  done:
    ret acc
"#;
    let module = parse_and_validate(src).unwrap();
    let inlined = inline_module(&module);
    let main = inlined.functions.iter().find(|f| f.name == "main").unwrap();
    let calls = main
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .filter(|i| matches!(i, achainsaw_ir::Instruction::Call { .. }))
        .count();
    assert_eq!(calls, 0, "{}", to_air_text(&inlined));
    // The expanded module is valid AIR, and expanding it again changes nothing.
    let text = to_air_text(&inlined);
    let again = parse_and_validate(&text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    assert_eq!(to_air_text(&inline_module(&again)), text);
    // Inline functions stay, callable on their own.
    assert!(inlined
        .functions
        .iter()
        .any(|f| f.name == "clamp" && f.inline));
}

#[test]
fn scalar_ops_round_trip_and_type_rules() {
    let src = r#"
fn k(f:f32, d:f64, i:i32, l:i64)->f64
  b0:
    a = floor f
    b = ceil d
    c = round f
    e = roundeven d
    g = roundz f
    p = popcnt i
    z = clz l
    t = ctz i
    rl = rotl i, 3:i32
    rr = rotr l, l
    cs = copysign d, -1.0:f64
    m = fma d, d, b
    u = uitof i:f64
    v = ftoui d:i64
    vv = ld __unused_never[0]:f32
    ret m
"#;
    // (an undefined pointer is caught; the rest is checked below without it)
    assert_eq!(err_code(src), "ERR_UNDEFINED_REG");
    let src = src.replace("    vv = ld __unused_never[0]:f32\n", "");
    let text = round_trips(&src);
    for line in [
        "a = floor f",
        "e = roundeven d",
        "g = roundz f",
        "z = clz l",
        "rl = rotl i, __imm_0",
        "m = fma d, d, b",
        "u = uitof i:f64",
        "v = ftoui d:i64",
    ] {
        assert!(text.contains(line), "{line}\n{text}");
    }
    let one = |body: &str| format!("fn k(f:f32, i:i32)\n  b0:\n{body}    ret\n");
    // Binary ops report the op/type pair like `rem` on floats.
    for bad in ["    r = copysign i, i\n", "    r = rotl f, f\n"] {
        assert_eq!(err_code(&one(bad)), "ERR_INVALID_OP_FOR_TYPE", "{bad}");
    }
    for bad in [
        "    r = floor i\n",
        "    r = popcnt f\n",
        "    r = fma i, i, i\n",
        "    r = fma f, f, i\n",
        "    r = ftoui f:i16\n",
        "    r = uitof f:f32\n",
    ] {
        assert_eq!(err_code(&one(bad)), "ERR_TYPE_MISMATCH", "{bad}");
    }
    // The new mnemonics stay valid register names.
    let regs = "fn k(round:i32, floor:i32)->i32\n  b0:\n    clz = add round, floor\n    fma = mul clz, round\n    ret fma\n";
    assert!(parse_and_validate(regs).is_ok());
}
