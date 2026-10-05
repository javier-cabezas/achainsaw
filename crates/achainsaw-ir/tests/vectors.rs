//! Vector syntax, validation, and AIRB encoding.

use achainsaw_ir::opt::optimize_module;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};
use std::path::Path;

fn err_code(src: &str) -> String {
    match parse_and_validate(src) {
        Ok(_) => panic!("expected an error for:\n{src}"),
        Err(d) => d.error_code,
    }
}

const ALL_OPS: &str = r#"
fn k(pa:ptr, po:ptr)->f64
  b0:
    a = ld pa:v256
    b = ld pa:v256
    c = ld pa:v256
    s = vadd a, b:f32
    d = vsub s, b:i64
    m = vmul d, a:i16
    q = vdiv m, b:f64
    lo = vmin q, a:i8
    hi = vmax lo, a:f32
    x = vand hi, a:i32
    y = vor x, a:i32
    z = vxor y, a:i64
    f = vfma z, b, c:f32
    mask = vlt f, a:f32
    e = veq mask, a:i8
    r = vsel e, a, b
    st po, r
    w = splat 1.5:f64:v512
    t = vsum w:f64
    u = vmaxr a:i32
    v = vminr a:i16
    n = vl f32
    lane = extlane r, 7:f32
    fi = vitof a:f32
    if = vftoi fi:i32
    wl = vwidenlo a:i16
    wh = vwidenhi a:i64
    ex = vexp fi:f32
    nr = vnarrow wl, wh:i8
    amt = cst 3:i32
    sl = vshl nr, amt:i32
    sr = vshr sl, 5:i16
    su = vushr sr, 9:i64:i8
    st po, su
    ret t
"#;

#[test]
fn every_example_has_a_matching_airb() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut checked = 0;
    for dir in [examples.clone(), examples.join("kernels")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "air") {
                assert!(
                    path.with_extension("airb").exists(),
                    "{path:?} has no .airb; run `achainsaw assemble` on it"
                );
            }
            let airb = path;
            if airb.extension().is_none_or(|e| e != "airb") {
                continue;
            }
            let bytes = std::fs::read(&airb).unwrap();
            let decoded = decode_module(&bytes).unwrap_or_else(|d| panic!("{airb:?}: {d:?}"));
            let source = std::fs::read_to_string(airb.with_extension("air")).unwrap();
            let parsed = parse_and_validate(&source).unwrap();
            assert_eq!(to_air_text(&decoded), to_air_text(&parsed), "{airb:?}");
            checked += 1;
        }
    }
    assert!(checked >= 7, "only {checked} .airb files found");
}

#[test]
fn airb_with_another_version_is_rejected() {
    let module = parse_and_validate("fn f()->i32\n  b0:\n    ret 1:i32\n").unwrap();
    let mut bytes = encode_module(&module).unwrap();
    bytes[4..6].copy_from_slice(&(achainsaw_ir::binary::VERSION + 1).to_le_bytes());
    let err = decode_module(&bytes).unwrap_err();
    assert!(
        err.message.contains("Unsupported AIRB version"),
        "{}",
        err.message
    );
}

#[test]
fn removed_spellings_are_rejected() {
    // Every vector op names its types: no untyped `splat`, no `vfadd`-style aliases.
    assert_eq!(
        err_code("fn k(x:f32)\n  b0:\n    a = splat x\n    ret\n"),
        "ERR_EXPECTED_TYPE"
    );
    for alias in [
        "vfadd a, a",
        "viadd a, a",
        "vfsum a:f32",
        "visum a:i32",
        "vfmax a:f32",
    ] {
        let src = format!("fn k(x:f32)\n  b0:\n    a = splat x:v128\n    b = {alias}\n    ret\n");
        assert!(
            parse_and_validate(&src).is_err(),
            "`{alias}` should not parse"
        );
    }
}

#[test]
fn text_and_airb_round_trip() {
    let module = parse_and_validate(ALL_OPS).unwrap();
    let text = to_air_text(&module);
    // Canonical text re-parses to the same program.
    assert_eq!(to_air_text(&parse_and_validate(&text).unwrap()), text);
    // AIRB preserves every instruction.
    let bytes = encode_module(&module).unwrap();
    assert_eq!(
        u16::from_le_bytes([bytes[4], bytes[5]]),
        achainsaw_ir::binary::VERSION
    );
    assert_eq!(to_air_text(&decode_module(&bytes).unwrap()), text);
    assert!(text.contains("w = splat __imm_0:v512"), "{text}");
    assert!(text.contains("n = vl f32"));
    assert!(text.contains("r = vsel e, a, b"));
    assert!(text.contains("f = vfma z, b, c:f32"));
    assert!(text.contains("fi = vitof a:f32"), "{text}");
    assert!(text.contains("nr = vnarrow wl, wh:i8"));
    assert!(text.contains("sl = vshl nr, amt:i32"));
}

#[test]
fn conversion_shift_and_exp_rules() {
    let one = |body: &str| {
        format!("fn k(p:ptr)\n  b0:\n    a = ld p:v256\n    b = ld p:v256\n{body}    st p, r\n    ret\n")
    };
    // Conversions name the result lane type and support only some.
    assert_eq!(
        err_code(&one("    r = vitof a:f64\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&one("    r = vftoi a:i64\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&one("    r = vwidenlo a:i8\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&one("    r = vexp a:f64\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&one("    r = vnarrow a, b:i32\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&one("    r = vshl a, 1:f32\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(err_code(&one("    r = vexp a\n")), "ERR_EXPECTED_LANE_TYPE");
    // Operands: a vector to convert, a scalar shift amount, same-width narrow operands.
    assert_eq!(
        err_code("fn k(p:ptr, x:f32)\n  b0:\n    r = vexp x:f32\n    st p, r\n    ret\n"),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&one("    r = vshl a, b:i32\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code("fn k(p:ptr)\n  b0:\n    a = ld p:v256\n    b = ld p:v128\n    r = vnarrow a, b:i8\n    st p, r\n    ret\n"),
        "ERR_TYPE_MISMATCH"
    );
    assert!(parse_and_validate(&one("    n = cst 4:i64\n    r = vushr a, n:i8\n")).is_ok());
    assert!(parse_and_validate(&one("    r = vshl a, 31:i32\n")).is_ok());
    // The new mnemonics are still ordinary register names.
    let src = "fn k(vexp:i32, vshl:i32)->i32\n  b0:\n    vnarrow = add vexp, vshl\n    vitof = mul vnarrow, vexp\n    ret vitof\n";
    assert!(parse_and_validate(src).is_ok());
}

#[test]
fn mnemonics_remain_valid_register_names() {
    let src = "fn k(vl:i32, vadd:i32)->i32\n  b0:\n    vsum = add vl, vadd\n    vsel = mul vsum, vl\n    ret vsel\n";
    assert!(parse_and_validate(src).is_ok());
}

#[test]
fn lane_type_rules() {
    let bin = |op: &str, lane: &str| {
        format!("fn k(p:ptr)\n  b0:\n    a = ld p:v128\n    r = {op} a, a:{lane}\n    st p, r\n    ret\n")
    };
    assert_eq!(err_code(&bin("vdiv", "i32")), "ERR_INVALID_LANE_TYPE");
    assert_eq!(err_code(&bin("vmul", "i8")), "ERR_INVALID_LANE_TYPE");
    assert_eq!(err_code(&bin("vmin", "i64")), "ERR_INVALID_LANE_TYPE");
    assert_eq!(err_code(&bin("vadd", "ptr")), "ERR_INVALID_LANE_TYPE");
    assert_eq!(err_code(&bin("vadd", "v128")), "ERR_INVALID_LANE_TYPE");
    assert!(parse_and_validate(&bin("vmul", "i64")).is_ok());
    assert!(parse_and_validate(&bin("vxor", "f64")).is_ok());
    assert_eq!(
        err_code("fn k(p:ptr)\n  b0:\n    a = ld p:v128\n    r = vfma a, a, a:i32\n    st p, r\n    ret\n"),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(
            "fn k(p:ptr)\n  b0:\n    a = ld p:v128\n    r = vadd a, a\n    st p, r\n    ret\n"
        ),
        "ERR_EXPECTED_LANE_TYPE"
    );
}

#[test]
fn vector_type_rules() {
    // Mixed widths.
    assert_eq!(
        err_code("fn k(p:ptr)\n  b0:\n    a = ld p:v128\n    b = ld p:v256\n    r = vadd a, b:f32\n    st p, r\n    ret\n"),
        "ERR_TYPE_MISMATCH"
    );
    // Vector ops on scalars and scalar ops on vectors.
    assert_eq!(
        err_code("fn k(x:f32)->f32\n  b0:\n    r = vadd x, x:f32\n    ret r\n"),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code("fn k(p:ptr)\n  b0:\n    a = ld p:v512\n    r = add a, a\n    st p, r\n    ret\n"),
        "ERR_TYPE_MISMATCH"
    );
    // Splat needs a vector width and a lane-typed source.
    assert_eq!(
        err_code("fn k(x:f32)->i32\n  b0:\n    r = splat x:i32\n    ret r\n"),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code("fn k(p:ptr)\n  b0:\n    r = splat p:v256\n    st p, r\n    ret\n"),
        "ERR_TYPE_MISMATCH"
    );
    // Vectors are untyped; bitcast cannot change vector type.
    assert_eq!(
        err_code(
            "fn k(p:ptr)\n  b0:\n    a = ld p:v256\n    r = bitcast a:v128\n    st p, r\n    ret\n"
        ),
        "ERR_TYPE_MISMATCH"
    );
}

#[test]
fn extract_lane_bounds_use_guaranteed_width() {
    let ext = |ty: &str, lane: u32, lt: &str| {
        format!("fn k(p:ptr)->{lt}\n  b0:\n    a = ld p:{ty}\n    r = extlane a, {lane}:{lt}\n    ret r\n")
    };
    assert!(parse_and_validate(&ext("v256", 7, "f32")).is_ok());
    assert_eq!(err_code(&ext("v256", 8, "f32")), "ERR_OUT_OF_BOUNDS_LANE");
    assert!(parse_and_validate(&ext("v512", 63, "i8")).is_ok());
    assert!(parse_and_validate(&ext("vx", 3, "f32")).is_ok());
    let d = parse_and_validate(&ext("vx", 4, "f32")).unwrap_err();
    assert_eq!(d.error_code, "ERR_OUT_OF_BOUNDS_LANE");
    assert!(
        d.message.contains("only guaranteed 128 bits"),
        "{}",
        d.message
    );
}

#[test]
fn signature_rules() {
    // AIR functions may take and return vx; C functions may not.
    assert!(parse_and_validate("fn k(v:vx)\n  b0:\n    ret\n").is_ok());
    assert!(parse_and_validate("fn k(p:ptr)->vx\n  b0:\n    v = ld p:vx\n    ret v\n").is_ok());
    assert_eq!(
        err_code("extfn ext(v:vx)\nfn k()\n  b0:\n    ret\n"),
        "ERR_SCALABLE_IN_SIGNATURE"
    );
    assert_eq!(
        err_code("extfn ext()->vx\nfn k()\n  b0:\n    ret\n"),
        "ERR_SCALABLE_IN_SIGNATURE"
    );
    assert_eq!(
        err_code("extfn ext(v:v256)\nfn k()\n  b0:\n    ret\n"),
        "ERR_WIDE_VECTOR_IN_EXTERN"
    );
    // Fixed-width vectors are fine between AIR functions; vx is fine inside bodies.
    assert!(parse_and_validate(
        "fn id(v:v512)->v512\n  b0:\n    ret v\n\nfn k(p:ptr)\n  b0:\n    v = ld p:vx\n    jmp b1(v)\n  b1(w:vx):\n    st p, w\n    ret\n"
    )
    .is_ok());
}

#[test]
fn optimizer_handles_vector_ops() {
    let src = r#"
fn k(p:ptr, x:i32)
  b0:
    zero = cst 0:i32
    y = add x, zero
    v = splat y:v256
    dead = vmul v, v:i32
    dead_mask = vlt v, v:i32
    dead_n = vl i8
    w = vadd v, v:i32
    st p, w
    ret
"#;
    let mut module = parse_and_validate(src).unwrap();
    let stats = optimize_module(&mut module);
    let text = to_air_text(&module);
    assert!(stats.dead_instructions_removed >= 3, "{stats:?}\n{text}");
    assert!(!text.contains("dead"), "{text}");
    assert!(
        text.contains("v = splat x:v256"),
        "substitution through splat:\n{text}"
    );
    assert!(
        parse_and_validate(&text).is_ok(),
        "optimized output re-validates:\n{text}"
    );
}

// ---------------------------------------------------------------------------------------
// f16/bf16, ldm/stm, mm
// ---------------------------------------------------------------------------------------

const MASKED_HALF_MM: &str = r#"
fn k(p:ptr, q:ptr, n:i64)
  b0:
    v = ldm p:vx, n:f32
    w = ldm p:v512, 3:i64
    stm q, v, n:f32
    stm q, w, 5:i8
    mm q, p, p, 4, n, 16:i64:bf16
    mm q, p, p, n, n, n:i8
    h = ld p:f16
    f = fext h:f32
    b = ftrunc f:bf16
    st q, b
    i = bitcast b:i16
    g = bitcast i:f16
    s = select n, g, h
    st q, s
    ret
"#;

#[test]
fn masked_half_mm_syntax_round_trips() {
    let module = parse_and_validate(MASKED_HALF_MM).unwrap();
    let text = to_air_text(&module);
    for expected in [
        "v = ldm p:vx, n:f32",
        "w = ldm p:v512, __imm_0:i64",
        "stm q, v, n:f32",
        "stm q, w, __imm_1:i8",
        "mm q, p, p, __imm_2, n, __imm_3:bf16",
        "mm q, p, p, n, n, n:i8",
        "f = fext h:f32",
        "b = ftrunc f:bf16",
    ] {
        assert!(text.contains(expected), "missing '{expected}' in:\n{text}");
    }
    assert_eq!(to_air_text(&parse_and_validate(&text).unwrap()), text);
    let bytes = encode_module(&module).unwrap();
    assert_eq!(to_air_text(&decode_module(&bytes).unwrap()), text);
    // The literal counts became i64 immediates; `16:i64:bf16` kept its explicit type.
    let consts: Vec<String> = text
        .lines()
        .filter(|l| l.contains("cst"))
        .map(str::trim)
        .map(String::from)
        .collect();
    assert_eq!(
        consts,
        [
            "__imm_0 = cst 3:i64",
            "__imm_1 = cst 5:i64",
            "__imm_2 = cst 4:i64",
            "__imm_3 = cst 16:i64"
        ]
    );
}

#[test]
fn ldm_stm_mm_remain_valid_register_names() {
    let src =
        "fn k(ldm:i32)->i32\n  b0:\n    stm = add ldm, ldm\n    mm = mul stm, ldm\n    ret mm\n";
    assert!(parse_and_validate(src).is_ok());
}

#[test]
fn masked_and_mm_type_rules() {
    let with = |body: &str| format!("fn k(p:ptr, n:i64, c:i32, x:f32)\n  b0:\n{body}    ret\n");
    assert_eq!(
        err_code(&with("    v = ldm p:vx, c:f32\n    st p, v\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    v = ldm p:f32, n:f32\n    st p, v\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    v = ldm p:vx, n:bf16\n    st p, v\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&with("    v = ldm n:vx, n:f32\n    st p, v\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    stm p, x, n:f32\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    v = ldm p:vx, n\n    st p, v\n")),
        "ERR_EXPECTED_LANE_TYPE"
    );
    assert_eq!(
        err_code(&with("    mm p, p, p, n, n, n:i32\n")),
        "ERR_INVALID_LANE_TYPE"
    );
    assert_eq!(
        err_code(&with("    mm p, p, p, c, n, n:f32\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    mm p, n, p, n, n, n:f32\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    mm p, p, p, n, n, n\n")),
        "ERR_EXPECTED_LANE_TYPE"
    );
}

#[test]
fn half_types_are_storage_only() {
    let with = |body: &str| format!("fn k(p:ptr)\n  b0:\n    h = ld p:f16\n{body}    ret\n");
    assert!(parse_and_validate(&with(
        "    f = fext h:f32\n    g = ftrunc f:f16\n    st p, g\n"
    ))
    .is_ok());
    assert_eq!(
        err_code(&with("    s = add h, h\n    st p, s\n")),
        "ERR_INVALID_OP_FOR_TYPE"
    );
    assert_eq!(
        err_code(&with("    s = lt h, h\n    st p, s\n")),
        "ERR_INVALID_OP_FOR_TYPE"
    );
    assert_eq!(
        err_code(&with("    s = neg h\n    st p, s\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    f = fext h:f64\n    st p, f\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    f = itof h:f32\n    st p, f\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code(&with("    v = splat h:v128\n    st p, v\n")),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code("fn k(p:ptr)\n  b0:\n    c = cst 1.0:bf16\n    st p, c\n    ret\n"),
        "ERR_TYPE_MISMATCH"
    );
    assert_eq!(
        err_code("fn k(h:f16)\n  b0:\n    ret\n"),
        "ERR_HALF_IN_SIGNATURE"
    );
    assert_eq!(
        err_code("fn k(p:ptr)->bf16\n  b0:\n    h = ld p:bf16\n    ret h\n"),
        "ERR_HALF_IN_SIGNATURE"
    );
    assert_eq!(
        err_code("extfn e(h:f16)\nfn k()\n  b0:\n    ret\n"),
        "ERR_HALF_IN_SIGNATURE"
    );
}

#[test]
fn masked_loads_and_mm_survive_dead_code_elimination() {
    let src = "fn k(p:ptr, n:i64)\n  b0:\n    unused = ldm p:v256, n:f32\n    mm p, p, p, n, n, n:f32\n    ret\n";
    let mut module = parse_and_validate(src).unwrap();
    optimize_module(&mut module);
    let text = to_air_text(&module);
    assert!(
        text.contains("unused = ldm") && text.contains("mm p, p, p"),
        "{text}"
    );
}

/// `vexp`'s constants are exactly the f32 values of the Cephes expf constants (results are
/// bit-identical across backends only while these bits stay fixed).
#[test]
fn vexp_constants_are_pinned() {
    use achainsaw_ir::ast::vexp;
    let bits: Vec<u32> = [vexp::LOG2E, vexp::NEG_LN2_HI, vexp::LN2_LO]
        .iter()
        .chain(&vexp::POLY)
        .map(|c| c.to_bits())
        .collect();
    assert_eq!(
        bits,
        [
            0x3fb8_aa3b,
            0xbf31_8000,
            0x395e_8083,
            0x3950_6967,
            0x3ab7_43ce,
            0x3c08_8908,
            0x3d2a_a9c1,
            0x3e2a_aaaa,
            0x3f00_0000
        ]
    );
}
