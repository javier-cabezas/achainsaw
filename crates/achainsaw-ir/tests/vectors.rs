//! AIR v2 vector syntax, validation, AIRB v2 encoding, and backward compatibility.

use achainsaw_ir::opt::optimize_module;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};
use std::path::Path;

fn err_code(src: &str) -> String {
    match parse_and_validate(src) {
        Ok(_) => panic!("expected an error for:\n{src}"),
        Err(d) => d.error_code,
    }
}

fn roundtrip_text(src: &str) -> String {
    to_air_text(&parse_and_validate(src).expect("valid"))
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
    ret t
"#;

#[test]
fn every_example_airb_decodes_including_v1() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    // Kernels assembled before AIRB v2, kept unchanged as decoder compatibility fixtures.
    let v1 = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1");
    let mut checked = 0;
    let mut v1_files = 0;
    for dir in [examples.clone(), examples.join("kernels"), v1] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let airb = entry.unwrap().path();
            if airb.extension().is_none_or(|e| e != "airb") {
                continue;
            }
            let bytes = std::fs::read(&airb).unwrap();
            if u16::from_le_bytes([bytes[4], bytes[5]]) == 1 {
                v1_files += 1;
            }
            let decoded = decode_module(&bytes).unwrap_or_else(|d| panic!("{airb:?}: {d:?}"));
            let source = std::fs::read_to_string(airb.with_extension("air")).unwrap();
            let parsed = parse_and_validate(&source).unwrap();
            assert_eq!(to_air_text(&decoded), to_air_text(&parsed), "{airb:?}");
            checked += 1;
        }
    }
    assert!(checked >= 7, "only {checked} .airb files found");
    assert!(v1_files >= 5, "only {v1_files} v1 .airb fixtures left");
}

#[test]
fn legacy_vector_syntax_normalizes_to_v2() {
    let text = roundtrip_text(
        "fn k(x:f32, n:i32)->f32\n  b0:\n    a = splat x\n    b = vfadd a, a\n    c = vfmul b, a\n    d = vfsub c, a\n    e = vfdiv d, a\n    i = splat n\n    j = viadd i, i\n    k = vimul j, i\n    l = visub k, i\n    s = visum l:i32\n    t = vfsum e:f32\n    u = vfmax e:f32\n    r = add t, u\n    ret r\n",
    );
    for expected in [
        "a = splat x\n",
        "b = vadd a, a:f32",
        "c = vmul b, a:f32",
        "d = vsub c, a:f32",
        "e = vdiv d, a:f32",
        "j = vadd i, i:i32",
        "k = vmul j, i:i32",
        "l = vsub k, i:i32",
        "s = vsum l:i32",
        "t = vsum e:f32",
        "u = vmaxr e:f32",
    ] {
        assert!(text.contains(expected), "missing '{expected}' in:\n{text}");
    }
}

#[test]
fn v2_text_and_airb_round_trip() {
    let module = parse_and_validate(ALL_OPS).unwrap();
    let text = to_air_text(&module);
    // Canonical text re-parses to the same program.
    assert_eq!(to_air_text(&parse_and_validate(&text).unwrap()), text);
    // AIRB v2 preserves every new instruction.
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
    assert_eq!(
        err_code("fn k(v:vx)\n  b0:\n    ret\n"),
        "ERR_SCALABLE_IN_SIGNATURE"
    );
    assert_eq!(
        err_code("fn k(p:ptr)->vx\n  b0:\n    v = ld p:vx\n    ret v\n"),
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
// f16/bf16, ldm/stm, mm (AIRB v3)
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
