use achainsaw_ir::{
    ast::{Constant, Instruction},
    decode_module, encode_module,
    opt::optimize_module,
    parse_and_validate, to_air_text,
    types::Type,
};

#[test]
fn test_inline_immediates_desugaring() {
    let code = r#"
    fn add_one(x:i32)->i32
      b0:
        y = add x, 1:i32
        ret y
    "#;
    let module = parse_and_validate(code).expect("should parse and validate with inline immediate");
    let fn_0 = &module.functions[0];
    assert_eq!(fn_0.blocks[0].instructions.len(), 2);
    // Instruction 0 should be synthesized constant: __imm_0 = cst 1:i32
    match &fn_0.blocks[0].instructions[0] {
        Instruction::AssignConst { dst, val, ty, .. } => {
            assert!(dst.starts_with("__imm_"));
            assert_eq!(*val, Constant::Int(1));
            assert_eq!(*ty, Type::I32);
        }
        other => panic!("Expected synthesized AssignConst, got {:?}", other),
    }
}

#[test]
fn test_type_conversions_and_intrinsics_parsing_and_binary() {
    let code = r#"
    fn test_ops(a:i32, f:f32, v:v128)->i32
      b0:
        f_val = itof a:f32
        i_val = ftoi f:i32
        wide = sext a:i64
        uwide = zext a:i64
        short = trunc a:i16
        double_f = fext f:f64
        narrow_f = ftrunc double_f:f32
        raw_bits = bitcast f:i32

        s = select a, 10:i32, 20:i32
        n = neg a
        ab = abs a
        sq = sqrt f
        m = min a, 5:i32
        mx = max a, 100:i32

        ud = udiv a, 2:i32
        ur = urem a, 3:i32
        us = ushr a, 1:i32
        is_lt = ult a, 10:i32

        vsum = vsum v:f32
        vmax = vmaxr v:f32
        vis = vsum v:i32
        ret i_val
    "#;
    let module = parse_and_validate(code).expect("should parse and validate phase 4 instructions");

    // AIRB encode/decode round trip
    let bytes = encode_module(&module).expect("encode should succeed");
    let decoded = decode_module(&bytes).expect("decode should succeed");

    assert_eq!(module.functions.len(), decoded.functions.len());
    let orig_fn = &module.functions[0];
    let dec_fn = &decoded.functions[0];
    assert_eq!(orig_fn.name, dec_fn.name);
    assert_eq!(
        orig_fn.blocks[0].instructions.len(),
        dec_fn.blocks[0].instructions.len()
    );

    // Disassembly round trip
    let text = to_air_text(&decoded);
    let reparsed = parse_and_validate(&text).expect("reparsing disassembly should succeed");
    assert_eq!(reparsed.functions.len(), 1);
}

#[test]
fn test_phase4_constant_folding() {
    let code = r#"
    fn fold_all()->i32
      b0:
        c1 = cst 10:i32
        c2 = cst 20:i32
        m = min c1, c2
        mx = max c1, c2
        ab = abs c1
        neg_val = neg c1

        cond = cst 1:i32
        sel = select cond, 100:i32, 200:i32

        ud = udiv c2, c1
        ur = urem c2, c1
        us = ushr c2, 1:i32
        lt_res = ult c1, c2

        cf = itof c1:f32
        ci = ftoi cf:i32
        ret sel
    "#;
    let mut module = parse_and_validate(code).expect("valid code");
    let stats = optimize_module(&mut module);

    assert!(
        stats.constants_folded >= 5,
        "Expected constant folding of min, max, select, etc., got {}",
        stats.constants_folded
    );
}
