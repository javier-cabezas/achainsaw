use achainsaw_ir::{
    ast::{BinaryOp, Constant, Instruction},
    decode_module, encode_module,
    opt::optimize_module,
    parse_and_validate,
    types::Type,
};

#[test]
fn test_repro_f1_integer_overflow_wrapping() {
    let code = r#"
    fn f()->i32
      b0:
        a = cst 2147483647:i32
        b = cst 1:i32
        c = add a, b
        ret c
    "#;
    let mut module = parse_and_validate(code).expect("should parse and validate");
    optimize_module(&mut module);

    // After optimization, c = cst -2147483648:i32
    let fn_f = &module.functions[0];
    let inst = &fn_f.blocks[0].instructions[0];
    match inst {
        Instruction::AssignConst {
            val: Constant::Int(val),
            ty,
            ..
        } => {
            assert_eq!(*val, i32::MIN as i64);
            assert_eq!(*ty, Type::I32);
        }
        other => panic!("Expected folded constant Int(-2147483648), got {:?}", other),
    }
}

#[test]
fn test_repro_f2_float_sub_preserved() {
    let code = r#"
    fn f(x:f32)->f32
      b0:
        d = sub x, x
        ret d
    "#;
    let mut module = parse_and_validate(code).expect("should parse and validate");
    let stats = optimize_module(&mut module);

    // sub x, x on floats must NOT be folded to 0.0 because NaN - NaN != 0.0
    assert_eq!(stats.algebraic_simplifications, 0);
    let fn_f = &module.functions[0];
    assert_eq!(fn_f.blocks[0].instructions.len(), 1);
    match &fn_f.blocks[0].instructions[0] {
        Instruction::Binary { op, .. } => {
            assert_eq!(*op, BinaryOp::Sub);
        }
        other => panic!("Expected instruction to remain sub x, x, got {:?}", other),
    }
}

#[test]
fn test_repro_f3_nan_inf_roundtrip() {
    let code = r#"
    fn f()->f32
      b0:
        n = cst nan:f32
        p = cst inf:f32
        m = cst -inf:f32
        ret n
    "#;
    let module = parse_and_validate(code).expect("should parse and validate");
    let encoded = encode_module(&module).expect("should encode");
    let decoded = decode_module(&encoded).expect("should decode");
    assert_eq!(decoded.functions.len(), 1);
}

#[test]
fn test_repro_f4_dominance_violation() {
    let code = r#"
    fn f(c:i32)->i32
      b0:
        br c, b1, b2
      b1:
        x = cst 7:i32
        jmp b2
      b2:
        ret x
    "#;
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_NON_DOMINATING_USE");
    assert_eq!(err.span.line, 9);
}

#[test]
fn test_repro_h1_op_type_legality() {
    let code = r#"
    fn f(a:f32, b:f32)->f32
      b0:
        c = shl a, b
        ret c
    "#;
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_INVALID_OP_FOR_TYPE");
}

#[test]
fn test_repro_h2_constant_range_check() {
    let code = r#"
    fn f()->i8
      b0:
        c = cst 300:i8
        ret c
    "#;
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_CONSTANT_OUT_OF_RANGE");
}

#[test]
fn test_repro_h3_entry_block_branch_rejected() {
    let code = r#"
    fn f(c:i32)->i32
      b0:
        br c, b1, b0
      b1:
        ret c
    "#;
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_BRANCH_TO_ENTRY");
}

#[test]
fn test_repro_lane_out_of_bounds() {
    let code = r#"
    fn f(v:v128)->f32
      b0:
        l = extlane v, 4:f32
        ret l
    "#;
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_OUT_OF_BOUNDS_LANE");
}

#[test]
fn test_repro_available_registers_omit_immediates() {
    // The registers the parser makes for inline immediates are not suggested
    let code = r#"
    fn f(n:i64)->i64
      b0:
        jmp b1(0:i64, 0:i64)
      b1(i:i64, acc:i64):
        more = lt i, n
        br more, b2, b3
      b2:
        i2 = add i, 1:i64
        jmp b1(i2, acc)
      b3:
        ret total
    "#;
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_UNDEFINED_REG");
    let ctx = err.context.expect("context");
    assert_eq!(
        ctx["available_registers"],
        serde_json::json!(["acc", "i", "more", "n"])
    );
}

/// Op mnemonics are register names outside the position after `=`: agents name the
/// remainder of a loop `rem`.
#[test]
fn test_repro_op_mnemonics_are_operands() {
    let code = r#"
fn f(n:i64, i:i64)->i64
  b0:
    rem = sub n, i
    min = min rem, 8:i64
    ret min
"#;
    parse_and_validate(code).unwrap_or_else(|d| panic!("{d:?}"));
}

/// Every instruction mnemonic is a register name outside the position after `=`, those with
/// their own tokens too (`cst`, `ld`, `select`, the casts, `neg`/`abs`/`sqrt`): as a
/// destination, an operand, a parameter and a label, and in the printed text.
#[test]
fn test_repro_mnemonics_are_destinations() {
    let code = r#"
fn f(ld:i64, cst:i64)->i64
  entry:
    neg = lt ld, 0:i64
    abs = select neg, cst, ld
    sext = add abs, 1:i64
    splat = alloc 64:i64
    st splat[ld], sext
    alloc = ld splat[ld]:i64
    sqrt = add alloc, cst
    bitcast = itof sqrt:f64
    extlane = ftoi bitcast:i64
    free splat
    br neg, select(extlane), select(abs)
  select(itof:i64):
    ret itof
"#;
    let module = parse_and_validate(code).unwrap_or_else(|d| panic!("{d:?}"));
    let text = achainsaw_ir::to_air_text(&module);
    let again = parse_and_validate(&text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    assert_eq!(achainsaw_ir::to_air_text(&again), text);
}

/// A mnemonic that starts a line without `=` is an instruction missing its destination.
#[test]
fn test_repro_missing_destination() {
    for (line, op) in [
        ("neg x", "neg"),
        ("add x, x", "add"),
        ("vadd v, v:f32", "vadd"),
        ("floor x", "floor"),
    ] {
        let code = format!("fn g(x:f32, v:v128)->f32\n  entry:\n    {line}\n    ret x\n");
        let err = parse_and_validate(&code).unwrap_err();
        assert_eq!(err.error_code, "ERR_MISSING_DESTINATION", "{line}: {err:?}");
        assert_eq!(err.context.expect("context")["op"], op);
        assert!(
            err.message.contains(&format!("r = {op}")),
            "{}",
            err.message
        );
    }
}

/// An integer literal of a float type is that float: `cst 0:f32`, `add x, 1:f32`.
#[test]
fn test_repro_integer_literals_of_float_type() {
    let code = r#"
fn f(x:f32)->f32
  b0:
    z = cst 0:f32
    y = add x, 1:f32
    r = add y, z
    ret r
"#;
    let module = parse_and_validate(code).unwrap_or_else(|d| panic!("{d:?}"));
    let consts: Vec<&Constant> = module.functions[0].blocks[0]
        .instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::AssignConst { val, .. } => Some(val),
            _ => None,
        })
        .collect();
    assert!(
        matches!(consts[..], [Constant::Float(z), Constant::Float(o)] if *z == 0.0 && *o == 1.0)
    );
}

fn parse_error(code: &str) -> achainsaw_ir::Diagnostic {
    parse_and_validate(code).expect_err("should not parse")
}

/// The errors agents hit most in the agent benchmark name the fix.
#[test]
fn test_repro_diagnostics_name_the_fix() {
    let splat = parse_error("fn f()->vx\n  b0:\n    z = splat 0:vx\n    ret z\n");
    assert_eq!(splat.error_code, "ERR_EXPECTED_TYPE");
    assert!(
        splat.message.contains("`z = splat 0:i32:vx`"),
        "{}",
        splat.message
    );

    let cast = parse_error("fn f(s:i32)->i64\n  b0:\n    w = zext s\n    ret w\n");
    assert_eq!(cast.error_code, "ERR_EXPECTED_TYPE");
    assert!(
        cast.message.contains("`w = zext s:i64`"),
        "{}",
        cast.message
    );

    let load = parse_error("fn f(p:ptr, i:i64)->i8\n  b0:\n    a = ld p[i:i8]\n    ret a\n");
    assert_eq!(load.error_code, "ERR_EXPECTED_TYPE");
    assert!(
        load.message.contains("`a = ld p[i]:f32`"),
        "{}",
        load.message
    );

    let op = parse_error("fn f(a:i64)->i64\n  b0:\n    h = sdiv a, 2:i64\n    ret h\n");
    assert_eq!(op.error_code, "ERR_UNKNOWN_OP");
    assert_eq!(op.context.expect("context")["did_you_mean"], "div");
}

/// A result of a call with several results, used where its definition does not dominate,
/// is reported as such (the dominance pre-pass records every destination).
#[test]
fn multi_result_use_outside_its_dominance_is_non_dominating() {
    let code = "fn pair(x:i64)->(i64, i64)\n  b0:\n    ret x, x\n\nfn f(x:i64)->i64\n  b0:\n    c = gt x, 0:i64\n    br c, a, b\n  a:\n    p, q = call pair(x)\n    jmp done\n  b:\n    jmp done\n  done:\n    ret p\n";
    let err = parse_and_validate(code).unwrap_err();
    assert_eq!(err.error_code, "ERR_NON_DOMINATING_USE");
}
