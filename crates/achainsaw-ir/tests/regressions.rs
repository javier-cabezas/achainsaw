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
