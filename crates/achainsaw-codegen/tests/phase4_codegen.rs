use achainsaw_codegen::JitEngine;
use achainsaw_ir::parse_and_validate;

#[test]
fn test_jit_select_and_intrinsics() {
    let code = r#"
    fn eval_math(a:i32, b:i32)->i32
      b0:
        s = select a, b, 999:i32
        neg_s = neg s
        abs_neg = abs neg_s
        m = min abs_neg, 50:i32
        ret m
    "#;
    let module = parse_and_validate(code).unwrap();
    let mut jit = JitEngine::new().unwrap();
    jit.compile_module(&module).unwrap();

    let fn_ptr = jit.get_fn_ptr("eval_math").unwrap();
    let eval: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(fn_ptr) };

    // When a != 0, select picks b (42), neg is -42, abs is 42, min(42, 50) is 42
    assert_eq!(eval(1, 42), 42);
    // When a == 0, select picks 999, neg is -999, abs is 999, min(999, 50) is 50
    assert_eq!(eval(0, 42), 50);
}

#[test]
fn test_jit_casts_and_sqrt() {
    let code = r#"
    fn cast_and_sqrt(x:i32)->i32
      b0:
        f = itof x:f32
        root = sqrt f
        i = ftoi root:i32
        ret i
    "#;
    let module = parse_and_validate(code).unwrap();
    let mut jit = JitEngine::new().unwrap();
    jit.compile_module(&module).unwrap();

    let fn_ptr = jit.get_fn_ptr("cast_and_sqrt").unwrap();
    let eval: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(fn_ptr) };

    assert_eq!(eval(16), 4);
    assert_eq!(eval(100), 10);
    assert_eq!(eval(0), 0);
}

#[test]
fn test_jit_unsigned_ops() {
    let code = r#"
    fn test_unsigned(a:i32, b:i32)->i32
      b0:
        div_res = udiv a, b
        rem_res = urem a, b
        sum = add div_res, rem_res
        ret sum
    "#;
    let module = parse_and_validate(code).unwrap();
    let mut jit = JitEngine::new().unwrap();
    jit.compile_module(&module).unwrap();

    let fn_ptr = jit.get_fn_ptr("test_unsigned").unwrap();
    let eval: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(fn_ptr) };

    // -1 as i32 is 0xFFFFFFFF unsigned (4294967295)
    // 4294967295 / 2 = 2147483647
    // 4294967295 % 2 = 1
    // sum = 2147483648 -> wraps to -2147483648
    assert_eq!(eval(-1, 2), -2147483648);
    // Division by zero safety: returns 0 + 0 = 0
    assert_eq!(eval(100, 0), 0);
}

#[test]
fn test_jit_vector_reductions() {
    let code = r#"
    fn reduce_sum(ptr:ptr)->f32
      b0:
        v = ld ptr:v128
        res = vfsum v:f32
        ret res

    fn reduce_max(ptr:ptr)->f32
      b0:
        v = ld ptr:v128
        res = vfmax v:f32
        ret res
    "#;
    let module = parse_and_validate(code).unwrap();
    let mut jit = JitEngine::new().unwrap();
    jit.compile_module(&module).unwrap();

    let sum_ptr = jit.get_fn_ptr("reduce_sum").unwrap();
    let reduce_sum_fn: extern "C" fn(*const f32) -> f32 = unsafe { std::mem::transmute(sum_ptr) };

    let max_ptr = jit.get_fn_ptr("reduce_max").unwrap();
    let reduce_max_fn: extern "C" fn(*const f32) -> f32 = unsafe { std::mem::transmute(max_ptr) };

    let data: [f32; 4] = [1.5, 2.5, 3.0, 4.0];
    let sum = reduce_sum_fn(data.as_ptr());
    assert!((sum - 11.0).abs() < 1e-6, "Expected 11.0, got {}", sum);

    let max = reduce_max_fn(data.as_ptr());
    assert!((max - 4.0).abs() < 1e-6, "Expected 4.0, got {}", max);
}
