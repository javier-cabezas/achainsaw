pub mod aot;
pub mod jit;

pub use aot::{link_shared_library, AotCompiler};
pub use jit::{
    check_execution_status, get_allocated_memory, get_execution_status, get_global_symbol_address,
    get_remaining_fuel, load_global_library, register_global_symbol, reset_execution_status,
    set_execution_fuel, set_memory_quota, to_clif_type, ExecutionStatus, JitEngine, SymbolRegistry,
};

#[cfg(test)]
mod tests {
    use super::*;
    use achainsaw_ir::parse_and_validate;

    #[test]
    fn test_jit_arithmetic() {
        let code = r#"
fn add(a:i32, b:i32)->i32
  b0:
    res = add a, b
    ret res
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        unsafe {
            let res = engine.run_i32_2_to_i32("add", 19, 23).expect("Run add");
            assert_eq!(res, 42);
        }
    }

    #[test]
    fn test_jit_loop_sum() {
        let code = r#"
fn sum_to_n(n:i32)->i32
  b0:
    zero = cst 0:i32
    jmp b1(zero, zero)
  b1(i:i32, acc:i32):
    cond = lt i, n
    br cond, b2, b3
  b2:
    one = cst 1:i32
    next_i = add i, one
    next_acc = add acc, next_i
    jmp b1(next_i, next_acc)
  b3:
    ret acc
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        unsafe {
            let res = engine.run_i32_to_i32("sum_to_n", 10).expect("Run loop");
            assert_eq!(res, 55); // 1 + 2 + ... + 10 = 55
        }
    }

    #[test]
    fn test_jit_memory_dot_product() {
        let code = r#"
fn dot(p0:ptr, p1:ptr, n:i64)->f32
  b0:
    zero_f = cst 0.0:f32
    zero_i = cst 0:i64
    jmp b1(zero_i, zero_f)
  b1(i:i64, acc:f32):
    cond = lt i, n
    br cond, b2, b3
  b2:
    four = cst 4:i64
    byte_off = mul i, four
    p0_elem = add p0, byte_off
    p1_elem = add p1, byte_off
    v0 = ld p0_elem:f32
    v1 = ld p1_elem:f32
    prod = mul v0, v1
    next_acc = add acc, prod
    one = cst 1:i64
    next_i = add i, one
    jmp b1(next_i, next_acc)
  b3:
    ret acc
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        let a = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let b = [2.0f32, 0.5f32, 1.0f32, 3.0f32];
        // dot = 1*2 + 2*0.5 + 3*1 + 4*3 = 2 + 1 + 3 + 12 = 18.0

        unsafe {
            let res = engine
                .run_ptr_ptr_i64_to_f32("dot", a.as_ptr(), b.as_ptr(), 4)
                .expect("Run dot product");
            assert!((res - 18.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_jit_simd_vector_math() {
        let code = r#"
fn simd_kernel(x:f32, y:f32)->f32
  b0:
    vx = splat x
    vy = splat y
    vsum = vfadd vx, vy
    two = cst 2.0:f32
    vtwo = splat two
    vprod = vfmul vsum, vtwo
    lane0 = extlane vprod, 0:f32
    lane1 = extlane vprod, 1:f32
    res = add lane0, lane1
    ret res
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        // (3.0 + 4.0) * 2.0 = 14.0 for each lane. lane0 + lane1 = 28.0
        unsafe {
            let ptr = engine.get_fn_ptr("simd_kernel").expect("Found simd_kernel");
            let f: extern "C" fn(f32, f32) -> f32 = std::mem::transmute(ptr);
            let res = f(3.0, 4.0);
            assert!((res - 28.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_jit_dynamic_memory_alloc_free() {
        let code = r#"
fn test_heap(n:i32)->i32
  b0:
    bytes = cst 64:i64
    buf = alloc bytes
    val = cst 777:i32
    st buf, val
    read_val = ld buf:i32
    free buf
    ret read_val
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        unsafe {
            let res = engine
                .run_i32_to_i32("test_heap", 1)
                .expect("Run test_heap");
            assert_eq!(res, 777);
        }
    }

    #[test]
    fn test_jit_call_external_math_sinf() {
        let code = r#"
extfn sinf(x:f32)->f32
fn compute_sin(x:f32)->f32
  b0:
    res = call sinf(x)
    ret res
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        unsafe {
            let res = engine
                .run_f32_to_f32("compute_sin", 0.0f32)
                .expect("Run sin(0)");
            assert!((res - 0.0).abs() < 1e-6);

            let pi_half = std::f32::consts::FRAC_PI_2;
            let res = engine
                .run_f32_to_f32("compute_sin", pi_half)
                .expect("Run sin(pi/2)");
            assert!((res - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_jit_custom_symbol_registration() {
        unsafe extern "C" fn custom_multiply_add(a: i32, b: i32, c: i32) -> i32 {
            a * b + c
        }

        let code = r#"
extfn custom_multiply_add(a:i32, b:i32, c:i32)->i32
fn eval_fused(x:i32, y:i32, z:i32)->i32
  b0:
    res = call custom_multiply_add(x, y, z)
    ret res
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.register_symbol("custom_multiply_add", custom_multiply_add as *const u8);
        engine.compile_module(&module).expect("Compile module");

        unsafe {
            let ptr = engine.get_fn_ptr("eval_fused").expect("Found eval_fused");
            let f: extern "C" fn(i32, i32, i32) -> i32 = std::mem::transmute(ptr);
            let res = f(6, 7, 5);
            assert_eq!(res, 47);
        }
    }

    #[test]
    fn test_jit_global_symbol_and_variable_lookup() {
        let mut global_counter: i32 = 999;
        let counter_ptr = &mut global_counter as *mut i32;

        register_global_symbol("global_counter", counter_ptr as *const u8);

        let resolved = get_global_symbol_address("global_counter");
        assert_eq!(resolved, Some(counter_ptr as *const u8));

        let code = r#"
fn inc_global(p:ptr)->i32
  b0:
    val = ld p:i32
    one = cst 1:i32
    new_val = add val, one
    st p, new_val
    ret new_val
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        unsafe {
            let ptr = engine.get_fn_ptr("inc_global").expect("Found inc_global");
            let f: extern "C" fn(*mut i32) -> i32 = std::mem::transmute(ptr);
            let res = f(counter_ptr);
            assert_eq!(res, 1000);
            assert_eq!(global_counter, 1000);
        }
    }

    #[test]
    fn test_jit_fuel_exhaustion_infinite_loop() {
        let code = r#"
fn loop_forever(n:i32)->i32
  b0:
    jmp b1(n)
  b1(x:i32):
    one = cst 1:i32
    next_x = add x, one
    jmp b1(next_x)
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        // Set fuel budget to 500 steps
        engine.set_fuel(Some(500));

        unsafe {
            let res = engine.run_i32_to_i32("loop_forever", 0);
            assert!(
                res.is_err(),
                "Expected fuel exhaustion error, got: {:?}",
                res
            );
            let err_msg = res.unwrap_err().to_string();
            assert!(
                err_msg.contains("[ERR_OUT_OF_FUEL]"),
                "Expected [ERR_OUT_OF_FUEL] in: {err_msg}"
            );
        }

        // Reset fuel to unlimited
        engine.set_fuel(None);
    }

    #[test]
    fn test_jit_memory_quota_limit() {
        let code = r#"
fn allocate_huge(n:i32)->ptr
  b0:
    bytes = cst 104857600:i64
    buf = alloc bytes
    ret buf
"#;
        let module = parse_and_validate(code).expect("IR valid");
        let mut engine = JitEngine::new().expect("JIT init");
        engine.compile_module(&module).expect("Compile module");

        // Set memory quota to 10 MB (10485760 bytes), while request is 100 MB
        engine.set_memory_quota(10 * 1024 * 1024);

        unsafe {
            let ptr = engine
                .get_fn_ptr("allocate_huge")
                .expect("Found allocate_huge");
            reset_execution_status();
            let f: extern "C" fn(i32) -> *mut u8 = std::mem::transmute(ptr);
            let _ = f(1);
            let status_res = check_execution_status();
            assert!(status_res.is_err(), "Expected memory quota error");
            let err_msg = status_res.unwrap_err().to_string();
            assert!(
                err_msg.contains("[ERR_OUT_OF_MEMORY]"),
                "Expected [ERR_OUT_OF_MEMORY] in: {err_msg}"
            );
        }

        // Reset memory quota to unlimited
        engine.set_memory_quota(0);
    }
}
