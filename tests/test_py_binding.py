"""
Comprehensive Python integration test suite for achainsaw PyO3 bindings.
Tests in-process compilation, zero-copy buffer execution, and agent self-repair diagnostics.
"""

import sys
import unittest
import numpy as np
import achainsaw


class TestChainsawPy(unittest.TestCase):
    def test_version(self):
        ver = achainsaw.version()
        self.assertIsInstance(ver, str)
        self.assertEqual(ver, "0.1.0")

    def test_check_valid(self):
        source = """fn add(a:i32, b:i32)->i32
  b0:
    c = add a, b
    ret c
"""
        res = achainsaw.check(source)
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res.get("functions"), ["add"])
        self.assertEqual(res.get("function_count"), 1)
        self.assertEqual(res.get("block_count"), 1)

    def test_check_syntax_error(self):
        source = """fn invalid_syntax(a:i32
"""
        res = achainsaw.check(source)
        self.assertEqual(res.get("status"), "error")
        self.assertTrue("error_code" in res)
        self.assertTrue("message" in res)

    def test_check_undefined_register(self):
        source = """fn test(a:i32)->i32
  b0:
    c = add a, undef_reg
    ret c
"""
        res = achainsaw.check(source)
        self.assertEqual(res.get("status"), "error")
        self.assertEqual(res.get("error_code"), "ERR_UNDEFINED_REG")
        self.assertIn("undef_reg", res.get("message", ""))

    def test_compile_and_run_scalar(self):
        source = """fn math_op(x:i32, y:i32)->i32
  b0:
    m = mul x, y
    s = sub m, x
    ret s
"""
        k = achainsaw.compile(source)
        self.assertIn("math_op", k.get_function_names())
        # (10 * 5) - 10 = 40
        self.assertEqual(k.run("math_op", 10, 5), 40)

    def test_compile_and_run_fibonacci(self):
        with open("examples/fibonacci.air", "r", encoding="utf-8") as f:
            code = f.read()
        k = achainsaw.compile(code)
        self.assertEqual(k.run("fib", 0), 0)
        self.assertEqual(k.run("fib", 1), 1)
        self.assertEqual(k.run("fib", 10), 55)
        self.assertEqual(k.run("fib", 20), 6765)

    def test_simd_dot_numpy(self):
        with open("examples/simd_vector_dot.air", "r", encoding="utf-8") as f:
            code = f.read()
        k = achainsaw.compile(code)

        # 2 vectors of 4xf32 (8 elements)
        a = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        b = np.array([2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        res = k.run("simd_dot", a, b, 2)
        expected = float(np.dot(a, b))
        self.assertAlmostEqual(res, expected, places=5)

    def test_simd_in_place_scale_numpy(self):
        code = """fn simd_scale(p:ptr, factor:f32, n:i64)
  b0:
    vfactor = splat factor
    zero = cst 0:i64
    jmp b1(zero)
  b1(i:i64):
    cond = lt i, n
    br cond, b2, b3
  b2:
    sixteen = cst 16:i64
    off = mul i, sixteen
    elem_ptr = add p, off
    v = ld elem_ptr:v128
    vscaled = vfmul v, vfactor
    st elem_ptr, vscaled
    one = cst 1:i64
    next_i = add i, one
    jmp b1(next_i)
  b3:
    ret
"""
        k = achainsaw.compile(code)
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        factor = 3.0
        k.run("simd_scale", arr, factor, 2)

        expected = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0], dtype=np.float32)
        np.testing.assert_allclose(arr, expected)

    def test_bytearray_buffer_protocol(self):
        """Verify zero-copy buffer protocol works with Python's built-in bytearray."""
        # Simple kernel writing byte value
        code = """fn write_first_byte(p:ptr, val:i64)
  b0:
    st p, val
    ret
"""
        k = achainsaw.compile(code)
        buf = bytearray(8)
        k.run("write_first_byte", buf, 123)
        self.assertEqual(buf[0], 123)

    def test_compilation_error_diagnostic(self):
        bad_code = """fn test(a:i32)->i32
  b0:
    x = add a, bad_reg
    ret x
"""
        with self.assertRaises(achainsaw.CompilationError) as ctx:
            achainsaw.compile(bad_code)

        err = ctx.exception
        self.assertIn("ERR_UNDEFINED_REG", str(err))
        self.assertTrue(hasattr(err, "diagnostic"))
        diag = err.diagnostic
        self.assertIsInstance(diag, dict)
        self.assertEqual(diag.get("error_code"), "ERR_UNDEFINED_REG")
        self.assertEqual(diag.get("status"), "error")
        self.assertIn("context", diag)
        self.assertEqual(diag["context"].get("target"), "bad_reg")

    def test_assemble_disassemble_roundtrip(self):
        source = """fn add(a:i32, b:i32)->i32
  b0:
    c = add a, b
    ret c
"""
        binary = achainsaw.assemble(source)
        self.assertIsInstance(binary, bytes)
        self.assertTrue(binary.startswith(b"\x00AIR"))

        disassembled = achainsaw.disassemble(binary)
        self.assertIn("fn add(a:i32, b:i32)->i32", disassembled)
        self.assertIn("ret c", disassembled)

    def test_compile_binary(self):
        with open("examples/fibonacci.air", "r", encoding="utf-8") as f:
            code = f.read()
        binary = achainsaw.assemble(code)
        k = achainsaw.compile_binary(binary)
        self.assertEqual(k.run("fib", 10), 55)

    def test_compile_binary_invalid(self):
        with self.assertRaises(achainsaw.CompilationError) as ctx:
            achainsaw.compile_binary(b"INVALID_PAYLOAD")
        self.assertIn("ERR_INVALID_AIRB", str(ctx.exception))

    def test_runtime_errors(self):
        code = """fn add(a:i32, b:i32)->i32
  b0:
    c = add a, b
    ret c
"""
        k = achainsaw.compile(code)
        # Function name not found
        with self.assertRaises(KeyError):
            k.run("sub", 1, 2)

        # Wrong argument count
        with self.assertRaises(ValueError):
            k.run("add", 1)

        with self.assertRaises(ValueError):
            k.run("add", 1, 2, 3)

    def test_extfn_math_intrinsics(self):
        """Test calling pre-registered C math intrinsics (sinf, sqrtf)."""
        import math
        code = """extfn sinf(x:f32)->f32
extfn sqrtf(x:f32)->f32
fn compute_hypot_sin(x:f32)->f32
  b0:
    s = call sinf(x)
    two = cst 4.0:f32
    r = call sqrtf(two)
    res = mul s, r
    ret res
"""
        k = achainsaw.compile(code)
        val = 1.04719755  # pi / 3
        res = k.run("compute_hypot_sin", val)
        expected = math.sin(val) * 2.0
        self.assertAlmostEqual(res, expected, places=5)

    def test_extfn_binary_roundtrip(self):
        """Test AIRB assemble/disassemble and binary compilation with extfn."""
        code = """extfn cosf(x:f32)->f32
fn compute_cos(x:f32)->f32
  b0:
    c = call cosf(x)
    ret c
"""
        binary = achainsaw.assemble(code)
        disassembled = achainsaw.disassemble(binary)
        self.assertIn("extfn cosf(x:f32)->f32", disassembled)

        k = achainsaw.compile_binary(binary)
        res = k.run("compute_cos", 0.0)
        self.assertAlmostEqual(res, 1.0, places=5)

    def test_register_custom_symbol(self):
        """Test registering a custom C function pointer via ctypes and invoking it from IR."""
        import ctypes
        c_func_ty = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)

        def custom_fn(a, b, c):
            return a * b + c

        c_cb = c_func_ty(custom_fn)
        cb_addr = ctypes.cast(c_cb, ctypes.c_void_p).value

        achainsaw.register_symbol("py_custom_fma", cb_addr)

        code = """extfn py_custom_fma(a:i32, b:i32, c:i32)->i32
fn test_fma(x:i32, y:i32, z:i32)->i32
  b0:
    res = call py_custom_fma(x, y, z)
    ret res
"""
        k = achainsaw.compile(code)
        res = k.run("test_fma", 7, 8, 9)
        self.assertEqual(res, 65)  # 7 * 8 + 9 = 65

    def test_external_variable_access(self):
        """Test accessing and modifying external variables via pointers in IR."""
        import ctypes
        var = ctypes.c_int32(100)
        var_addr = ctypes.addressof(var)

        code = """fn add_to_var(p:ptr, delta:i32)->i32
  b0:
    cur = ld p:i32
    updated = add cur, delta
    st p, updated
    ret updated
"""
        k = achainsaw.compile(code)
        res = k.run("add_to_var", var_addr, 25)
        self.assertEqual(res, 125)
        self.assertEqual(var.value, 125)

    def test_load_library_and_symbol_address(self):
        """Test library loading and symbol address querying."""
        # Querying an existing pre-registered symbol
        sinf_addr = achainsaw.get_symbol_address("sinf")
        self.assertIsNotNone(sinf_addr)
        self.assertGreater(sinf_addr, 0)

        # Non-existent library raises RuntimeError
        with self.assertRaises(RuntimeError):
            achainsaw.load_library("non_existent_library_12345.dll")

    def test_fuel_exhaustion_in_python(self):
        """Test that infinite loops are safely caught and terminated by loop fuel budget."""
        code = """fn infinite_loop(n:i32)->i32
  b0:
    jmp b1(n)
  b1(x:i32):
    one = cst 1:i32
    next_x = add x, one
    jmp b1(next_x)
"""
        k = achainsaw.compile(code)

        # Set fuel to 500 loop iterations
        achainsaw.set_fuel(500)
        try:
            with self.assertRaises(achainsaw.CompilationError) as ctx:
                k.run("infinite_loop", 0)

            err = ctx.exception
            self.assertIn("ERR_OUT_OF_FUEL", str(err))
            self.assertTrue(hasattr(err, "diagnostic"))
            diag = err.diagnostic
            self.assertEqual(diag.get("error_code"), "ERR_OUT_OF_FUEL")
        finally:
            # Reset fuel to unlimited
            achainsaw.set_fuel(None)

    def test_memory_quota_in_python(self):
        """Test that heap allocation limits protect the host process from memory exhaustion."""
        code = """fn request_giant_heap(n:i32)->ptr
  b0:
    bytes = cst 104857600:i64
    buf = alloc bytes
    ret buf
"""
        k = achainsaw.compile(code)

        # Limit memory quota to 10 MB (allocation asks for 100 MB)
        achainsaw.set_memory_quota(10 * 1024 * 1024)
        try:
            with self.assertRaises(achainsaw.CompilationError) as ctx:
                k.run("request_giant_heap", 1)

            err = ctx.exception
            self.assertIn("ERR_OUT_OF_MEMORY", str(err))
            self.assertTrue(hasattr(err, "diagnostic"))
            diag = err.diagnostic
            self.assertEqual(diag.get("error_code"), "ERR_OUT_OF_MEMORY")
            self.assertEqual(diag.get("context", {}).get("quota_bytes"), 10 * 1024 * 1024)
        finally:
            # Reset memory quota to unlimited
            achainsaw.set_memory_quota(0)

    def test_optimize_in_python(self):
        """Test IR constant folding, algebraic simplification, and DCE in Python."""
        code = """fn unoptimized_calc(x:i32)->i32
  b0:
    c1 = cst 10:i32
    c2 = cst 20:i32
    c3 = add c1, c2
    zero = cst 0:i32
    res = add x, zero
    ret res
"""
        opt_res = achainsaw.optimize(code)
        self.assertIsInstance(opt_res, dict)
        self.assertGreaterEqual(opt_res.get("total_optimizations", 0), 2)
        opt_code = opt_res.get("code", "")
        self.assertIn("ret x", opt_code)


if __name__ == "__main__":
    unittest.main()

