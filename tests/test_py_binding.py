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


if __name__ == "__main__":
    unittest.main()
