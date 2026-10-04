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

    def test_cpu_features(self):
        report = achainsaw.cpu_features()
        self.assertEqual(report["status"], "ok")
        self.assertIn(report["host"]["arch"], ("x86_64", "aarch64", "other"))
        self.assertIsInstance(report["host"]["features"], list)
        self.assertEqual(report["backends"]["cranelift"]["vector_bits"], 128)

    def test_isa_cap(self):
        host = achainsaw.cpu_features()["host"]
        lowest = {"x86_64": "sse", "aarch64": "neon"}.get(host["arch"])
        if lowest is None:
            self.skipTest("no ISA levels for this architecture")
        try:
            achainsaw.set_isa_cap(lowest)
            self.assertEqual(achainsaw.get_isa_cap(), lowest)
            capped = achainsaw.cpu_features()["effective"]
            self.assertEqual(capped["max_isa"], lowest)
            self.assertTrue(set(capped["features"]) <= set(host["features"]))

            # Kernels compiled under the cap still produce correct results.
            with open("examples/simd_vector_dot.air", "r", encoding="utf-8") as f:
                k = achainsaw.compile(f.read())
            a = np.arange(8, dtype=np.float32)
            b = np.ones(8, dtype=np.float32)
            self.assertAlmostEqual(k.run("simd_dot", a, b, 8), 28.0, places=5)
        finally:
            achainsaw.set_isa_cap(None)
        self.assertIsNone(achainsaw.get_isa_cap())

        with self.assertRaises(ValueError):
            achainsaw.set_isa_cap("avx9000")
        foreign = "sve" if host["arch"] == "x86_64" else "avx2"
        with self.assertRaises(ValueError):
            achainsaw.set_isa_cap(foreign)

    def test_simd_dot_numpy(self):
        with open("examples/simd_vector_dot.air", "r", encoding="utf-8") as f:
            code = f.read()
        k = achainsaw.compile(code)

        # 8 elements: two 4-lane vectors
        a = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        b = np.array([2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        res = k.run("simd_dot", a, b, 8)
        expected = float(np.dot(a, b))
        self.assertAlmostEqual(res, expected, places=5)

    def test_simd_in_place_scale_numpy(self):
        code = """fn simd_scale(p:ptr, factor:f32, n:i64)
  b0:
    vfactor = splat factor:v128
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
    vscaled = vmul v, vfactor:f32
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

    def test_kernel_usable_from_other_threads(self):
        """A kernel compiled on one thread can be called from others."""
        import threading
        k = achainsaw.compile("fn sq(x:i64)->i64\n  b0:\n    y = mul x, x\n    ret y\n")
        results = {}

        def worker(i):
            results[i] = k.run("sq", i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, {i: i * i for i in range(8)})

    def test_reentrant_kernel_call_raises(self):
        """A host callback calling back into the kernel that is running gets a clear error."""
        import ctypes
        cb_ty = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)
        seen = {}

        def reenter(x):
            try:
                kernel.run("outer", x)
            except RuntimeError as e:
                seen["error"] = str(e)
            return x + 1

        cb = cb_ty(reenter)
        achainsaw.register_symbol("py_reenter", ctypes.cast(cb, ctypes.c_void_p).value)
        kernel = achainsaw.compile(
            "extfn py_reenter(x:i32)->i32\n"
            "fn outer(x:i32)->i32\n  b0:\n    r = call py_reenter(x)\n    ret r\n"
        )
        self.assertEqual(kernel.run("outer", 41), 42)
        self.assertIn("re-entrant", seen.get("error", ""))

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


    PAR_SQUARES = """fn fill(i:i64, out:ptr, scale:f32)
  b0:
    f = itof i:f32
    v = mul f, f
    w = mul v, scale
    off = mul i, 4:i64
    q = add out, off
    st q, w
    ret

fn squares(out:ptr, n:i64, scale:f32)
  b0:
    par n, fill(out, scale)
    ret
"""

    def test_par_over_numpy_buffer(self):
        """`par` fills a NumPy array from all cores; one thread gives the same result."""
        kernel = achainsaw.compile(self.PAR_SQUARES)
        self.assertGreaterEqual(kernel.threads, 1)
        want = np.arange(10000, dtype=np.float32) ** 2 * np.float32(0.5)
        for threads in [None, 1, 2]:
            kernel.set_threads(threads)
            out = np.zeros(10000, dtype=np.float32)
            kernel.run("squares", out, 10000, 0.5)
            np.testing.assert_array_equal(out, want)
        self.assertEqual(kernel.threads, 2 if kernel.threads >= 2 else 1)
        with self.assertRaises(ValueError):
            kernel.set_threads(0)

    def test_par_host_callback_from_workers(self):
        """Host callbacks run on `par` workers while the caller has released the GIL."""
        import ctypes
        import threading
        import time
        cb_ty = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)
        threads_seen = set()

        def triple(x):
            threads_seen.add(threading.get_ident())
            # Sleeping releases the GIL, so callbacks on other workers overlap, and the
            # calling thread cannot finish every index before the helpers join.
            time.sleep(0.002)
            return 3 * x

        cb = cb_ty(triple)
        achainsaw.register_symbol("py_triple", ctypes.cast(cb, ctypes.c_void_p).value)
        kernel = achainsaw.compile(
            "extfn py_triple(x:i64)->i64\n"
            "fn body(i:i64, out:ptr)\n  b0:\n    v = call py_triple(i)\n"
            "    off = mul i, 8:i64\n    q = add out, off\n    st q, v\n    ret\n"
            "fn run(out:ptr, n:i64)\n  b0:\n    par n, body(out)\n    ret\n"
        )
        # A `par` runs serially while another thread's `par` has the pool, so retry.
        for _ in range(5):
            threads_seen.clear()
            out = np.zeros(64, dtype=np.int64)
            kernel.run("run", out, 64)
            np.testing.assert_array_equal(out, 3 * np.arange(64, dtype=np.int64))
            if len(threads_seen) > 1:
                break
        if kernel.threads > 1:
            self.assertGreater(len(threads_seen), 1)

    def test_par_kernel_from_python_threads(self):
        """Python threads share one kernel: calls release the GIL and take turns."""
        from concurrent.futures import ThreadPoolExecutor
        kernel = achainsaw.compile(self.PAR_SQUARES)
        want = np.arange(5000, dtype=np.float32) ** 2

        def job(_):
            out = np.zeros(5000, dtype=np.float32)
            kernel.run("squares", out, 5000, 1.0)
            return np.array_equal(out, want)

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(pool.map(job, range(16))))

    def test_par_signature_diagnostic(self):
        res = achainsaw.check(
            "fn body(i:i32)\n  b0:\n    ret\nfn run()\n  b0:\n    par 4:i64, body()\n    ret\n"
        )
        self.assertEqual(res.get("status"), "error")
        self.assertEqual(res.get("error_code"), "ERR_PAR_SIGNATURE")
        self.assertEqual(res["context"]["expected_signature"], "fn body(i64)")


if __name__ == "__main__":
    unittest.main()

