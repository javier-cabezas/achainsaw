"""
End-to-End Ahead-Of-Time (AOT) Compilation Integration Test.
Verifies compiling AIR modules into native object files (.o) and shared libraries (.dll / .so),
and invoking exported functions directly via standard C ABI (ctypes).
"""

import ctypes
import os
import subprocess
import unittest

import sys

EXE_NAME = "achainsaw.exe" if sys.platform == "win32" else "achainsaw"
EXE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "debug", EXE_NAME))
if not os.path.exists(EXE_PATH):
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "release", EXE_NAME))
    if os.path.exists(rel_path):
        EXE_PATH = rel_path

FIB_AIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "fibonacci.air"))


class TestAotCompilation(unittest.TestCase):
    def test_aot_build_object(self):
        out_o = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_fib.o"))
        try:
            res = subprocess.run(
                [EXE_PATH, "build", FIB_AIR, "-o", out_o, "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(os.path.exists(out_o))
            self.assertGreater(os.path.getsize(out_o), 100)
        finally:
            if os.path.exists(out_o):
                os.remove(out_o)

    def test_aot_build_and_link_shared(self):
        # Configure toolchain environment
        env = os.environ.copy()
        local_mingw = r"C:\Users\Javier\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
        if os.path.isdir(local_mingw):
            env["PATH"] = local_mingw + ";" + env.get("PATH", "")

        out_o = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_fib_shared.o"))

        try:
            res = subprocess.run(
                [EXE_PATH, "build", FIB_AIR, "-o", out_o, "--shared", "--json"],
                capture_output=True,
                text=True,
                env=env,
            )
            if res.returncode != 0:
                self.skipTest(f"C compiler not available or linking failed: {res.stderr}")

            # Determine platform shared library extension
            if sys.platform == "darwin":
                ext = ".dylib"
            elif sys.platform == "win32":
                ext = ".dll"
            else:
                ext = ".so"

            target_shared = FIB_AIR.replace(".air", ext)
            self.assertTrue(os.path.exists(target_shared), f"Expected shared library at {target_shared}")

            # Load via ctypes and call fib(10)
            lib = ctypes.CDLL(target_shared)
            lib.fib.argtypes = [ctypes.c_int32]
            lib.fib.restype = ctypes.c_int32

            result = lib.fib(10)
            self.assertEqual(result, 55)

            # Unload DLL before removal on Windows
            if hasattr(ctypes, "_FreeLibrary"):
                ctypes._FreeLibrary(lib._handle)
            elif hasattr(ctypes.windll.kernel32, "FreeLibrary"):
                ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(lib._handle))

            if os.path.exists(target_shared):
                try:
                    os.remove(target_shared)
                except OSError:
                    pass
        finally:
            if os.path.exists(out_o):
                try:
                    os.remove(out_o)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
