# Calling C (`extfn`)

`achainsaw` can seamlessly call functions and access variables from external C/C++ libraries, Python extensions, and system shared objects (`.dll`, `.so`, `.dylib`).

## Declaring external functions
External functions are declared with the token-minimal `extfn` keyword and invoked with standard `call`:
```air
extfn sinf(x:f32)->f32
extfn cosf(x:f32)->f32
extfn sqrtf(x:f32)->f32

fn compute_sincos_norm(x:f32)->f32
  b0:
    s = call sinf(x)
    c = call cosf(x)
    s2 = mul s, s
    c2 = mul c, c
    sum = add s2, c2
    norm = call sqrtf(sum)
    ret norm
```

## C math functions
The JIT engine pre-registers high-performance standard C math functions out of the box with zero runtime overhead:
- **32-bit float:** `sinf`, `cosf`, `tanf`, `sqrtf`, `expf`, `logf`, `powf`, `fabsf`, `floorf`, `ceilf`, `roundf`
- **64-bit float:** `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `pow`, `fabs`, `floor`, `ceil`, `round`

## Loading shared libraries
Load any dynamic library at runtime. Exported C symbols are resolved automatically during JIT linking:
```python
import achainsaw

# Dynamically load a shared library
achainsaw.load_library("libcustom_math.so")  # or "custom_math.dll"

# Execute kernel calling symbols from the loaded library
kernel = achainsaw.compile("""
extfn custom_dsp_filter(input:ptr, output:ptr, len:i64)
fn process(a:ptr, b:ptr, n:i64)
  b0:
    call custom_dsp_filter(a, b, n)
    ret
""")
```

## Registering symbols and callbacks
Register any raw function pointer or host callback directly into `achainsaw`'s symbol table:
```python
import ctypes
import achainsaw

# Create a C callback from Python
callback_type = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
def custom_fma(a, b, c):
    return a * b + c
cb = callback_type(custom_fma)

# Register pointer in achainsaw
achainsaw.register_symbol("host_fma", ctypes.cast(cb, ctypes.c_void_p).value)

# Call registered symbol inside an AIR kernel
k = achainsaw.compile("""
extfn host_fma(a:i32, b:i32, c:i32)->i32
fn compute(x:i32, y:i32, z:i32)->i32
  b0:
    res = call host_fma(x, y, z)
    ret res
""")
print(k.run("compute", 3, 4, 5))  # 3 * 4 + 5 = 17
```

## External variables
External variables and exported global memory can be directly accessed and mutated via pointers (`ptr`):
- **Method A (Zero-Overhead Pointer Passing):** Pass the address of the external variable directly to the kernel, and read/write it at native CPU speed using `ld` and `st`:
  ```air
  fn increment_counter(p:ptr, delta:i32)->i32
    b0:
      val = ld p:i32
      new_val = add val, delta
      st p, new_val
      ret new_val
  ```
- **Method B (Symbol Address Resolution):** Query any registered or library-exported symbol address directly via `achainsaw.get_symbol_address(name)` or `kernel.lookup_symbol(name)`.
