use achainsaw_codegen::{
    get_global_symbol_address, load_global_library, register_global_symbol, JitEngine,
};
use achainsaw_ir::diag::Diagnostic;
use achainsaw_ir::types::Type;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

create_exception!(achainsaw, CompilationError, PyException);

struct BufferGuard {
    view: pyo3::ffi::Py_buffer,
}

impl Drop for BufferGuard {
    fn drop(&mut self) {
        unsafe {
            pyo3::ffi::PyBuffer_Release(&mut self.view);
        }
    }
}

#[pyclass(name = "Kernel")]
pub struct PyKernel {
    engine: JitEngine,
    signatures: HashMap<String, (Vec<Type>, Option<Type>)>,
}

#[pymethods]
impl PyKernel {
    pub fn get_function_names(&self) -> Vec<String> {
        self.signatures.keys().cloned().collect()
    }

    pub fn lookup_symbol(&self, name: &str) -> Option<usize> {
        self.engine.lookup_symbol(name).map(|ptr| ptr as usize)
    }

    #[pyo3(signature = (func_name, *args))]
    pub fn run(
        &self,
        py: Python<'_>,
        func_name: &str,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<PyObject> {
        let (param_types, ret_type) = self.signatures.get(func_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Function '{func_name}' not found"))
        })?;

        if args.len() != param_types.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Function '{}' expects {} arguments, received {}",
                func_name,
                param_types.len(),
                args.len()
            )));
        }

        let fn_ptr = self.engine.get_fn_ptr(func_name).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Function '{func_name}' has no compiled code pointer"
            ))
        })?;

        // Extract native arguments
        // On x86_64 C ABI (Windows and SysV):
        // Integers and pointers are passed in general purpose registers (rcx/rdx/r8/r9 or rdi/rsi/rdx/rcx/r8/r9).
        // Floats are passed in XMM0-XMM3.
        dispatch_call(py, fn_ptr, param_types, ret_type, args)
    }
}

fn dispatch_call(
    py: Python<'_>,
    fn_ptr: *const u8,
    param_types: &[Type],
    ret_type: &Option<Type>,
    args: &Bound<'_, PyTuple>,
) -> PyResult<PyObject> {
    // Collect pointer or 64-bit integer values
    let mut i_vals: Vec<i64> = Vec::new();
    let mut f_vals: Vec<f64> = Vec::new();

    // Keep active PyBuffer guards alive during call
    let mut _buffers: Vec<BufferGuard> = Vec::new();

    for (i, &ty) in param_types.iter().enumerate() {
        let arg = args.get_item(i)?;
        match ty {
            Type::Ptr => {
                // Try Python buffer protocol first
                let mut view: pyo3::ffi::Py_buffer = unsafe { std::mem::zeroed() };
                let is_buffer = unsafe {
                    pyo3::ffi::PyObject_GetBuffer(arg.as_ptr(), &mut view, pyo3::ffi::PyBUF_SIMPLE) == 0
                };

                if is_buffer {
                    let ptr = view.buf as i64;
                    _buffers.push(BufferGuard { view });
                    i_vals.push(ptr);
                } else {
                    unsafe {
                        pyo3::ffi::PyErr_Clear();
                    }
                    // Try __array_interface__ (NumPy, PyTorch, CuPy)
                    if let Ok(ai) = arg.getattr("__array_interface__") {
                        if let Ok(data) = ai.get_item("data") {
                            if let Ok(tuple) = data.extract::<(i64, bool)>() {
                                i_vals.push(tuple.0);
                                continue;
                            }
                        }
                    }
                    if let Ok(val) = arg.extract::<i64>() {
                        i_vals.push(val);
                    } else if let Ok(val) = arg.extract::<usize>() {
                        i_vals.push(val as i64);
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "Argument {i} expected pointer or buffer (NumPy array), found {:?}",
                            arg
                        )));
                    }
                }
            }
            Type::I64 => {
                let val: i64 = arg.extract()?;
                i_vals.push(val);
            }
            Type::I32 => {
                let val: i32 = arg.extract()?;
                i_vals.push(val as i64);
            }
            Type::I16 => {
                let val: i16 = arg.extract()?;
                i_vals.push(val as i64);
            }
            Type::I8 => {
                let val: i8 = arg.extract()?;
                i_vals.push(val as i64);
            }
            Type::F32 => {
                let val: f32 = arg.extract()?;
                f_vals.push(val as f64);
            }
            Type::F64 => {
                let val: f64 = arg.extract()?;
                f_vals.push(val);
            }
            Type::V128 => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Direct passing of v128 register arguments across Python boundary not supported; pass by pointer (ptr)",
                ));
            }
        }
    }

    // Dynamic execution dispatch based on parameter pattern
    unsafe {
        match (param_types, ret_type) {
            // Void returns
            ([], None) => {
                let f: extern "C" fn() = std::mem::transmute(fn_ptr);
                f();
                Ok(py.None())
            }
            ([Type::Ptr | Type::I64], None) => {
                let f: extern "C" fn(i64) = std::mem::transmute(fn_ptr);
                f(i_vals[0]);
                Ok(py.None())
            }
            ([Type::Ptr | Type::I64, Type::Ptr | Type::I64], None) => {
                let f: extern "C" fn(i64, i64) = std::mem::transmute(fn_ptr);
                f(i_vals[0], i_vals[1]);
                Ok(py.None())
            }
            ([Type::Ptr | Type::I64, Type::I32], None) => {
                let f: extern "C" fn(i64, i32) = std::mem::transmute(fn_ptr);
                f(i_vals[0], i_vals[1] as i32);
                Ok(py.None())
            }
            ([Type::Ptr | Type::I64, Type::F32, Type::I64 | Type::I32], None) => {
                let f: extern "C" fn(i64, f32, i64) = std::mem::transmute(fn_ptr);
                f(i_vals[0], f_vals[0] as f32, i_vals[1]);
                Ok(py.None())
            }
            ([Type::Ptr, Type::Ptr, Type::I64 | Type::I32], None) => {
                let f: extern "C" fn(i64, i64, i64) = std::mem::transmute(fn_ptr);
                f(i_vals[0], i_vals[1], i_vals[2]);
                Ok(py.None())
            }

            // Integer returns
            ([], Some(Type::I32)) => {
                let f: extern "C" fn() -> i32 = std::mem::transmute(fn_ptr);
                Ok(f().into_py(py))
            }
            ([Type::I32 | Type::I64], Some(Type::I32)) => {
                let f: extern "C" fn(i32) -> i32 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0] as i32).into_py(py))
            }
            ([Type::Ptr], Some(Type::I32)) => {
                let f: extern "C" fn(i64) -> i32 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0]).into_py(py))
            }
            ([Type::Ptr, Type::I32 | Type::I64], Some(Type::I32)) => {
                let f: extern "C" fn(i64, i32) -> i32 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0], i_vals[1] as i32).into_py(py))
            }
            ([Type::I32 | Type::I64, Type::I32 | Type::I64], Some(Type::I32)) => {
                let f: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0] as i32, i_vals[1] as i32).into_py(py))
            }
            ([Type::I32 | Type::I64, Type::I32 | Type::I64, Type::I32 | Type::I64], Some(Type::I32)) => {
                let f: extern "C" fn(i32, i32, i32) -> i32 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0] as i32, i_vals[1] as i32, i_vals[2] as i32).into_py(py))
            }
            ([Type::Ptr | Type::I64], Some(Type::I64)) => {
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0]).into_py(py))
            }
            ([Type::Ptr | Type::I64, Type::Ptr | Type::I64], Some(Type::I64)) => {
                let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(fn_ptr);
                Ok(f(i_vals[0], i_vals[1]).into_py(py))
            }

            // Float returns
            ([Type::F32], Some(Type::F32)) => {
                let f: extern "C" fn(f32) -> f32 = std::mem::transmute(fn_ptr);
                Ok(f(f_vals[0] as f32).into_py(py))
            }
            ([Type::F64], Some(Type::F64)) => {
                let f: extern "C" fn(f64) -> f64 = std::mem::transmute(fn_ptr);
                Ok(f(f_vals[0]).into_py(py))
            }
            ([Type::F32, Type::F32], Some(Type::F32)) => {
                let f: extern "C" fn(f32, f32) -> f32 = std::mem::transmute(fn_ptr);
                Ok(f(f_vals[0] as f32, f_vals[1] as f32).into_py(py))
            }
            ([Type::F64, Type::F64], Some(Type::F64)) => {
                let f: extern "C" fn(f64, f64) -> f64 = std::mem::transmute(fn_ptr);
                Ok(f(f_vals[0], f_vals[1]).into_py(py))
            }
            ([Type::Ptr, Type::Ptr, Type::I64 | Type::I32], Some(Type::F32)) => {
                let f: extern "C" fn(*const u8, *const u8, i64) -> f32 = std::mem::transmute(fn_ptr);
                let res = f(i_vals[0] as *const u8, i_vals[1] as *const u8, i_vals[2]);
                Ok(res.into_py(py))
            }
            ([Type::Ptr, Type::I64 | Type::I32], Some(Type::F32)) => {
                let f: extern "C" fn(*const u8, i64) -> f32 = std::mem::transmute(fn_ptr);
                let res = f(i_vals[0] as *const u8, i_vals[1]);
                Ok(res.into_py(py))
            }
            ([Type::Ptr], Some(Type::F32)) => {
                let f: extern "C" fn(*const u8) -> f32 = std::mem::transmute(fn_ptr);
                let res = f(i_vals[0] as *const u8);
                Ok(res.into_py(py))
            }

            // Pointer returns
            ([], Some(Type::Ptr)) => {
                let f: extern "C" fn() -> *mut u8 = std::mem::transmute(fn_ptr);
                Ok((f() as usize).into_py(py))
            }
            ([Type::I64 | Type::I32], Some(Type::Ptr)) => {
                let f: extern "C" fn(i64) -> *mut u8 = std::mem::transmute(fn_ptr);
                Ok((f(i_vals[0]) as usize).into_py(py))
            }

            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "Signature with params {:?} and return {:?} not yet mapped in dispatcher",
                param_types, ret_type
            ))),
        }
    }
}

fn diagnostic_to_py_err(py: Python<'_>, diag: Diagnostic) -> PyErr {
    let err_type = py.get_type_bound::<CompilationError>();
    let err_instance = err_type
        .call1((format!("[{}] {}", diag.error_code, diag.message),))
        .unwrap();

    let diag_json = serde_json::to_string(&diag).unwrap_or_else(|_| "{}".to_string());
    if let Ok(py_dict) = py
        .import_bound("json")
        .and_then(|m| m.call_method1("loads", (diag_json,)))
    {
        let _ = err_instance.setattr("diagnostic", py_dict);
    }

    PyErr::from_value_bound(err_instance)
}

#[pyfunction]
pub fn check(py: Python<'_>, source: &str) -> PyResult<PyObject> {
    match parse_and_validate(source) {
        Ok(module) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("status", "ok")?;
            let func_names: Vec<String> = module.functions.iter().map(|f| f.name.clone()).collect();
            dict.set_item("functions", func_names)?;
            dict.set_item("function_count", module.functions.len())?;
            let block_count: usize = module.functions.iter().map(|f| f.blocks.len()).sum();
            dict.set_item("block_count", block_count)?;
            Ok(dict.into_py(py))
        }
        Err(diag) => {
            let diag_json = serde_json::to_string(&diag).unwrap_or_else(|_| "{}".to_string());
            let json_mod = py.import_bound("json")?;
            let parsed = json_mod.call_method1("loads", (diag_json,))?;
            Ok(parsed.into_py(py))
        }
    }
}

#[pyfunction]
pub fn assemble(py: Python<'_>, source: &str) -> PyResult<PyObject> {
    let module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;
    let bytes = encode_module(&module);
    Ok(pyo3::types::PyBytes::new_bound(py, &bytes).into_py(py))
}

#[pyfunction]
pub fn disassemble(py: Python<'_>, bytes: &[u8]) -> PyResult<String> {
    let module = decode_module(bytes).map_err(|d| diagnostic_to_py_err(py, d))?;
    Ok(to_air_text(&module))
}

#[pyfunction]
pub fn compile_binary(py: Python<'_>, bytes: &[u8]) -> PyResult<PyKernel> {
    let module = decode_module(bytes).map_err(|d| diagnostic_to_py_err(py, d))?;
    let mut validator = achainsaw_ir::Validator::new();
    validator
        .validate_module(&module)
        .map_err(|d| diagnostic_to_py_err(py, d))?;

    let mut signatures = HashMap::new();
    for func in &module.functions {
        let p_types: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
        signatures.insert(func.name.clone(), (p_types, func.ret_type));
    }

    let mut engine = JitEngine::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    engine
        .compile_module(&module)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyKernel { engine, signatures })
}

#[pyfunction]
pub fn compile(py: Python<'_>, source: &str) -> PyResult<PyKernel> {
    let module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;

    let mut signatures = HashMap::new();
    for func in &module.functions {
        let p_types: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
        signatures.insert(func.name.clone(), (p_types, func.ret_type));
    }

    let mut engine = JitEngine::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    engine
        .compile_module(&module)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyKernel { engine, signatures })
}

#[pyfunction]
pub fn register_symbol(name: &str, addr: usize) {
    register_global_symbol(name, addr as *const u8);
}

#[pyfunction]
pub fn load_library(path: &str) -> PyResult<()> {
    load_global_library(path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn get_symbol_address(name: &str) -> Option<usize> {
    get_global_symbol_address(name).map(|ptr| ptr as usize)
}

#[pyfunction]
pub fn version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn achainsaw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernel>()?;
    m.add("CompilationError", m.py().get_type_bound::<CompilationError>())?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(assemble, m)?)?;
    m.add_function(wrap_pyfunction!(disassemble, m)?)?;
    m.add_function(wrap_pyfunction!(compile_binary, m)?)?;
    m.add_function(wrap_pyfunction!(register_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(load_library, m)?)?;
    m.add_function(wrap_pyfunction!(get_symbol_address, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
