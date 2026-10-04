use achainsaw_codegen::{
    get_global_symbol_address, load_global_library, register_global_symbol, JitEngine, RtValue,
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
create_exception!(achainsaw, ExecutionError, CompilationError);

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

    /// Code generator that compiled this kernel ("cranelift" or "llvm").
    #[getter]
    pub fn backend(&self) -> &'static str {
        self.engine.backend().as_str()
    }

    #[pyo3(signature = (fuel=None))]
    pub fn set_fuel(&mut self, fuel: Option<u64>) {
        self.engine.set_fuel(fuel);
    }

    pub fn set_memory_quota(&self, quota_bytes: usize) {
        self.engine.set_memory_quota(quota_bytes);
    }

    #[pyo3(signature = (func_name, *args))]
    pub fn run(
        &self,
        py: Python<'_>,
        func_name: &str,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<PyObject> {
        let (param_types, _ret_type) = self.signatures.get(func_name).ok_or_else(|| {
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

        let mut rt_args = Vec::with_capacity(param_types.len());
        let mut _buffers: Vec<BufferGuard> = Vec::new();

        for (i, &ty) in param_types.iter().enumerate() {
            let arg = args.get_item(i)?;
            let rt_val = match ty {
                Type::Ptr => {
                    let mut view: pyo3::ffi::Py_buffer = unsafe { std::mem::zeroed() };
                    let is_writable_buffer = unsafe {
                        pyo3::ffi::PyObject_GetBuffer(
                            arg.as_ptr(),
                            &mut view,
                            pyo3::ffi::PyBUF_WRITABLE | pyo3::ffi::PyBUF_ND,
                        ) == 0
                    };

                    let is_buffer = if is_writable_buffer {
                        true
                    } else {
                        unsafe { pyo3::ffi::PyErr_Clear() };
                        unsafe {
                            pyo3::ffi::PyObject_GetBuffer(
                                arg.as_ptr(),
                                &mut view,
                                pyo3::ffi::PyBUF_SIMPLE,
                            ) == 0
                        }
                    };

                    if is_buffer {
                        let ptr = view.buf as usize;
                        _buffers.push(BufferGuard { view });
                        RtValue::Ptr(ptr)
                    } else {
                        unsafe {
                            pyo3::ffi::PyErr_Clear();
                        }
                        if let Ok(ai) = arg.getattr("__array_interface__") {
                            if let Ok(data) = ai.get_item("data") {
                                if let Ok(tuple) = data.extract::<(usize, bool)>() {
                                    RtValue::Ptr(tuple.0)
                                } else if let Ok(tuple) = data.extract::<(i64, bool)>() {
                                    RtValue::Ptr(tuple.0 as usize)
                                } else {
                                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                        "Argument {i} expected pointer or buffer, found {:?}",
                                        arg
                                    )));
                                }
                            } else {
                                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                    "Argument {i} expected pointer or buffer, found {:?}",
                                    arg
                                )));
                            }
                        } else if let Ok(val) = arg.extract::<usize>() {
                            RtValue::Ptr(val)
                        } else if let Ok(val) = arg.extract::<i64>() {
                            RtValue::Ptr(val as usize)
                        } else {
                            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                "Argument {i} expected pointer or buffer (NumPy array), found {:?}",
                                arg
                            )));
                        }
                    }
                }
                Type::I8 => RtValue::I8(arg.extract()?),
                Type::I16 => RtValue::I16(arg.extract()?),
                Type::I32 => RtValue::I32(arg.extract()?),
                Type::I64 => RtValue::I64(arg.extract()?),
                Type::F32 => RtValue::F32(arg.extract()?),
                Type::F64 => RtValue::F64(arg.extract()?),
                Type::V128 | Type::V256 | Type::V512 | Type::Vx | Type::F16 | Type::BF16 => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Direct passing of {ty} arguments across Python boundary not supported; pass by pointer (ptr)",
                    )));
                }
            };
            rt_args.push(rt_val);
        }

        let res_rt = unsafe { self.engine.call_typed(func_name, &rt_args) }.map_err(|e| {
            check_execution_status_py(py).err().unwrap_or_else(|| {
                let err_type = py.get_type_bound::<ExecutionError>();
                PyErr::from_value_bound(err_type.call1((e.to_string(),)).unwrap())
            })
        })?;

        drop(_buffers);

        match res_rt {
            Some(RtValue::I8(n)) => Ok(n.into_py(py)),
            Some(RtValue::I16(n)) => Ok(n.into_py(py)),
            Some(RtValue::I32(n)) => Ok(n.into_py(py)),
            Some(RtValue::I64(n)) => Ok(n.into_py(py)),
            Some(RtValue::Ptr(p)) => Ok(p.into_py(py)),
            Some(RtValue::F32(f)) => Ok(f.into_py(py)),
            Some(RtValue::F64(f)) => Ok(f.into_py(py)),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (*args))]
    pub fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        if !args.is_empty() {
            if let Ok(name) = args.get_item(0)?.extract::<String>() {
                if self.signatures.contains_key(&name) {
                    let sub_args = args.get_slice(1, args.len());
                    return self.run(py, &name, &sub_args);
                }
            }
        }
        if self.signatures.len() == 1 {
            let func_name = self.signatures.keys().next().unwrap().clone();
            return self.run(py, &func_name, args);
        }
        if self.signatures.contains_key("main") {
            return self.run(py, "main", args);
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "Kernel has multiple functions. Specify function name as first argument or call kernel.run('func_name', ...)",
        ))
    }
}

fn check_execution_status_py(py: Python<'_>) -> PyResult<()> {
    achainsaw_codegen::check_execution_status().map_err(|e| {
        let status = achainsaw_codegen::get_execution_status();
        match status {
            achainsaw_codegen::ExecutionStatus::OutOfFuel => {
                let err_type = py.get_type_bound::<ExecutionError>();
                let msg = "[ERR_OUT_OF_FUEL] Execution halted: loop fuel budget exhausted";
                let err_instance = err_type.call1((msg,)).unwrap();
                let diag = serde_json::json!({
                    "status": "error",
                    "error_code": "ERR_OUT_OF_FUEL",
                    "message": "Execution halted: loop fuel budget exhausted",
                });
                if let Ok(py_dict) = py
                    .import_bound("json")
                    .and_then(|m| m.call_method1("loads", (diag.to_string(),)))
                {
                    let _ = err_instance.setattr("diagnostic", py_dict);
                }
                PyErr::from_value_bound(err_instance)
            }
            achainsaw_codegen::ExecutionStatus::OutOfMemory { requested, limit } => {
                let err_type = py.get_type_bound::<ExecutionError>();
                let msg = format!(
                    "[ERR_OUT_OF_MEMORY] Allocation of {requested} bytes exceeded memory quota of {limit} bytes"
                );
                let err_instance = err_type.call1((msg.clone(),)).unwrap();
                let diag = serde_json::json!({
                    "status": "error",
                    "error_code": "ERR_OUT_OF_MEMORY",
                    "message": msg,
                    "context": {
                        "requested_bytes": requested,
                        "quota_bytes": limit,
                    }
                });
                if let Ok(py_dict) = py
                    .import_bound("json")
                    .and_then(|m| m.call_method1("loads", (diag.to_string(),)))
                {
                    let _ = err_instance.setattr("diagnostic", py_dict);
                }
                PyErr::from_value_bound(err_instance)
            }
            _ => {
                let err_type = py.get_type_bound::<ExecutionError>();
                PyErr::from_value_bound(err_type.call1((e.to_string(),)).unwrap())
            }
        }
    })
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
    let bytes = encode_module(&module).map_err(|d| diagnostic_to_py_err(py, d))?;
    Ok(pyo3::types::PyBytes::new_bound(py, &bytes).into_py(py))
}

#[pyfunction]
pub fn disassemble(py: Python<'_>, bytes: &[u8]) -> PyResult<String> {
    let module = decode_module(bytes).map_err(|d| diagnostic_to_py_err(py, d))?;
    Ok(to_air_text(&module))
}

fn build_kernel(module: &achainsaw_ir::Module, backend: Option<&str>) -> PyResult<PyKernel> {
    let runtime_err = |e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());

    let mut signatures = HashMap::new();
    for func in &module.functions {
        let p_types: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
        signatures.insert(func.name.clone(), (p_types, func.ret_type));
    }

    let mut engine = JitEngine::for_backend(backend).map_err(runtime_err)?;
    engine.compile_module(module).map_err(runtime_err)?;

    Ok(PyKernel { engine, signatures })
}

/// Compiles AIRB bytecode. `backend` is "cranelift", "llvm" or "auto" (the default, which
/// follows ACHAINSAW_BACKEND and otherwise uses Cranelift).
#[pyfunction]
#[pyo3(signature = (bytes, backend=None))]
pub fn compile_binary(py: Python<'_>, bytes: &[u8], backend: Option<&str>) -> PyResult<PyKernel> {
    let module = decode_module(bytes).map_err(|d| diagnostic_to_py_err(py, d))?;
    let mut validator = achainsaw_ir::Validator::new();
    validator
        .validate_module(&module)
        .map_err(|d| diagnostic_to_py_err(py, d))?;
    build_kernel(&module, backend)
}

/// Compiles AIR source text. `backend` is "cranelift", "llvm" or "auto" (the default,
/// which follows ACHAINSAW_BACKEND and otherwise uses Cranelift).
#[pyfunction]
#[pyo3(signature = (source, backend=None))]
pub fn compile(py: Python<'_>, source: &str, backend: Option<&str>) -> PyResult<PyKernel> {
    let module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;
    build_kernel(&module, backend)
}

/// Backends compiled into this build, e.g. ["cranelift", "llvm"].
#[pyfunction]
pub fn available_backends() -> Vec<&'static str> {
    achainsaw_codegen::Backend::available()
        .iter()
        .map(|b| b.as_str())
        .collect()
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
#[pyo3(signature = (fuel=None))]
pub fn set_fuel(fuel: Option<u64>) {
    achainsaw_codegen::set_execution_fuel(fuel);
}

#[pyfunction]
pub fn get_remaining_fuel() -> Option<u64> {
    achainsaw_codegen::get_remaining_fuel()
}

#[pyfunction]
pub fn set_memory_quota(quota_bytes: usize) {
    achainsaw_codegen::set_memory_quota(quota_bytes);
}

#[pyfunction]
pub fn get_allocated_memory() -> usize {
    achainsaw_codegen::get_allocated_memory()
}

#[pyfunction]
pub fn version() -> &'static str {
    "0.1.0"
}

#[pyfunction]
#[pyo3(signature = (source))]
pub fn optimize(py: Python<'_>, source: &str) -> PyResult<PyObject> {
    let mut module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;
    let stats = achainsaw_ir::opt::optimize_module(&mut module);
    let optimized_code = to_air_text(&module);

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("code", optimized_code)?;
    dict.set_item("constants_folded", stats.constants_folded)?;
    dict.set_item("algebraic_simplifications", stats.algebraic_simplifications)?;
    dict.set_item("branches_folded", stats.branches_folded)?;
    dict.set_item("dead_instructions_removed", stats.dead_instructions_removed)?;
    dict.set_item("dead_blocks_removed", stats.dead_blocks_removed)?;
    dict.set_item("total_optimizations", stats.total_optimizations())?;
    Ok(dict.into_py(py))
}

/// Host CPU vector features, active ISA cap, and backend vector widths.
#[pyfunction]
pub fn cpu_features(py: Python<'_>) -> PyResult<PyObject> {
    let report = achainsaw_codegen::cpu::target_report()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let parsed = py
        .import_bound("json")?
        .call_method1("loads", (report.to_string(),))?;
    Ok(parsed.into_py(py))
}

/// Caps the vector ISA for kernels compiled afterwards (`None` removes the cap).
#[pyfunction]
#[pyo3(signature = (level))]
pub fn set_isa_cap(level: Option<&str>) -> PyResult<()> {
    let cap = level
        .map(|l| l.parse::<achainsaw_codegen::cpu::IsaLevel>())
        .transpose()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    achainsaw_codegen::cpu::set_isa_cap(cap)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn get_isa_cap() -> PyResult<Option<String>> {
    achainsaw_codegen::cpu::isa_cap()
        .map(|c| c.map(|l| l.as_str().to_string()))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn achainsaw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernel>()?;
    m.add(
        "CompilationError",
        m.py().get_type_bound::<CompilationError>(),
    )?;
    m.add("ExecutionError", m.py().get_type_bound::<ExecutionError>())?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(assemble, m)?)?;
    m.add_function(wrap_pyfunction!(disassemble, m)?)?;
    m.add_function(wrap_pyfunction!(compile_binary, m)?)?;
    m.add_function(wrap_pyfunction!(register_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(load_library, m)?)?;
    m.add_function(wrap_pyfunction!(get_symbol_address, m)?)?;
    m.add_function(wrap_pyfunction!(set_fuel, m)?)?;
    m.add_function(wrap_pyfunction!(get_remaining_fuel, m)?)?;
    m.add_function(wrap_pyfunction!(set_memory_quota, m)?)?;
    m.add_function(wrap_pyfunction!(get_allocated_memory, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    m.add_function(wrap_pyfunction!(set_isa_cap, m)?)?;
    m.add_function(wrap_pyfunction!(get_isa_cap, m)?)?;
    Ok(())
}
