use achainsaw_codegen::{
    get_global_symbol_address, load_global_library, register_global_symbol, JitEngine, RtValue,
};
use achainsaw_ir::diag::Diagnostic;
use achainsaw_ir::types::Type;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, TryLockError};
use std::thread::ThreadId;

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
    /// pyo3 requires `#[pyclass]` types to be `Sync`; the engine (JIT state, fuel counter)
    /// is not, so calls go through a mutex. Calls release the GIL, so other Python threads
    /// (and host callbacks on `par` workers) keep running; a second thread calling the same
    /// kernel waits for it.
    engine: Mutex<JitEngine>,
    /// Thread currently running AIR code of this kernel, to turn re-entry (a host callback
    /// calling the same kernel while it runs) into a Python error instead of a deadlock.
    running: Mutex<Option<ThreadId>>,
    signatures: HashMap<String, (Vec<Type>, Option<Type>)>,
}

/// Lets a `!Sync` value cross into `Python::detach`; the caller keeps it alive and
/// unaliased for the duration.
struct Unshared<T>(*const T);
unsafe impl<T> Send for Unshared<T> {}
impl<T> Unshared<T> {
    unsafe fn get(&self) -> &T {
        &*self.0
    }
}

impl PyKernel {
    fn engine(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, JitEngine>> {
        let reentrant = || {
            pyo3::exceptions::PyRuntimeError::new_err(
                "kernel is already running (re-entrant call from a host callback)",
            )
        };
        let me = std::thread::current().id();
        loop {
            match self.engine.try_lock() {
                Ok(guard) => return Ok(guard),
                // A panic during an earlier call (raised in Python as PanicException) poisons
                // the lock but leaves the engine intact.
                Err(TryLockError::Poisoned(p)) => return Ok(p.into_inner()),
                Err(TryLockError::WouldBlock) => {
                    let running = *self.running.lock().unwrap_or_else(|e| e.into_inner());
                    // Waiting on the running thread, or on a `par` worker, would deadlock.
                    if running == Some(me) || achainsaw_codegen::in_par_worker() {
                        return Err(reentrant());
                    }
                    // Another Python thread is running it: wait without holding the GIL.
                    py.detach(|| drop(self.engine.lock()));
                }
            }
        }
    }

    /// Runs `func_name` with the GIL released, so host callbacks from `par` workers can
    /// take it.
    fn call(
        &self,
        py: Python<'_>,
        func_name: &str,
        args: &[RtValue],
    ) -> PyResult<anyhow::Result<Option<RtValue>>> {
        let engine = self.engine(py)?;
        let set_running = |t| *self.running.lock().unwrap_or_else(|e| e.into_inner()) = t;
        set_running(Some(std::thread::current().id()));
        let shared = Unshared(&*engine as *const JitEngine);
        // SAFETY: `engine` holds the lock until the call returns.
        let res = py.detach(move || unsafe { shared.get().call_typed(func_name, args) });
        set_running(None);
        Ok(res)
    }
}

#[pymethods]
impl PyKernel {
    pub fn get_function_names(&self) -> Vec<String> {
        self.signatures.keys().cloned().collect()
    }

    pub fn lookup_symbol(&self, py: Python<'_>, name: &str) -> PyResult<Option<usize>> {
        Ok(self.engine(py)?.lookup_symbol(name).map(|ptr| ptr as usize))
    }

    /// Code generator that compiled this kernel ("cranelift" or "llvm").
    #[getter]
    pub fn backend(&self, py: Python<'_>) -> PyResult<&'static str> {
        Ok(self.engine(py)?.backend().as_str())
    }

    /// Whether float min/max were compiled as compare and select (see `compile`).
    #[getter]
    pub fn fast_math(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.engine(py)?.fast_math())
    }

    /// Most threads `par` loops may use (`None`: all cores; 1 runs them serially).
    #[pyo3(signature = (threads=None))]
    pub fn set_threads(&self, py: Python<'_>, threads: Option<usize>) -> PyResult<()> {
        if threads == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threads must be at least 1",
            ));
        }
        self.engine(py)?.set_threads(threads);
        Ok(())
    }

    /// Threads `par` loops run on.
    #[getter]
    pub fn threads(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.engine(py)?.threads())
    }

    #[pyo3(signature = (fuel=None))]
    pub fn set_fuel(&self, py: Python<'_>, fuel: Option<u64>) -> PyResult<()> {
        self.engine(py)?.set_fuel(fuel);
        Ok(())
    }

    pub fn set_memory_quota(&self, py: Python<'_>, quota_bytes: usize) -> PyResult<()> {
        self.engine(py)?.set_memory_quota(quota_bytes);
        Ok(())
    }

    #[pyo3(signature = (func_name, *args))]
    pub fn run(
        &self,
        py: Python<'_>,
        func_name: &str,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyAny>> {
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

        let res_rt = self.call(py, func_name, &rt_args)?.map_err(|e| {
            check_execution_status_py(py).err().unwrap_or_else(|| {
                let err_type = py.get_type::<ExecutionError>();
                PyErr::from_value(err_type.call1((e.to_string(),)).unwrap())
            })
        })?;

        drop(_buffers);

        match res_rt {
            Some(RtValue::I8(n)) => n.into_py_any(py),
            Some(RtValue::I16(n)) => n.into_py_any(py),
            Some(RtValue::I32(n)) => n.into_py_any(py),
            Some(RtValue::I64(n)) => n.into_py_any(py),
            Some(RtValue::Ptr(p)) => p.into_py_any(py),
            Some(RtValue::F32(f)) => f.into_py_any(py),
            Some(RtValue::F64(f)) => f.into_py_any(py),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (*args))]
    pub fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
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
                let err_type = py.get_type::<ExecutionError>();
                let msg = "[ERR_OUT_OF_FUEL] Execution halted: loop fuel budget exhausted";
                let err_instance = err_type.call1((msg,)).unwrap();
                let diag = serde_json::json!({
                    "status": "error",
                    "error_code": "ERR_OUT_OF_FUEL",
                    "message": "Execution halted: loop fuel budget exhausted",
                });
                if let Ok(py_dict) = py
                    .import("json")
                    .and_then(|m| m.call_method1("loads", (diag.to_string(),)))
                {
                    let _ = err_instance.setattr("diagnostic", py_dict);
                }
                PyErr::from_value(err_instance)
            }
            achainsaw_codegen::ExecutionStatus::OutOfMemory { requested, limit } => {
                let err_type = py.get_type::<ExecutionError>();
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
                    .import("json")
                    .and_then(|m| m.call_method1("loads", (diag.to_string(),)))
                {
                    let _ = err_instance.setattr("diagnostic", py_dict);
                }
                PyErr::from_value(err_instance)
            }
            _ => {
                let err_type = py.get_type::<ExecutionError>();
                PyErr::from_value(err_type.call1((e.to_string(),)).unwrap())
            }
        }
    })
}

fn diagnostic_to_py_err(py: Python<'_>, diag: Diagnostic) -> PyErr {
    let err_type = py.get_type::<CompilationError>();
    let err_instance = err_type
        .call1((format!("[{}] {}", diag.error_code, diag.message),))
        .unwrap();

    let diag_json = serde_json::to_string(&diag).unwrap_or_else(|_| "{}".to_string());
    if let Ok(py_dict) = py
        .import("json")
        .and_then(|m| m.call_method1("loads", (diag_json,)))
    {
        let _ = err_instance.setattr("diagnostic", py_dict);
    }

    PyErr::from_value(err_instance)
}

#[pyfunction]
pub fn check(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    match parse_and_validate(source) {
        Ok(module) => {
            let dict = PyDict::new(py);
            dict.set_item("status", "ok")?;
            let func_names: Vec<String> = module.functions.iter().map(|f| f.name.clone()).collect();
            dict.set_item("functions", func_names)?;
            dict.set_item("function_count", module.functions.len())?;
            let block_count: usize = module.functions.iter().map(|f| f.blocks.len()).sum();
            dict.set_item("block_count", block_count)?;
            Ok(dict.into_any().unbind())
        }
        Err(diag) => {
            let diag_json = serde_json::to_string(&diag).unwrap_or_else(|_| "{}".to_string());
            let json_mod = py.import("json")?;
            let parsed = json_mod.call_method1("loads", (diag_json,))?;
            Ok(parsed.unbind())
        }
    }
}

#[pyfunction]
pub fn assemble(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    let module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;
    let bytes = encode_module(&module).map_err(|d| diagnostic_to_py_err(py, d))?;
    Ok(PyBytes::new(py, &bytes).into_any().unbind())
}

#[pyfunction]
pub fn disassemble(py: Python<'_>, bytes: &[u8]) -> PyResult<String> {
    let module = decode_module(bytes).map_err(|d| diagnostic_to_py_err(py, d))?;
    Ok(to_air_text(&module))
}

fn build_kernel(
    module: &achainsaw_ir::Module,
    backend: Option<&str>,
    fast_math: bool,
) -> PyResult<PyKernel> {
    let runtime_err = |e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());

    let mut signatures = HashMap::new();
    for func in &module.functions {
        let p_types: Vec<Type> = func.params.iter().map(|(_, ty)| *ty).collect();
        signatures.insert(func.name.clone(), (p_types, func.ret_type));
    }

    let mut engine = JitEngine::for_module(backend, module).map_err(runtime_err)?;
    engine.set_fast_math(fast_math);
    engine.compile_module(module).map_err(runtime_err)?;

    Ok(PyKernel {
        engine: Mutex::new(engine),
        running: Mutex::new(None),
        signatures,
    })
}

/// Compiles AIRB bytecode; `backend` and `fast_math` as in `compile`.
#[pyfunction]
#[pyo3(signature = (bytes, backend=None, fast_math=false))]
pub fn compile_binary(
    py: Python<'_>,
    bytes: &[u8],
    backend: Option<&str>,
    fast_math: bool,
) -> PyResult<PyKernel> {
    let module = decode_module(bytes).map_err(|d| diagnostic_to_py_err(py, d))?;
    let mut validator = achainsaw_ir::Validator::new();
    validator
        .validate_module(&module)
        .map_err(|d| diagnostic_to_py_err(py, d))?;
    build_kernel(&module, backend, fast_math)
}

/// Compiles AIR source text. `backend` is "cranelift", "llvm" or "auto" (the default:
/// follows ACHAINSAW_BACKEND, else LLVM for wide-vector modules when built in, else
/// Cranelift). `fast_math` compiles float min/max (min, max, vmin, vmax, vminr, vmaxr) as
/// compare and select, `max(a, b) = a > b ? a : b`, so a NaN operand or two zeros give `b`;
/// by default they propagate NaN and order -0.0 below +0.0. It also lets bf16 `mm` use AMX or
/// SME, which treat bf16 subnormal inputs as zero; by default bf16 `mm` is exact.
#[pyfunction]
#[pyo3(signature = (source, backend=None, fast_math=false))]
pub fn compile(
    py: Python<'_>,
    source: &str,
    backend: Option<&str>,
    fast_math: bool,
) -> PyResult<PyKernel> {
    let module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;
    build_kernel(&module, backend, fast_math)
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
pub fn optimize(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    let mut module = parse_and_validate(source).map_err(|d| diagnostic_to_py_err(py, d))?;
    let stats = achainsaw_ir::opt::optimize_module(&mut module);
    let optimized_code = to_air_text(&module);

    let dict = PyDict::new(py);
    dict.set_item("code", optimized_code)?;
    dict.set_item("constants_folded", stats.constants_folded)?;
    dict.set_item("algebraic_simplifications", stats.algebraic_simplifications)?;
    dict.set_item("branches_folded", stats.branches_folded)?;
    dict.set_item("dead_instructions_removed", stats.dead_instructions_removed)?;
    dict.set_item("dead_blocks_removed", stats.dead_blocks_removed)?;
    dict.set_item("total_optimizations", stats.total_optimizations())?;
    Ok(dict.into_any().unbind())
}

/// Host CPU vector features, active ISA cap, and backend vector widths.
#[pyfunction]
pub fn cpu_features(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let report = achainsaw_codegen::cpu::target_report()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let parsed = py
        .import("json")?
        .call_method1("loads", (report.to_string(),))?;
    Ok(parsed.unbind())
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
    m.add("CompilationError", m.py().get_type::<CompilationError>())?;
    m.add("ExecutionError", m.py().get_type::<ExecutionError>())?;
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
