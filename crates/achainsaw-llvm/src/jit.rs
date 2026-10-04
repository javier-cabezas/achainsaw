use crate::lower::{self, lower_module, trampoline_name, LowerOptions};
use crate::target::{optimize, target_machine, TargetSpec};
use achainsaw_ir::ast::Module;
use anyhow::{anyhow, Result};
use inkwell::context::Context;
use inkwell::llvm_sys::error::{LLVMDisposeErrorMessage, LLVMErrorRef, LLVMGetErrorMessage};
use inkwell::llvm_sys::orc2::lljit::*;
use inkwell::llvm_sys::orc2::*;
use std::collections::HashMap;
use std::ffi::{CStr, CString};

/// Addresses of the host runtime hooks JIT code calls (see `achainsaw-codegen/src/jit.rs`).
#[derive(Debug, Clone, Copy)]
pub struct RuntimeHooks {
    pub malloc: usize,
    pub free: usize,
    pub fuel_exhausted: usize,
    pub consume_fuel: usize,
    pub sandbox_fault: usize,
    pub stack_check: usize,
    pub sandbox_check_mm: usize,
    pub par_for: usize,
    pub fuel_counter: usize,
}

/// One AIR module JIT compiled with ORC LLJIT. Code stays valid until this is dropped.
pub struct LlvmJit {
    jit: LLVMOrcLLJITRef,
    functions: HashMap<String, usize>,
    trampolines: HashMap<String, usize>,
}

// The LLJIT instance is only used through `&self` lookups that are already resolved.
unsafe impl Send for LlvmJit {}

fn take_error(err: LLVMErrorRef) -> Result<()> {
    if err.is_null() {
        return Ok(());
    }
    unsafe {
        let msg = LLVMGetErrorMessage(err);
        let text = CStr::from_ptr(msg).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(msg);
        Err(anyhow!("{text}"))
    }
}

impl LlvmJit {
    /// Lowers, optimizes and JIT compiles `air` for the host CPU described by `target`
    /// (`triple` must be `None`). Externs are resolved with `resolve`, falling back to
    /// symbols of the current process.
    pub fn compile(
        air: &Module,
        target: &TargetSpec,
        opts: &LowerOptions,
        hooks: &RuntimeHooks,
        resolve: &dyn Fn(&str) -> Option<usize>,
    ) -> Result<Self> {
        let tm = target_machine(target, true)?;
        let ctx = Context::create();
        let raw_module = {
            let opts = LowerOptions {
                aot: false,
                target_cpu: target.cpu.clone(),
                target_features: target.features.clone(),
                ..opts.clone()
            };
            let module = lower_module(&ctx, air, &opts)?;
            optimize(&module, &tm)?;
            let raw = module.as_mut_ptr();
            // Ownership moves to the thread-safe module below.
            std::mem::forget(module);
            raw
        };

        unsafe {
            let tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx.raw());
            std::mem::forget(ctx);
            let tsm = LLVMOrcCreateNewThreadSafeModule(raw_module, tsc);
            LLVMOrcDisposeThreadSafeContext(tsc);

            let jtmb = LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm.as_mut_ptr());
            std::mem::forget(tm);
            let builder = LLVMOrcCreateLLJITBuilder();
            LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(builder, jtmb);
            let mut jit = std::ptr::null_mut();
            if let Err(e) = take_error(LLVMOrcCreateLLJIT(&mut jit, builder)) {
                LLVMOrcDisposeThreadSafeModule(tsm);
                return Err(anyhow!("[ERR_LLVM_JIT] Cannot create LLVM JIT: {e}"));
            }
            let mut this = LlvmJit {
                jit,
                functions: HashMap::new(),
                trampolines: HashMap::new(),
            };
            let jd = LLVMOrcLLJITGetMainJITDylib(jit);

            let mut symbols: Vec<(String, usize)> = vec![
                (lower::RT_MALLOC.into(), hooks.malloc),
                (lower::RT_FREE.into(), hooks.free),
                (lower::RT_FUEL_EXHAUSTED.into(), hooks.fuel_exhausted),
                (lower::RT_CONSUME_FUEL.into(), hooks.consume_fuel),
                (lower::RT_SANDBOX_FAULT.into(), hooks.sandbox_fault),
                (lower::RT_STACK_CHECK.into(), hooks.stack_check),
                (lower::RT_SANDBOX_CHECK_MM.into(), hooks.sandbox_check_mm),
                (lower::RT_PAR_FOR.into(), hooks.par_for),
                (lower::RT_FUEL_COUNTER.into(), hooks.fuel_counter),
            ];
            for ext in &air.extern_functions {
                if let Some(addr) = resolve(&ext.name) {
                    symbols.push((ext.name.clone(), addr));
                }
            }
            symbols.extend(sme_abi_fallbacks());
            let names: Vec<CString> = symbols
                .iter()
                .map(|(n, _)| CString::new(n.as_str()).unwrap())
                .collect();
            let mut pairs: Vec<LLVMOrcCSymbolMapPair> = names
                .iter()
                .zip(&symbols)
                .map(|(name, (_, addr))| LLVMOrcCSymbolMapPair {
                    Name: LLVMOrcLLJITMangleAndIntern(jit, name.as_ptr()),
                    Sym: LLVMJITEvaluatedSymbol {
                        Address: *addr as u64,
                        Flags: LLVMJITSymbolFlags {
                            GenericFlags:
                                LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8
                                    | LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable
                                        as u8,
                            TargetFlags: 0,
                        },
                    },
                })
                .collect();
            let mu = LLVMOrcAbsoluteSymbols(pairs.as_mut_ptr(), pairs.len());
            take_error(LLVMOrcJITDylibDefine(jd, mu))?;

            // Libcalls LLVM may emit (memcpy, fmaf without FMA, ...) and other externs.
            let mut generator = std::ptr::null_mut();
            take_error(LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut generator,
                LLVMOrcLLJITGetGlobalPrefix(jit),
                None,
                std::ptr::null_mut(),
            ))?;
            LLVMOrcJITDylibAddGenerator(jd, generator);

            take_error(LLVMOrcLLJITAddLLVMIRModule(jit, jd, tsm))?;

            for func in &air.functions {
                let addr = this.lookup(&func.name)?;
                this.functions.insert(func.name.clone(), addr);
                let tramp = trampoline_name(&func.name);
                let scalar_only = func
                    .params
                    .iter()
                    .map(|(_, t)| *t)
                    .chain(func.ret_type)
                    .all(|t| !t.is_vector());
                if scalar_only {
                    let addr = this.lookup(&tramp)?;
                    this.trampolines.insert(func.name.clone(), addr);
                }
            }
            Ok(this)
        }
    }

    fn lookup(&self, name: &str) -> Result<usize> {
        let c = CString::new(name)?;
        let mut addr = 0;
        unsafe { take_error(LLVMOrcLLJITLookup(self.jit, &mut addr, c.as_ptr())) }
            .map_err(|e| anyhow!("[ERR_LLVM_JIT] Cannot materialize '{name}': {e}"))?;
        Ok(addr as usize)
    }

    /// Address of the compiled AIR function `name`.
    pub fn function(&self, name: &str) -> Option<usize> {
        self.functions.get(name).copied()
    }

    /// Address of `name`'s scalar trampoline `void(u64 *args, u64 *ret)`, absent for
    /// functions that take or return vectors.
    pub fn trampoline(&self, name: &str) -> Option<usize> {
        self.trampolines.get(name).copied()
    }

    pub fn function_names(&self) -> impl Iterator<Item = &str> {
        self.functions.keys().map(String::as_str)
    }
}

/// SME support routines that ZA-using code (the SME `mm` kernel) calls, for processes whose
/// C runtime lacks them (libgcc gained them in GCC 14). Only routines missing from the
/// process are returned; ORC resolves the others from it.
fn sme_abi_fallbacks() -> Vec<(String, usize)> {
    #[cfg(all(target_arch = "aarch64", unix))]
    {
        /// Called on entry to a function with new ZA state only when a caller left a lazy
        /// ZA save pending (TPIDR2_EL0 != 0). Nothing that calls into JIT code here sets one
        /// up, so reaching this means unsupported ZA use by the host: stop rather than
        /// corrupt the caller's ZA state.
        extern "C" fn tpidr2_save_unsupported() {
            eprintln!(
                "achainsaw: __arm_tpidr2_save called with a pending ZA lazy save; \
                 link a C runtime with SME support (GCC 14+ libgcc or compiler-rt)"
            );
            std::process::abort();
        }
        let name = c"__arm_tpidr2_save";
        // SAFETY: dlsym with a valid NUL-terminated name only performs a lookup.
        let found = unsafe { !libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()).is_null() };
        if !found {
            return vec![(
                "__arm_tpidr2_save".to_string(),
                tpidr2_save_unsupported as *const () as usize,
            )];
        }
    }
    Vec::new()
}

impl Drop for LlvmJit {
    fn drop(&mut self) {
        unsafe {
            let _ = take_error(LLVMOrcDisposeLLJIT(self.jit));
        }
    }
}
