use crate::lower::{lower_module, LowerOptions};
use crate::target::{optimize, target_machine, TargetSpec};
use achainsaw_ir::ast::Module;
use anyhow::Result;
use inkwell::context::Context;
use inkwell::targets::FileType;

/// Compiles `air` to a relocatable object file for `spec`, returning the object and its
/// target triple. Like the Cranelift AOT backend, `alloc`/`free` call libc `malloc`/`free`
/// and there is no fuel or sandboxing.
pub fn compile_object(
    air: &Module,
    spec: &TargetSpec,
    opts: &LowerOptions,
) -> Result<(Vec<u8>, String)> {
    emit(air, spec, opts, FileType::Object)
}

/// Like [`compile_object`], but returns target assembly text.
pub fn compile_assembly(
    air: &Module,
    spec: &TargetSpec,
    opts: &LowerOptions,
) -> Result<(String, String)> {
    let (bytes, triple) = emit(air, spec, opts, FileType::Assembly)?;
    Ok((String::from_utf8_lossy(&bytes).into_owned(), triple))
}

fn emit(
    air: &Module,
    spec: &TargetSpec,
    opts: &LowerOptions,
    file_type: FileType,
) -> Result<(Vec<u8>, String)> {
    let tm = target_machine(spec, false)?;
    let ctx = Context::create();
    let opts = LowerOptions {
        aot: true,
        fuel: false,
        sandbox: None,
        target_cpu: spec.cpu.clone(),
        target_features: spec.features.clone(),
        ..opts.clone()
    };
    let module = lower_module(&ctx, air, &opts)?;
    optimize(&module, &tm)?;
    let buf = tm
        .write_to_memory_buffer(&module, file_type)
        .map_err(|e| anyhow::anyhow!("LLVM object emission failed: {e}"))?;
    let triple = tm.get_triple().as_str().to_string_lossy().into_owned();
    Ok((buf.as_slice().to_vec(), triple))
}
