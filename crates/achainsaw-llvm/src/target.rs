use anyhow::{anyhow, Result};
use inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::OptimizationLevel;
use std::sync::Once;

/// What to generate code for. `triple: None` means the host.
#[derive(Debug, Clone, Default)]
pub struct TargetSpec {
    pub triple: Option<String>,
    /// LLVM CPU name (`znver4`, `x86-64-v3`, `neoverse-v2`, ...).
    pub cpu: String,
    /// LLVM feature string (`+avx2,-avx512f,...`).
    pub features: String,
}

/// LLVM's name for the host CPU, e.g. `znver4`.
pub fn host_cpu_name() -> String {
    init_targets();
    TargetMachine::get_host_cpu_name().to_string()
}

pub(crate) fn init_targets() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let cfg = InitializationConfig::default();
        Target::initialize_native(&cfg).expect("LLVM native target");
        Target::initialize_x86(&cfg);
        Target::initialize_aarch64(&cfg);
    });
}

pub(crate) fn target_machine(spec: &TargetSpec, jit: bool) -> Result<TargetMachine> {
    init_targets();
    let triple = match &spec.triple {
        Some(t) => TargetTriple::create(t),
        None => TargetMachine::get_default_triple(),
    };
    let target = Target::from_triple(&triple).map_err(|e| {
        anyhow!(
            "[ERR_UNSUPPORTED_TARGET] No LLVM backend for '{}': {e}",
            triple.as_str().to_string_lossy()
        )
    })?;
    let (reloc, model) = if jit {
        (RelocMode::Default, CodeModel::JITDefault)
    } else {
        (RelocMode::PIC, CodeModel::Default)
    };
    target
        .create_target_machine(
            &triple,
            &spec.cpu,
            &spec.features,
            OptimizationLevel::Default,
            reloc,
            model,
        )
        .ok_or_else(|| anyhow!("[ERR_INVALID_TARGET] LLVM rejected target {spec:?}"))
}

/// Applies the standard optimization pipeline for `tm` to `module`.
pub(crate) fn optimize(module: &inkwell::module::Module, tm: &TargetMachine) -> Result<()> {
    module.set_triple(&tm.get_triple());
    module.set_data_layout(&tm.get_target_data().get_data_layout());
    module
        .run_passes(
            "default<O2>",
            tm,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .map_err(|e| anyhow!("LLVM optimization failed: {e}"))
}
