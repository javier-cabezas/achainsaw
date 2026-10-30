use anyhow::{anyhow, Result};
use inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::{AnyTypeEnum, BasicTypeEnum};
use inkwell::values::{
    AnyValueEnum, BasicValue, BasicValueEnum, InstructionOpcode, InstructionValue,
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
    split_scalable_load_bitcasts(module)?;
    module
        .run_passes(
            "default<O2>",
            tm,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .map_err(|e| anyhow!("LLVM optimization failed: {e}"))
}

/// Replaces a scalable vector load whose users are all bitcasts by one load of each type it
/// is bitcast to, at the same place. LLVM 22's VectorCombine (`scalarizeLoadBitcast`)
/// aborts on such a load, since it sizes the vector as a fixed width (`Cannot implicitly
/// convert a scalable size`). AIR vectors load as i32 lanes and are bitcast to each op's
/// lane type; InstCombine folds a single bitcast into the load, so the crash took a vector
/// loaded once and used at two other lane types (`vadd a, b:i8` then `vadd x, b:i16`).
/// The new loads have users of their own type, and so does the one load GVN may merge them
/// back into. Loads with any other user are left alone, so code that compiled before is
/// unchanged.
fn split_scalable_load_bitcasts(module: &inkwell::module::Module) -> Result<()> {
    let builder = module.get_context().create_builder();
    for f in module.get_functions() {
        for bb in f.get_basic_blocks() {
            let mut next = bb.get_first_instruction();
            while let Some(load) = next {
                next = load.get_next_instruction();
                if load.get_opcode() != InstructionOpcode::Load
                    || !matches!(load.get_type(), AnyTypeEnum::ScalableVectorType(_))
                {
                    continue;
                }
                let mut bitcasts: Vec<InstructionValue> = Vec::new();
                let mut only_bitcasts = true;
                let mut u = load.get_first_use();
                while let Some(use_) = u {
                    u = use_.get_next_use();
                    let user = match use_.get_user() {
                        AnyValueEnum::InstructionValue(i) => Some(i),
                        other => BasicValueEnum::try_from(other)
                            .ok()
                            .and_then(|v| v.as_instruction_value()),
                    };
                    match user.filter(|u| u.get_opcode() == InstructionOpcode::BitCast) {
                        Some(cast) => bitcasts.push(cast),
                        None => only_bitcasts = false,
                    }
                }
                if !only_bitcasts || bitcasts.is_empty() {
                    continue;
                }
                let ptr = load
                    .get_operand(0)
                    .and_then(|o| o.value())
                    .ok_or_else(|| anyhow!("vector load without a pointer operand"))?
                    .into_pointer_value();
                let align = load.get_alignment().map_err(|e| anyhow!("{e}"))?;
                // The new loads go right before the original, which the casts leave unused.
                builder.position_before(&load);
                let mut loads: Vec<(AnyTypeEnum, InstructionValue)> = Vec::new();
                for cast in bitcasts {
                    let ty = cast.get_type();
                    let typed = match loads.iter().find(|(t, _)| *t == ty) {
                        Some(&(_, l)) => l,
                        None => {
                            let elem = BasicTypeEnum::try_from(ty)
                                .map_err(|_| anyhow!("bitcast to a non-first-class type"))?;
                            let l = builder
                                .build_load(elem, ptr, "")?
                                .as_instruction_value()
                                .expect("a load is an instruction");
                            l.set_alignment(align).map_err(|e| anyhow!("{e}"))?;
                            loads.push((ty, l));
                            l
                        }
                    };
                    cast.replace_all_uses_with(&typed);
                    cast.erase_from_basic_block();
                }
                next = load.get_next_instruction();
                load.erase_from_basic_block();
            }
        }
    }
    Ok(())
}
