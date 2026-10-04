use crate::backend::Backend;
use crate::cpu;
use crate::lower::{lower_function, push_abi_params, LowerConfig};
use achainsaw_ir::ast::Module;
use anyhow::{anyhow, Result};
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module as ClifModule};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

pub struct AotCompiler {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: ObjectModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
    ignored_features: Vec<&'static str>,
}

/// Code generation target for AOT builds. All fields default to the (capped) host.
#[derive(Debug, Clone, Default)]
pub struct AotTarget {
    /// Target triple, e.g. `aarch64-unknown-linux-gnu`.
    pub triple: Option<String>,
    /// CPU preset using LLVM names, e.g. `znver4` or `x86-64-v3`.
    pub cpu: Option<String>,
    /// LLVM-style feature overrides, e.g. `+avx2,-avx512f`.
    pub features: Option<String>,
}

impl AotCompiler {
    pub fn new(target_triple: Option<&str>) -> Result<Self> {
        Self::with_target(&AotTarget {
            triple: target_triple.map(str::to_string),
            ..Default::default()
        })
    }

    pub fn with_target(target: &AotTarget) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("is_pic", "true")?;
        flag_builder.set("opt_level", "speed")?;
        // v256/v512 values span 2/4 vector registers, more than ABIs return in registers;
        // spill extra return values through an implicit struct-return pointer.
        flag_builder.set("enable_multi_ret_implicit_sret", "true")?;

        let (isa_builder, ignored_features) = cpu::target_isa_builder(
            target.triple.as_deref(),
            target.cpu.as_deref(),
            target.features.as_deref(),
        )?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        let builder = ObjectBuilder::new(isa, "achainsaw_aot", default_libcall_names())?;
        let mut module = ObjectModule::new(builder);

        let mut alloc_sig = module.make_signature();
        alloc_sig.params.push(AbiParam::new(types::I64));
        alloc_sig.returns.push(AbiParam::new(types::I64));
        let rt_malloc_id = module.declare_function("malloc", Linkage::Import, &alloc_sig)?;

        let mut free_sig = module.make_signature();
        free_sig.params.push(AbiParam::new(types::I64));
        let rt_free_id = module.declare_function("free", Linkage::Import, &free_sig)?;

        let ctx = module.make_context();

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            module,
            rt_malloc_id,
            rt_free_id,
            ignored_features,
        })
    }

    /// Requested target features that Cranelift cannot use (e.g. `avx512bw`, `sve`).
    pub fn ignored_features(&self) -> &[&'static str] {
        &self.ignored_features
    }

    /// Triple of the ISA being compiled for.
    pub fn triple(&self) -> String {
        self.module.isa().triple().to_string()
    }

    pub fn compile_module(&mut self, ir_mod: &Module) -> Result<()> {
        let mut func_ids = HashMap::new();
        let mut func_returns = HashMap::new();

        // 1. Declare external functions (Linkage::Import)
        for ext_fn in &ir_mod.extern_functions {
            let mut sig = self.module.make_signature();
            for (_, p_ty) in &ext_fn.params {
                push_abi_params(&mut sig.params, *p_ty);
            }
            if let Some(r_ty) = ext_fn.ret_type {
                push_abi_params(&mut sig.returns, r_ty);
            }
            func_returns.insert(ext_fn.name.clone(), ext_fn.ret_type);

            let func_id = self
                .module
                .declare_function(&ext_fn.name, Linkage::Import, &sig)?;
            func_ids.insert(ext_fn.name.clone(), func_id);
        }

        // 2. Declare internal functions (Linkage::Export)
        for func in &ir_mod.functions {
            let mut sig = self.module.make_signature();
            for (_, p_ty) in &func.params {
                push_abi_params(&mut sig.params, *p_ty);
            }
            if let Some(r_ty) = func.ret_type {
                push_abi_params(&mut sig.returns, r_ty);
            }
            func_returns.insert(func.name.clone(), func.ret_type);

            let func_id = self
                .module
                .declare_function(&func.name, Linkage::Export, &sig)?;
            func_ids.insert(func.name.clone(), func_id);
        }

        // 3. Lower and define each function
        let config = LowerConfig {
            fuel_check: None,
            fuel_consume_func_id: None,
            rt_malloc_id: self.rt_malloc_id,
            rt_free_id: self.rt_free_id,
            sandbox: None,
        };

        for func in &ir_mod.functions {
            let func_id = func_ids[&func.name];
            lower_function(
                &mut self.module,
                &mut self.ctx,
                &mut self.builder_context,
                func,
                &func_ids,
                &func_returns,
                &config,
            )?;
            self.module.define_function(func_id, &mut self.ctx)?;
            self.module.clear_context(&mut self.ctx);
        }

        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        let product = self.module.finish();
        let bytes = product.emit()?;
        Ok(bytes)
    }

    pub fn emit_to_file(self, path: &Path) -> Result<usize> {
        let bytes = self.finish()?;
        std::fs::write(path, &bytes)?;
        Ok(bytes.len())
    }
}

/// An AOT-compiled object file.
#[derive(Debug, Clone)]
pub struct AotObject {
    pub bytes: Vec<u8>,
    /// Triple the object was compiled for.
    pub triple: String,
    /// Requested features the backend could not use (always empty for LLVM).
    pub ignored_features: Vec<&'static str>,
}

/// Target assembly for `ir_mod` (LLVM backend only), and its triple.
pub fn compile_assembly(
    ir_mod: &Module,
    target: &AotTarget,
    backend: Backend,
) -> Result<(String, String)> {
    match backend {
        #[cfg(feature = "llvm")]
        Backend::Llvm => {
            let (spec, opts) = llvm_aot_options(target)?;
            achainsaw_llvm::compile_assembly(ir_mod, &spec, &opts)
        }
        _ => {
            let _ = (ir_mod, target);
            Err(anyhow!(
                "[ERR_UNSUPPORTED_EMIT] Assembly output needs the llvm backend (--backend llvm)"
            ))
        }
    }
}

#[cfg(feature = "llvm")]
fn llvm_aot_options(
    target: &AotTarget,
) -> Result<(achainsaw_llvm::TargetSpec, achainsaw_llvm::LowerOptions)> {
    let (spec, features) = cpu::llvm_aot_target(target)?;
    let (vx, vector_width) = features.llvm_vector_shape(false);
    let opts = achainsaw_llvm::LowerOptions {
        vx,
        vector_width,
        matrix: features.llvm_matrix_units(),
        ..Default::default()
    };
    Ok((spec, opts))
}

/// Compiles `ir_mod` to a native object for `target` with `backend`.
pub fn compile_object(ir_mod: &Module, target: &AotTarget, backend: Backend) -> Result<AotObject> {
    match backend {
        Backend::Cranelift => {
            let mut compiler = AotCompiler::with_target(target)?;
            let triple = compiler.triple();
            let ignored_features = compiler.ignored_features().to_vec();
            compiler.compile_module(ir_mod)?;
            Ok(AotObject {
                bytes: compiler.finish()?,
                triple,
                ignored_features,
            })
        }
        #[cfg(feature = "llvm")]
        Backend::Llvm => {
            let (spec, opts) = llvm_aot_options(target)?;
            let (bytes, triple) = achainsaw_llvm::compile_object(ir_mod, &spec, &opts)?;
            Ok(AotObject {
                bytes,
                triple,
                ignored_features: Vec::new(),
            })
        }
        #[cfg(not(feature = "llvm"))]
        Backend::Llvm => Err(anyhow!(
            "[ERR_BACKEND_UNAVAILABLE] The llvm backend is not in this build"
        )),
    }
}

pub fn link_shared_library(object_path: &Path, shared_path: &Path) -> Result<()> {
    let shared_arg = if cfg!(target_os = "macos") {
        "-dynamiclib"
    } else {
        "-shared"
    };

    // Try gcc first
    let gcc_res = Command::new("gcc")
        .arg(shared_arg)
        .arg("-o")
        .arg(shared_path)
        .arg(object_path)
        .output();

    if let Ok(out) = gcc_res {
        if out.status.success() {
            return Ok(());
        }
    }

    // Try clang as fallback
    let clang_res = Command::new("clang")
        .arg(shared_arg)
        .arg("-o")
        .arg(shared_path)
        .arg(object_path)
        .output();

    if let Ok(out) = clang_res {
        if out.status.success() {
            return Ok(());
        }
    }

    // Try cc as fallback
    let cc_res = Command::new("cc")
        .arg(shared_arg)
        .arg("-o")
        .arg(shared_path)
        .arg(object_path)
        .output();

    if let Ok(out) = cc_res {
        if out.status.success() {
            return Ok(());
        }
    }

    // Try link.exe on Windows MSVC
    if cfg!(target_os = "windows") {
        if let Ok(out) = Command::new("link.exe")
            .arg("/DLL")
            .arg(format!("/OUT:{}", shared_path.display()))
            .arg(object_path)
            .output()
        {
            if out.status.success() {
                return Ok(());
            }
        }
    }

    Err(anyhow!(
        "Failed to link shared library: neither gcc, clang, cc, nor link.exe succeeded"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use achainsaw_ir::parse_and_validate;
    use object::{Object, ObjectSymbol};

    #[test]
    fn test_aot_compile_native_object() {
        let code = r#"
        fn add(a:i32, b:i32)->i32
          b0:
            res = add a, b
            ret res
        "#;
        let module = parse_and_validate(code).unwrap();
        let mut compiler = AotCompiler::new(None).unwrap();
        compiler.compile_module(&module).unwrap();

        let object_bytes = compiler.finish().unwrap();
        assert!(!object_bytes.is_empty());

        // Parse with object crate and inspect exported symbols
        let obj = object::File::parse(&*object_bytes).expect("Valid native object format");
        let symbols: Vec<String> = obj
            .symbols()
            .filter_map(|s| s.name().ok().map(|n| n.to_string()))
            .collect();

        assert!(
            symbols.iter().any(|s| s.contains("add")),
            "Expected 'add' symbol in object file, found: {:?}",
            symbols
        );
    }

    const SIMD_CODE: &str = r#"
    fn scale(x:f32, y:f32)->f32
      b0:
        vx = splat x
        vy = splat y
        vp = vfmul vx, vy
        vs = vfadd vp, vx
        s = vfsum vs:f32
        ret s
    "#;

    fn text_section(target: &AotTarget) -> (Vec<u8>, Vec<&'static str>) {
        use object::ObjectSection;
        let module = parse_and_validate(SIMD_CODE).unwrap();
        let mut compiler = AotCompiler::with_target(target).unwrap();
        let ignored = compiler.ignored_features().to_vec();
        compiler.compile_module(&module).unwrap();
        let bytes = compiler.finish().unwrap();
        let obj = object::File::parse(&*bytes).unwrap();
        let text = obj
            .section_by_name(".text")
            .unwrap()
            .data()
            .unwrap()
            .to_vec();
        (text, ignored)
    }

    fn x86_target(cpu: &str, features: Option<&str>) -> AotTarget {
        AotTarget {
            triple: Some("x86_64-unknown-linux-gnu".into()),
            cpu: Some(cpu.into()),
            features: features.map(str::to_string),
        }
    }

    #[test]
    fn test_aot_target_cpu_changes_codegen() {
        let (v2, _) = text_section(&x86_target("x86-64-v2", None));
        let (v3, _) = text_section(&x86_target("x86-64-v3", None));
        assert_ne!(v2, v3, "AVX2 target should emit VEX-encoded vector code");
        // `-avx` also drops AVX2/FMA (they require it), leaving the same SSE code as v2.
        let (v3_no_avx, _) = text_section(&x86_target("x86-64-v3", Some("-avx")));
        assert_eq!(v2, v3_no_avx);
        // Named presets and generic levels agree once prerequisites are completed.
        let (haswell, _) = text_section(&x86_target("haswell", None));
        assert_eq!(v3, haswell);
    }

    #[test]
    fn test_aot_reports_features_cranelift_cannot_use() {
        let (_, ignored) = text_section(&x86_target(
            "x86-64-v4",
            Some("+avx512bw,+amx-tile,-avx512f"),
        ));
        assert_eq!(ignored, vec!["avx512bw", "amx-tile"]);

        let arm = AotTarget {
            triple: Some("aarch64-unknown-linux-gnu".into()),
            cpu: Some("generic".into()),
            features: Some("+sve2,+dotprod".into()),
        };
        let (text, ignored) = text_section(&arm);
        assert!(!text.is_empty());
        assert_eq!(ignored, vec!["sve2"]);
    }

    #[test]
    fn test_aot_target_errors() {
        let err = |t: AotTarget| AotCompiler::with_target(&t).err().unwrap().to_string();
        assert!(err(x86_target("pentium9", None)).contains("ERR_UNKNOWN_TARGET_CPU"));
        assert!(err(x86_target("x86-64-v3", Some("+sve"))).contains("ERR_ISA_ARCH_MISMATCH"));
        assert!(
            err(x86_target("x86-64-v3", Some("+avx1024"))).contains("ERR_UNKNOWN_TARGET_FEATURE")
        );
        let bad_triple = AotTarget {
            triple: Some("not-a-triple-at-all".into()),
            ..Default::default()
        };
        assert!(err(bad_triple).contains("ERR_INVALID_TARGET"));
    }
}
