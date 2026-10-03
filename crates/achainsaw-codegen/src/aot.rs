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
use std::str::FromStr;
use target_lexicon::Triple;

use crate::lower::{lower_function, to_clif_type, LowerConfig};

pub struct AotCompiler {
    builder_context: FunctionBuilderContext,
    ctx: cranelift_codegen::Context,
    module: ObjectModule,
    rt_malloc_id: FuncId,
    rt_free_id: FuncId,
}

impl AotCompiler {
    pub fn new(target_triple: Option<&str>) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("is_pic", "true")?;
        flag_builder.set("opt_level", "speed")?;

        let isa = if let Some(triple_str) = target_triple {
            let triple =
                Triple::from_str(triple_str).map_err(|e| anyhow!("Invalid target triple: {e}"))?;
            cranelift_codegen::isa::lookup(triple)?.finish(settings::Flags::new(flag_builder))?
        } else {
            let isa_builder = cranelift_native::builder()
                .map_err(|msg| anyhow!("Host machine not supported by Cranelift: {msg}"))?;
            isa_builder.finish(settings::Flags::new(flag_builder))?
        };

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
        })
    }

    pub fn compile_module(&mut self, ir_mod: &Module) -> Result<()> {
        let mut func_ids = HashMap::new();
        let mut func_returns = HashMap::new();

        // 1. Declare external functions (Linkage::Import)
        for ext_fn in &ir_mod.extern_functions {
            let mut sig = self.module.make_signature();
            for (_, p_ty) in &ext_fn.params {
                sig.params.push(AbiParam::new(to_clif_type(*p_ty)));
            }
            if let Some(r_ty) = ext_fn.ret_type {
                sig.returns.push(AbiParam::new(to_clif_type(r_ty)));
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
                sig.params.push(AbiParam::new(to_clif_type(*p_ty)));
            }
            if let Some(r_ty) = func.ret_type {
                sig.returns.push(AbiParam::new(to_clif_type(r_ty)));
            }
            func_returns.insert(func.name.clone(), func.ret_type);

            let func_id = self
                .module
                .declare_function(&func.name, Linkage::Export, &sig)?;
            func_ids.insert(func.name.clone(), func_id);
        }

        // 3. Lower and define each function
        let config = LowerConfig {
            fuel_check_func_id: None,
            rt_malloc_id: self.rt_malloc_id,
            rt_free_id: self.rt_free_id,
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
}
