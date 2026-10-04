use achainsaw_codegen::cpu::{self, IsaLevel};
use achainsaw_codegen::{
    compile_assembly, compile_object, link_shared_library, AotObject, AotTarget, Backend,
    JitEngine, RtValue,
};
use achainsaw_ir::diag::Diagnostic;
use achainsaw_ir::types::Type;
use achainsaw_ir::{
    decode_module, encode_module, parse_and_validate, to_air_text, Module, Validator,
};
use anyhow::{anyhow, Result};
use clap::{Parser as ClapParser, Subcommand};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod mcp;

#[derive(ClapParser)]
#[command(name = "achainsaw")]
#[command(about = "High-performance agent-native toolchain & JIT compiler", long_about = None)]
struct Cli {
    /// Cap the vector ISA used for code generation (sse, avx, avx2, avx512, amx, neon, sve,
    /// sve2, sme). Overrides ACHAINSAW_MAX_ISA.
    #[arg(long, global = true, value_name = "LEVEL", value_parser = clap::value_parser!(IsaLevel))]
    isa: Option<IsaLevel>,
    /// Code generator for run, bench and build: cranelift, llvm (needs an LLVM-enabled
    /// build), or auto (default: ACHAINSAW_BACKEND if set, else llvm for modules using
    /// v256/v512/vx/mm when built in, else cranelift).
    #[arg(long, global = true, value_name = "BACKEND")]
    backend: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate IR module syntax and SSA invariants (.air or .airb)
    Check {
        /// Path to .air or .airb file
        path: PathBuf,
        /// Emit structured machine-readable JSON output
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Compile and execute an IR function via the JIT (.air or .airb)
    Run {
        /// Path to .air or .airb file
        path: PathBuf,
        /// Name of the function to execute
        #[arg(long, default_value = "main")]
        func: String,
        /// Arguments passed to the function (comma or space separated)
        #[arg(short, long, value_delimiter = ',')]
        args: Vec<String>,
        /// Loop fuel instruction budget (prevents infinite loops)
        #[arg(long)]
        fuel: Option<u64>,
        /// Maximum heap allocation quota in megabytes
        #[arg(long)]
        max_memory_mb: Option<usize>,
        /// Most threads `par` loops may use (default: all cores; 1 runs them serially)
        #[arg(long)]
        threads: Option<usize>,
        /// Emit structured machine-readable JSON output
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Benchmark compilation and execution latency
    Bench {
        /// Path to .air or .airb file
        path: PathBuf,
        /// Number of compilation iterations
        #[arg(short, long, default_value_t = 100)]
        iters: u32,
    },

    /// Assemble text AIR (.air) into compact binary bytecode (.airb)
    Assemble {
        /// Path to input .air file
        input: PathBuf,
        /// Path to output .airb file (defaults to replacing .air with .airb)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Emit structured JSON telemetry
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Disassemble binary bytecode (.airb) into text AIR format (.air)
    Disassemble {
        /// Path to input .airb file
        input: PathBuf,
        /// Path to output .air file (defaults to printing to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Optimize IR module (constant folding, algebraic simplification, DCE)
    Opt {
        /// Path to input .air or .airb file
        input: PathBuf,
        /// Path to output file (defaults to printing to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Emit structured machine-readable JSON telemetry
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Ahead-Of-Time (AOT) compile AIR module into native object (.o) or shared library (.so / .dll)
    Build {
        /// Path to input .air or .airb file
        input: PathBuf,
        /// Path to output object file (.o) or shared library
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Target triple (defaults to host architecture)
        #[arg(long)]
        target: Option<String>,
        /// Target CPU using LLVM names, e.g. x86-64-v3, znver4, sapphirerapids (defaults to host)
        #[arg(long)]
        target_cpu: Option<String>,
        /// LLVM-style target feature overrides, e.g. +avx2,-avx512f
        #[arg(long, allow_hyphen_values = true)]
        target_features: Option<String>,
        /// Link into shared library (.so / .dll) using host C compiler
        #[arg(long, default_value_t = false)]
        shared: bool,
        /// Output kind: obj (native object, default) or asm (target assembly text, written
        /// to a .s file; needs the llvm backend)
        #[arg(long, default_value = "obj", value_parser = ["obj", "asm"])]
        emit: String,
        /// Emit structured machine-readable JSON telemetry
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Report host CPU vector features, the active ISA cap, and backend vector widths
    Cpu {
        /// Emit structured machine-readable JSON output
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Run Model Context Protocol (MCP) server over standard I/O (stdio)
    Mcp,
}

/// Splits an `[ERR_CODE] message` error into its code and message.
fn split_error_code(e: &anyhow::Error, default_code: &'static str) -> (String, String) {
    let text = e.to_string();
    if let Some(rest) = text.strip_prefix('[') {
        if let Some((code, msg)) = rest.split_once("] ") {
            if code.starts_with("ERR_") {
                return (code.to_string(), msg.to_string());
            }
        }
    }
    (default_code.to_string(), text)
}

fn print_error_json(e: &anyhow::Error, default_code: &'static str) {
    let (code, message) = split_error_code(e, default_code);
    let err_json = json!({
        "status": "error",
        "error_code": code,
        "message": message,
    });
    println!("{}", serde_json::to_string_pretty(&err_json).unwrap());
}

fn main() {
    let cli = Cli::parse();

    if let Some(level) = cli.isa {
        if let Err(e) = cpu::set_isa_cap(Some(level)) {
            print_error_json(&e, "ERR_INVALID_ISA_LEVEL");
            std::process::exit(1);
        }
    }

    let backend = cli.backend.as_deref();

    match cli.command {
        Commands::Cpu { json } => match cpu::target_report() {
            Ok(report) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                } else {
                    let eff = &report["effective"];
                    println!("arch: {}", eff["arch"].as_str().unwrap_or("?"));
                    println!("max ISA: {}", eff["max_isa"].as_str().unwrap_or("none"));
                    println!("ISA cap: {}", report["isa_cap"].as_str().unwrap_or("none"));
                    println!("native vector bits: {}", eff["native_vector_bits"]);
                    println!(
                        "features: {}",
                        eff["features"]
                            .as_array()
                            .map(|a| a
                                .iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(" "))
                            .unwrap_or_default()
                    );
                }
            }
            Err(e) => {
                print_error_json(&e, "ERR_CPU_DETECTION");
                std::process::exit(1);
            }
        },
        Commands::Mcp => {
            if let Err(e) = mcp::run_mcp_server() {
                eprintln!("MCP server error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Check { path, json } => match run_check(&path) {
            Ok(stats) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                } else {
                    println!("Validation OK: {}", stats["function_count"]);
                }
            }
            Err(diag) => {
                if json {
                    println!("{}", diag.to_json());
                } else {
                    eprintln!("Error [{}]: {}", diag.error_code, diag.message);
                }
                std::process::exit(1);
            }
        },
        Commands::Run {
            path,
            func,
            args,
            json,
            fuel,
            max_memory_mb,
            threads,
        } => match run_exec(&path, &func, &args, fuel, max_memory_mb, threads, backend) {
            Ok(output) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&output).unwrap());
                } else {
                    println!("Result: {}", output["result"]);
                }
            }
            Err(e) => {
                if json {
                    let err_json = json!({
                        "status": "error",
                        "error_code": "ERR_EXECUTION",
                        "message": e.to_string(),
                    });
                    println!("{}", serde_json::to_string_pretty(&err_json).unwrap());
                } else {
                    eprintln!("Execution error: {e}");
                }
                std::process::exit(1);
            }
        },
        Commands::Bench { path, iters } => match run_bench(&path, iters, backend) {
            Ok(bench_res) => println!("{}", serde_json::to_string_pretty(&bench_res).unwrap()),
            Err(e) => {
                eprintln!("Benchmark error: {e}");
                std::process::exit(1);
            }
        },
        Commands::Assemble {
            input,
            output,
            json,
        } => match run_assemble(&input, output) {
            Ok(stats) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                } else {
                    println!(
                        "Assembled {} ({} bytes) -> {} ({} bytes)",
                        stats["input"],
                        stats["source_bytes"],
                        stats["output"],
                        stats["binary_bytes"]
                    );
                }
            }
            Err(e) => {
                if json {
                    let err_json = json!({
                        "status": "error",
                        "error_code": "ERR_ASSEMBLE",
                        "message": e.to_string(),
                    });
                    println!("{}", serde_json::to_string_pretty(&err_json).unwrap());
                } else {
                    eprintln!("Assembly error: {e}");
                }
                std::process::exit(1);
            }
        },
        Commands::Disassemble { input, output } => match run_disassemble(&input, output) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Disassembly error: {e}");
                std::process::exit(1);
            }
        },
        Commands::Opt {
            input,
            output,
            json,
        } => match run_optimize(&input, output) {
            Ok(stats) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                } else if stats
                    .get("output_written")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    println!(
                        "Optimized {} -> {} (total transformations: {})",
                        stats["input"], stats["output"], stats["total_optimizations"]
                    );
                } else {
                    print!("{}", stats["code"].as_str().unwrap_or(""));
                }
            }
            Err(e) => {
                if json {
                    let err_json = json!({
                        "status": "error",
                        "error_code": "ERR_OPTIMIZE",
                        "message": e.to_string(),
                    });
                    println!("{}", serde_json::to_string_pretty(&err_json).unwrap());
                } else {
                    eprintln!("Optimization error: {e}");
                }
                std::process::exit(1);
            }
        },
        Commands::Build {
            input,
            output,
            target,
            target_cpu,
            target_features,
            shared,
            emit,
            json,
        } => match if emit == "asm" {
            run_build_asm(
                &input,
                output,
                &AotTarget {
                    triple: target,
                    cpu: target_cpu,
                    features: target_features,
                },
                shared,
                backend,
            )
        } else {
            run_build(
                &input,
                output,
                &AotTarget {
                    triple: target,
                    cpu: target_cpu,
                    features: target_features,
                },
                shared,
                backend,
            )
        } {
            Ok(stats) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                } else {
                    println!(
                        "Built {} -> {} ({} bytes, shared: {})",
                        stats["input"], stats["output"], stats["object_bytes"], stats["shared"]
                    );
                }
            }
            Err(e) => {
                if json {
                    print_error_json(&e, "ERR_BUILD");
                } else {
                    eprintln!("Build error: {e}");
                }
                std::process::exit(1);
            }
        },
    }
}

fn load_module(path: &Path) -> Result<Module, Diagnostic> {
    let bytes = fs::read(path).map_err(|e| {
        Diagnostic::error(
            "ERR_FILE_IO",
            format!("Could not read file: {e}"),
            Default::default(),
        )
    })?;

    if bytes.starts_with(b"\x00AIR") {
        let module = decode_module(&bytes)?;
        let mut validator = Validator::new();
        validator.validate_module(&module)?;
        Ok(module)
    } else {
        let source = std::str::from_utf8(&bytes).map_err(|e| {
            Diagnostic::error(
                "ERR_INVALID_UTF8",
                format!("Source not valid UTF-8: {e}"),
                Default::default(),
            )
        })?;
        parse_and_validate(source)
    }
}

fn run_check(path: &Path) -> Result<serde_json::Value, Diagnostic> {
    let module = load_module(path)?;
    let total_blocks: usize = module.functions.iter().map(|f| f.blocks.len()).sum();
    let total_instructions: usize = module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.instructions.len())
        .sum();

    Ok(json!({
        "status": "ok",
        "functions": module.functions.iter().map(|f| &f.name).collect::<Vec<_>>(),
        "function_count": module.functions.len(),
        "extern_functions": module.extern_functions.iter().map(|f| &f.name).collect::<Vec<_>>(),
        "extern_function_count": module.extern_functions.len(),
        "block_count": total_blocks,
        "instruction_count": total_instructions,
    }))
}

fn run_exec(
    path: &Path,
    func_name: &str,
    args: &[String],
    fuel: Option<u64>,
    max_memory_mb: Option<usize>,
    threads: Option<usize>,
    backend: Option<&str>,
) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    let module = load_module(path)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let parse_time_us = t0.elapsed().as_micros();

    let func = module
        .functions
        .iter()
        .find(|f| f.name == func_name)
        .ok_or_else(|| anyhow!("Function '{func_name}' not found in module"))?;

    if args.len() != func.params.len() {
        return Err(anyhow!(
            "Function '{func_name}' expects {} arguments, received {}",
            func.params.len(),
            args.len()
        ));
    }

    let mut parsed_args = Vec::with_capacity(args.len());
    for (arg_str, (_, param_ty)) in args.iter().zip(&func.params) {
        let trimmed = arg_str.trim();
        let val = match param_ty {
            Type::I8 => RtValue::I8(
                trimmed
                    .parse::<i8>()
                    .map_err(|e| anyhow!("Invalid i8 argument '{trimmed}': {e}"))?,
            ),
            Type::I16 => RtValue::I16(
                trimmed
                    .parse::<i16>()
                    .map_err(|e| anyhow!("Invalid i16 argument '{trimmed}': {e}"))?,
            ),
            Type::I32 => RtValue::I32(
                trimmed
                    .parse::<i32>()
                    .map_err(|e| anyhow!("Invalid i32 argument '{trimmed}': {e}"))?,
            ),
            Type::I64 => RtValue::I64(
                trimmed
                    .parse::<i64>()
                    .map_err(|e| anyhow!("Invalid i64 argument '{trimmed}': {e}"))?,
            ),
            Type::Ptr => RtValue::Ptr(
                trimmed
                    .parse::<usize>()
                    .map_err(|e| anyhow!("Invalid ptr argument '{trimmed}': {e}"))?,
            ),
            Type::F32 => RtValue::F32(
                trimmed
                    .parse::<f32>()
                    .map_err(|e| anyhow!("Invalid f32 argument '{trimmed}': {e}"))?,
            ),
            Type::F64 => RtValue::F64(
                trimmed
                    .parse::<f64>()
                    .map_err(|e| anyhow!("Invalid f64 argument '{trimmed}': {e}"))?,
            ),
            Type::V128 | Type::V256 | Type::V512 | Type::Vx | Type::F16 | Type::BF16 => {
                return Err(anyhow!(
                    "Direct passing of {param_ty} arguments not supported via CLI; pass a pointer instead"
                ))
            }
        };
        parsed_args.push(val);
    }

    let t1 = Instant::now();
    let backend = Backend::resolve_for(backend, &module)?;
    let mut engine = JitEngine::with_backend(backend, &cpu::CpuFeatures::effective()?)?;
    if let Some(f) = fuel {
        engine.set_fuel(Some(f));
    }
    if let Some(mb) = max_memory_mb {
        engine.set_memory_quota(mb * 1024 * 1024);
    }
    engine.set_threads(threads);
    engine.compile_module(&module)?;
    let compile_time_us = t1.elapsed().as_micros();

    let t2 = Instant::now();
    let rt_res = unsafe { engine.call_typed(func_name, &parsed_args)? };
    let exec_time_us = t2.elapsed().as_micros();

    let res_json = match rt_res {
        Some(RtValue::I8(v)) => json!(v),
        Some(RtValue::I16(v)) => json!(v),
        Some(RtValue::I32(v)) => json!(v),
        Some(RtValue::I64(v)) => json!(v),
        Some(RtValue::Ptr(v)) => json!(v),
        Some(RtValue::F32(v)) => json!(v),
        Some(RtValue::F64(v)) => json!(v),
        None => json!(null),
    };

    Ok(json!({
        "status": "ok",
        "function": func_name,
        "result": res_json,
        "backend": backend.as_str(),
        "threads": engine.threads(),
        "parse_time_us": parse_time_us,
        "compile_time_us": compile_time_us,
        "exec_time_us": exec_time_us,
        "total_time_ms": (t0.elapsed().as_micros() as f64) / 1000.0,
    }))
}

fn run_bench(path: &Path, iters: u32, backend: Option<&str>) -> Result<serde_json::Value> {
    let module = load_module(path)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let backend = Backend::resolve_for(backend, &module)?;

    let features = cpu::CpuFeatures::effective()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut engine = JitEngine::with_backend(backend, &features)?;
        engine.compile_module(&module)?;
    }
    let total_duration = t0.elapsed();
    let avg_compile_time_ms = total_duration.as_secs_f64() * 1000.0 / (iters as f64);
    let throughput_compiles_per_sec = (iters as f64) / total_duration.as_secs_f64();

    Ok(json!({
        "status": "ok",
        "backend": backend.as_str(),
        "iterations": iters,
        "total_duration_sec": total_duration.as_secs_f64(),
        "avg_compile_time_ms": avg_compile_time_ms,
        "throughput_compiles_per_sec": throughput_compiles_per_sec,
    }))
}

fn run_assemble(input: &Path, output: Option<PathBuf>) -> Result<serde_json::Value> {
    let source = fs::read_to_string(input)?;
    let module = parse_and_validate(&source)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;

    let binary = encode_module(&module)
        .map_err(|d| anyhow!("AIRB encoding failed: [{}] {}", d.error_code, d.message))?;

    let out_path = output.unwrap_or_else(|| input.with_extension("airb"));
    fs::write(&out_path, &binary)?;

    Ok(json!({
        "status": "ok",
        "input": input.to_string_lossy(),
        "output": out_path.to_string_lossy(),
        "source_bytes": source.len(),
        "binary_bytes": binary.len(),
        "compression_ratio": (binary.len() as f64) / (source.len().max(1) as f64),
    }))
}

fn run_disassemble(input: &Path, output: Option<PathBuf>) -> Result<()> {
    let bytes = fs::read(input)?;
    let module = decode_module(&bytes)
        .map_err(|d| anyhow!("AIRB decode failed: [{}] {}", d.error_code, d.message))?;

    let text = to_air_text(&module);
    if let Some(out) = output {
        fs::write(out, text)?;
    } else {
        print!("{text}");
    }
    Ok(())
}

fn run_optimize(input: &Path, output: Option<PathBuf>) -> Result<serde_json::Value> {
    let mut module = load_module(input)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;

    let stats = achainsaw_ir::optimize_module(&mut module);

    let optimized_code = to_air_text(&module);
    let mut output_written = false;
    if let Some(ref out) = output {
        if out.extension().and_then(|e| e.to_str()) == Some("airb") {
            let binary = encode_module(&module)
                .map_err(|d| anyhow!("AIRB encoding failed: [{}] {}", d.error_code, d.message))?;
            fs::write(out, binary)?;
        } else {
            fs::write(out, &optimized_code)?;
        }
        output_written = true;
    }

    Ok(json!({
        "status": "ok",
        "input": input.to_string_lossy(),
        "output": output.map(|p| p.to_string_lossy().to_string()),
        "output_written": output_written,
        "constants_folded": stats.constants_folded,
        "algebraic_simplifications": stats.algebraic_simplifications,
        "branches_folded": stats.branches_folded,
        "dead_instructions_removed": stats.dead_instructions_removed,
        "dead_blocks_removed": stats.dead_blocks_removed,
        "total_optimizations": stats.total_optimizations(),
        "iterations": stats.iterations,
        "code": optimized_code,
    }))
}

/// `build --emit asm`: writes target assembly (LLVM backend) to `output` or `<input>.s`.
fn run_build_asm(
    input: &Path,
    output: Option<PathBuf>,
    target: &AotTarget,
    shared: bool,
    backend: Option<&str>,
) -> Result<serde_json::Value> {
    if shared {
        return Err(anyhow!(
            "[ERR_UNSUPPORTED_EMIT] --shared cannot be combined with --emit asm"
        ));
    }
    let module = load_module(input)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let backend = Backend::resolve_for(backend, &module)?;
    let (asm, triple) = compile_assembly(&module, target, backend)?;
    let out = output.unwrap_or_else(|| input.with_extension("s"));
    fs::write(&out, &asm)?;
    Ok(json!({
        "status": "ok",
        "input": input.to_string_lossy(),
        "output": out.to_string_lossy(),
        "object_bytes": asm.len(),
        "shared": false,
        "emit": "asm",
        "backend": backend.as_str(),
        "target": triple,
        "target_cpu": target.cpu,
        "target_features": target.features,
    }))
}

fn run_build(
    input: &Path,
    output: Option<PathBuf>,
    target: &AotTarget,
    shared: bool,
    backend: Option<&str>,
) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    let module = load_module(input)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let parse_time_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let backend = Backend::resolve_for(backend, &module)?;
    let AotObject {
        bytes,
        triple,
        ignored_features,
    } = compile_object(&module, target, backend)?;
    let compile_time_us = t1.elapsed().as_micros();

    let sh_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    let (obj_path, target_shared) = if shared {
        match output {
            Some(ref out) => {
                if out.extension().is_some_and(|e| e == "o" || e == "obj") {
                    let sh = input.with_extension(sh_ext);
                    (out.clone(), sh)
                } else if out.extension().is_some_and(|e| e == sh_ext) {
                    let obj = out.with_extension("o");
                    (obj, out.clone())
                } else {
                    let obj = out.with_extension("o");
                    let sh = out.with_extension(sh_ext);
                    (obj, sh)
                }
            }
            None => {
                let obj = input.with_extension("o");
                let sh = input.with_extension(sh_ext);
                (obj, sh)
            }
        }
    } else {
        let obj = output.unwrap_or_else(|| input.with_extension("o"));
        (obj.clone(), obj)
    };

    fs::write(&obj_path, &bytes)?;

    let mut link_time_us = 0;
    if shared {
        let t2 = Instant::now();
        link_shared_library(&obj_path, &target_shared)?;
        link_time_us = t2.elapsed().as_micros();
    }

    Ok(json!({
        "status": "ok",
        "input": input.to_string_lossy(),
        "output": target_shared.to_string_lossy(),
        "object_bytes": bytes.len(),
        "shared": shared,
        "backend": backend.as_str(),
        "target": triple,
        "target_cpu": target.cpu,
        "target_features": target.features,
        "ignored_features": ignored_features,
        "parse_time_us": parse_time_us,
        "compile_time_us": compile_time_us,
        "link_time_us": link_time_us,
        "total_time_ms": (t0.elapsed().as_micros() as f64) / 1000.0,
    }))
}
