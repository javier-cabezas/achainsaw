use achainsaw_codegen::{link_shared_library, AotCompiler, JitEngine};
use achainsaw_ir::diag::Diagnostic;
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

    /// Compile and execute an IR function via Cranelift JIT (.air or .airb)
    Run {
        /// Path to .air or .airb file
        path: PathBuf,
        /// Name of the function to execute
        #[arg(long, default_value = "main")]
        func: String,
        /// Integer arguments passed to the function
        #[arg(short, long, value_delimiter = ' ')]
        args: Vec<i64>,
        /// Loop fuel instruction budget (prevents infinite loops)
        #[arg(long)]
        fuel: Option<u64>,
        /// Maximum heap allocation quota in megabytes
        #[arg(long)]
        max_memory_mb: Option<usize>,
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
        /// Link into shared library (.so / .dll) using host C compiler
        #[arg(long, default_value_t = false)]
        shared: bool,
        /// Emit structured machine-readable JSON telemetry
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Run Model Context Protocol (MCP) server over standard I/O (stdio)
    Mcp,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Mcp => {
            if let Err(e) = mcp::run_mcp_server() {
                eprintln!("MCP server error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Check { path, json } => {
            let res = run_check(&path);
            if json {
                match res {
                    Ok(stats) => println!("{}", serde_json::to_string_pretty(&stats).unwrap()),
                    Err(diag) => println!("{}", diag.to_json()),
                }
            } else {
                match res {
                    Ok(stats) => println!("Validation OK: {}", stats["function_count"]),
                    Err(diag) => eprintln!("Error [{}]: {}", diag.error_code, diag.message),
                }
            }
        }
        Commands::Run {
            path,
            func,
            args,
            json,
            fuel,
            max_memory_mb,
        } => match run_exec(&path, &func, &args, fuel, max_memory_mb) {
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
            }
        },
        Commands::Bench { path, iters } => match run_bench(&path, iters) {
            Ok(bench_res) => println!("{}", serde_json::to_string_pretty(&bench_res).unwrap()),
            Err(e) => eprintln!("Benchmark error: {e}"),
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
                        stats["input"], stats["source_bytes"], stats["output"], stats["binary_bytes"]
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
            }
        },
        Commands::Disassemble { input, output } => match run_disassemble(&input, output) {
            Ok(_) => {}
            Err(e) => eprintln!("Disassembly error: {e}"),
        },
        Commands::Opt { input, output, json } => match run_optimize(&input, output) {
            Ok(stats) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                } else if stats.get("output_written").and_then(|v| v.as_bool()).unwrap_or(false) {
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
            }
        },
        Commands::Build {
            input,
            output,
            target,
            shared,
            json,
        } => match run_build(&input, output, target.as_deref(), shared) {
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
                    let err_json = json!({
                        "status": "error",
                        "error_code": "ERR_BUILD",
                        "message": e.to_string(),
                    });
                    println!("{}", serde_json::to_string_pretty(&err_json).unwrap());
                } else {
                    eprintln!("Build error: {e}");
                }
            }
        },
    }
}

fn load_module(path: &Path) -> Result<Module, Diagnostic> {
    let bytes = fs::read(path).map_err(|e| {
        Diagnostic::error("ERR_FILE_IO", format!("Could not read file: {e}"), Default::default())
    })?;

    if bytes.starts_with(b"\x00AIR") {
        let module = decode_module(&bytes)?;
        let mut validator = Validator::new();
        validator.validate_module(&module)?;
        Ok(module)
    } else {
        let source = std::str::from_utf8(&bytes).map_err(|e| {
            Diagnostic::error("ERR_INVALID_UTF8", format!("Source not valid UTF-8: {e}"), Default::default())
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
    args: &[i64],
    fuel: Option<u64>,
    max_memory_mb: Option<usize>,
) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    let module = load_module(path)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let parse_time_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let mut engine = JitEngine::new()?;
    if let Some(f) = fuel {
        engine.set_fuel(Some(f));
    }
    if let Some(mb) = max_memory_mb {
        engine.set_memory_quota(mb * 1024 * 1024);
    }
    engine.compile_module(&module)?;
    let compile_time_us = t1.elapsed().as_micros();

    let t2 = Instant::now();
    let res = unsafe {
        match args.len() {
            0 => {
                let func_ptr = engine
                    .get_fn_ptr(func_name)
                    .ok_or_else(|| anyhow!("Function '{func_name}' not found"))?;
                achainsaw_codegen::reset_execution_status();
                let f: extern "C" fn() -> i32 = std::mem::transmute(func_ptr);
                let val = f() as i64;
                achainsaw_codegen::check_execution_status()?;
                val
            }
            1 => engine.run_i32_to_i32(func_name, args[0] as i32)? as i64,
            2 => engine.run_i32_2_to_i32(func_name, args[0] as i32, args[1] as i32)? as i64,
            _ => return Err(anyhow!("CLI runner supports up to 2 direct arguments")),
        }
    };
    let exec_time_us = t2.elapsed().as_micros();

    Ok(json!({
        "status": "ok",
        "function": func_name,
        "result": res,
        "parse_time_us": parse_time_us,
        "compile_time_us": compile_time_us,
        "exec_time_us": exec_time_us,
        "total_time_ms": (t0.elapsed().as_micros() as f64) / 1000.0,
    }))
}

fn run_bench(path: &Path, iters: u32) -> Result<serde_json::Value> {
    let module = load_module(path)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;

    let t0 = Instant::now();
    for _ in 0..iters {
        let mut engine = JitEngine::new()?;
        engine.compile_module(&module)?;
    }
    let total_duration = t0.elapsed();
    let avg_compile_time_ms = total_duration.as_secs_f64() * 1000.0 / (iters as f64);
    let throughput_compiles_per_sec = (iters as f64) / total_duration.as_secs_f64();

    Ok(json!({
        "status": "ok",
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

    let binary = encode_module(&module);

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
    if let Some(out_path) = output {
        fs::write(out_path, text)?;
    } else {
        print!("{text}");
    }
    Ok(())
}

fn run_optimize(input: &Path, output: Option<PathBuf>) -> Result<serde_json::Value> {
    let mut module = load_module(input)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;

    let stats = achainsaw_ir::opt::optimize_module(&mut module);
    let optimized_code = to_air_text(&module);

    let output_written = if let Some(out_path) = &output {
        fs::write(out_path, &optimized_code)?;
        true
    } else {
        false
    };

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

fn run_build(
    input: &Path,
    output: Option<PathBuf>,
    target: Option<&str>,
    shared: bool,
) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    let module = load_module(input)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let parse_time_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let mut compiler = AotCompiler::new(target)?;
    compiler.compile_module(&module)?;
    let bytes = compiler.finish()?;
    let compile_time_us = t1.elapsed().as_micros();

    let obj_path = output.clone().unwrap_or_else(|| input.with_extension("o"));
    fs::write(&obj_path, &bytes)?;

    let mut final_path = obj_path.clone();
    let mut shared_linked = false;
    let mut link_time_us = 0;

    if shared {
        let t2 = Instant::now();
        let target_shared = if cfg!(target_os = "windows") {
            input.with_extension("dll")
        } else if cfg!(target_os = "macos") {
            input.with_extension("dylib")
        } else {
            input.with_extension("so")
        };
        link_shared_library(&obj_path, &target_shared)?;
        link_time_us = t2.elapsed().as_micros();
        final_path = target_shared;
        shared_linked = true;
    }

    Ok(json!({
        "status": "ok",
        "input": input.to_string_lossy(),
        "output": final_path.to_string_lossy(),
        "object_bytes": bytes.len(),
        "shared": shared_linked,
        "parse_time_us": parse_time_us,
        "compile_time_us": compile_time_us,
        "link_time_us": link_time_us,
        "total_time_ms": (t0.elapsed().as_micros() as f64) / 1000.0,
    }))
}
