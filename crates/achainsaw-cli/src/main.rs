use achainsaw_codegen::JitEngine;
use achainsaw_ir::diag::Diagnostic;
use achainsaw_ir::parse_and_validate;
use anyhow::{anyhow, Result};
use clap::{Parser as ClapParser, Subcommand};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(ClapParser)]
#[command(name = "achainsaw")]
#[command(about = "High-performance agent-native toolchain & JIT compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate IR module syntax and SSA invariants
    Check {
        /// Path to .air file
        path: PathBuf,
        /// Emit structured machine-readable JSON output
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Compile and execute an IR function via Cranelift JIT
    Run {
        /// Path to .air file
        path: PathBuf,
        /// Name of the function to execute
        #[arg(long, default_value = "main")]
        func: String,
        /// Integer arguments passed to the function
        #[arg(short, long, value_delimiter = ' ')]
        args: Vec<i64>,
        /// Emit structured machine-readable JSON output
        #[arg(long, default_value_t = true)]
        json: bool,
    },

    /// Benchmark compilation and execution latency
    Bench {
        /// Path to .air file
        path: PathBuf,
        /// Number of compilation iterations
        #[arg(short, long, default_value_t = 100)]
        iters: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
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
        } => match run_exec(&path, &func, &args) {
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
    }
}

fn run_check(path: &PathBuf) -> Result<serde_json::Value, Diagnostic> {
    let source = fs::read_to_string(path).map_err(|e| {
        Diagnostic::error("ERR_FILE_IO", format!("Could not read file: {e}"), Default::default())
    })?;

    let module = parse_and_validate(&source)?;
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
        "block_count": total_blocks,
        "instruction_count": total_instructions,
    }))
}

fn run_exec(path: &PathBuf, func_name: &str, args: &[i64]) -> Result<serde_json::Value> {
    let source = fs::read_to_string(path)?;

    let t0 = Instant::now();
    let module = parse_and_validate(&source)
        .map_err(|d| anyhow!("Validation failed: [{}] {}", d.error_code, d.message))?;
    let parse_time_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let mut engine = JitEngine::new()?;
    engine.compile_module(&module)?;
    let compile_time_us = t1.elapsed().as_micros();

    let t2 = Instant::now();
    let res = unsafe {
        match args.len() {
            0 => {
                let func_ptr = engine
                    .get_fn_ptr(func_name)
                    .ok_or_else(|| anyhow!("Function '{func_name}' not found"))?;
                let f: extern "C" fn() -> i32 = std::mem::transmute(func_ptr);
                f() as i64
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

fn run_bench(path: &PathBuf, iters: u32) -> Result<serde_json::Value> {
    let source = fs::read_to_string(path)?;
    let module = parse_and_validate(&source)
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
