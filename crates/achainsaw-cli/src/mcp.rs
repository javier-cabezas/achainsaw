use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::diag::Diagnostic;
use achainsaw_ir::{
    decode_module, encode_module, parse_and_validate, to_air_text, Module, Type, Validator,
};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// MCP protocol revisions this server speaks, newest first.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// AIR primer returned in the `initialize` result. MCP clients such as Claude Code and
/// Claude Desktop inject it into the model context, so agents can write valid AIR
/// without having seen the language before. Keep in sync with the parser and validator.
pub const SERVER_INSTRUCTIONS: &str = "\
achainsaw compiles AIR (Agent Intermediate Representation), a flat SSA IR, to native code via Cranelift.
Workflow: write AIR -> air_check -> fix using the JSON diagnostic (error_code, span, context.available_registers) -> air_run. air_optimize shows simplified IR; air_assemble/air_disassemble convert to and from base64 AIRB bytecode; air_target reports host vector features (backends currently generate 128-bit vector code).

AIR syntax:
- Types: i8 i16 i32 i64 f32 f64 ptr; f16 bf16 are storage-only (ld/st, `f = fext h:f32`, `h = ftrunc f:f16`, not in signatures); vectors v128 v256 v512 vx (scalable, >=128 bits, lane count via `vl`). Vectors are untyped bits; each vector op names its lane type.
- Function: `fn name(a:i32, b:f32)->i32` (omit `->ty` for void), then indented blocks `label:` or `label(x:i64, acc:f32):`.
- First block is the entry: no params, cannot be a branch target; function params are in scope. Use a separate loop-header block.
- One instruction per line. Every register is assigned exactly once (SSA); merge values through block params, not reassignment.
- Each block ends with exactly one terminator: `jmp b(args)` | `br cond, b_then(args), b_else(args)` | `ret v` | `ret`.
- Constants: `x = cst 5:i32`, `f = cst 1.5:f32`. Operands may be inline immediates: `y = add x, 1:i32`.
- Binary (operands must share a type): add sub mul div rem and or xor shl shr min max udiv urem ushr umin umax.
- Compare -> i32 0/1: eq ne lt gt le ge ult ugt ule uge.
- Pointers: `p2 = add p, off` with off:i64; `v = ld p:f32`; `st p, v`; `p = alloc n` (n:i64 bytes); `free p`.
- Other: `s = select c, a, b`; unary neg abs sqrt; casts `itof x:f32` ftoi sext zext trunc fext ftrunc bitcast (`dst = op src:ty`).
- Vectors: `v = ld p:v256`, `st p, v` (any alignment); `v = splat x:v256` (plain `splat x` is v128); `r = vadd a, b:f32` (vsub vmul vdiv vmin vmax vand vor vxor; lane types i8 i16 i32 i64 f32 f64, but vmul has no i8, vdiv is float-only, vmin/vmax have no i64); `r = vfma a, b, c:f32` (a*b+c, f32/f64); `m = vlt a, b:f32` (veq vne vgt vle vge, all-ones lanes when true); `r = vsel m, a, b`; `s = vsum v:f32` (vmaxr vminr); `e = extlane v, 7:f32`; `n = vl f32` (i64 lanes in vx); tail-masked `v = ldm p:vx, n:f32` / `stm p, v, n:f32` touch only the first n lanes (rest of a load is zero).
- Matmul: `mm pc, pa, pb, m, n, k:bf16` does C[m x n] += A[m x k] * B[k x n], row-major contiguous; dtype bf16/f16/f32 with f32 C, or i8 with i32 C; m n k are i64; costs 1 fuel per 1024 multiply-adds. vx cannot appear in function signatures; extfn takes no v256/v512. Legacy vfadd/viadd/vfsum/visum/vfmax still parse.
- Calls: `r = call f(a, b)` or `call f(a)`; external C functions need `extfn sinf(x:f32)->f32` at the top. air_run only permits C math externs (sinf cosf tanf sqrtf expf logf powf fabsf floorf ceilf roundf and f64 sin cos tan sqrt exp log pow fabs floor ceil round).
- Comments: `#` or `//`. Names starting with `__` are reserved.

air_run: `func` defaults to \"main\"; `args` are numbers coerced to the parameter types (v128 params are not allowed); fuel defaults to 1000000 and ERR_OUT_OF_FUEL means a runaway loop.

Example:
fn sum_to(n:i64)->i64
  b0:
    jmp b1(0:i64, 0:i64)
  b1(i:i64, acc:i64):
    c = lt i, n
    br c, b2, b3
  b2:
    acc2 = add acc, i
    i2 = add i, 1:i64
    jmp b1(i2, acc2)
  b3:
    ret acc
";

/// Picks the protocol version for the `initialize` response: the client's requested
/// version if we support it, otherwise our latest (the client then decides whether to proceed).
pub fn negotiate_protocol_version(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|r| SUPPORTED_PROTOCOL_VERSIONS.iter().find(|&&v| v == r))
        .copied()
        .unwrap_or(SUPPORTED_PROTOCOL_VERSIONS[0])
}

pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };
        out.push(B64_CHARS[(b0 >> 2) as usize] as char);
        out.push(B64_CHARS[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_CHARS[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn b64_decode(input: &str) -> Result<Vec<u8>, String> {
    let clean = input.trim();
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for c in clean.chars() {
        if c == '=' || c.is_whitespace() {
            continue;
        }
        let val = match c {
            'A'..='Z' => (c as u32) - ('A' as u32),
            'a'..='z' => (c as u32) - ('a' as u32) + 26,
            '0'..='9' => (c as u32) - ('0' as u32) + 52,
            '+' => 62,
            '/' => 63,
            _ => return Err(format!("Invalid base64 character: {c}")),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

pub fn load_module_from_code(code: &str) -> Result<Module, Diagnostic> {
    let trimmed = code.trim();
    if let Ok(bytes) = b64_decode(trimmed) {
        if bytes.starts_with(b"\x00AIR") {
            let module = decode_module(&bytes)?;
            let mut validator = Validator::new();
            validator.validate_module(&module)?;
            return Ok(module);
        }
    }
    parse_and_validate(trimmed)
}

pub fn execute_ir(
    module: &Module,
    func_name: &str,
    args: &[f64],
    fuel: Option<u64>,
    max_memory_mb: Option<usize>,
) -> Result<Value> {
    const ALLOWED_EXTERNALS: &[&str] = &[
        "sinf", "cosf", "tanf", "sqrtf", "expf", "logf", "powf", "fabsf", "floorf", "ceilf",
        "roundf", "sin", "cos", "tan", "sqrt", "exp", "log", "pow", "fabs", "floor", "ceil",
        "round",
    ];

    for ext in &module.extern_functions {
        if !ALLOWED_EXTERNALS.contains(&ext.name.as_str()) {
            return Err(anyhow!(
                "[ERR_UNAUTHORIZED_EXTERN] External function '{}' is not permitted by MCP security allowlist",
                ext.name
            ));
        }
    }

    let func = module
        .functions
        .iter()
        .find(|f| f.name == func_name)
        .ok_or_else(|| anyhow!("Function '{func_name}' not defined in module"))?;

    let t1 = Instant::now();
    let mut engine = JitEngine::new()?;
    // Enforce default fuel budget of 1_000_000 instructions to prevent runaway LLM code
    let effective_fuel = fuel.or(Some(1_000_000));
    engine.set_fuel(effective_fuel);

    if let Some(mb) = max_memory_mb {
        engine.set_memory_quota(mb * 1024 * 1024);
    }
    engine.compile_module(module)?;
    let compile_time_us = t1.elapsed().as_micros();

    let mut rt_args = Vec::with_capacity(func.params.len());
    for (i, (_, p_ty)) in func.params.iter().enumerate() {
        let val_num = args.get(i).copied().unwrap_or(0.0);
        let rt_val = match p_ty {
            Type::I8 => RtValue::I8(val_num as i8),
            Type::I16 => RtValue::I16(val_num as i16),
            Type::I32 => RtValue::I32(val_num as i32),
            Type::I64 => RtValue::I64(val_num as i64),
            Type::Ptr => RtValue::Ptr(val_num as usize),
            Type::F32 => RtValue::F32(val_num as f32),
            Type::F64 => RtValue::F64(val_num),
            Type::V128 | Type::V256 | Type::V512 | Type::Vx | Type::F16 | Type::BF16 => {
                return Err(anyhow!("Cannot pass {p_ty} directly via MCP"))
            }
        };
        rt_args.push(rt_val);
    }

    let t2 = Instant::now();
    let res_rt = unsafe { engine.call_typed(func_name, &rt_args)? };
    let exec_time_us = t2.elapsed().as_micros();

    let res_val = match res_rt {
        Some(RtValue::I8(n)) => json!(n),
        Some(RtValue::I16(n)) => json!(n),
        Some(RtValue::I32(n)) => json!(n),
        Some(RtValue::I64(n)) => json!(n),
        Some(RtValue::Ptr(p)) => json!(p),
        Some(RtValue::F32(f)) => json!(f),
        Some(RtValue::F64(f)) => json!(f),
        None => json!(null),
    };

    Ok(json!({
        "status": "ok",
        "function": func_name,
        "result": res_val,
        "compile_time_us": compile_time_us,
        "exec_time_us": exec_time_us,
    }))
}

fn json_tool_result(val: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string())
            }
        ],
        "isError": false
    })
}

fn json_tool_error(val: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string())
            }
        ],
        "isError": true
    })
}

fn error_response(msg: &str) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": msg
            }
        ],
        "isError": true
    })
}

pub fn handle_air_check(arguments: &Value) -> Value {
    let code = match arguments.get("code").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return error_response("Missing required parameter 'code'"),
    };
    match load_module_from_code(code) {
        Ok(module) => {
            let total_blocks: usize = module.functions.iter().map(|f| f.blocks.len()).sum();
            let total_instructions: usize = module
                .functions
                .iter()
                .flat_map(|f| &f.blocks)
                .map(|b| b.instructions.len())
                .sum();
            json_tool_result(json!({
                "status": "ok",
                "functions": module.functions.iter().map(|f| &f.name).collect::<Vec<_>>(),
                "function_count": module.functions.len(),
                "extern_functions": module.extern_functions.iter().map(|f| &f.name).collect::<Vec<_>>(),
                "extern_function_count": module.extern_functions.len(),
                "block_count": total_blocks,
                "instruction_count": total_instructions,
            }))
        }
        Err(diag) => json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| {
            json!({
                "status": "error",
                "error_code": diag.error_code,
                "message": diag.message
            })
        })),
    }
}

pub fn handle_air_run(arguments: &Value) -> Value {
    let code = match arguments.get("code").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return error_response("Missing required parameter 'code'"),
    };
    let func = arguments
        .get("func")
        .and_then(|v| v.as_str())
        .unwrap_or("main");
    let fuel = arguments.get("fuel").and_then(|v| v.as_u64());
    let max_memory_mb = arguments
        .get("max_memory_mb")
        .and_then(|v| v.as_u64())
        .map(|m| m as usize);
    let mut args = Vec::new();
    if let Some(arr) = arguments.get("args").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(num) = item.as_f64() {
                args.push(num);
            }
        }
    }

    let module = match load_module_from_code(code) {
        Ok(m) => m,
        Err(diag) => {
            return json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| {
                json!({
                    "status": "error",
                    "error_code": diag.error_code,
                    "message": diag.message
                })
            }))
        }
    };

    match execute_ir(&module, func, &args, fuel, max_memory_mb) {
        Ok(val) => json_tool_result(val),
        Err(e) => json_tool_error(json!({
            "status": "error",
            "error_code": "ERR_EXECUTION",
            "message": e.to_string(),
        })),
    }
}

pub fn handle_air_assemble(arguments: &Value) -> Value {
    let code = match arguments.get("code").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return error_response("Missing required parameter 'code'"),
    };
    match parse_and_validate(code) {
        Ok(module) => {
            let binary = match encode_module(&module) {
                Ok(b) => b,
                Err(diag) => {
                    return json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(
                        |_| {
                            json!({
                                "status": "error",
                                "error_code": diag.error_code,
                                "message": diag.message
                            })
                        },
                    ));
                }
            };
            let b64 = b64_encode(&binary);
            let ratio = (binary.len() as f64) / (code.len().max(1) as f64);
            json_tool_result(json!({
                "status": "ok",
                "binary_base64": b64,
                "source_bytes": code.len(),
                "binary_bytes": binary.len(),
                "compression_ratio": ratio,
            }))
        }
        Err(diag) => json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| {
            json!({
                "status": "error",
                "error_code": diag.error_code,
                "message": diag.message
            })
        })),
    }
}

pub fn handle_air_disassemble(arguments: &Value) -> Value {
    let b64 = match arguments.get("binary_base64").and_then(|v| v.as_str()) {
        Some(b) => b,
        None => return error_response("Missing required parameter 'binary_base64'"),
    };
    let bytes = match b64_decode(b64) {
        Ok(b) => b,
        Err(e) => return error_response(&format!("Base64 decoding failed: {e}")),
    };
    match decode_module(&bytes) {
        Ok(module) => {
            let text = to_air_text(&module);
            json_tool_result(json!({
                "status": "ok",
                "code": text,
            }))
        }
        Err(diag) => json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| {
            json!({
                "status": "error",
                "error_code": diag.error_code,
                "message": diag.message
            })
        })),
    }
}

pub fn handle_air_optimize(arguments: &Value) -> Value {
    let code = match arguments.get("code").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return error_response("Missing required parameter 'code'"),
    };
    let mut module = match load_module_from_code(code) {
        Ok(m) => m,
        Err(diag) => {
            return json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| {
                json!({
                    "status": "error",
                    "error_code": diag.error_code,
                    "message": diag.message
                })
            }))
        }
    };

    let stats = achainsaw_ir::opt::optimize_module(&mut module);
    let optimized_code = to_air_text(&module);

    json_tool_result(json!({
        "status": "ok",
        "code": optimized_code,
        "constants_folded": stats.constants_folded,
        "algebraic_simplifications": stats.algebraic_simplifications,
        "branches_folded": stats.branches_folded,
        "dead_instructions_removed": stats.dead_instructions_removed,
        "dead_blocks_removed": stats.dead_blocks_removed,
        "total_optimizations": stats.total_optimizations(),
        "iterations": stats.iterations,
    }))
}

pub fn handle_air_target(_arguments: &Value) -> Value {
    match achainsaw_codegen::cpu::target_report() {
        Ok(report) => json_tool_result(report),
        Err(e) => json_tool_error(json!({
            "status": "error",
            "error_code": "ERR_CPU_DETECTION",
            "message": e.to_string(),
        })),
    }
}

pub fn handle_initialize(params: &Value) -> Value {
    let requested = params.get("protocolVersion").and_then(|v| v.as_str());
    json!({
        "protocolVersion": negotiate_protocol_version(requested),
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "achainsaw-mcp",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": SERVER_INSTRUCTIONS
    })
}

pub fn get_tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "air_check",
                "description": "Validate AIR text IR or binary bytecode syntax and SSA invariants. Returns structured diagnostic or validation metrics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "AIR IR text representation or base64 encoded AIRB bytecode"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "air_run",
                "description": "JIT compile and execute an AIR function with optional arguments, loop fuel budget, and memory quota.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "AIR IR text representation or base64 encoded AIRB bytecode"
                        },
                        "func": {
                            "type": "string",
                            "description": "Name of function to execute (default: 'main')"
                        },
                        "args": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Numeric arguments passed to function"
                        },
                        "fuel": {
                            "type": "integer",
                            "description": "Loop fuel instruction budget (prevents infinite loops)"
                        },
                        "max_memory_mb": {
                            "type": "integer",
                            "description": "Maximum heap memory quota in megabytes"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "air_assemble",
                "description": "Assemble AIR textual IR into compact binary bytecode (AIRB), returned as base64 string.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "AIR textual IR source code"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "air_disassemble",
                "description": "Disassemble binary bytecode (AIRB) from base64 string into canonical, agent-readable AIR text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "binary_base64": {
                            "type": "string",
                            "description": "Base64 encoded AIRB binary bytecode"
                        }
                    },
                    "required": ["binary_base64"]
                }
            },
            {
                "name": "air_target",
                "description": "Report the host CPU's vector features (AVX/AVX2/AVX-512/AMX, NEON/SVE/SME), the active ISA cap, and the vector width each code generation backend uses.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "air_optimize",
                "description": "Run IR optimization engine passes (constant folding, algebraic simplification, branch folding, dead code & block elimination).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "AIR textual IR source code or base64 encoded AIRB bytecode"
                        }
                    },
                    "required": ["code"]
                }
            }
        ]
    })
}

pub fn run_mcp_server() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());

    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {e}")
                    }
                });
                serde_json::to_writer(&mut writer, &err_resp)?;
                writer.write_all(b"\n")?;
                writer.flush()?;
                line.clear();
                continue;
            }
        };

        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

        // If it's a notification without an ID, do not send a response
        if id.is_none() || id.as_ref().is_some_and(|v| v.is_null()) {
            line.clear();
            continue;
        }
        let id_val = id.unwrap();

        let response = match method {
            "initialize" => {
                let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
                json!({
                    "jsonrpc": "2.0",
                    "id": id_val,
                    "result": handle_initialize(&params)
                })
            }
            "ping" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id_val,
                    "result": {}
                })
            }
            "tools/list" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id_val,
                    "result": get_tools_list()
                })
            }
            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));

                let tool_res = match name {
                    "air_check" => handle_air_check(&arguments),
                    "air_run" => handle_air_run(&arguments),
                    "air_assemble" => handle_air_assemble(&arguments),
                    "air_disassemble" => handle_air_disassemble(&arguments),
                    "air_optimize" => handle_air_optimize(&arguments),
                    "air_target" => handle_air_target(&arguments),
                    unknown => error_response(&format!("Unknown tool: '{unknown}'")),
                };

                json!({
                    "jsonrpc": "2.0",
                    "id": id_val,
                    "result": tool_res
                })
            }
            unknown_method => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id_val,
                    "error": {
                        "code": -32601,
                        "message": format!("Method not found: '{unknown_method}'")
                    }
                })
            }
        };

        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;

        line.clear();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_roundtrip() {
        let test_cases: &[&[u8]] = &[
            b"",
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
            b"\x00AIR\x01\x00\x02\x03\xff\xfe",
        ];
        for data in test_cases {
            let encoded = b64_encode(data);
            let decoded = b64_decode(&encoded).expect("decode failed");
            assert_eq!(*data, &decoded[..]);
        }
    }

    #[test]
    fn test_protocol_version_negotiation() {
        assert_eq!(negotiate_protocol_version(Some("2024-11-05")), "2024-11-05");
        assert_eq!(negotiate_protocol_version(Some("2025-06-18")), "2025-06-18");
        assert_eq!(
            negotiate_protocol_version(Some("1999-01-01")),
            SUPPORTED_PROTOCOL_VERSIONS[0]
        );
        assert_eq!(
            negotiate_protocol_version(None),
            SUPPORTED_PROTOCOL_VERSIONS[0]
        );
    }

    #[test]
    fn test_mcp_initialize() {
        let res = handle_initialize(&json!({ "protocolVersion": "2025-06-18" }));
        assert_eq!(res["protocolVersion"], "2025-06-18");
        assert_eq!(res["serverInfo"]["name"], "achainsaw-mcp");
        assert!(res["capabilities"]["tools"].is_object());
        assert_eq!(res["instructions"], SERVER_INSTRUCTIONS);
    }

    #[test]
    fn test_server_instructions_example_is_valid_air() {
        let example = SERVER_INSTRUCTIONS
            .split_once("Example:\n")
            .expect("instructions must contain an example")
            .1;
        let res = handle_air_run(&json!({
            "code": example,
            "func": "sum_to",
            "args": [10]
        }));
        assert_eq!(res["isError"], false, "{res}");
        let text = res["content"][0]["text"].as_str().unwrap();
        let payload: Value = serde_json::from_str(text).unwrap();
        assert_eq!(payload["result"], 45);
    }

    #[test]
    fn test_mcp_air_target() {
        let res = handle_air_target(&json!({}));
        assert_eq!(res["isError"], false);
        let text = res["content"][0]["text"].as_str().unwrap();
        let report: Value = serde_json::from_str(text).unwrap();
        assert_eq!(report["status"], "ok");
        assert!(report["host"]["features"].is_array());
        assert_eq!(report["backends"]["cranelift"]["vector_bits"], 128);
        let tools = get_tools_list();
        assert!(tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "air_target"));
    }

    #[test]
    fn test_mcp_air_check() {
        let code = r#"
        fn calc(v0:i32)->i32
          b0:
            ten = cst 10:i32
            v1 = add v0, ten
            ret v1
        "#;
        let res = handle_air_check(&json!({ "code": code }));
        assert_eq!(res["isError"], false);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"function_count\": 1"));

        // Invalid code
        let bad = handle_air_check(&json!({ "code": "invalid syntax error" }));
        assert_eq!(bad["isError"], true);
    }

    #[test]
    fn test_mcp_air_run() {
        let code = r#"
        fn main(v0:i32, v1:i32)->i32
          b0:
            v2 = mul v0, v1
            ret v2
        "#;
        let res = handle_air_run(&json!({
            "code": code,
            "func": "main",
            "args": [6, 7]
        }));
        assert_eq!(res["isError"], false);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"result\": 42"));
    }

    #[test]
    fn test_mcp_air_run_fuel_trap() {
        let loop_code = r#"
        fn main()->i32
          b0:
            zero = cst 0:i32
            jmp b1(zero)
          b1(x:i32):
            one = cst 1:i32
            next_x = add x, one
            jmp b1(next_x)
        "#;
        let res = handle_air_run(&json!({
            "code": loop_code,
            "func": "main",
            "fuel": 200
        }));
        assert_eq!(res["isError"], true);
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("ERR_OUT_OF_FUEL"));
    }

    #[test]
    fn test_mcp_air_assemble_and_disassemble() {
        let code = "fn add_five(v0:i32)->i32\n  b0:\n    five = cst 5:i32\n    v1 = add v0, five\n    ret v1\n";
        let asm_res = handle_air_assemble(&json!({ "code": code }));
        assert_eq!(asm_res["isError"], false);
        let asm_text = asm_res["content"][0]["text"].as_str().unwrap();
        let asm_json: Value = serde_json::from_str(asm_text).unwrap();
        let b64 = asm_json["binary_base64"].as_str().unwrap();

        let disasm_res = handle_air_disassemble(&json!({ "binary_base64": b64 }));
        assert_eq!(disasm_res["isError"], false);
        let disasm_text = disasm_res["content"][0]["text"].as_str().unwrap();
        let disasm_json: Value = serde_json::from_str(disasm_text).unwrap();
        let decompiled = disasm_json["code"].as_str().unwrap();
        assert!(decompiled.contains("fn add_five"));
        assert!(decompiled.contains("add v0, five"));
    }

    #[test]
    fn test_mcp_air_optimize() {
        let unopt = r#"
        fn unoptimized(x:i32)->i32
          b0:
            a = cst 10:i32
            b = cst 20:i32
            c = add a, b
            zero = cst 0:i32
            d = add x, zero
            ret d
        "#;
        let res = handle_air_optimize(&json!({ "code": unopt }));
        assert_eq!(res["isError"], false);
        let text = res["content"][0]["text"].as_str().unwrap();
        let stats: Value = serde_json::from_str(text).unwrap();
        assert!(stats["total_optimizations"].as_u64().unwrap() >= 2);
        assert!(stats["code"].as_str().unwrap().contains("ret x"));
    }
}
