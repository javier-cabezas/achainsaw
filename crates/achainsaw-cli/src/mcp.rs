use achainsaw_codegen::JitEngine;
use achainsaw_ir::diag::Diagnostic;
use achainsaw_ir::{
    decode_module, encode_module, parse_and_validate, to_air_text, Module, Type, Validator,
};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
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
    let func = module
        .functions
        .iter()
        .find(|f| f.name == func_name)
        .ok_or_else(|| anyhow!("Function '{func_name}' not defined in module"))?;

    let t1 = Instant::now();
    let mut engine = JitEngine::new()?;
    if let Some(f) = fuel {
        engine.set_fuel(Some(f));
    }
    if let Some(mb) = max_memory_mb {
        engine.set_memory_quota(mb * 1024 * 1024);
    }
    engine.compile_module(module)?;
    let compile_time_us = t1.elapsed().as_micros();

    let func_ptr = engine
        .get_fn_ptr(func_name)
        .ok_or_else(|| anyhow!("Function pointer for '{func_name}' not found"))?;

    let t2 = Instant::now();
    achainsaw_codegen::reset_execution_status();

    let res_val: Value = unsafe {
        match (func.params.len(), func.ret_type) {
            (0, None) => {
                let f: extern "C" fn() = std::mem::transmute(func_ptr);
                f();
                json!(null)
            }
            (0, Some(Type::I32)) => {
                let f: extern "C" fn() -> i32 = std::mem::transmute(func_ptr);
                json!(f())
            }
            (0, Some(Type::I64)) => {
                let f: extern "C" fn() -> i64 = std::mem::transmute(func_ptr);
                json!(f())
            }
            (0, Some(Type::F32)) => {
                let f: extern "C" fn() -> f32 = std::mem::transmute(func_ptr);
                json!(f())
            }
            (0, Some(Type::F64)) => {
                let f: extern "C" fn() -> f64 = std::mem::transmute(func_ptr);
                json!(f())
            }
            (1, Some(Type::I32)) => {
                let arg0 = args.first().copied().unwrap_or(0.0) as i32;
                let f: extern "C" fn(i32) -> i32 = std::mem::transmute(func_ptr);
                json!(f(arg0))
            }
            (1, Some(Type::I64)) => {
                let arg0 = args.first().copied().unwrap_or(0.0) as i64;
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(func_ptr);
                json!(f(arg0))
            }
            (1, Some(Type::F32)) => {
                let arg0 = args.first().copied().unwrap_or(0.0) as f32;
                let f: extern "C" fn(f32) -> f32 = std::mem::transmute(func_ptr);
                json!(f(arg0))
            }
            (1, Some(Type::F64)) => {
                let arg0 = args.first().copied().unwrap_or(0.0);
                let f: extern "C" fn(f64) -> f64 = std::mem::transmute(func_ptr);
                json!(f(arg0))
            }
            (2, Some(Type::I32)) => {
                let arg0 = args.first().copied().unwrap_or(0.0) as i32;
                let arg1 = args.get(1).copied().unwrap_or(0.0) as i32;
                let f: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(func_ptr);
                json!(f(arg0, arg1))
            }
            (2, Some(Type::I64)) => {
                let arg0 = args.first().copied().unwrap_or(0.0) as i64;
                let arg1 = args.get(1).copied().unwrap_or(0.0) as i64;
                let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(func_ptr);
                json!(f(arg0, arg1))
            }
            (2, Some(Type::F32)) => {
                let arg0 = args.first().copied().unwrap_or(0.0) as f32;
                let arg1 = args.get(1).copied().unwrap_or(0.0) as f32;
                let f: extern "C" fn(f32, f32) -> f32 = std::mem::transmute(func_ptr);
                json!(f(arg0, arg1))
            }
            (2, Some(Type::F64)) => {
                let arg0 = args.first().copied().unwrap_or(0.0);
                let arg1 = args.get(1).copied().unwrap_or(0.0);
                let f: extern "C" fn(f64, f64) -> f64 = std::mem::transmute(func_ptr);
                json!(f(arg0, arg1))
            }
            _ => {
                return Err(anyhow!(
                    "Function signature ({:?} params -> {:?}) not directly invokable via standard MCP runner",
                    func.params.len(),
                    func.ret_type
                ));
            }
        }
    };

    achainsaw_codegen::check_execution_status()?;
    let exec_time_us = t2.elapsed().as_micros();

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
        Err(diag) => json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| json!({
            "status": "error",
            "error_code": diag.error_code,
            "message": diag.message
        }))),
    }
}

pub fn handle_air_run(arguments: &Value) -> Value {
    let code = match arguments.get("code").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return error_response("Missing required parameter 'code'"),
    };
    let func = arguments.get("func").and_then(|v| v.as_str()).unwrap_or("main");
    let fuel = arguments.get("fuel").and_then(|v| v.as_u64());
    let max_memory_mb = arguments.get("max_memory_mb").and_then(|v| v.as_u64()).map(|m| m as usize);
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
        Err(diag) => return json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| json!({
            "status": "error",
            "error_code": diag.error_code,
            "message": diag.message
        }))),
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
            let binary = encode_module(&module);
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
        Err(diag) => json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| json!({
            "status": "error",
            "error_code": diag.error_code,
            "message": diag.message
        }))),
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
        Err(diag) => json_tool_error(serde_json::from_str(&diag.to_json()).unwrap_or_else(|_| json!({
            "status": "error",
            "error_code": diag.error_code,
            "message": diag.message
        }))),
    }
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
        if id.is_none() || id.as_ref().map_or(false, |v| v.is_null()) {
            line.clear();
            continue;
        }
        let id_val = id.unwrap();

        let response = match method {
            "initialize" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id_val,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "achainsaw-mcp",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }
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
                let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

                let tool_res = match name {
                    "air_check" => handle_air_check(&arguments),
                    "air_run" => handle_air_run(&arguments),
                    "air_assemble" => handle_air_assemble(&arguments),
                    "air_disassemble" => handle_air_disassemble(&arguments),
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
}
