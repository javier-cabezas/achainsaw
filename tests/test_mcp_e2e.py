"""
End-to-End Integration Test for achainsaw Model Context Protocol (MCP) Server.
Tests JSON-RPC 2.0 stdio communication, tool listing, validation, assembly, and JIT execution.
"""

import json
import os
import subprocess
import sys

EXE_NAME = "achainsaw.exe" if sys.platform == "win32" else "achainsaw"
EXE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "debug", EXE_NAME))
if not os.path.exists(EXE_PATH):
    rel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "release", EXE_NAME))
    if os.path.exists(rel_path):
        EXE_PATH = rel_path


def test_mcp_server():
    proc = subprocess.Popen(
        [EXE_PATH, "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def send_req(req_obj):
        line = json.dumps(req_obj) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()
        resp_line = proc.stdout.readline()
        return json.loads(resp_line)

    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    init_res = send_req(init_req)
    assert init_res["id"] == 1, f"Init failed: {init_res}"
    assert init_res["result"]["serverInfo"]["name"] == "achainsaw-mcp"
    assert init_res["result"]["protocolVersion"] == "2024-11-05"
    assert "air_check" in init_res["result"]["instructions"]
    print("[PASS] MCP initialize")

    # 1b. Initialized notification (no response expected)
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    proc.stdin.flush()

    # 2. Ping
    ping_res = send_req({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping_res["id"] == 2
    print("[PASS] MCP ping")

    # 3. Tools List
    list_res = send_req({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
    tools = list_res["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "air_check" in tool_names
    assert "air_run" in tool_names
    assert "air_assemble" in tool_names
    assert "air_disassemble" in tool_names
    assert "air_target" in tool_names
    print(f"[PASS] MCP tools/list ({len(tools)} tools: {', '.join(tool_names)})")

    # 4. Tool Call: air_check
    sample_air = """fn mul_add(a:i32, b:i32, c:i32)->i32
  b0:
    prod = mul a, b
    res = add prod, c
    ret res
"""
    check_res = send_req({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "air_check",
            "arguments": {"code": sample_air}
        }
    })
    assert check_res["result"]["isError"] is False
    check_payload = json.loads(check_res["result"]["content"][0]["text"])
    assert check_payload["function_count"] == 1
    assert check_payload["functions"] == ["mul_add"]
    print("[PASS] MCP tools/call air_check")

    # 5. Tool Call: air_assemble
    asm_res = send_req({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "air_assemble",
            "arguments": {"code": sample_air}
        }
    })
    assert asm_res["result"]["isError"] is False
    asm_payload = json.loads(asm_res["result"]["content"][0]["text"])
    b64_bytes = asm_payload["binary_base64"]
    assert len(b64_bytes) > 0
    print(f"[PASS] MCP tools/call air_assemble (bytecode b64: {b64_bytes[:20]}...)")

    # 6. Tool Call: air_disassemble
    disasm_res = send_req({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "air_disassemble",
            "arguments": {"binary_base64": b64_bytes}
        }
    })
    assert disasm_res["result"]["isError"] is False
    disasm_payload = json.loads(disasm_res["result"]["content"][0]["text"])
    assert "fn mul_add" in disasm_payload["code"]
    print("[PASS] MCP tools/call air_disassemble")

    # 7. Tool Call: air_run (execute text AIR)
    run_air = """fn main(x:i32, y:i32)->i32
  b0:
    sum = add x, y
    ret sum
"""
    run_res = send_req({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "air_run",
            "arguments": {
                "code": run_air,
                "func": "main",
                "args": [19, 23]
            }
        }
    })
    assert run_res["result"]["isError"] is False
    run_payload = json.loads(run_res["result"]["content"][0]["text"])
    assert run_payload["result"] == 42
    print(f"[PASS] MCP tools/call air_run result={run_payload['result']}")

    # 8. Tool Call: air_run with fuel trap
    infinite_loop = """fn main()->i32
  b0:
    zero = cst 0:i32
    jmp b1(zero)
  b1(x:i32):
    one = cst 1:i32
    next_x = add x, one
    jmp b1(next_x)
"""
    fuel_res = send_req({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "air_run",
            "arguments": {
                "code": infinite_loop,
                "fuel": 500
            }
        }
    })
    assert fuel_res["result"]["isError"] is True
    err_text = fuel_res["result"]["content"][0]["text"]
    assert "ERR_OUT_OF_FUEL" in err_text
    print("[PASS] MCP tools/call air_run fuel trap enforcement")

    # 8b. Sandbox: out-of-bounds access and unbounded recursion are errors, and the
    # server keeps running afterwards (later requests below still succeed).
    sandbox_cases = [
        ("ERR_MEMORY_VIOLATION", """fn main()->i32
  b0:
    p = alloc 16:i64
    q = add p, 1048576:i64
    v = ld q:i32
    ret v
"""),
        ("ERR_STACK_OVERFLOW", """fn main()->i32
  b0:
    r = call main()
    ret r
"""),
    ]
    for i, (code, air) in enumerate(sandbox_cases):
        res = send_req({
            "jsonrpc": "2.0",
            "id": 80 + i,
            "method": "tools/call",
            "params": {
                "name": "air_run",
                "arguments": {"code": air, "max_memory_mb": 1}
            }
        })
        assert res["result"]["isError"] is True
        assert code in res["result"]["content"][0]["text"]
        print(f"[PASS] MCP tools/call air_run sandbox {code}")

    # 8c. par: a fork-join loop gives the same result on all cores and on one thread, and
    # a fault in one worker is reported like any other sandbox violation.
    par_air = """fn square(i:i64, p:ptr)
  b0:
    off = mul i, 8:i64
    q = add p, off
    v = mul i, i
    st q, v
    ret

fn main(n:i64)->i64
  b0:
    bytes = mul n, 8:i64
    p = alloc bytes
    par n, square(p)
    jmp b1(0:i64, 0:i64)
  b1(i:i64, acc:i64):
    more = lt i, n
    br more, b2, b3
  b2:
    off = mul i, 8:i64
    q = add p, off
    v = ld q:i64
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp b1(i2, acc2)
  b3:
    ret acc
"""
    for i, threads in enumerate([None, 1]):
        arguments = {"code": par_air, "args": [1000]}
        if threads is not None:
            arguments["threads"] = threads
        res = send_req({
            "jsonrpc": "2.0",
            "id": 85 + i,
            "method": "tools/call",
            "params": {"name": "air_run", "arguments": arguments}
        })
        assert res["result"]["isError"] is False, res
        payload = json.loads(res["result"]["content"][0]["text"])
        assert payload["result"] == sum(k * k for k in range(1000))
        assert threads is None or payload["threads"] == threads
        print(f"[PASS] MCP tools/call air_run par (threads={payload['threads']})")
    res = send_req({
        "jsonrpc": "2.0",
        "id": 87,
        "method": "tools/call",
        "params": {
            "name": "air_run",
            "arguments": {"code": par_air.replace("off = mul i, 8:i64\n    q = add p, off\n    v = mul i, i",
                                                  "off = mul i, 1048576:i64\n    q = add p, off\n    v = mul i, i", 1),
                          "args": [64], "max_memory_mb": 1}
        }
    })
    assert res["result"]["isError"] is True
    assert "ERR_MEMORY_VIOLATION" in res["result"]["content"][0]["text"]
    print("[PASS] MCP tools/call air_run par worker violation")

    # 9. Tool Call: air_optimize
    unopt_air = """fn opt_me(x:i32)->i32
  b0:
    c1 = cst 10:i32
    c2 = cst 20:i32
    c3 = add c1, c2
    zero = cst 0:i32
    d = add x, zero
    ret d
"""
    opt_res = send_req({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "air_optimize",
            "arguments": {"code": unopt_air}
        }
    })
    assert opt_res["result"]["isError"] is False
    opt_payload = json.loads(opt_res["result"]["content"][0]["text"])
    assert opt_payload["total_optimizations"] >= 2
    assert "ret x" in opt_payload["code"]
    print("[PASS] MCP tools/call air_optimize")

    # 10. Tool Call: air_target
    target_res = send_req({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "air_target", "arguments": {}}
    })
    assert target_res["result"]["isError"] is False
    target_payload = json.loads(target_res["result"]["content"][0]["text"])
    assert target_payload["status"] == "ok"
    assert isinstance(target_payload["host"]["features"], list)
    assert target_payload["backends"]["cranelift"]["vector_bits"] == 128
    assert target_payload["par_threads"] >= 1
    print(f"[PASS] MCP tools/call air_target (max_isa={target_payload['host']['max_isa']})")
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
    print("All MCP end-to-end integration tests passed!")


if __name__ == "__main__":
    test_mcp_server()
