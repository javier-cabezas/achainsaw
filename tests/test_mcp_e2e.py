"""
End-to-End Integration Test for achainsaw Model Context Protocol (MCP) Server.
Tests JSON-RPC 2.0 stdio communication, tool listing, validation, assembly, and JIT execution.
"""

import json
import os
import subprocess
import sys

EXE_PATH = os.path.join(os.path.dirname(__file__), "..", "target", "debug", "achainsaw.exe")


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
    print("[PASS] MCP initialize")

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

    proc.terminate()
    print("All MCP end-to-end integration tests passed!")


if __name__ == "__main__":
    test_mcp_server()
