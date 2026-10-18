# MCP server

`achainsaw` includes a native Model Context Protocol (MCP) server communicating over JSON-RPC 2.0 stdio, exposing compiler tools directly to LLM agents and IDE assistants:
```bash
achainsaw mcp
```

**Tools:**
- **`air_check`**: Validates AIR textual IR or base64 AIRB bytecode syntax and SSA invariants. Returns structured diagnostic metrics or error payloads with line/column pointers and self-repair hints.
- **`air_run`**: JIT compiles and executes AIR functions with arguments, loop fuel budget, and memory quota sandboxing. Code runs in a sandbox: memory comes only from `alloc` within a private arena (`max_memory_mb`, default 64), out-of-bounds accesses and bad `free`s fail with `ERR_MEMORY_VIOLATION`, unbounded recursion with `ERR_STACK_OVERFLOW`, and pointer parameters are rejected.
- **`air_assemble`**: Assembles textual AIR into compact base64-encoded AIRB bytecode with compression metrics.
- **`air_disassemble`**: Decompiles base64 AIRB bytecode back into canonical, human/agent-readable textual AIR.
- **`air_optimize`**: Optimizes IR using constant folding, algebraic simplification, branch folding, and DCE to a fixpoint.
- **`air_target`**: Reports the host's vector features (AVX/AVX2/AVX-512/AMX, NEON/SVE/SME), the active ISA cap, and each backend's vector width (see [CPU features and ISA targeting](cli.md#cpu-features-and-isa-targeting-achainsaw-cpu)).
- **`air_std`**: Lists the standard library's functions with their signatures and documentation, or returns one's AIR source by name.

The `initialize` response includes server `instructions`, a compact AIR syntax primer that MCP clients inject into the model's context. That lets an agent write valid AIR on the first attempt without other documentation.

## Claude Code
The repository ships a project-scoped [`.mcp.json`](../.mcp.json) that runs the server via `cargo run --release`. Build once so the first startup does not exceed the MCP connection timeout, then start Claude Code in the repo and approve the `achainsaw` server when prompted:
```bash
cargo build --release -p achainsaw
claude
```

To make the tools available in every project, register an installed binary at user scope instead:
```bash
cargo install --path crates/achainsaw-cli
claude mcp add --scope user achainsaw -- achainsaw mcp
```

## Claude Desktop
Add the server to `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), using the absolute path to the binary, then restart Claude Desktop:
```json
{
  "mcpServers": {
    "achainsaw": {
      "command": "/absolute/path/to/achainsaw",
      "args": ["mcp"]
    }
  }
}
```
