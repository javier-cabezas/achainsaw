# Security

## What the sandbox does

Code run through the MCP server's `air_run` (and with `JitEngine::enable_sandbox` from Rust)
is compiled with checks meant to contain a model's mistakes:

- every load, store, gather, scatter, atomic and `mm` is checked against a private memory
  arena (`max_memory_mb`, default 64 MB), and writes to the module's read-only `data` fail;
- `free` must name a live allocation;
- fuel bounds the work a run does (1,000,000 units by default), and function entry checks
  the stack depth;
- pointer arguments are rejected, and the only external functions allowed are C math
  functions (`sinf`, `sqrt`, ...).

A violation ends the run with an error code (`ERR_MEMORY_VIOLATION`, `ERR_OUT_OF_FUEL`,
`ERR_STACK_OVERFLOW`, ...) and never crashes the server. The MCP tools are fuzzed with
mutated programs and bytecode (`fuzz_mcp_tools`, and weekly on both backends).

## What it does not do

The sandbox has not been audited against hostile code. It runs in the server's own process,
does not bound compile time or wall-clock time apart from fuel, and does not protect against
side channels. Run code from sources you do not trust in a container or a VM as well.

Code compiled without the sandbox (the CLI's `run`, `bench` and `build`, the Python package,
AOT objects and libraries) is as trusted as any native code you run.

## Reporting a vulnerability

A way for code to read or write memory outside its arena, call something other than an
allowed external, or crash or hang the server despite its limits is a vulnerability. Please
report it privately through GitHub's [private vulnerability reporting](https://github.com/javier-cabezas/achainsaw/security/advisories/new)
rather than in a public issue, with the AIR program (or bytecode) and the backend.

Only the latest release is supported with fixes.
