//! Fuzzing of the MCP tools an agent reaches: mutated AIR programs and AIRB bytecode through
//! `air_check`, `air_assemble`/`air_disassemble`, `air_optimize` and `air_run` (sandboxed,
//! with small fuel and memory budgets). Every tool must answer with a result or a diagnostic
//! and never panic, a program it accepts must survive the AIRB round trip, and running one
//! must never crash the process.
//!
//! The inputs are mutations of every `.air` file in `examples/` (lines deleted, duplicated or
//! swapped, tokens replaced or inserted from the corpus's own vocabulary, numbers replaced by
//! edge values, characters replaced, text truncated) and byte mutations of their bytecode.
//! `ACHAINSAW_FUZZ_CASES` sets the number of cases (default 400) and `ACHAINSAW_FUZZ_SEED`
//! the seed (default 1), so a longer run is `ACHAINSAW_FUZZ_CASES=100000 cargo test --release
//! -p achainsaw fuzz`. Each case is written to `achainsaw-fuzz-case.air` in the temporary
//! directory before it runs, so a crash leaves its input there.

use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

/// xorshift64*: a small, seedable generator, so a failing seed reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Numbers that sit at the ends of ranges, where bounds and overflow checks live.
const EDGES: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "31",
    "32",
    "63",
    "64",
    "127",
    "128",
    "255",
    "256",
    "65535",
    "65536",
    "2147483647",
    "-2147483648",
    "4294967295",
    "9223372036854775807",
    "-9223372036854775808",
    "1.0e38",
    "-0.0",
    "1.5",
];

/// Scalar arguments air_run gets: small values and the ends of the integer ranges.
const ARGS: &[f64] = &[
    0.0,
    1.0,
    -1.0,
    3.0,
    17.0,
    100.0,
    1000.0,
    65536.0,
    2147483647.0,
    -2147483648.0,
    1.0e18,
];

fn examples() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|e| e == "air") {
                out.push((p.clone(), std::fs::read_to_string(&p).unwrap()));
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut out = Vec::new();
    walk(&root, &mut out);
    assert!(out.len() > 10, "no examples under {}", root.display());
    out
}

/// The corpus's tokens: identifiers, mnemonics, types and numbers, and the punctuation
/// between them.
fn vocabulary(corpus: &[(PathBuf, String)]) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for (_, text) in corpus {
        for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
            for tok in split_tokens(line) {
                if !v.contains(&tok) {
                    v.push(tok);
                }
            }
        }
    }
    v
}

fn split_tokens(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in line.chars() {
        if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Mutations of `text`: one most of the time, up to four. Half replace a number literal with
/// an edge value, which usually leaves the program valid, so its run sees extreme constants
/// (sizes, trip counts, shifts, indices); the rest change its structure.
fn mutate(rng: &mut Rng, text: &str, vocab: &[String]) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let n = [1, 1, 1, 2, 2, 3, 4][rng.below(7)];
    for _ in 0..n {
        if lines.is_empty() {
            break;
        }
        if rng.below(2) == 0 {
            replace_number(rng, &mut lines);
            continue;
        }
        let i = rng.below(lines.len());
        match rng.below(8) {
            0 => {
                lines.remove(i);
            }
            1 => {
                let l = lines[i].clone();
                lines.insert(i, l);
            }
            2 => {
                let j = rng.below(lines.len());
                lines.swap(i, j);
            }
            3 | 4 => {
                // A token replaced by another of the corpus's, or an edge value for a number.
                let mut toks = split_tokens(&lines[i]);
                if toks.is_empty() {
                    continue;
                }
                let t = rng.below(toks.len());
                toks[t] = if toks[t].chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    EDGES[rng.below(EDGES.len())].to_string()
                } else {
                    vocab[rng.below(vocab.len())].clone()
                };
                lines[i] = format!("    {}", join_tokens(&toks));
            }
            5 => {
                let mut toks = split_tokens(&lines[i]);
                let t = rng.below(toks.len() + 1);
                toks.insert(t, vocab[rng.below(vocab.len())].clone());
                lines[i] = format!("    {}", join_tokens(&toks));
            }
            6 => {
                // A character replaced by a printable one, or by any character.
                let mut chars: Vec<char> = lines[i].chars().collect();
                if chars.is_empty() {
                    continue;
                }
                let c = rng.below(chars.len());
                chars[c] = if rng.below(4) == 0 {
                    char::from_u32(rng.below(0x3000) as u32).unwrap_or('?')
                } else {
                    (b' ' + rng.below(95) as u8) as char
                };
                lines[i] = chars.into_iter().collect();
            }
            _ => {
                lines.truncate(i + 1);
                let chars: Vec<char> = lines[i].chars().collect();
                lines[i] = chars[..rng.below(chars.len() + 1)].iter().collect();
            }
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// One number literal of the program (outside comments) replaced by an edge value, in place.
fn replace_number(rng: &mut Rng, lines: &mut [String]) {
    let mut spots: Vec<(usize, usize, usize)> = Vec::new();
    for (li, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let b = line.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let starts = b[i].is_ascii_digit()
                && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'));
            if starts {
                let s = if i > 0 && b[i - 1] == b'-' { i - 1 } else { i };
                let mut e = i;
                while e < b.len() && (b[e].is_ascii_alphanumeric() || b[e] == b'.') {
                    e += 1;
                }
                spots.push((li, s, e));
                i = e;
            } else {
                i += 1;
            }
        }
    }
    if spots.is_empty() {
        return;
    }
    let (li, s, e) = spots[rng.below(spots.len())];
    let edge = EDGES[rng.below(EDGES.len())];
    lines[li].replace_range(s..e, edge);
}

/// Tokens back into a line: words space-separated, punctuation where AIR puts it.
fn join_tokens(toks: &[String]) -> String {
    let mut s = String::new();
    for t in toks {
        let punct = matches!(t.as_str(), "," | ":" | "(" | ")" | "[" | "]");
        if !s.is_empty() && !punct && !s.ends_with('(') && !s.ends_with('[') {
            s.push(' ');
        }
        if t == "," {
            s.push_str(", ");
            continue;
        }
        s.push_str(t);
    }
    s.replace("  ", " ")
}

fn is_ok(v: &Value) -> bool {
    v.get("isError").and_then(Value::as_bool) == Some(false)
}

fn text_of(v: &Value) -> String {
    v["content"][0]["text"].as_str().unwrap_or("").to_string()
}

/// Runs `tool` on `args`, recording a panic (with the input) instead of unwinding.
fn guarded(
    failures: &mut Vec<String>,
    tool: &str,
    input: &str,
    f: impl FnOnce() -> Value,
) -> Option<Value> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Some(v),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            failures.push(format!("{tool} panicked: {msg}\n--- input ---\n{input}"));
            None
        }
    }
}

/// The functions air_run can call: every parameter a scalar.
fn runnable(module: &Module) -> Vec<(String, usize)> {
    module
        .functions
        .iter()
        .filter(|f| {
            f.params.iter().all(|(_, t)| {
                matches!(
                    t,
                    Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F32 | Type::F64
                )
            })
        })
        .map(|f| (f.name.clone(), f.params.len()))
        .collect()
}

/// How many cases the tools accepted, and how many air_run calls they made.
#[derive(Default)]
struct Stats {
    accepted: usize,
    runs: usize,
}

/// Checks, round-trips, optimizes and runs one program (AIR text or base64 AIRB).
fn exercise(rng: &mut Rng, code: &str, failures: &mut Vec<String>, stats: &mut Stats) {
    let case = std::env::temp_dir().join("achainsaw-fuzz-case.air");
    let _ = std::fs::write(&case, code);
    let Some(check) = guarded(failures, "air_check", code, || {
        handle_air_check(&json!({ "code": code }))
    }) else {
        return;
    };
    if !is_ok(&check) {
        return;
    }
    stats.accepted += 1;
    // Accepted AIR text: AIRB and back must be accepted again (bytecode inputs, which
    // air_check also takes, are not text to assemble).
    let is_bytecode = b64_decode(code.trim()).is_ok_and(|b| b.starts_with(b"\x00AIR"));
    if is_bytecode {
    } else if let Some(asm) = guarded(failures, "air_assemble", code, || {
        handle_air_assemble(&json!({ "code": code }))
    }) {
        if is_ok(&asm) {
            let b64 = serde_json::from_str::<Value>(&text_of(&asm)).unwrap()["binary_base64"]
                .as_str()
                .unwrap()
                .to_string();
            if let Some(dis) = guarded(failures, "air_disassemble", code, || {
                handle_air_disassemble(&json!({ "binary_base64": b64 }))
            }) {
                let back = serde_json::from_str::<Value>(&text_of(&dis))
                    .ok()
                    .and_then(|v| v["code"].as_str().map(str::to_string))
                    .unwrap_or_default();
                if !is_ok(&dis) || !is_ok(&handle_air_check(&json!({ "code": back }))) {
                    failures.push(format!(
                        "AIRB round trip rejected an accepted program\n--- input ---\n{code}\n--- back ---\n{back}"
                    ));
                }
            }
        } else {
            failures.push(format!(
                "air_assemble rejected an accepted program: {}\n--- input ---\n{code}",
                text_of(&asm)
            ));
        }
    }
    guarded(failures, "air_optimize", code, || {
        handle_air_optimize(&json!({ "code": code }))
    });
    // Run up to 3 functions with random scalar arguments, under small budgets. A crash here
    // ends the test process; the case file holds its input.
    let Ok(module) = load_module_from_code(code) else {
        return;
    };
    let mut funcs = runnable(&module);
    for _ in 0..funcs.len().saturating_sub(3) {
        funcs.remove(rng.below(funcs.len()));
    }
    for (name, arity) in funcs {
        let args: Vec<f64> = (0..arity).map(|_| ARGS[rng.below(ARGS.len())]).collect();
        let _ = std::fs::write(
            &case,
            format!("# air_run func={name} args={args:?}\n{code}"),
        );
        stats.runs += 1;
        guarded(failures, &format!("air_run {name} {args:?}"), code, || {
            handle_air_run(&json!({
                "code": code,
                "func": name,
                "args": args,
                "fuel": 200_000,
                "max_memory_mb": 4,
                "threads": 2,
            }))
        });
    }
}

/// Byte mutations of a program's bytecode: flips, truncation, insertions.
fn mutate_bytes(rng: &mut Rng, bytes: &[u8]) -> Vec<u8> {
    let mut b = bytes.to_vec();
    for _ in 0..1 + rng.below(4) {
        if b.is_empty() {
            break;
        }
        let i = rng.below(b.len());
        match rng.below(4) {
            0 => b[i] ^= 1 << rng.below(8),
            1 => b[i] = rng.next() as u8,
            2 => b.truncate(i.max(5)),
            _ => b.insert(i, rng.next() as u8),
        }
    }
    b
}

#[test]
fn fuzz_mcp_tools() {
    let cases: usize = std::env::var("ACHAINSAW_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let seed: u64 = std::env::var("ACHAINSAW_FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let corpus = examples();
    let vocab = vocabulary(&corpus);
    // Each example's bytecode, for the byte mutations.
    let bytecode: Vec<Vec<u8>> = corpus
        .iter()
        .filter_map(|(_, text)| {
            let asm = handle_air_assemble(&json!({ "code": text }));
            is_ok(&asm).then(|| {
                let b64 = serde_json::from_str::<Value>(&text_of(&asm)).unwrap()["binary_base64"]
                    .as_str()
                    .unwrap()
                    .to_string();
                b64_decode(&b64).unwrap()
            })
        })
        .collect();
    assert!(!bytecode.is_empty());
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut failures: Vec<String> = Vec::new();
    let mut stats = Stats::default();
    // The examples themselves first, then mutations.
    for (_, text) in &corpus {
        exercise(&mut rng, text, &mut failures, &mut stats);
    }
    for c in 0..cases {
        if c % 4 == 3 {
            let b = &bytecode[rng.below(bytecode.len())];
            let code = b64_encode(&mutate_bytes(&mut rng, b));
            exercise(&mut rng, &code, &mut failures, &mut stats);
        } else {
            let (_, text) = &corpus[rng.below(corpus.len())];
            let code = mutate(&mut rng, text, &vocab);
            exercise(&mut rng, &code, &mut failures, &mut stats);
        }
        if failures.len() >= 5 {
            break;
        }
    }
    eprintln!(
        "fuzz: {} examples and {cases} cases (seed {seed}): {} accepted, {} air_run calls",
        corpus.len(),
        stats.accepted,
        stats.runs
    );
    assert!(
        failures.is_empty(),
        "{} failure(s) (seed {seed}):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}
