//! The AIR standard library: functions written in AIR (`std/std.air`, compiled into the
//! toolchain) that a module imports with a top-level `use` line:
//!
//! ```text
//! use quantize_q8, qmat_chunk
//! ```
//!
//! Linking happens when the module is parsed: each named function, and every library function
//! it calls or runs with `par`, is copied into the module, which then holds ordinary functions.
//! So imported code is validated, encoded in AIRB, printed by `to_air_text` and compiled by
//! every backend like the module's own.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use serde_json::json;

use crate::ast::{Function, Instruction, Module};
use crate::diag::{Diagnostic, Span};

/// Source of the standard library.
pub const SOURCE: &str = include_str!("../std/std.air");

/// The library, parsed and validated once.
fn library() -> &'static Module {
    static LIBRARY: OnceLock<Module> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        crate::parse_and_validate(SOURCE)
            .unwrap_or_else(|d| panic!("the AIR standard library must validate: {d:?}"))
    })
}

/// A library function, for listings: its name, signature line and documentation comment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StdFunction {
    pub name: String,
    pub signature: String,
    pub doc: String,
}

/// Every library function, in source order, with the comment lines right above it as its
/// documentation.
pub fn functions() -> Vec<StdFunction> {
    let lines: Vec<&str> = SOURCE.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("fn ") else {
            continue;
        };
        let name = rest.split('(').next().unwrap_or_default().to_string();
        let mut doc = Vec::new();
        for prev in lines[..i].iter().rev() {
            match prev.strip_prefix('#') {
                Some(text) => doc.push(text.strip_prefix(' ').unwrap_or(text)),
                None => break,
            }
        }
        doc.reverse();
        out.push(StdFunction {
            name,
            signature: line.to_string(),
            doc: doc.join("\n"),
        });
    }
    out
}

/// The AIR source of library function `name`: its documentation comment and definition.
pub fn source_of(name: &str) -> Option<String> {
    let lines: Vec<&str> = SOURCE.lines().collect();
    let start = lines.iter().position(|l| {
        l.strip_prefix("fn ")
            .is_some_and(|r| r.starts_with(&format!("{name}(")))
    })?;
    let mut first = start;
    while first > 0 && lines[first - 1].starts_with('#') {
        first -= 1;
    }
    let mut end = start + 1;
    while end < lines.len() && (lines[end].starts_with(' ') || lines[end].is_empty()) {
        if lines[end].is_empty() && lines.get(end + 1).is_none_or(|l| !l.starts_with(' ')) {
            break;
        }
        end += 1;
    }
    Some(lines[first..end].join("\n") + "\n")
}

/// Functions that `f` calls or runs with `par`.
fn callees(f: &Function) -> impl Iterator<Item = &str> {
    f.blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i {
            Instruction::Call { func, .. } | Instruction::Par { func, .. } => Some(func.as_str()),
            _ => None,
        })
}

/// Adds the library functions named by `uses` (each with the span of its `use` line), and
/// the library functions they depend on, to `module`.
///
/// Errors: `ERR_UNKNOWN_STD_FUNCTION` for a name the library does not have (the context lists
/// the ones it has), and `ERR_DUPLICATE_FUNCTION` when the module itself defines or declares a
/// function the import brings in.
pub fn link(module: &mut Module, uses: &[(String, Span)]) -> Result<(), Diagnostic> {
    if uses.is_empty() {
        return Ok(());
    }
    let lib = library();
    let by_name: HashMap<&str, &Function> =
        lib.functions.iter().map(|f| (f.name.as_str(), f)).collect();
    // Which `use` brings in each function (directly or through a dependency).
    let mut wanted: HashMap<&str, (&str, Span)> = HashMap::new();
    for (name, span) in uses {
        if !by_name.contains_key(name.as_str()) {
            let available: Vec<&str> = lib.functions.iter().map(|f| f.name.as_str()).collect();
            return Err(Diagnostic::error(
                "ERR_UNKNOWN_STD_FUNCTION",
                format!(
                    "'{name}' is not in the standard library; available: {}",
                    available.join(", ")
                ),
                *span,
            )
            .with_context(json!({ "name": name, "available": available })));
        }
        let mut stack = vec![name.as_str()];
        while let Some(n) = stack.pop() {
            if wanted.contains_key(n) {
                continue;
            }
            wanted.insert(n, (name.as_str(), *span));
            stack.extend(callees(by_name[n]).filter(|c| by_name.contains_key(c)));
        }
    }
    let declared: BTreeSet<&str> = module
        .functions
        .iter()
        .map(|f| f.name.as_str())
        .chain(module.extern_functions.iter().map(|f| f.name.as_str()))
        .collect();
    let mut imported = Vec::new();
    for f in &lib.functions {
        let Some(&(via, span)) = wanted.get(f.name.as_str()) else {
            continue;
        };
        if declared.contains(f.name.as_str()) {
            let how = if via == f.name {
                String::new()
            } else {
                format!(" (needed by '{via}')")
            };
            return Err(Diagnostic::error(
                "ERR_DUPLICATE_FUNCTION",
                format!(
                    "'{}' is defined in this module and also imported from the standard library{how}; rename the module's function",
                    f.name
                ),
                span,
            )
            .with_context(json!({ "function": f.name, "imported_by": via })));
        }
        imported.push(f.clone());
    }
    module.functions.extend(imported);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_module, encode_module, parse_and_validate, to_air_text};

    #[test]
    fn library_validates_and_lists_documented_functions() {
        let lib = library();
        let listed = functions();
        assert_eq!(listed.len(), lib.functions.len());
        for (f, l) in lib.functions.iter().zip(&listed) {
            assert_eq!(f.name, l.name);
            assert!(!l.doc.is_empty(), "{} has no documentation comment", f.name);
            assert!(l.signature.starts_with(&format!("fn {}(", f.name)));
        }
    }

    #[test]
    fn source_of_returns_each_function_whole() {
        for f in functions() {
            let src = source_of(&f.name).unwrap();
            assert!(src.contains(&f.signature));
            // It parses alone unless it calls other library functions, which `use` links.
            let calls_library = src.contains("call qmat_chunk");
            if !calls_library {
                let module =
                    parse_and_validate(&src).unwrap_or_else(|d| panic!("{}: {d:?}", f.name));
                assert_eq!(module.functions.len(), 1, "{}", f.name);
            }
            let module = parse_and_validate(&format!("use {}\n", f.name))
                .unwrap_or_else(|d| panic!("{}: {d:?}", f.name));
            assert!(module.functions.iter().any(|m| m.name == f.name));
            assert_eq!(module.functions.len() > 1, calls_library, "{}", f.name);
        }
        assert!(source_of("nope").is_none());
    }

    #[test]
    fn use_links_functions_and_their_dependencies() {
        let src = "use exp_shift\n\nfn main(p:ptr, n:i64)->f32\n  b0:\n    s = call exp_shift(p, n, 1.0:f32)\n    ret s\n";
        let module = parse_and_validate(src).unwrap();
        let names: Vec<&str> = module.functions.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["main", "exp_shift"]);
        // The linked module is ordinary AIR: text and AIRB round-trip it.
        let text = to_air_text(&module);
        assert!(text.contains("fn exp_shift(p:ptr, n:i64, shift:f32)->f32"));
        assert!(!text.contains("use "));
        assert_eq!(to_air_text(&parse_and_validate(&text).unwrap()), text);
        let decoded = decode_module(&encode_module(&module).unwrap()).unwrap();
        assert_eq!(to_air_text(&decoded), text);
        // `use` is still a register name.
        let src = "fn f(use:i32)->i32\n  b0:\n    r = add use, use\n    ret r\n";
        assert!(parse_and_validate(src).is_ok());
    }

    #[test]
    fn use_errors_are_diagnostics() {
        let d = parse_and_validate("use nope\nfn f()\n  b0:\n    ret\n").unwrap_err();
        assert_eq!(d.error_code, "ERR_UNKNOWN_STD_FUNCTION");
        let ctx = d.context.unwrap();
        assert!(ctx["available"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n == "quantize_q8"));
        assert_eq!(d.span.line, 1);

        let src = "use silu_mul\n\nfn silu_mul(a:ptr, b:ptr, c:ptr, n:i64)\n  b0:\n    ret\n";
        let d = parse_and_validate(src).unwrap_err();
        assert_eq!(d.error_code, "ERR_DUPLICATE_FUNCTION");
        assert_eq!(d.context.unwrap()["function"], "silu_mul");

        let d = parse_and_validate("use quantize_q8,\nfn f()\n  b0:\n    ret\n").unwrap_err();
        assert_eq!(d.error_code, "ERR_EXPECTED_IDENTIFIER");
    }
}
