//! `par n, f(args)` syntax, validation, optimization, and AIRB encoding.

use achainsaw_ir::ast::Instruction;
use achainsaw_ir::opt::optimize_module;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};

fn diag(src: &str) -> achainsaw_ir::diag::Diagnostic {
    match parse_and_validate(src) {
        Ok(_) => panic!("expected an error for:\n{src}"),
        Err(d) => d,
    }
}

const BODY: &str = "fn body(i:i64, p:ptr, s:f32)\n  b0:\n    ret\n\n";

fn with_main(main: &str) -> String {
    format!("{BODY}fn main(p:ptr, n:i64)\n  b0:\n{main}    ret\n")
}

#[test]
fn par_parses_and_round_trips() {
    let src = with_main("    par n, body(p, 0.5:f32)\n    par 8:i64, body(p, 1.5:f32)\n");
    let module = parse_and_validate(&src).unwrap();
    let text = to_air_text(&module);
    assert!(text.contains("par n, body(p, __imm_0)"), "{text}");
    assert_eq!(to_air_text(&parse_and_validate(&text).unwrap()), text);
    let bytes = encode_module(&module).unwrap();
    assert_eq!(to_air_text(&decode_module(&bytes).unwrap()), text);
    assert!(module.uses_par());
    assert!(!parse_and_validate(BODY).unwrap().uses_par());
}

#[test]
fn par_is_never_removed_by_the_optimizer() {
    let mut module = parse_and_validate(&with_main("    par n, body(p, 0.5:f32)\n")).unwrap();
    optimize_module(&mut module);
    assert!(module.uses_par());
}

#[test]
fn par_remains_a_valid_register_name() {
    let src = "fn k(par:i64)->i64\n  b0:\n    x = add par, 1:i64\n    ret x\n";
    assert!(parse_and_validate(src).is_ok());
    let src = "fn k(a:i64)->i64\n  b0:\n    par = add a, 1:i64\n    ret par\n";
    assert!(parse_and_validate(src).is_ok());
}

#[test]
fn par_diagnostics() {
    let cases = [
        // Count must be i64.
        (
            "    c = cst 4:i32\n    par c, body(p, 0.5:f32)\n",
            "ERR_TYPE_MISMATCH",
        ),
        ("    par n, nope(p)\n", "ERR_UNDEFINED_FUNCTION"),
        ("    par n, body(p)\n", "ERR_ARITY_MISMATCH"),
        ("    par n, body(p, n)\n", "ERR_TYPE_MISMATCH"),
        ("    par n, body(p, 0.5:f32\n", "ERR_UNEXPECTED_TOKEN"),
    ];
    for (main, code) in cases {
        let src = with_main(main);
        assert_eq!(diag(&src).error_code, code, "{src}");
    }
}

#[test]
fn par_body_signature_is_checked() {
    let bodies = [
        // Index must come first and be i64.
        "fn f(i:i32)\n  b0:\n    ret\n",
        "fn f(p:ptr, i:i64)\n  b0:\n    ret\n",
        // No return value.
        "fn f(i:i64)->i64\n  b0:\n    ret i\n",
        // No arguments at all.
        "fn f()\n  b0:\n    ret\n",
    ];
    for body in bodies {
        let src = format!("{body}\nfn main(p:ptr)\n  b0:\n    par 4:i64, f()\n    ret\n");
        let d = diag(&src);
        assert_eq!(d.error_code, "ERR_PAR_SIGNATURE", "{src}");
        let ctx = d.context.expect("context");
        assert_eq!(ctx["expected_signature"], "fn f(i64)", "{src}");
    }

    // External functions may not be thread-safe.
    let src = "extfn sinf(x:f32)->f32\n\nfn main()\n  b0:\n    par 4:i64, sinf()\n    ret\n";
    assert_eq!(diag(src).error_code, "ERR_PAR_SIGNATURE");

    // Vectors cannot pass through the runtime's scalar argument slots.
    let src = "fn f(i:i64, v:v128)\n  b0:\n    ret\n\nfn main(p:ptr)\n  b0:\n    v = ld p:v128\n    par 4:i64, f(v)\n    ret\n";
    assert_eq!(diag(src).error_code, "ERR_PAR_SIGNATURE");
}

#[test]
fn par_has_no_destination() {
    let module = parse_and_validate(&with_main("    par n, body(p, 0.5:f32)\n")).unwrap();
    let par = module.functions[1].blocks[0]
        .instructions
        .iter()
        .find(|i| matches!(i, Instruction::Par { .. }))
        .unwrap();
    assert_eq!(par.dst(), None);
    let operands: Vec<&str> = par.operands().into_iter().map(String::as_str).collect();
    assert_eq!(operands, ["n", "p", "__imm_0"]);
}
