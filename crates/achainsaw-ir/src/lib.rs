pub mod ast;
pub mod binary;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod types;
pub mod validator;

pub use ast::*;
pub use binary::{decode_module, encode_module, to_air_text};
pub use diag::{Diagnostic, Span};
pub use lexer::Lexer;
pub use parser::Parser;
pub use types::Type;
pub use validator::Validator;

pub fn parse_and_validate(source: &str) -> Result<Module, Diagnostic> {
    let mut parser = Parser::new(source)?;
    let module = parser.parse_module()?;
    let mut validator = Validator::new();
    validator.validate_module(&module)?;
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_arithmetic() {
        let code = r#"
fn add3(a:i32, b:i32, c:i32)->i32
  b0:
    t0 = add a, b
    t1 = add t0, c
    ret t1
"#;
        let module = parse_and_validate(code).expect("Should parse and validate");
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "add3");
    }

    #[test]
    fn test_valid_loop() {
        let code = r#"
fn sum_loop(n:i32)->i32
  b0:
    zero = cst 0:i32
    jmp b1(zero, zero)
  b1(i:i32, acc:i32):
    cond = lt i, n
    br cond, b2, b3
  b2:
    new_acc = add acc, i
    one = cst 1:i32
    new_i = add i, one
    jmp b1(new_i, new_acc)
  b3:
    ret acc
"#;
        let module = parse_and_validate(code).expect("Loop should validate");
        assert_eq!(module.functions[0].blocks.len(), 4);
    }

    #[test]
    fn test_extern_function_declaration_and_call() {
        let code = r#"
extfn sinf(x:f32)->f32
extfn puts(s:ptr)->i32

fn compute_wave(freq:f32)->f32
  b0:
    res = call sinf(freq)
    ret res
"#;
        let module = parse_and_validate(code).expect("External function call should validate");
        assert_eq!(module.extern_functions.len(), 2);
        assert_eq!(module.extern_functions[0].name, "sinf");
        assert_eq!(module.functions.len(), 1);

        // Binary roundtrip
        let bytes = encode_module(&module);
        let decoded = decode_module(&bytes).expect("Should decode with extern_functions");
        assert_eq!(decoded.extern_functions.len(), 2);
        assert_eq!(decoded.extern_functions[0].name, "sinf");

        let text = to_air_text(&decoded);
        assert!(text.contains("extfn sinf(x:f32)->f32"));
        assert!(text.contains("extfn puts(s:ptr)->i32"));
    }

    #[test]
    fn test_binary_encode_decode_roundtrip() {
        let code = r#"
fn simd_scale(p:ptr, factor:f32, n:i64)
  b0:
    vfactor = splat factor
    zero = cst 0:i64
    jmp b1(zero)
  b1(i:i64):
    cond = lt i, n
    br cond, b2, b3
  b2:
    sixteen = cst 16:i64
    off = mul i, sixteen
    elem_ptr = add p, off
    v = ld elem_ptr:v128
    vscaled = vfmul v, vfactor
    st elem_ptr, vscaled
    one = cst 1:i64
    next_i = add i, one
    jmp b1(next_i)
  b3:
    ret
"#;
        let original_module = parse_and_validate(code).expect("Valid AIR");
        let bytes = encode_module(&original_module);

        // Verify magic and minimum size
        assert!(bytes.starts_with(b"\x00AIR"));
        println!("bytes.len() = {}, text.len() = {}", bytes.len(), code.len());
        assert!(bytes.len() < 500, "AIRB binary should be compact: {}", bytes.len());

        // Decode back
        let decoded_module = decode_module(&bytes).expect("Should decode AIRB cleanly");
        assert_eq!(decoded_module.functions.len(), original_module.functions.len());
        assert_eq!(decoded_module.functions[0].name, original_module.functions[0].name);
        assert_eq!(decoded_module.functions[0].blocks.len(), original_module.functions[0].blocks.len());

        // Validate decoded module
        let mut validator = Validator::new();
        validator.validate_module(&decoded_module).expect("Decoded module should be valid SSA");

        // Disassemble back to text and re-validate
        let disassembled_text = to_air_text(&decoded_module);
        let roundtrip_module = parse_and_validate(&disassembled_text).expect("Disassembled text must validate");
        assert_eq!(roundtrip_module.functions[0].name, "simd_scale");
    }

    #[test]
    fn test_invalid_binary_magic() {
        let bad_bytes = b"BADMAGICDATA";
        let res = decode_module(bad_bytes);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.error_code, "ERR_INVALID_AIRB");
    }

    #[test]
    fn test_diagnostic_undefined_reg() {
        let code = r#"
fn fail(a:i32)->i32
  b0:
    res = add a, undefined_var
    ret res
"#;
        let err = parse_and_validate(code).unwrap_err();
        assert_eq!(err.error_code, "ERR_UNDEFINED_REG");
        assert!(err.context.is_some());
        let json = err.to_json();
        assert!(json.contains("ERR_UNDEFINED_REG"));
    }

    #[test]
    fn test_diagnostic_type_mismatch() {
        let code = r#"
fn fail(a:i32, b:f32)->i32
  b0:
    res = add a, b
    ret res
"#;
        let err = parse_and_validate(code).unwrap_err();
        assert_eq!(err.error_code, "ERR_TYPE_MISMATCH");
    }

    #[test]
    fn test_diagnostic_ssa_violation() {
        let code = r#"
fn fail(a:i32)->i32
  b0:
    x = cst 1:i32
    x = cst 2:i32
    ret x
"#;
        let err = parse_and_validate(code).unwrap_err();
        assert_eq!(err.error_code, "ERR_SSA_REDEFINITION");
    }
}


