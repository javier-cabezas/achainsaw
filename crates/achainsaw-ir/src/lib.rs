pub mod ast;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod types;
pub mod validator;

pub use ast::*;
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

