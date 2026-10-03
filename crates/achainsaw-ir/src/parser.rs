use crate::ast::*;
use crate::diag::{Diagnostic, Span};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::types::Type;

pub struct Parser<'a> {
    _source: &'a str,
    tokens: Vec<Token>,
    cursor: usize,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Result<Self, Diagnostic> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize_all()?;
        Ok(Self {
            _source: source,
            tokens,
            cursor: 0,
        })
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.cursor].kind
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.cursor];
        if self.cursor + 1 < self.tokens.len() {
            self.cursor += 1;
        }
        tok
    }

    fn skip_newlines(&mut self) {
        while self.peek_kind() == &TokenKind::Newline {
            self.advance();
        }
    }

    /// Statements (instructions and terminators) must end at a line break or EOF.
    fn expect_eol(&mut self) -> Result<(), Diagnostic> {
        match self.peek_kind() {
            TokenKind::Newline => {
                self.skip_newlines();
                Ok(())
            }
            TokenKind::Eof => Ok(()),
            other => Err(Diagnostic::error(
                "ERR_EXPECTED_NEWLINE",
                format!("Expected end of line after statement, found {other:?}"),
                self.peek().span,
            )),
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), Diagnostic> {
        match self.peek_kind().clone() {
            TokenKind::Ident(s) | TokenKind::Op(s) => {
                let span = self.advance().span;
                Ok((s, span))
            }
            _ => {
                let span = self.peek().span;
                Err(Diagnostic::error(
                    "ERR_EXPECTED_IDENTIFIER",
                    format!("Expected identifier, found {:?}", self.peek_kind()),
                    span,
                ))
            }
        }
    }

    fn expect(&mut self, expected: TokenKind) -> Result<Span, Diagnostic> {
        if self.peek_kind() == &expected {
            Ok(self.advance().span)
        } else {
            let span = self.peek().span;
            Err(Diagnostic::error(
                "ERR_UNEXPECTED_TOKEN",
                format!("Expected {expected:?}, found {:?}", self.peek_kind()),
                span,
            ))
        }
    }

    fn parse_type(&mut self) -> Result<Type, Diagnostic> {
        match self.peek_kind().clone() {
            TokenKind::Ident(s) => {
                if let Some(ty) = Type::from_str_token(&s) {
                    self.advance();
                    Ok(ty)
                } else {
                    Err(Diagnostic::error(
                        "ERR_UNKNOWN_TYPE",
                        format!("Unknown type: '{s}'. Expected one of: i8, i16, i32, i64, f32, f64, ptr, v128"),
                        self.peek().span,
                    ))
                }
            }
            _ => Err(Diagnostic::error(
                "ERR_EXPECTED_TYPE",
                format!("Expected type, found {:?}", self.peek_kind()),
                self.peek().span,
            )),
        }
    }

    /// Parses `name:type, name:type, ...` up to (not including) the closing `)`.
    fn parse_typed_params(&mut self) -> Result<Vec<(String, Type)>, Diagnostic> {
        let mut params = Vec::new();
        if self.peek_kind() != &TokenKind::RParen {
            loop {
                let (name, _) = self.expect_ident()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                params.push((name, ty));
                if self.peek_kind() == &TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        Ok(params)
    }

    /// Parses `(a, b, c)`, where the parentheses are required.
    fn parse_paren_idents(&mut self) -> Result<Vec<String>, Diagnostic> {
        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        if self.peek_kind() != &TokenKind::RParen {
            loop {
                let (arg, _) = self.expect_ident()?;
                args.push(arg);
                if self.peek_kind() == &TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(args)
    }

    /// Parses `(a, b, c)` if present, otherwise returns an empty list.
    fn parse_optional_paren_idents(&mut self) -> Result<Vec<String>, Diagnostic> {
        if self.peek_kind() == &TokenKind::LParen {
            self.parse_paren_idents()
        } else {
            Ok(Vec::new())
        }
    }

    pub fn parse_module(&mut self) -> Result<Module, Diagnostic> {
        let mut extern_functions = Vec::new();
        let mut functions = Vec::new();
        self.skip_newlines();

        while self.peek_kind() != &TokenKind::Eof {
            if self.peek_kind() == &TokenKind::ExtFn {
                extern_functions.push(self.parse_extern_function()?);
            } else if self.peek_kind() == &TokenKind::Fn {
                functions.push(self.parse_function()?);
            } else {
                return Err(Diagnostic::error(
                    "ERR_UNEXPECTED_TOKEN",
                    format!("Expected 'fn' or 'extfn', found {:?}", self.peek_kind()),
                    self.peek().span,
                ));
            }
            self.skip_newlines();
        }

        Ok(Module {
            extern_functions,
            functions,
        })
    }

    fn parse_signature(
        &mut self,
    ) -> Result<(String, Vec<(String, Type)>, Option<Type>), Diagnostic> {
        let (name, _) = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_typed_params()?;
        self.expect(TokenKind::RParen)?;

        let mut ret_type = None;
        if self.peek_kind() == &TokenKind::Arrow {
            self.advance();
            ret_type = Some(self.parse_type()?);
        }
        Ok((name, params, ret_type))
    }

    fn parse_extern_function(&mut self) -> Result<ExternFunction, Diagnostic> {
        let fn_span = self.expect(TokenKind::ExtFn)?;
        let (name, params, ret_type) = self.parse_signature()?;
        self.skip_newlines();

        Ok(ExternFunction {
            name,
            params,
            ret_type,
            span: fn_span,
        })
    }

    fn parse_function(&mut self) -> Result<Function, Diagnostic> {
        let fn_span = self.expect(TokenKind::Fn)?;
        let (name, params, ret_type) = self.parse_signature()?;
        self.skip_newlines();

        let mut blocks = Vec::new();
        while self.peek_kind() != &TokenKind::Fn
            && self.peek_kind() != &TokenKind::ExtFn
            && self.peek_kind() != &TokenKind::Eof
        {
            blocks.push(self.parse_block()?);
            self.skip_newlines();
        }

        if blocks.is_empty() {
            return Err(Diagnostic::error(
                "ERR_EMPTY_FUNCTION",
                format!("Function '{name}' must have at least one basic block"),
                fn_span,
            ));
        }

        let end_span = blocks.last().map(|b| b.span).unwrap_or(fn_span);
        Ok(Function {
            name,
            params,
            ret_type,
            blocks,
            span: Span {
                start: fn_span.start,
                end: end_span.end,
                line: fn_span.line,
                column: fn_span.column,
            },
        })
    }

    fn parse_block(&mut self) -> Result<Block, Diagnostic> {
        let (label, label_span) = self.expect_ident()?;

        let mut params = Vec::new();
        if self.peek_kind() == &TokenKind::LParen {
            self.advance();
            params = self.parse_typed_params()?;
            self.expect(TokenKind::RParen)?;
        }

        self.expect(TokenKind::Colon)?;
        self.skip_newlines();

        let mut instructions = Vec::new();
        let terminator = loop {
            self.skip_newlines();
            // Check for block terminators
            match self.peek_kind() {
                TokenKind::Jmp => {
                    let span = self.advance().span;
                    let (target, _) = self.expect_ident()?;
                    let args = self.parse_optional_paren_idents()?;
                    self.expect_eol()?;
                    break Terminator::Jmp { target, args, span };
                }
                TokenKind::Br => {
                    let span = self.advance().span;
                    let (cond, _) = self.expect_ident()?;
                    self.expect(TokenKind::Comma)?;
                    let (then_block, _) = self.expect_ident()?;
                    let then_args = self.parse_optional_paren_idents()?;
                    self.expect(TokenKind::Comma)?;
                    let (else_block, _) = self.expect_ident()?;
                    let else_args = self.parse_optional_paren_idents()?;
                    self.expect_eol()?;
                    break Terminator::Br {
                        cond,
                        then_block,
                        then_args,
                        else_block,
                        else_args,
                        span,
                    };
                }
                TokenKind::Ret => {
                    let span = self.advance().span;
                    let val = if self.peek_kind() != &TokenKind::Newline
                        && self.peek_kind() != &TokenKind::Eof
                    {
                        let (v, _) = self.expect_ident()?;
                        Some(v)
                    } else {
                        None
                    };
                    self.expect_eol()?;
                    break Terminator::Ret { val, span };
                }
                TokenKind::Eof => {
                    return Err(Diagnostic::error(
                        "ERR_MISSING_TERMINATOR",
                        format!("Block '{label}' ends without a terminator (jmp, br or ret)"),
                        self.peek().span,
                    ));
                }
                _ => {
                    // Regular instruction
                    instructions.push(self.parse_instruction()?);
                }
            }
        };

        Ok(Block {
            label,
            params,
            instructions,
            terminator,
            span: label_span,
        })
    }

    fn parse_instruction(&mut self) -> Result<Instruction, Diagnostic> {
        let first_tok = self.peek().clone();

        // Check if store: st ptr, val
        if first_tok.kind == TokenKind::St {
            let span = self.advance().span;
            let (ptr, _) = self.expect_ident()?;
            self.expect(TokenKind::Comma)?;
            let (val, _) = self.expect_ident()?;
            self.expect_eol()?;
            return Ok(Instruction::Store { ptr, val, span });
        }

        // Check if free: free ptr
        if first_tok.kind == TokenKind::Free {
            let span = self.advance().span;
            let (ptr, _) = self.expect_ident()?;
            self.expect_eol()?;
            return Ok(Instruction::Free { ptr, span });
        }

        // Check if call without dst: call func(args...)
        if first_tok.kind == TokenKind::Call {
            let span = self.advance().span;
            let (func, _) = self.expect_ident()?;
            let args = self.parse_paren_idents()?;
            self.expect_eol()?;
            return Ok(Instruction::Call {
                dst: None,
                func,
                args,
                span,
            });
        }

        // Otherwise: dst = <op> ...
        let (dst, dst_span) = self.expect_ident()?;
        self.expect(TokenKind::Equal)?;

        match self.peek_kind().clone() {
            TokenKind::Cst => {
                self.advance();
                let val = match self.peek_kind().clone() {
                    TokenKind::IntLit(n) => {
                        self.advance();
                        Constant::Int(n)
                    }
                    TokenKind::FloatLit(f) => {
                        self.advance();
                        Constant::Float(f)
                    }
                    TokenKind::Ident(name) if name == "inf" => {
                        self.advance();
                        Constant::Float(f64::INFINITY)
                    }
                    TokenKind::Ident(name) if name == "nan" => {
                        self.advance();
                        Constant::Float(f64::NAN)
                    }
                    _ => {
                        return Err(Diagnostic::error(
                            "ERR_EXPECTED_NUMBER",
                            format!("Expected numeric literal, found {:?}", self.peek_kind()),
                            self.peek().span,
                        ))
                    }
                };
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                self.expect_eol()?;
                Ok(Instruction::AssignConst {
                    dst,
                    val,
                    ty,
                    span: dst_span,
                })
            }
            TokenKind::Ld => {
                self.advance();
                let (ptr, _) = self.expect_ident()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                self.expect_eol()?;
                Ok(Instruction::Load {
                    dst,
                    ptr,
                    ty,
                    span: dst_span,
                })
            }
            TokenKind::Call => {
                self.advance();
                let (func, _) = self.expect_ident()?;
                let args = self.parse_paren_idents()?;
                self.expect_eol()?;
                Ok(Instruction::Call {
                    dst: Some(dst),
                    func,
                    args,
                    span: dst_span,
                })
            }
            TokenKind::Alloc => {
                self.advance();
                let (size, _) = self.expect_ident()?;
                self.expect_eol()?;
                Ok(Instruction::Alloc {
                    dst,
                    size,
                    span: dst_span,
                })
            }
            TokenKind::Splat => {
                self.advance();
                let (src, _) = self.expect_ident()?;
                self.expect_eol()?;
                Ok(Instruction::Splat {
                    dst,
                    src,
                    span: dst_span,
                })
            }
            TokenKind::Extlane => {
                self.advance();
                let (vec, _) = self.expect_ident()?;
                self.expect(TokenKind::Comma)?;
                let lane = match self.peek_kind() {
                    TokenKind::IntLit(n) => u32::try_from(*n).map_err(|_| {
                        Diagnostic::error(
                            "ERR_EXPECTED_LANE_INDEX",
                            format!("Lane index {n} is out of range"),
                            self.peek().span,
                        )
                    })?,
                    _ => {
                        return Err(Diagnostic::error(
                            "ERR_EXPECTED_LANE_INDEX",
                            format!("Expected lane index integer, found {:?}", self.peek_kind()),
                            self.peek().span,
                        ));
                    }
                };
                self.advance();
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                self.expect_eol()?;
                Ok(Instruction::ExtractLane {
                    dst,
                    vec,
                    lane,
                    ty,
                    span: dst_span,
                })
            }
            TokenKind::Op(op_name) => {
                self.advance();
                let op = op_name.parse::<BinaryOp>().map_err(|_| {
                    Diagnostic::error(
                        "ERR_UNKNOWN_OP",
                        format!("Unknown binary op '{op_name}'"),
                        dst_span,
                    )
                })?;
                let (lhs, _) = self.expect_ident()?;
                self.expect(TokenKind::Comma)?;
                let (rhs, _) = self.expect_ident()?;
                self.expect_eol()?;
                Ok(Instruction::Binary {
                    op,
                    dst,
                    lhs,
                    rhs,
                    span: dst_span,
                })
            }
            _ => Err(Diagnostic::error(
                "ERR_EXPECTED_RVALUE",
                format!("Expected cst, ld, call, alloc, splat, extlane, or binary op after '=', found {:?}", self.peek_kind()),
                self.peek().span,
            )),
        }
    }
}
