use crate::ast::*;
use crate::diag::{Diagnostic, Span};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::types::Type;

/// Function signature: name, typed params, optional return type.
type Signature = (String, Vec<(String, Type)>, Option<Type>);

pub struct Parser<'a> {
    _source: &'a str,
    tokens: Vec<Token>,
    cursor: usize,
    imm_counter: usize,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Result<Self, Diagnostic> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize_all()?;
        Ok(Self {
            _source: source,
            tokens,
            cursor: 0,
            imm_counter: 0,
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
            TokenKind::Ident(s) | TokenKind::Op(s) | TokenKind::VOp(s) => {
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
                        format!("Unknown type: '{s}'. Expected one of: i8, i16, i32, i64, f32, f64, ptr, v128, v256, v512, vx"),
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

    fn parse_operand(
        &mut self,
        instructions: &mut Vec<Instruction>,
        fallback_ty: Option<Type>,
    ) -> Result<String, Diagnostic> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::IntLit(n) => {
                self.advance();
                let ty = if self.peek_kind() == &TokenKind::Colon {
                    self.advance();
                    self.parse_type()?
                } else {
                    fallback_ty.unwrap_or(Type::I64)
                };
                let imm_reg = format!("__imm_{}", self.imm_counter);
                self.imm_counter += 1;
                instructions.push(Instruction::AssignConst {
                    dst: imm_reg.clone(),
                    val: Constant::Int(n),
                    ty,
                    span: tok.span,
                });
                Ok(imm_reg)
            }
            TokenKind::FloatLit(f) => {
                self.advance();
                let ty = if self.peek_kind() == &TokenKind::Colon {
                    self.advance();
                    self.parse_type()?
                } else {
                    fallback_ty.unwrap_or(Type::F32)
                };
                let imm_reg = format!("__imm_{}", self.imm_counter);
                self.imm_counter += 1;
                instructions.push(Instruction::AssignConst {
                    dst: imm_reg.clone(),
                    val: Constant::Float(f),
                    ty,
                    span: tok.span,
                });
                Ok(imm_reg)
            }
            TokenKind::Ident(ref name) if name == "inf" || name == "nan" => {
                let name = name.clone();
                self.advance();
                let val = if name == "inf" {
                    Constant::Float(f64::INFINITY)
                } else {
                    Constant::Float(f64::NAN)
                };
                let ty = if self.peek_kind() == &TokenKind::Colon {
                    self.advance();
                    self.parse_type()?
                } else {
                    fallback_ty.unwrap_or(Type::F32)
                };
                let imm_reg = format!("__imm_{}", self.imm_counter);
                self.imm_counter += 1;
                instructions.push(Instruction::AssignConst {
                    dst: imm_reg.clone(),
                    val,
                    ty,
                    span: tok.span,
                });
                Ok(imm_reg)
            }
            TokenKind::Ident(name) | TokenKind::VOp(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(Diagnostic::error(
                "ERR_EXPECTED_OPERAND",
                format!("Expected register name or immediate constant, found {other:?}"),
                tok.span,
            )),
        }
    }

    fn parse_paren_operands(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Vec<String>, Diagnostic> {
        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        if self.peek_kind() != &TokenKind::RParen {
            loop {
                let arg = self.parse_operand(instructions, None)?;
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

    fn parse_optional_paren_operands(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Vec<String>, Diagnostic> {
        if self.peek_kind() == &TokenKind::LParen {
            self.parse_paren_operands(instructions)
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

    fn parse_signature(&mut self) -> Result<Signature, Diagnostic> {
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
                    let args = self.parse_optional_paren_operands(&mut instructions)?;
                    self.expect_eol()?;
                    break Terminator::Jmp { target, args, span };
                }
                TokenKind::Br => {
                    let span = self.advance().span;
                    let cond = self.parse_operand(&mut instructions, Some(Type::I32))?;
                    self.expect(TokenKind::Comma)?;
                    let (then_block, _) = self.expect_ident()?;
                    let then_args = self.parse_optional_paren_operands(&mut instructions)?;
                    self.expect(TokenKind::Comma)?;
                    let (else_block, _) = self.expect_ident()?;
                    let else_args = self.parse_optional_paren_operands(&mut instructions)?;
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
                        Some(self.parse_operand(&mut instructions, None)?)
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
                    let inst = self.parse_instruction(&mut instructions)?;
                    instructions.push(inst);
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

    fn peek_at(&self, offset: usize) -> Option<&TokenKind> {
        self.tokens.get(self.cursor + offset).map(|t| &t.kind)
    }

    /// Parses the trailing `:type` of a vector op; `example` shows the expected syntax.
    fn parse_lane_suffix(&mut self, example: &str) -> Result<Type, Diagnostic> {
        if self.peek_kind() != &TokenKind::Colon {
            return Err(Diagnostic::error(
                "ERR_EXPECTED_LANE_TYPE",
                format!("Expected a type suffix, e.g. '{example}'"),
                self.peek().span,
            ));
        }
        self.advance();
        self.parse_type()
    }

    /// Parses an i64 count operand: a register or an integer literal. A literal only takes
    /// an explicit type (`16:i64`) when a comma or a second `:suffix` follows, so the
    /// trailing `:lane` suffix of `ldm`/`stm`/`mm` is never mistaken for the literal's type
    /// (`n:f32` vs `16:i64:f32`).
    fn parse_count_operand(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<String, Diagnostic> {
        let TokenKind::IntLit(n) = self.peek_kind().clone() else {
            return self.parse_operand(instructions, Some(Type::I64));
        };
        let span = self.advance().span;
        let mut ty = Type::I64;
        if self.peek_kind() == &TokenKind::Colon
            && matches!(self.peek_at(2), Some(TokenKind::Comma | TokenKind::Colon))
        {
            if let Some(TokenKind::Ident(t)) = self.peek_at(1) {
                if let Some(explicit) = Type::from_str_token(t) {
                    ty = explicit;
                    self.advance();
                    self.advance();
                }
            }
        }
        let imm_reg = format!("__imm_{}", self.imm_counter);
        self.imm_counter += 1;
        instructions.push(Instruction::AssignConst {
            dst: imm_reg.clone(),
            val: Constant::Int(n),
            ty,
            span,
        });
        Ok(imm_reg)
    }

    /// `stm p, v, n:f32` and `mm pc, pa, pb, m, n, k:bf16` (statements without a result).
    fn parse_vector_statement(
        &mut self,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Instruction, Diagnostic> {
        let span = self.advance().span;
        if name == "stm" {
            let (ptr, _) = self.expect_ident()?;
            self.expect(TokenKind::Comma)?;
            let val = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            let count = self.parse_count_operand(instructions)?;
            let lane = self.parse_lane_suffix("stm p, v, n:f32")?;
            self.expect_eol()?;
            return Ok(Instruction::MaskedStore {
                ptr,
                val,
                count,
                lane,
                span,
            });
        }
        let mut regs = Vec::with_capacity(6);
        for i in 0..6 {
            if i > 0 {
                self.expect(TokenKind::Comma)?;
            }
            regs.push(if i < 3 {
                self.parse_operand(instructions, None)?
            } else {
                self.parse_count_operand(instructions)?
            });
        }
        let dtype = self.parse_lane_suffix("mm pc, pa, pb, m, n, k:bf16")?;
        self.expect_eol()?;
        let mut regs = regs.into_iter();
        let mut next = || regs.next().unwrap();
        Ok(Instruction::MatMul {
            pc: next(),
            pa: next(),
            pb: next(),
            m: next(),
            n: next(),
            k: next(),
            dtype,
            span,
        })
    }

    /// Parses the operands of a vector op whose mnemonic `name` was just consumed.
    fn parse_vector_op(
        &mut self,
        name: &str,
        dst: String,
        span: Span,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Instruction, Diagnostic> {
        if name == "vl" {
            let lane = self.parse_type()?;
            self.expect_eol()?;
            return Ok(Instruction::VLen { dst, lane, span });
        }
        if let Some(op) = VectorReduceOp::from_str_opt(name) {
            let src = self.parse_operand(instructions, Some(Type::V128))?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            self.expect_eol()?;
            return Ok(Instruction::VectorReduce {
                op,
                dst,
                src,
                ty,
                span,
            });
        }
        if name == "vsel" {
            let mask = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            let then_val = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            let else_val = self.parse_operand(instructions, None)?;
            self.expect_eol()?;
            return Ok(Instruction::VSelect {
                dst,
                mask,
                then_val,
                else_val,
                span,
            });
        }

        if name == "ldm" {
            let (ptr, _) = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            self.expect(TokenKind::Comma)?;
            let count = self.parse_count_operand(instructions)?;
            let lane = self.parse_lane_suffix("ldm p:vx, n:f32")?;
            self.expect_eol()?;
            return Ok(Instruction::MaskedLoad {
                dst,
                ptr,
                count,
                ty,
                lane,
                span,
            });
        }

        let arity = if name == "vfma" { 3 } else { 2 };
        let mut regs = Vec::with_capacity(arity);
        for i in 0..arity {
            if i > 0 {
                self.expect(TokenKind::Comma)?;
            }
            regs.push(self.parse_operand(instructions, None)?);
        }
        let lane = self.parse_lane_suffix(&format!("{name} a, b:f32"))?;
        self.expect_eol()?;

        let mut regs = regs.into_iter();
        let mut next = || regs.next().unwrap();
        if name == "vfma" {
            return Ok(Instruction::VFma {
                dst,
                a: next(),
                b: next(),
                c: next(),
                lane,
                span,
            });
        }
        if let Some(op) = VCmpOp::from_str_opt(name) {
            return Ok(Instruction::VCmp {
                op,
                dst,
                lhs: next(),
                rhs: next(),
                lane,
                span,
            });
        }
        let op = VBinOp::from_str_opt(name).ok_or_else(|| {
            Diagnostic::error(
                "ERR_UNKNOWN_OP",
                format!("Unknown vector op '{name}'"),
                span,
            )
        })?;
        Ok(Instruction::VBinary {
            op,
            dst,
            lhs: next(),
            rhs: next(),
            lane,
            span,
        })
    }

    fn parse_instruction(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Instruction, Diagnostic> {
        let first_tok = self.peek().clone();

        // `stm`/`mm` statements; followed by `=` they are ordinary register names.
        if let TokenKind::VOp(name) = &first_tok.kind {
            if (name == "stm" || name == "mm") && self.peek_at(1) != Some(&TokenKind::Equal) {
                let name = name.clone();
                return self.parse_vector_statement(&name, instructions);
            }
        }

        // Check if store: st ptr, val
        if first_tok.kind == TokenKind::St {
            let span = self.advance().span;
            let (ptr, _) = self.expect_ident()?;
            self.expect(TokenKind::Comma)?;
            let val = self.parse_operand(instructions, None)?;
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
            let args = self.parse_paren_operands(instructions)?;
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
                let args = self.parse_paren_operands(instructions)?;
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
                let size = self.parse_operand(instructions, Some(Type::I64))?;
                self.expect_eol()?;
                Ok(Instruction::Alloc {
                    dst,
                    size,
                    span: dst_span,
                })
            }
            TokenKind::Splat => {
                self.advance();
                let src = self.parse_operand(instructions, None)?;
                // The vector type is required: `splat x:v128`, `splat x:vx`.
                if self.peek_kind() != &TokenKind::Colon {
                    return Err(Diagnostic::error(
                        "ERR_EXPECTED_TYPE",
                        "splat needs its vector type, e.g. `v = splat x:v128` or `v = splat x:vx`",
                        self.peek().span,
                    ));
                }
                self.advance();
                let ty = self.parse_type()?;
                self.expect_eol()?;
                Ok(Instruction::Splat {
                    dst,
                    src,
                    ty,
                    span: dst_span,
                })
            }
            TokenKind::VOp(name) => {
                self.advance();
                self.parse_vector_op(&name, dst, dst_span, instructions)
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
            TokenKind::Select => {
                self.advance();
                let cond = self.parse_operand(instructions, Some(Type::I32))?;
                self.expect(TokenKind::Comma)?;
                let then_val = self.parse_operand(instructions, None)?;
                self.expect(TokenKind::Comma)?;
                let else_val = self.parse_operand(instructions, None)?;
                self.expect_eol()?;
                Ok(Instruction::Select {
                    dst,
                    cond,
                    then_val,
                    else_val,
                    span: dst_span,
                })
            }
            TokenKind::Cast(op) => {
                self.advance();
                let src = self.parse_operand(instructions, None)?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                self.expect_eol()?;
                Ok(Instruction::Cast {
                    op,
                    dst,
                    src,
                    ty,
                    span: dst_span,
                })
            }
            TokenKind::Unary(op) => {
                self.advance();
                let src = self.parse_operand(instructions, None)?;
                self.expect_eol()?;
                Ok(Instruction::Unary {
                    op,
                    dst,
                    src,
                    span: dst_span,
                })
            }
            TokenKind::Op(op_name) => {
                let op = op_name.parse::<BinaryOp>().map_err(|_| {
                    Diagnostic::error(
                        "ERR_UNKNOWN_OP",
                        format!("Unknown binary op '{op_name}'"),
                        dst_span,
                    )
                })?;
                self.advance();
                let lhs = self.parse_operand(instructions, None)?;
                self.expect(TokenKind::Comma)?;
                let rhs = self.parse_operand(instructions, None)?;
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
                format!("Expected cst, ld, call, alloc, splat, extlane, select, cast, unary, binary op, or vector op after '=', found {:?}", self.peek_kind()),
                self.peek().span,
            )),
        }
    }
}
