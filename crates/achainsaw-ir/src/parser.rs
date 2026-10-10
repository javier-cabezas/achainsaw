use crate::ast::*;
use crate::diag::{Diagnostic, Span};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::types::Type;

/// Function signature: name, typed params, optional return type.
type Signature = (String, Vec<(String, Type)>, Vec<Type>);

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
        let mut uses = Vec::new();
        self.skip_newlines();

        while self.peek_kind() != &TokenKind::Eof {
            if matches!(self.peek_kind(), TokenKind::Ident(s) if s == "use") {
                // `use name, ...`: standard library functions to link in (see `stdlib`).
                self.advance();
                loop {
                    uses.push(self.expect_ident()?);
                    if self.peek_kind() != &TokenKind::Comma {
                        break;
                    }
                    self.advance();
                }
                self.expect_eol()?;
            } else if self.peek_kind() == &TokenKind::ExtFn {
                extern_functions.push(self.parse_extern_function()?);
            } else if self.peek_kind() == &TokenKind::Fn {
                functions.push(self.parse_function()?);
            } else if self.at_inline_fn() {
                self.advance();
                let mut f = self.parse_function()?;
                f.inline = true;
                functions.push(f);
            } else {
                return Err(Diagnostic::error(
                    "ERR_UNEXPECTED_TOKEN",
                    format!(
                        "Expected 'fn', 'inline fn', 'extfn' or 'use', found {:?}",
                        self.peek_kind()
                    ),
                    self.peek().span,
                ));
            }
            self.skip_newlines();
        }

        let mut module = Module {
            extern_functions,
            functions,
        };
        crate::stdlib::link(&mut module, &uses)?;
        Ok(module)
    }

    /// Whether the next tokens are `inline fn`.
    fn at_inline_fn(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(s) if s == "inline")
            && self.peek_at(1) == Some(&TokenKind::Fn)
    }

    /// `name(params)`, then `->ty`, `->(ty, ty, ...)` or nothing.
    fn parse_signature(&mut self) -> Result<Signature, Diagnostic> {
        let (name, _) = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_typed_params()?;
        self.expect(TokenKind::RParen)?;

        let mut rets = Vec::new();
        if self.peek_kind() == &TokenKind::Arrow {
            self.advance();
            if self.peek_kind() == &TokenKind::LParen {
                self.advance();
                loop {
                    rets.push(self.parse_type()?);
                    if self.peek_kind() != &TokenKind::Comma {
                        break;
                    }
                    self.advance();
                }
                self.expect(TokenKind::RParen)?;
            } else {
                rets.push(self.parse_type()?);
            }
        }
        Ok((name, params, rets))
    }

    fn parse_extern_function(&mut self) -> Result<ExternFunction, Diagnostic> {
        let fn_span = self.expect(TokenKind::ExtFn)?;
        let (name, params, rets) = self.parse_signature()?;
        if rets.len() > 1 {
            return Err(Diagnostic::error(
                "ERR_MULTI_RETURN_EXTERN",
                format!(
                    "External function '{name}' returns {} values; C functions return at most one",
                    rets.len()
                ),
                fn_span,
            ));
        }
        self.skip_newlines();

        Ok(ExternFunction {
            name,
            params,
            ret_type: rets.first().copied(),
            span: fn_span,
        })
    }

    fn parse_function(&mut self) -> Result<Function, Diagnostic> {
        let fn_span = self.expect(TokenKind::Fn)?;
        let (name, params, rets) = self.parse_signature()?;
        self.skip_newlines();

        let mut blocks = Vec::new();
        while self.peek_kind() != &TokenKind::Fn
            && self.peek_kind() != &TokenKind::ExtFn
            && self.peek_kind() != &TokenKind::Eof
            && !self.at_inline_fn()
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
            rets,
            inline: false,
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
                    let mut vals = Vec::new();
                    if self.peek_kind() != &TokenKind::Newline
                        && self.peek_kind() != &TokenKind::Eof
                    {
                        loop {
                            vals.push(self.parse_operand(&mut instructions, None)?);
                            if self.peek_kind() != &TokenKind::Comma {
                                break;
                            }
                            self.advance();
                        }
                    }
                    self.expect_eol()?;
                    break Terminator::Ret { vals, span };
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
            let (ptr, index) = self.parse_mem_ref(instructions)?;
            self.expect(TokenKind::Comma)?;
            let val = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            let count = self.parse_count_operand(instructions)?;
            let lane = self.parse_lane_suffix("stm p, v, n:f32")?;
            self.expect_eol()?;
            return Ok(Instruction::MaskedStore {
                ptr,
                index,
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

    /// Parses a memory operand: a pointer register, optionally indexed as `p[i]` or
    /// `p[i:unit]` (the index an i64 register or integer literal).
    fn parse_mem_ref(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(String, Option<Index>), Diagnostic> {
        let (ptr, _) = self.expect_ident()?;
        if self.peek_kind() != &TokenKind::LBracket {
            return Ok((ptr, None));
        }
        self.advance();
        let reg = match self.peek_kind().clone() {
            // A literal index is an i64; a `:` after it names the unit, not its type.
            TokenKind::IntLit(n) => {
                let span = self.advance().span;
                let imm = format!("__imm_{}", self.imm_counter);
                self.imm_counter += 1;
                instructions.push(Instruction::AssignConst {
                    dst: imm.clone(),
                    val: Constant::Int(n),
                    ty: Type::I64,
                    span,
                });
                imm
            }
            _ => self.expect_ident()?.0,
        };
        let unit = if self.peek_kind() == &TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokenKind::RBracket)?;
        Ok((ptr, Some(Index { reg, unit })))
    }

    /// Mnemonics recognized only after `=` (so older programs may use them as registers).
    fn is_contextual_op(name: &str) -> bool {
        matches!(
            name,
            "floor"
                | "ceil"
                | "round"
                | "roundeven"
                | "roundz"
                | "popcnt"
                | "clz"
                | "ctz"
                | "copysign"
                | "rotl"
                | "rotr"
                | "fma"
                | "uitof"
                | "ftoui"
        )
    }

    /// Parses `v, 3:f32` (the operands of `extlane` and `vdup`) up to the end of the line.
    fn parse_lane_ref(&mut self) -> Result<(String, u32, Type), Diagnostic> {
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
        Ok((vec, lane, ty))
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
        if let Some(op) = VUnaryOp::from_str_opt(name) {
            let src = self.parse_operand(instructions, None)?;
            let lane = self.parse_lane_suffix(&format!("{name} v:f32"))?;
            self.expect_eol()?;
            return Ok(Instruction::VUnary {
                op,
                dst,
                src,
                lane,
                span,
            });
        }
        if let Some(op) = VShiftOp::from_str_opt(name) {
            let src = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            // A literal amount defaults to i64; `:lane` after it is the lane type.
            let amount = self.parse_count_operand(instructions)?;
            let lane = self.parse_lane_suffix(&format!("{name} v, n:i32"))?;
            self.expect_eol()?;
            return Ok(Instruction::VShift {
                op,
                dst,
                src,
                amount,
                lane,
                span,
            });
        }
        if name == "vdup" {
            let (vec, lane, ty) = self.parse_lane_ref()?;
            return Ok(Instruction::VDup {
                dst,
                vec,
                lane,
                ty,
                span,
            });
        }
        if let Some(op) = VZipOp::from_str_opt(name) {
            let lhs = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            let rhs = self.parse_operand(instructions, None)?;
            let lane = self.parse_lane_suffix(&format!("{name} a, b:f32"))?;
            self.expect_eol()?;
            return Ok(Instruction::VZip {
                op,
                dst,
                lhs,
                rhs,
                lane,
                span,
            });
        }
        if name == "vnarrow" {
            let lo = self.parse_operand(instructions, None)?;
            self.expect(TokenKind::Comma)?;
            let hi = self.parse_operand(instructions, None)?;
            let lane = self.parse_lane_suffix("vnarrow a, b:i8")?;
            self.expect_eol()?;
            return Ok(Instruction::VNarrow {
                dst,
                lo,
                hi,
                lane,
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
            let (ptr, index) = self.parse_mem_ref(instructions)?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            self.expect(TokenKind::Comma)?;
            let count = self.parse_count_operand(instructions)?;
            let lane = self.parse_lane_suffix("ldm p:vx, n:f32")?;
            self.expect_eol()?;
            return Ok(Instruction::MaskedLoad {
                dst,
                ptr,
                index,
                count,
                ty,
                lane,
                span,
            });
        }

        let arity = if name == "vfma" || name == "vdot" || name == "vdotu" {
            3
        } else {
            2
        };
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
        if name == "vdot" || name == "vdotu" {
            return Ok(Instruction::VDot {
                dst,
                acc: next(),
                a: next(),
                b: next(),
                lane,
                unsigned: name == "vdotu",
                span,
            });
        }
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

        // `par n, f(args...)`; followed by `=` it is an ordinary register name.
        if matches!(&first_tok.kind, TokenKind::Ident(name) if name == "par")
            && self.peek_at(1) != Some(&TokenKind::Equal)
        {
            let span = self.advance().span;
            let count = self.parse_operand(instructions, Some(Type::I64))?;
            self.expect(TokenKind::Comma)?;
            let (func, _) = self.expect_ident()?;
            let args = self.parse_paren_operands(instructions)?;
            self.expect_eol()?;
            return Ok(Instruction::Par {
                count,
                func,
                args,
                span,
            });
        }

        // Check if store: st ptr, val
        if first_tok.kind == TokenKind::St {
            let span = self.advance().span;
            let (ptr, index) = self.parse_mem_ref(instructions)?;
            self.expect(TokenKind::Comma)?;
            let val = self.parse_operand(instructions, None)?;
            self.expect_eol()?;
            return Ok(Instruction::Store {
                ptr,
                index,
                val,
                span,
            });
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
                dsts: Vec::new(),
                func,
                args,
                span,
            });
        }

        // Otherwise: dst = <op> ..., or `a, b = call f(...)` for several results.
        let (dst, dst_span) = self.expect_ident()?;
        if self.peek_kind() == &TokenKind::Comma {
            let mut dsts = vec![dst];
            while self.peek_kind() == &TokenKind::Comma {
                self.advance();
                dsts.push(self.expect_ident()?.0);
            }
            self.expect(TokenKind::Equal)?;
            if self.peek_kind() != &TokenKind::Call {
                return Err(Diagnostic::error(
                    "ERR_EXPECTED_RVALUE",
                    format!(
                        "Only 'call' assigns several registers ({}), found {:?}",
                        dsts.join(", "),
                        self.peek_kind()
                    ),
                    self.peek().span,
                ));
            }
            self.advance();
            let (func, _) = self.expect_ident()?;
            let args = self.parse_paren_operands(instructions)?;
            self.expect_eol()?;
            return Ok(Instruction::Call {
                dsts,
                func,
                args,
                span: dst_span,
            });
        }
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
                let (ptr, index) = self.parse_mem_ref(instructions)?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                self.expect_eol()?;
                Ok(Instruction::Load {
                    dst,
                    ptr,
                    index,
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
                    dsts: vec![dst],
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
                let (vec, lane, ty) = self.parse_lane_ref()?;
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
            // Scalar ops added after the reserved mnemonics are contextual: an identifier
            // right after `=` names an op, and stays a valid register name elsewhere.
            TokenKind::Ident(name) if Self::is_contextual_op(&name) => {
                self.advance();
                if let Some(op) = UnaryOp::from_str_opt(&name) {
                    let src = self.parse_operand(instructions, None)?;
                    self.expect_eol()?;
                    return Ok(Instruction::Unary {
                        op,
                        dst,
                        src,
                        span: dst_span,
                    });
                }
                if let Some(op) = CastOp::from_str_opt(&name) {
                    let src = self.parse_operand(instructions, None)?;
                    self.expect(TokenKind::Colon)?;
                    let ty = self.parse_type()?;
                    self.expect_eol()?;
                    return Ok(Instruction::Cast {
                        op,
                        dst,
                        src,
                        ty,
                        span: dst_span,
                    });
                }
                let lhs = self.parse_operand(instructions, None)?;
                self.expect(TokenKind::Comma)?;
                let rhs = self.parse_operand(instructions, None)?;
                if name == "fma" {
                    self.expect(TokenKind::Comma)?;
                    let c = self.parse_operand(instructions, None)?;
                    self.expect_eol()?;
                    return Ok(Instruction::Fma {
                        dst,
                        a: lhs,
                        b: rhs,
                        c,
                        span: dst_span,
                    });
                }
                self.expect_eol()?;
                Ok(Instruction::Binary {
                    op: BinaryOp::from_str_opt(&name).expect("contextual binary op"),
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
