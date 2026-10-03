use crate::diag::{Diagnostic, Span};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords / Instructions
    Fn,
    ExtFn,
    Cst,
    Ld,
    St,
    Call,
    Jmp,
    Br,
    Ret,
    Splat,
    Extlane,
    Alloc,
    Free,

    // Operators
    Op(String),

    // Phase 4 Keywords
    Select,
    Cast(crate::ast::CastOp),
    Unary(crate::ast::UnaryOp),
    VectorReduce(crate::ast::VectorReduceOp),

    // Symbols
    Colon,
    Comma,
    Equal,
    LParen,
    RParen,
    Arrow,

    // Literals & Identifiers
    Ident(String),
    IntLit(i64),
    FloatLit(f64),

    // Structural
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub struct Lexer<'a> {
    source: &'a str,
    chars: Vec<(usize, char)>,
    cursor: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        let chars = source.char_indices().collect();
        Self {
            source,
            chars,
            cursor: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.cursor).map(|&(_, c)| c)
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.get(self.cursor + 1).map(|&(_, c)| c)
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(&(_, c)) = self.chars.get(self.cursor) {
            self.cursor += 1;
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            Some(c)
        } else {
            None
        }
    }

    fn current_byte_pos(&self) -> usize {
        self.chars
            .get(self.cursor)
            .map(|&(pos, _)| pos)
            .unwrap_or(self.source.len())
    }

    pub fn next_token(&mut self) -> Result<Token, Diagnostic> {
        // Skip horizontal whitespace and comments
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\r' {
                self.advance();
            } else if c == '#' || (c == '/' && self.peek_next() == Some('/')) {
                // Line comment
                while let Some(nc) = self.peek() {
                    if nc == '\n' {
                        break;
                    }
                    self.advance();
                }
            } else {
                break;
            }
        }

        let start_pos = self.current_byte_pos();
        let start_line = self.line;
        let start_col = self.col;

        let Some(c) = self.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                span: Span {
                    start: start_pos,
                    end: start_pos,
                    line: start_line,
                    column: start_col,
                },
            });
        };

        if c == '\n' {
            self.advance();
            return Ok(Token {
                kind: TokenKind::Newline,
                span: Span {
                    start: start_pos,
                    end: self.current_byte_pos(),
                    line: start_line,
                    column: start_col,
                },
            });
        }

        if c == '-' && self.peek_next() == Some('>') {
            self.advance();
            self.advance();
            return Ok(Token {
                kind: TokenKind::Arrow,
                span: Span {
                    start: start_pos,
                    end: self.current_byte_pos(),
                    line: start_line,
                    column: start_col,
                },
            });
        }

        // Single punctuation
        let single_kind = match c {
            ':' => Some(TokenKind::Colon),
            ',' => Some(TokenKind::Comma),
            '=' => Some(TokenKind::Equal),
            '(' => Some(TokenKind::LParen),
            ')' => Some(TokenKind::RParen),
            _ => None,
        };

        if let Some(kind) = single_kind {
            self.advance();
            return Ok(Token {
                kind,
                span: Span {
                    start: start_pos,
                    end: self.current_byte_pos(),
                    line: start_line,
                    column: start_col,
                },
            });
        }

        // `-inf` (a register can never start with '-', so this is unambiguous)
        if c == '-' && self.peek_next() == Some('i') {
            self.advance();
            let mut ident = String::new();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ident.push(self.advance().unwrap());
                } else {
                    break;
                }
            }
            let span = Span {
                start: start_pos,
                end: self.current_byte_pos(),
                line: start_line,
                column: start_col,
            };
            return if ident == "inf" {
                Ok(Token {
                    kind: TokenKind::FloatLit(f64::NEG_INFINITY),
                    span,
                })
            } else {
                Err(Diagnostic::error(
                    "ERR_LEXICAL",
                    format!("Unexpected token '-{ident}' at {start_line}:{start_col}"),
                    span,
                ))
            };
        }

        // Number (integer or float, signed or unsigned)
        if c.is_ascii_digit()
            || (c == '-' && self.peek_next().is_some_and(|next| next.is_ascii_digit()))
        {
            let mut num_str = String::new();
            if c == '-' {
                num_str.push(self.advance().unwrap());
            }
            let mut is_float = false;
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    num_str.push(self.advance().unwrap());
                } else if ch == '.'
                    && !is_float
                    && self.peek_next().is_some_and(|next| next.is_ascii_digit())
                {
                    is_float = true;
                    num_str.push(self.advance().unwrap());
                } else if (ch == 'e' || ch == 'E') && self.exponent_follows() {
                    is_float = true;
                    num_str.push(self.advance().unwrap());
                    if matches!(self.peek(), Some('+') | Some('-')) {
                        num_str.push(self.advance().unwrap());
                    }
                    while let Some(d) = self.peek() {
                        if d.is_ascii_digit() {
                            num_str.push(self.advance().unwrap());
                        } else {
                            break;
                        }
                    }
                    break;
                } else {
                    break;
                }
            }

            let span = Span {
                start: start_pos,
                end: self.current_byte_pos(),
                line: start_line,
                column: start_col,
            };

            return if is_float {
                let val: f64 = num_str.parse().map_err(|e| {
                    Diagnostic::error("ERR_LEXICAL", format!("Invalid float {num_str}: {e}"), span)
                })?;
                Ok(Token {
                    kind: TokenKind::FloatLit(val),
                    span,
                })
            } else {
                let val: i64 = num_str.parse().map_err(|e| {
                    Diagnostic::error(
                        "ERR_LEXICAL",
                        format!("Invalid integer {num_str}: {e}"),
                        span,
                    )
                })?;
                Ok(Token {
                    kind: TokenKind::IntLit(val),
                    span,
                })
            };
        }

        // Identifiers / Keywords / Ops
        if c.is_ascii_alphanumeric() || c == '_' {
            let mut ident = String::new();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ident.push(self.advance().unwrap());
                } else {
                    break;
                }
            }

            let kind = match ident.as_str() {
                "fn" => TokenKind::Fn,
                "extfn" => TokenKind::ExtFn,
                "cst" => TokenKind::Cst,
                "ld" => TokenKind::Ld,
                "st" => TokenKind::St,
                "call" => TokenKind::Call,
                "jmp" => TokenKind::Jmp,
                "br" => TokenKind::Br,
                "ret" => TokenKind::Ret,
                "splat" => TokenKind::Splat,
                "extlane" => TokenKind::Extlane,
                "alloc" => TokenKind::Alloc,
                "free" => TokenKind::Free,
                "select" => TokenKind::Select,
                "itof" => TokenKind::Cast(crate::ast::CastOp::Itof),
                "ftoi" => TokenKind::Cast(crate::ast::CastOp::Ftoi),
                "sext" => TokenKind::Cast(crate::ast::CastOp::Sext),
                "zext" => TokenKind::Cast(crate::ast::CastOp::Zext),
                "trunc" => TokenKind::Cast(crate::ast::CastOp::Trunc),
                "fext" => TokenKind::Cast(crate::ast::CastOp::Fext),
                "ftrunc" => TokenKind::Cast(crate::ast::CastOp::Ftrunc),
                "bitcast" => TokenKind::Cast(crate::ast::CastOp::Bitcast),
                "sqrt" => TokenKind::Unary(crate::ast::UnaryOp::Sqrt),
                "neg" => TokenKind::Unary(crate::ast::UnaryOp::Neg),
                "abs" => TokenKind::Unary(crate::ast::UnaryOp::Abs),
                "vfsum" => TokenKind::VectorReduce(crate::ast::VectorReduceOp::VfSum),
                "vfmax" => TokenKind::VectorReduce(crate::ast::VectorReduceOp::VfMax),
                "visum" => TokenKind::VectorReduce(crate::ast::VectorReduceOp::ViSum),
                "add" | "sub" | "mul" | "div" | "rem" | "and" | "or" | "xor" | "shl" | "shr"
                | "eq" | "ne" | "lt" | "gt" | "le" | "ge" | "vfadd" | "vfsub" | "vfmul"
                | "vfdiv" | "viadd" | "visub" | "vimul" | "min" | "max" | "umin" | "umax"
                | "udiv" | "urem" | "ushr" | "ult" | "ugt" | "ule" | "uge" => TokenKind::Op(ident),
                _ => TokenKind::Ident(ident),
            };

            return Ok(Token {
                kind,
                span: Span {
                    start: start_pos,
                    end: self.current_byte_pos(),
                    line: start_line,
                    column: start_col,
                },
            });
        }

        // Consume the offending character so the span has non-zero width.
        self.advance();
        Err(Diagnostic::error(
            "ERR_LEXICAL",
            format!("Unexpected character: '{c}' at {start_line}:{start_col}"),
            Span {
                start: start_pos,
                end: self.current_byte_pos(),
                line: start_line,
                column: start_col,
            },
        ))
    }

    /// True if the cursor is at an `e`/`E` that starts a valid exponent (`e5`, `e-5`, `E+5`).
    fn exponent_follows(&self) -> bool {
        match self.chars.get(self.cursor + 1).map(|&(_, c)| c) {
            Some(d) if d.is_ascii_digit() => true,
            Some('+') | Some('-') => self
                .chars
                .get(self.cursor + 2)
                .is_some_and(|&(_, d)| d.is_ascii_digit()),
            _ => false,
        }
    }

    pub fn tokenize_all(&mut self) -> Result<Vec<Token>, Diagnostic> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }
}
