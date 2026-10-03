use crate::diag::Span;

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

    pub fn next_token(&mut self) -> Result<Token, String> {
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

        // Number (integer or float, signed or unsigned)
        if c.is_ascii_digit() || (c == '-' && self.peek_next().map_or(false, |next| next.is_ascii_digit())) {
            let mut num_str = String::new();
            if c == '-' {
                num_str.push(self.advance().unwrap());
            }
            let mut is_float = false;
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    num_str.push(self.advance().unwrap());
                } else if ch == '.' && !is_float && self.peek_next().map_or(false, |next| next.is_ascii_digit()) {
                    is_float = true;
                    num_str.push(self.advance().unwrap());
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
                let val: f64 = num_str.parse().map_err(|e| format!("Invalid float {num_str}: {e}"))?;
                Ok(Token {
                    kind: TokenKind::FloatLit(val),
                    span,
                })
            } else {
                let val: i64 = num_str.parse().map_err(|e| format!("Invalid integer {num_str}: {e}"))?;
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
                "add" | "sub" | "mul" | "div" | "rem" | "and" | "or" | "xor" | "shl" | "shr"
                | "eq" | "ne" | "lt" | "gt" | "le" | "ge" | "vfadd" | "vfsub" | "vfmul"
                | "vfdiv" | "viadd" | "visub" | "vimul" => TokenKind::Op(ident),
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

        Err(format!("Unexpected character: '{c}' at {start_line}:{start_col}"))
    }

    pub fn tokenize_all(&mut self) -> Result<Vec<Token>, String> {
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
