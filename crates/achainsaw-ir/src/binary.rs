//! AIRB: Agent Intermediate Representation Binary Specification & Codec
//!
//! Provides high-density zero-allocation binary serialization and deserialization
//! for AIR modules, enabling instant (<1 us) caching, IPC, and re-execution.

use crate::ast::*;
use crate::diag::{Diagnostic, Span};
use crate::types::Type;
use std::collections::HashMap;

pub const MAGIC: &[u8; 4] = b"\x00AIR";
pub const VERSION: u16 = 1;

/// Serializes an in-memory AIR `Module` into compact AIRB binary bytes.
pub fn encode_module(module: &Module) -> Vec<u8> {
    let mut encoder = BinaryEncoder::new();
    encoder.encode(module);
    encoder.finish()
}

/// Deserializes AIRB binary bytes into an in-memory `Module`.
pub fn decode_module(bytes: &[u8]) -> Result<Module, Diagnostic> {
    let mut decoder = BinaryDecoder::new(bytes);
    decoder.decode()
}

/// Converts an in-memory `Module` back into canonical AIR text format (disassembler).
pub fn to_air_text(module: &Module) -> String {
    let mut out = String::new();
    for (f_idx, func) in module.functions.iter().enumerate() {
        if f_idx > 0 {
            out.push('\n');
        }
        out.push_str("fn ");
        out.push_str(&func.name);
        out.push('(');
        for (p_idx, (p_name, p_ty)) in func.params.iter().enumerate() {
            if p_idx > 0 {
                out.push_str(", ");
            }
            out.push_str(p_name);
            out.push(':');
            out.push_str(p_ty.as_str());
        }
        out.push(')');
        if let Some(ret_ty) = func.ret_type {
            out.push_str("->");
            out.push_str(ret_ty.as_str());
        }
        out.push('\n');

        for block in &func.blocks {
            out.push_str("  ");
            out.push_str(&block.label);
            if !block.params.is_empty() {
                out.push('(');
                for (b_idx, (bp_name, bp_ty)) in block.params.iter().enumerate() {
                    if b_idx > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(bp_name);
                    out.push(':');
                    out.push_str(bp_ty.as_str());
                }
                out.push(')');
            }
            out.push_str(":\n");

            for inst in &block.instructions {
                out.push_str("    ");
                match inst {
                    Instruction::AssignConst { dst, val, ty, .. } => {
                        out.push_str(dst);
                        out.push_str(" = cst ");
                        match val {
                            Constant::Int(n) => out.push_str(&n.to_string()),
                            Constant::Float(f) => {
                                if f.fract() == 0.0 {
                                    out.push_str(&format!("{f:.1}"));
                                } else {
                                    out.push_str(&f.to_string());
                                }
                            }
                        }
                        out.push(':');
                        out.push_str(ty.as_str());
                    }
                    Instruction::Binary {
                        op,
                        dst,
                        lhs,
                        rhs,
                        ..
                    } => {
                        out.push_str(dst);
                        out.push_str(" = ");
                        out.push_str(binary_op_to_str(*op));
                        out.push(' ');
                        out.push_str(lhs);
                        out.push_str(", ");
                        out.push_str(rhs);
                    }
                    Instruction::Load { dst, ptr, ty, .. } => {
                        out.push_str(dst);
                        out.push_str(" = ld ");
                        out.push_str(ptr);
                        out.push(':');
                        out.push_str(ty.as_str());
                    }
                    Instruction::Store { ptr, val, .. } => {
                        out.push_str("st ");
                        out.push_str(ptr);
                        out.push_str(", ");
                        out.push_str(val);
                    }
                    Instruction::Call {
                        dst,
                        func,
                        args,
                        ..
                    } => {
                        if let Some(d) = dst {
                            out.push_str(d);
                            out.push_str(" = ");
                        }
                        out.push_str("call ");
                        out.push_str(func);
                        out.push('(');
                        for (a_idx, arg) in args.iter().enumerate() {
                            if a_idx > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(arg);
                        }
                        out.push(')');
                    }
                    Instruction::Splat { dst, src, .. } => {
                        out.push_str(dst);
                        out.push_str(" = splat ");
                        out.push_str(src);
                    }
                    Instruction::ExtractLane {
                        dst,
                        vec,
                        lane,
                        ty,
                        ..
                    } => {
                        out.push_str(dst);
                        out.push_str(" = extlane ");
                        out.push_str(vec);
                        out.push_str(", ");
                        out.push_str(&lane.to_string());
                        out.push(':');
                        out.push_str(ty.as_str());
                    }
                    Instruction::Alloc { dst, size, .. } => {
                        out.push_str(dst);
                        out.push_str(" = alloc ");
                        out.push_str(size);
                    }
                    Instruction::Free { ptr, .. } => {
                        out.push_str("free ");
                        out.push_str(ptr);
                    }
                }
                out.push('\n');
            }

            out.push_str("    ");
            match &block.terminator {
                Terminator::Jmp { target, args, .. } => {
                    out.push_str("jmp ");
                    out.push_str(target);
                    if !args.is_empty() {
                        out.push('(');
                        for (a_idx, a) in args.iter().enumerate() {
                            if a_idx > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(a);
                        }
                        out.push(')');
                    }
                }
                Terminator::Br {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                    ..
                } => {
                    out.push_str("br ");
                    out.push_str(cond);
                    out.push_str(", ");
                    out.push_str(then_block);
                    if !then_args.is_empty() {
                        out.push('(');
                        for (a_idx, a) in then_args.iter().enumerate() {
                            if a_idx > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(a);
                        }
                        out.push(')');
                    }
                    out.push_str(", ");
                    out.push_str(else_block);
                    if !else_args.is_empty() {
                        out.push('(');
                        for (a_idx, a) in else_args.iter().enumerate() {
                            if a_idx > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(a);
                        }
                        out.push(')');
                    }
                }
                Terminator::Ret { val, .. } => {
                    out.push_str("ret");
                    if let Some(v) = val {
                        out.push(' ');
                        out.push_str(v);
                    }
                }
            }
            out.push('\n');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Binary Encoding
// ---------------------------------------------------------------------------

struct BinaryEncoder {
    buf: Vec<u8>,
    string_map: HashMap<String, u32>,
    strings: Vec<String>,
}

impl BinaryEncoder {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            string_map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_map.get(s) {
            id
        } else {
            let id = self.strings.len() as u32;
            self.string_map.insert(s.to_string(), id);
            self.strings.push(s.to_string());
            id
        }
    }

    fn encode(&mut self, module: &Module) {
        // Collect all strings first
        for func in &module.functions {
            self.intern(&func.name);
            for (p_name, _) in &func.params {
                self.intern(p_name);
            }
            for block in &func.blocks {
                self.intern(&block.label);
                for (bp_name, _) in &block.params {
                    self.intern(bp_name);
                }
                for inst in &block.instructions {
                    match inst {
                        Instruction::AssignConst { dst, .. } => {
                            self.intern(dst);
                        }
                        Instruction::Binary { dst, lhs, rhs, .. } => {
                            self.intern(dst);
                            self.intern(lhs);
                            self.intern(rhs);
                        }
                        Instruction::Load { dst, ptr, .. } => {
                            self.intern(dst);
                            self.intern(ptr);
                        }
                        Instruction::Store { ptr, val, .. } => {
                            self.intern(ptr);
                            self.intern(val);
                        }
                        Instruction::Call { dst, func, args, .. } => {
                            if let Some(d) = dst {
                                self.intern(d);
                            }
                            self.intern(func);
                            for a in args {
                                self.intern(a);
                            }
                        }
                        Instruction::Splat { dst, src, .. } => {
                            self.intern(dst);
                            self.intern(src);
                        }
                        Instruction::ExtractLane { dst, vec, .. } => {
                            self.intern(dst);
                            self.intern(vec);
                        }
                        Instruction::Alloc { dst, size, .. } => {
                            self.intern(dst);
                            self.intern(size);
                        }
                        Instruction::Free { ptr, .. } => {
                            self.intern(ptr);
                        }
                    }
                }
                match &block.terminator {
                    Terminator::Jmp { target, args, .. } => {
                        self.intern(target);
                        for a in args {
                            self.intern(a);
                        }
                    }
                    Terminator::Br {
                        cond,
                        then_block,
                        then_args,
                        else_block,
                        else_args,
                        ..
                    } => {
                        self.intern(cond);
                        self.intern(then_block);
                        for a in then_args {
                            self.intern(a);
                        }
                        self.intern(else_block);
                        for a in else_args {
                            self.intern(a);
                        }
                    }
                    Terminator::Ret { val, .. } => {
                        if let Some(v) = val {
                            self.intern(v);
                        }
                    }
                }
            }
        }

        // 1. Header
        self.buf.extend_from_slice(MAGIC);
        self.buf.extend_from_slice(&VERSION.to_le_bytes());
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // flags

        // 2. String Pool
        self.buf.extend_from_slice(&(self.strings.len() as u32).to_le_bytes());
        for s in &self.strings {
            let bytes = s.as_bytes();
            self.buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            self.buf.extend_from_slice(bytes);
        }

        // 3. Functions
        self.buf.extend_from_slice(&(module.functions.len() as u32).to_le_bytes());
        for func in &module.functions {
            let fn_name_id = self.string_map[&func.name];
            self.buf.extend_from_slice(&fn_name_id.to_le_bytes());

            // Parameters
            self.buf.extend_from_slice(&(func.params.len() as u32).to_le_bytes());
            for (p_name, ty) in &func.params {
                let p_id = self.string_map[p_name];
                self.buf.extend_from_slice(&p_id.to_le_bytes());
                self.buf.push(encode_type(*ty));
            }

            // Return Type
            if let Some(ret_ty) = func.ret_type {
                self.buf.push(1);
                self.buf.push(encode_type(ret_ty));
            } else {
                self.buf.push(0);
            }

            // Blocks
            self.buf.extend_from_slice(&(func.blocks.len() as u32).to_le_bytes());
            for block in &func.blocks {
                let label_id = self.string_map[&block.label];
                self.buf.extend_from_slice(&label_id.to_le_bytes());

                // Block parameters
                self.buf.extend_from_slice(&(block.params.len() as u32).to_le_bytes());
                for (bp_name, ty) in &block.params {
                    let bp_id = self.string_map[bp_name];
                    self.buf.extend_from_slice(&bp_id.to_le_bytes());
                    self.buf.push(encode_type(*ty));
                }

                // Instructions
                self.buf.extend_from_slice(&(block.instructions.len() as u32).to_le_bytes());
                for inst in &block.instructions {
                    self.encode_instruction(inst);
                }

                // Terminator
                self.encode_terminator(&block.terminator);
            }
        }
    }

    fn encode_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::AssignConst { dst, val, ty, .. } => match val {
                Constant::Int(n) => {
                    self.buf.push(0x01);
                    self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                    self.buf.extend_from_slice(&n.to_le_bytes());
                    self.buf.push(encode_type(*ty));
                }
                Constant::Float(f) => {
                    self.buf.push(0x02);
                    self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                    self.buf.extend_from_slice(&f.to_bits().to_le_bytes());
                    self.buf.push(encode_type(*ty));
                }
            },
            Instruction::Binary {
                op,
                dst,
                lhs,
                rhs,
                ..
            } => {
                self.buf.push(0x03);
                self.buf.push(encode_binary_op(*op));
                self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[lhs].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[rhs].to_le_bytes());
            }
            Instruction::Load { dst, ptr, ty, .. } => {
                self.buf.push(0x04);
                self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[ptr].to_le_bytes());
                self.buf.push(encode_type(*ty));
            }
            Instruction::Store { ptr, val, .. } => {
                self.buf.push(0x05);
                self.buf.extend_from_slice(&self.string_map[ptr].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[val].to_le_bytes());
            }
            Instruction::Call {
                dst,
                func,
                args,
                ..
            } => {
                self.buf.push(0x06);
                if let Some(d) = dst {
                    self.buf.push(1);
                    self.buf.extend_from_slice(&self.string_map[d].to_le_bytes());
                } else {
                    self.buf.push(0);
                }
                self.buf.extend_from_slice(&self.string_map[func].to_le_bytes());
                self.buf.extend_from_slice(&(args.len() as u32).to_le_bytes());
                for a in args {
                    self.buf.extend_from_slice(&self.string_map[a].to_le_bytes());
                }
            }
            Instruction::Splat { dst, src, .. } => {
                self.buf.push(0x07);
                self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[src].to_le_bytes());
            }
            Instruction::ExtractLane {
                dst,
                vec,
                lane,
                ty,
                ..
            } => {
                self.buf.push(0x08);
                self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[vec].to_le_bytes());
                self.buf.extend_from_slice(&lane.to_le_bytes());
                self.buf.push(encode_type(*ty));
            }
            Instruction::Alloc { dst, size, .. } => {
                self.buf.push(0x09);
                self.buf.extend_from_slice(&self.string_map[dst].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[size].to_le_bytes());
            }
            Instruction::Free { ptr, .. } => {
                self.buf.push(0x0A);
                self.buf.extend_from_slice(&self.string_map[ptr].to_le_bytes());
            }
        }
    }

    fn encode_terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Jmp { target, args, .. } => {
                self.buf.push(0x10);
                self.buf.extend_from_slice(&self.string_map[target].to_le_bytes());
                self.buf.extend_from_slice(&(args.len() as u32).to_le_bytes());
                for a in args {
                    self.buf.extend_from_slice(&self.string_map[a].to_le_bytes());
                }
            }
            Terminator::Br {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                self.buf.push(0x11);
                self.buf.extend_from_slice(&self.string_map[cond].to_le_bytes());
                self.buf.extend_from_slice(&self.string_map[then_block].to_le_bytes());
                self.buf.extend_from_slice(&(then_args.len() as u32).to_le_bytes());
                for a in then_args {
                    self.buf.extend_from_slice(&self.string_map[a].to_le_bytes());
                }
                self.buf.extend_from_slice(&self.string_map[else_block].to_le_bytes());
                self.buf.extend_from_slice(&(else_args.len() as u32).to_le_bytes());
                for a in else_args {
                    self.buf.extend_from_slice(&self.string_map[a].to_le_bytes());
                }
            }
            Terminator::Ret { val, .. } => {
                self.buf.push(0x12);
                if let Some(v) = val {
                    self.buf.push(1);
                    self.buf.extend_from_slice(&self.string_map[v].to_le_bytes());
                } else {
                    self.buf.push(0);
                }
            }
        }
    }

    fn finish(self) -> Vec<u8> {
        self.buf
    }
}

// ---------------------------------------------------------------------------
// Binary Decoding
// ---------------------------------------------------------------------------

struct BinaryDecoder<'a> {
    bytes: &'a [u8],
    pos: usize,
    strings: Vec<String>,
}

impl<'a> BinaryDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            strings: Vec::new(),
        }
    }

    fn err(&self, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::error("ERR_INVALID_AIRB", msg, Span::default())
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], Diagnostic> {
        if self.pos + n > self.bytes.len() {
            return Err(self.err("Unexpected end of AIRB binary stream"));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, Diagnostic> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_u16(&mut self) -> Result<u16, Diagnostic> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, Diagnostic> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, Diagnostic> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, Diagnostic> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, Diagnostic> {
        let id = self.read_u32()?;
        self.strings
            .get(id as usize)
            .cloned()
            .ok_or_else(|| self.err(format!("Invalid string ID {id} in AIRB stream")))
    }

    fn decode(&mut self) -> Result<Module, Diagnostic> {
        // 1. Verify Magic
        let magic = self.read_bytes(4)?;
        if magic != MAGIC {
            return Err(self.err(format!(
                "Invalid AIRB magic: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        // 2. Version
        let version = self.read_u16()?;
        if version != VERSION {
            return Err(self.err(format!(
                "Unsupported AIRB version: expected {VERSION}, got {version}"
            )));
        }

        // Flags
        let _flags = self.read_u16()?;

        // 3. String Pool
        let string_count = self.read_u32()?;
        for _ in 0..string_count {
            let len = self.read_u16()? as usize;
            let str_bytes = self.read_bytes(len)?;
            let s = std::str::from_utf8(str_bytes)
                .map_err(|e| self.err(format!("Invalid UTF-8 in string pool: {e}")))?;
            self.strings.push(s.to_string());
        }

        // 4. Functions
        let func_count = self.read_u32()?;
        let mut functions = Vec::with_capacity(func_count as usize);

        for _ in 0..func_count {
            let name = self.read_string()?;

            let param_count = self.read_u32()?;
            let mut params = Vec::with_capacity(param_count as usize);
            for _ in 0..param_count {
                let p_name = self.read_string()?;
                let p_ty = decode_type(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid parameter type code in AIRB"))?;
                params.push((p_name, p_ty));
            }

            let has_ret = self.read_u8()?;
            let ret_type = if has_ret != 0 {
                let r_ty = decode_type(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid return type code in AIRB"))?;
                Some(r_ty)
            } else {
                None
            };

            let block_count = self.read_u32()?;
            let mut blocks = Vec::with_capacity(block_count as usize);

            for _ in 0..block_count {
                let label = self.read_string()?;

                let bp_count = self.read_u32()?;
                let mut bp_params = Vec::with_capacity(bp_count as usize);
                for _ in 0..bp_count {
                    let bp_name = self.read_string()?;
                    let bp_ty = decode_type(self.read_u8()?)
                        .ok_or_else(|| self.err("Invalid block param type in AIRB"))?;
                    bp_params.push((bp_name, bp_ty));
                }

                let inst_count = self.read_u32()?;
                let mut instructions = Vec::with_capacity(inst_count as usize);
                for _ in 0..inst_count {
                    instructions.push(self.decode_instruction()?);
                }

                let terminator = self.decode_terminator()?;

                blocks.push(Block {
                    label,
                    params: bp_params,
                    instructions,
                    terminator,
                    span: Span::default(),
                });
            }

            functions.push(Function {
                name,
                params,
                ret_type,
                blocks,
                span: Span::default(),
            });
        }

        Ok(Module { functions })
    }

    fn decode_instruction(&mut self) -> Result<Instruction, Diagnostic> {
        let tag = self.read_u8()?;
        let span = Span::default();
        match tag {
            0x01 => {
                let dst = self.read_string()?;
                let val = self.read_i64()?;
                let ty = decode_type(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid const type in AIRB"))?;
                Ok(Instruction::AssignConst {
                    dst,
                    val: Constant::Int(val),
                    ty,
                    span,
                })
            }
            0x02 => {
                let dst = self.read_string()?;
                let bits = self.read_u64()?;
                let ty = decode_type(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid const float type in AIRB"))?;
                Ok(Instruction::AssignConst {
                    dst,
                    val: Constant::Float(f64::from_bits(bits)),
                    ty,
                    span,
                })
            }
            0x03 => {
                let op = decode_binary_op(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid binary op code in AIRB"))?;
                let dst = self.read_string()?;
                let lhs = self.read_string()?;
                let rhs = self.read_string()?;
                Ok(Instruction::Binary {
                    op,
                    dst,
                    lhs,
                    rhs,
                    span,
                })
            }
            0x04 => {
                let dst = self.read_string()?;
                let ptr = self.read_string()?;
                let ty = decode_type(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid load type in AIRB"))?;
                Ok(Instruction::Load {
                    dst,
                    ptr,
                    ty,
                    span,
                })
            }
            0x05 => {
                let ptr = self.read_string()?;
                let val = self.read_string()?;
                Ok(Instruction::Store { ptr, val, span })
            }
            0x06 => {
                let has_dst = self.read_u8()?;
                let dst = if has_dst != 0 {
                    Some(self.read_string()?)
                } else {
                    None
                };
                let func = self.read_string()?;
                let arg_count = self.read_u32()?;
                let mut args = Vec::with_capacity(arg_count as usize);
                for _ in 0..arg_count {
                    args.push(self.read_string()?);
                }
                Ok(Instruction::Call {
                    dst,
                    func,
                    args,
                    span,
                })
            }
            0x07 => {
                let dst = self.read_string()?;
                let src = self.read_string()?;
                Ok(Instruction::Splat { dst, src, span })
            }
            0x08 => {
                let dst = self.read_string()?;
                let vec = self.read_string()?;
                let lane = self.read_u32()?;
                let ty = decode_type(self.read_u8()?)
                    .ok_or_else(|| self.err("Invalid extlane type in AIRB"))?;
                Ok(Instruction::ExtractLane {
                    dst,
                    vec,
                    lane,
                    ty,
                    span,
                })
            }
            0x09 => {
                let dst = self.read_string()?;
                let size = self.read_string()?;
                Ok(Instruction::Alloc { dst, size, span })
            }
            0x0A => {
                let ptr = self.read_string()?;
                Ok(Instruction::Free { ptr, span })
            }
            _ => Err(self.err(format!("Unknown instruction opcode tag 0x{tag:02X} in AIRB"))),
        }
    }

    fn decode_terminator(&mut self) -> Result<Terminator, Diagnostic> {
        let tag = self.read_u8()?;
        let span = Span::default();
        match tag {
            0x10 => {
                let target = self.read_string()?;
                let arg_count = self.read_u32()?;
                let mut args = Vec::with_capacity(arg_count as usize);
                for _ in 0..arg_count {
                    args.push(self.read_string()?);
                }
                Ok(Terminator::Jmp { target, args, span })
            }
            0x11 => {
                let cond = self.read_string()?;
                let then_block = self.read_string()?;
                let then_count = self.read_u32()?;
                let mut then_args = Vec::with_capacity(then_count as usize);
                for _ in 0..then_count {
                    then_args.push(self.read_string()?);
                }
                let else_block = self.read_string()?;
                let else_count = self.read_u32()?;
                let mut else_args = Vec::with_capacity(else_count as usize);
                for _ in 0..else_count {
                    else_args.push(self.read_string()?);
                }
                Ok(Terminator::Br {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                    span,
                })
            }
            0x12 => {
                let has_val = self.read_u8()?;
                let val = if has_val != 0 {
                    Some(self.read_string()?)
                } else {
                    None
                };
                Ok(Terminator::Ret { val, span })
            }
            _ => Err(self.err(format!("Unknown terminator opcode tag 0x{tag:02X} in AIRB"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn encode_type(ty: Type) -> u8 {
    match ty {
        Type::I8 => 1,
        Type::I16 => 2,
        Type::I32 => 3,
        Type::I64 => 4,
        Type::F32 => 5,
        Type::F64 => 6,
        Type::Ptr => 7,
        Type::V128 => 8,
    }
}

fn decode_type(code: u8) -> Option<Type> {
    match code {
        1 => Some(Type::I8),
        2 => Some(Type::I16),
        3 => Some(Type::I32),
        4 => Some(Type::I64),
        5 => Some(Type::F32),
        6 => Some(Type::F64),
        7 => Some(Type::Ptr),
        8 => Some(Type::V128),
        _ => None,
    }
}

fn encode_binary_op(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add => 1,
        BinaryOp::Sub => 2,
        BinaryOp::Mul => 3,
        BinaryOp::Div => 4,
        BinaryOp::Rem => 5,
        BinaryOp::And => 6,
        BinaryOp::Or => 7,
        BinaryOp::Xor => 8,
        BinaryOp::Shl => 9,
        BinaryOp::Shr => 10,
        BinaryOp::Eq => 11,
        BinaryOp::Ne => 12,
        BinaryOp::Lt => 13,
        BinaryOp::Gt => 14,
        BinaryOp::Le => 15,
        BinaryOp::Ge => 16,
        BinaryOp::VfAdd => 17,
        BinaryOp::VfSub => 18,
        BinaryOp::VfMul => 19,
        BinaryOp::VfDiv => 20,
        BinaryOp::ViAdd => 21,
        BinaryOp::ViSub => 22,
        BinaryOp::ViMul => 23,
    }
}

fn decode_binary_op(code: u8) -> Option<BinaryOp> {
    match code {
        1 => Some(BinaryOp::Add),
        2 => Some(BinaryOp::Sub),
        3 => Some(BinaryOp::Mul),
        4 => Some(BinaryOp::Div),
        5 => Some(BinaryOp::Rem),
        6 => Some(BinaryOp::And),
        7 => Some(BinaryOp::Or),
        8 => Some(BinaryOp::Xor),
        9 => Some(BinaryOp::Shl),
        10 => Some(BinaryOp::Shr),
        11 => Some(BinaryOp::Eq),
        12 => Some(BinaryOp::Ne),
        13 => Some(BinaryOp::Lt),
        14 => Some(BinaryOp::Gt),
        15 => Some(BinaryOp::Le),
        16 => Some(BinaryOp::Ge),
        17 => Some(BinaryOp::VfAdd),
        18 => Some(BinaryOp::VfSub),
        19 => Some(BinaryOp::VfMul),
        20 => Some(BinaryOp::VfDiv),
        21 => Some(BinaryOp::ViAdd),
        22 => Some(BinaryOp::ViSub),
        23 => Some(BinaryOp::ViMul),
        _ => None,
    }
}

fn binary_op_to_str(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Rem => "rem",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::Xor => "xor",
        BinaryOp::Shl => "shl",
        BinaryOp::Shr => "shr",
        BinaryOp::Eq => "eq",
        BinaryOp::Ne => "ne",
        BinaryOp::Lt => "lt",
        BinaryOp::Gt => "gt",
        BinaryOp::Le => "le",
        BinaryOp::Ge => "ge",
        BinaryOp::VfAdd => "vfadd",
        BinaryOp::VfSub => "vfsub",
        BinaryOp::VfMul => "vfmul",
        BinaryOp::VfDiv => "vfdiv",
        BinaryOp::ViAdd => "viadd",
        BinaryOp::ViSub => "visub",
        BinaryOp::ViMul => "vimul",
    }
}
