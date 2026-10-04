//! AIR -> LLVM IR lowering.
//!
//! Semantics match the Cranelift backend (`achainsaw-codegen/src/lower.rs`) exactly: safe
//! integer division, masked shift amounts, NaN-propagating float min/max, saturating
//! float-to-int casts, the same f16/bf16 rounding, the canonical reduction tree, and the same
//! fuel-check placement (so a fuel budget runs out at the same point on both backends).

use std::collections::HashMap;

use achainsaw_ir::ast::{
    BinaryOp, CastOp, Constant, Function, Instruction, Module, Terminator, UnaryOp, VBinOp, VCmpOp,
    VectorReduceOp,
};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::llvm_sys::LLVMTailCallKind;
use inkwell::module::{Linkage, Module as LModule};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType, VectorType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PhiValue,
    PointerValue, VectorValue,
};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};

/// Runtime hook symbols referenced by JIT code (see `achainsaw-codegen/src/jit.rs`).
pub const RT_MALLOC: &str = "__achainsaw_rt_malloc";
pub const RT_FREE: &str = "__achainsaw_rt_free";
pub const RT_CHECK_FUEL: &str = "__achainsaw_rt_check_fuel";
pub const RT_CONSUME_FUEL: &str = "__achainsaw_rt_consume_fuel";
pub const RT_SANDBOX_FAULT: &str = "__achainsaw_rt_sandbox_fault";
pub const RT_STACK_CHECK: &str = "__achainsaw_rt_stack_check";
pub const RT_SANDBOX_CHECK_MM: &str = "__achainsaw_rt_sandbox_check_mm";

/// Name of the scalar host-call trampoline generated for `func`.
pub fn trampoline_name(func: &str) -> String {
    format!("__achainsaw_trampoline_{func}")
}

/// Sandbox arena `[base, base + len)` that every memory access is checked against.
#[derive(Debug, Clone, Copy)]
pub struct SandboxBounds {
    pub base: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Default)]
pub struct LowerOptions {
    /// Call the fuel hooks at every branch and before `mm`.
    pub fuel: bool,
    /// Bounds-check memory accesses and stack depth (implies the halt checks at branches).
    pub sandbox: Option<SandboxBounds>,
    /// AOT objects: `alloc`/`free` call libc `malloc`/`free`, no runtime hooks, and no
    /// trampolines.
    pub aot: bool,
    /// Value of the `target-cpu` / `target-features` function attributes.
    pub target_cpu: String,
    pub target_features: String,
}

/// Width in bits of a vector type on this backend. `vx` is 128 bits until the LLVM tier
/// targets wider vectors natively.
fn vec_bits(ty: Type) -> u32 {
    match ty {
        Type::V256 => 256,
        Type::V512 => 512,
        _ => 128,
    }
}

fn lane_bits(lane: Type) -> u32 {
    lane.bit_width().expect("lane type has a width")
}

fn lane_count(ty: Type, lane: Type) -> u32 {
    vec_bits(ty) / lane_bits(lane)
}

pub fn lower_module<'ctx>(
    ctx: &'ctx Context,
    air: &Module,
    opts: &LowerOptions,
) -> Result<LModule<'ctx>> {
    let module = ctx.create_module("achainsaw");
    let lw = ModuleLowerer {
        ctx,
        module: &module,
        builder: ctx.create_builder(),
        opts,
    };
    lw.declare_runtime();
    for ext in &air.extern_functions {
        let fn_ty = lw.fn_type(&ext.params, ext.ret_type);
        module.add_function(&ext.name, fn_ty, Some(Linkage::External));
    }
    for func in &air.functions {
        let fn_ty = lw.fn_type(&func.params, func.ret_type);
        let f = module.add_function(&func.name, fn_ty, Some(Linkage::External));
        lw.add_target_attributes(f);
    }
    for func in &air.functions {
        lw.lower_function(func, air)?;
    }
    if !opts.aot {
        for func in &air.functions {
            let scalar_only = func
                .params
                .iter()
                .map(|(_, t)| *t)
                .chain(func.ret_type)
                .all(|t| !t.is_vector());
            if scalar_only {
                lw.lower_trampoline(func)?;
            }
        }
    }
    module
        .verify()
        .map_err(|e| anyhow!("[ERR_LLVM_VERIFY] Generated invalid LLVM IR: {e}"))?;
    Ok(module)
}

struct ModuleLowerer<'a, 'ctx> {
    ctx: &'ctx Context,
    module: &'a LModule<'ctx>,
    builder: Builder<'ctx>,
    opts: &'a LowerOptions,
}

/// One AIR register: its LLVM value and AIR type.
type Values<'ctx> = HashMap<String, (BasicValueEnum<'ctx>, Type)>;

/// Per-function lowering state.
struct FnState<'ctx> {
    func: FunctionValue<'ctx>,
    values: Values<'ctx>,
    blocks: HashMap<String, (BasicBlock<'ctx>, Vec<PhiValue<'ctx>>)>,
    /// Returns zeroes once a runtime check fails; the host then reports the status.
    trap: Option<BasicBlock<'ctx>>,
    /// Reports a sandbox violation `(addr, size)` and jumps to `trap`.
    fault: Option<(BasicBlock<'ctx>, PhiValue<'ctx>, PhiValue<'ctx>)>,
}

impl<'a, 'ctx> ModuleLowerer<'a, 'ctx> {
    // ---------------------------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------------------------

    fn i1(&self) -> IntType<'ctx> {
        self.ctx.bool_type()
    }
    fn i16(&self) -> IntType<'ctx> {
        self.ctx.i16_type()
    }
    fn i32(&self) -> IntType<'ctx> {
        self.ctx.i32_type()
    }
    fn i64(&self) -> IntType<'ctx> {
        self.ctx.i64_type()
    }
    fn int_of_bits(&self, bits: u32) -> IntType<'ctx> {
        match bits {
            8 => self.ctx.i8_type(),
            16 => self.i16(),
            32 => self.i32(),
            _ => self.i64(),
        }
    }

    /// LLVM type of a scalar AIR type or a vector lane. Pointers are i64 addresses, and
    /// f16/bf16 are carried as their raw bits, as on Cranelift.
    fn scalar_type(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::I8 => self.ctx.i8_type().into(),
            Type::I16 | Type::F16 | Type::BF16 => self.i16().into(),
            Type::I32 => self.i32().into(),
            Type::I64 | Type::Ptr => self.i64().into(),
            Type::F32 => self.ctx.f32_type().into(),
            Type::F64 => self.ctx.f64_type().into(),
            Type::V128 | Type::V256 | Type::V512 | Type::Vx => unreachable!("vector as scalar"),
        }
    }

    /// Canonical representation of an (untyped) AIR vector: i32 lanes.
    fn canon_vec(&self, ty: Type) -> VectorType<'ctx> {
        self.i32().vec_type(vec_bits(ty) / 32)
    }

    fn lane_vec(&self, ty: Type, lane: Type) -> VectorType<'ctx> {
        let n = lane_count(ty, lane);
        match self.scalar_type(lane) {
            BasicTypeEnum::IntType(t) => t.vec_type(n),
            BasicTypeEnum::FloatType(t) => t.vec_type(n),
            _ => unreachable!(),
        }
    }

    fn air_type(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        if ty.is_vector() {
            self.canon_vec(ty).into()
        } else {
            self.scalar_type(ty)
        }
    }

    fn fn_type(
        &self,
        params: &[(String, Type)],
        ret: Option<Type>,
    ) -> inkwell::types::FunctionType<'ctx> {
        let ps: Vec<BasicMetadataTypeEnum> = params
            .iter()
            .map(|(_, t)| self.air_type(*t).into())
            .collect();
        match ret {
            Some(r) => self.air_type(r).fn_type(&ps, false),
            None => self.ctx.void_type().fn_type(&ps, false),
        }
    }

    fn add_target_attributes(&self, f: FunctionValue<'ctx>) {
        let attrs = [
            ("target-cpu", self.opts.target_cpu.as_str()),
            ("target-features", self.opts.target_features.as_str()),
            // Probe large frames page by page so they cannot skip the guard page.
            ("probe-stack", "inline-asm"),
        ];
        for (k, v) in attrs {
            if !v.is_empty() {
                f.add_attribute(
                    AttributeLoc::Function,
                    self.ctx.create_string_attribute(k, v),
                );
            }
        }
        f.add_attribute(AttributeLoc::Function, self.enum_attr("nounwind"));
    }

    fn enum_attr(&self, name: &str) -> Attribute {
        self.ctx
            .create_enum_attribute(Attribute::get_named_enum_kind_id(name), 0)
    }

    fn declare_runtime(&self) {
        let i64t = self.i64();
        let i32t = self.i32();
        let void = self.ctx.void_type();
        let decl = |name: &str, ty| {
            self.module.add_function(name, ty, Some(Linkage::External));
        };
        if self.opts.aot {
            decl("malloc", i64t.fn_type(&[i64t.into()], false));
            decl("free", void.fn_type(&[i64t.into()], false));
            return;
        }
        decl(RT_MALLOC, i64t.fn_type(&[i64t.into()], false));
        decl(RT_FREE, void.fn_type(&[i64t.into()], false));
        decl(RT_CHECK_FUEL, i32t.fn_type(&[], false));
        decl(RT_CONSUME_FUEL, i32t.fn_type(&[i64t.into()], false));
        decl(
            RT_SANDBOX_FAULT,
            void.fn_type(&[i64t.into(), i64t.into()], false),
        );
        decl(RT_STACK_CHECK, i32t.fn_type(&[], false));
        decl(RT_SANDBOX_CHECK_MM, i32t.fn_type(&[i64t.into(); 7], false));
    }

    fn runtime_fn(&self, name: &str) -> FunctionValue<'ctx> {
        self.module
            .get_function(name)
            .expect("runtime hook declared")
    }

    fn malloc_fn(&self) -> FunctionValue<'ctx> {
        self.runtime_fn(if self.opts.aot { "malloc" } else { RT_MALLOC })
    }

    fn free_fn(&self) -> FunctionValue<'ctx> {
        self.runtime_fn(if self.opts.aot { "free" } else { RT_FREE })
    }

    fn intrinsic(&self, name: &str, types: &[BasicTypeEnum<'ctx>]) -> FunctionValue<'ctx> {
        Intrinsic::find(name)
            .unwrap_or_else(|| panic!("unknown intrinsic {name}"))
            .get_declaration(self.module, types)
            .unwrap_or_else(|| panic!("bad intrinsic overload {name}"))
    }

    // ---------------------------------------------------------------------------------
    // Small builders
    // ---------------------------------------------------------------------------------

    fn c64(&self, v: i64) -> IntValue<'ctx> {
        self.i64().const_int(v as u64, true)
    }
    fn c32(&self, v: u32) -> IntValue<'ctx> {
        self.i32().const_int(v as u64, false)
    }

    fn call(
        &self,
        f: FunctionValue<'ctx>,
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<Option<BasicValueEnum<'ctx>>> {
        let args: Vec<BasicMetadataValueEnum> = args.iter().map(|a| (*a).into()).collect();
        let cs = self.builder.build_call(f, &args, "")?;
        Ok(cs.try_as_basic_value().basic())
    }

    fn call1(
        &self,
        f: FunctionValue<'ctx>,
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<BasicValueEnum<'ctx>> {
        self.call(f, args)?
            .ok_or_else(|| anyhow!("call to {:?} returned void", f.get_name()))
    }

    fn intr(
        &self,
        name: &str,
        overload: &[BasicTypeEnum<'ctx>],
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<BasicValueEnum<'ctx>> {
        self.call1(self.intrinsic(name, overload), args)
    }

    fn bitcast(
        &self,
        v: BasicValueEnum<'ctx>,
        ty: impl BasicType<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        if v.get_type() == ty.as_basic_type_enum() {
            return Ok(v);
        }
        Ok(self.builder.build_bit_cast(v, ty, "")?)
    }

    fn as_lanes(&self, v: BasicValueEnum<'ctx>, ty: Type, lane: Type) -> Result<VectorValue<'ctx>> {
        Ok(self
            .bitcast(v, self.lane_vec(ty, lane))?
            .into_vector_value())
    }

    fn to_canon(&self, v: impl BasicValue<'ctx>, ty: Type) -> Result<BasicValueEnum<'ctx>> {
        self.bitcast(v.as_basic_value_enum(), self.canon_vec(ty))
    }

    fn int_ptr(&self, addr: IntValue<'ctx>) -> Result<PointerValue<'ctx>> {
        Ok(self
            .builder
            .build_int_to_ptr(addr, self.ctx.ptr_type(AddressSpace::default()), "")?)
    }

    /// Unaligned load: AIR pointers into user buffers may have any alignment.
    fn load(&self, ty: BasicTypeEnum<'ctx>, addr: IntValue<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let p = self.int_ptr(addr)?;
        let v = self.builder.build_load(ty, p, "")?;
        v.as_instruction_value()
            .unwrap()
            .set_alignment(1)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(v)
    }

    fn store(&self, v: BasicValueEnum<'ctx>, addr: IntValue<'ctx>) -> Result<()> {
        let p = self.int_ptr(addr)?;
        self.builder
            .build_store(p, v)?
            .set_alignment(1)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    fn zext_bool(&self, b: IntValue<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        Ok(self.builder.build_int_z_extend(b, self.i32(), "")?.into())
    }

    /// Splats `x` into every lane of a `ty`-wide vector whose lanes have `x`'s type.
    fn splat(&self, x: BasicValueEnum<'ctx>, n: u32) -> Result<VectorValue<'ctx>> {
        let vt = match x.get_type() {
            BasicTypeEnum::IntType(t) => t.vec_type(n),
            BasicTypeEnum::FloatType(t) => t.vec_type(n),
            _ => unreachable!(),
        };
        let one = self
            .builder
            .build_insert_element(vt.get_poison(), x, self.c32(0), "")?;
        let zeros = self.i32().vec_type(n).const_zero();
        Ok(self
            .builder
            .build_shuffle_vector(one, vt.get_poison(), zeros, "")?)
    }

    // ---------------------------------------------------------------------------------
    // f16 / bf16 (bit-exact ports of the Cranelift sequences)
    // ---------------------------------------------------------------------------------

    fn f16_to_f32(&self, h: IntValue<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let x = b.build_int_z_extend(h, self.i32(), "")?;
        let sign = b.build_and(x, self.c32(0x8000), "")?;
        let sign = b.build_left_shift(sign, self.c32(16), "")?;
        let exp = b.build_right_shift(x, self.c32(10), false, "")?;
        let exp = b.build_and(exp, self.c32(0x1f), "")?;
        let mant = b.build_and(x, self.c32(0x3ff), "")?;
        let mant13 = b.build_left_shift(mant, self.c32(13), "")?;
        let exp32 = b.build_int_add(exp, self.c32(112), "")?;
        let exp32 = b.build_left_shift(exp32, self.c32(23), "")?;
        let normal = b.build_or(exp32, mant13, "")?;
        let infnan = b.build_or(mant13, self.c32(0x7f80_0000), "")?;
        let f32t = self.ctx.f32_type();
        let mant_f = b.build_unsigned_int_to_float(mant, f32t, "")?;
        let scale = f32t.const_float(f32::from_bits(0x3380_0000) as f64);
        let sub_f = b.build_float_mul(mant_f, scale, "")?;
        let sub = b.build_bit_cast(sub_f, self.i32(), "")?.into_int_value();
        let is_sub = b.build_int_compare(IntPredicate::EQ, exp, self.c32(0), "")?;
        let is_max = b.build_int_compare(IntPredicate::EQ, exp, self.c32(31), "")?;
        let mag = b.build_select(is_max, infnan, normal, "")?.into_int_value();
        let mag = b.build_select(is_sub, sub, mag, "")?.into_int_value();
        let bits = b.build_or(mag, sign, "")?;
        Ok(b.build_bit_cast(bits, f32t, "")?)
    }

    fn f32_to_f16(&self, f: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let f32t = self.ctx.f32_type();
        let x = b.build_bit_cast(f, self.i32(), "")?.into_int_value();
        let sign = b.build_and(x, self.c32(0x8000_0000), "")?;
        let a = b.build_xor(x, sign, "")?;
        let is_big = b.build_int_compare(IntPredicate::UGE, a, self.c32(0x4780_0000), "")?;
        let is_nan = b.build_int_compare(IntPredicate::UGT, a, self.c32(0x7f80_0000), "")?;
        let big = b
            .build_select(is_nan, self.c32(0x7e00), self.c32(0x7c00), "")?
            .into_int_value();
        let is_sub = b.build_int_compare(IntPredicate::ULT, a, self.c32(0x3880_0000), "")?;
        let af = b.build_bit_cast(a, f32t, "")?.into_float_value();
        let magic = f32t.const_float(f32::from_bits(0x3f00_0000) as f64);
        let sum = b.build_float_add(af, magic, "")?;
        let sum_bits = b.build_bit_cast(sum, self.i32(), "")?.into_int_value();
        let sub = b.build_int_sub(sum_bits, self.c32(0x3f00_0000), "")?;
        let odd = b.build_right_shift(a, self.c32(13), false, "")?;
        let odd = b.build_and(odd, self.c32(1), "")?;
        let bias = self.c32((((15i32 - 127) << 23) + 0xfff) as u32);
        let t = b.build_int_add(a, bias, "")?;
        let t = b.build_int_add(t, odd, "")?;
        let normal = b.build_right_shift(t, self.c32(13), false, "")?;
        let mag = b.build_select(is_sub, sub, normal, "")?.into_int_value();
        let mag = b.build_select(is_big, big, mag, "")?.into_int_value();
        let sign16 = b.build_right_shift(sign, self.c32(16), false, "")?;
        let bits = b.build_or(mag, sign16, "")?;
        Ok(b.build_int_truncate(bits, self.i16(), "")?.into())
    }

    fn bf16_to_f32(&self, h: IntValue<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let x = b.build_int_z_extend(h, self.i32(), "")?;
        let x = b.build_left_shift(x, self.c32(16), "")?;
        Ok(b.build_bit_cast(x, self.ctx.f32_type(), "")?)
    }

    fn f32_to_bf16(&self, f: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let x = b.build_bit_cast(f, self.i32(), "")?.into_int_value();
        let hi = b.build_right_shift(x, self.c32(16), false, "")?;
        let lsb = b.build_and(hi, self.c32(1), "")?;
        let r = b.build_int_add(x, self.c32(0x7fff), "")?;
        let r = b.build_int_add(r, lsb, "")?;
        let rounded = b.build_right_shift(r, self.c32(16), false, "")?;
        let quiet = b.build_or(hi, self.c32(0x40), "")?;
        let abs = b.build_and(x, self.c32(0x7fff_ffff), "")?;
        let is_nan = b.build_int_compare(IntPredicate::UGT, abs, self.c32(0x7f80_0000), "")?;
        let bits = b.build_select(is_nan, quiet, rounded, "")?.into_int_value();
        Ok(b.build_int_truncate(bits, self.i16(), "")?.into())
    }

    // ---------------------------------------------------------------------------------
    // Runtime checks
    // ---------------------------------------------------------------------------------

    /// Branches to the fault block unless `[addr, addr + size)` is inside the sandbox arena
    /// or `size` is 0, leaving the builder in the passing block.
    fn bounds_check(
        &self,
        st: &FnState<'ctx>,
        addr: IntValue<'ctx>,
        size: IntValue<'ctx>,
    ) -> Result<()> {
        let (Some(sb), Some((fault, addr_phi, size_phi))) = (self.opts.sandbox, st.fault) else {
            return Ok(());
        };
        let b = &self.builder;
        let off = b.build_int_sub(addr, self.c64(sb.base as i64), "")?;
        let room = b.build_int_sub(self.c64(sb.len as i64), size, "")?;
        let fits = b.build_int_compare(IntPredicate::ULE, off, room, "")?;
        let empty = b.build_int_compare(IntPredicate::EQ, size, self.c64(0), "")?;
        let ok = b.build_or(fits, empty, "")?;
        let pass = self.ctx.append_basic_block(st.func, "inbounds");
        let here = b.get_insert_block().unwrap();
        b.build_conditional_branch(ok, pass, fault)?;
        addr_phi.add_incoming(&[(&addr, here)]);
        size_phi.add_incoming(&[(&size, here)]);
        b.position_at_end(pass);
        Ok(())
    }

    /// Calls a hook returning i32 and branches to the trap block when it is non-zero.
    fn check_hook(
        &self,
        st: &FnState<'ctx>,
        hook: &str,
        args: &[BasicValueEnum<'ctx>],
        next: BasicBlock<'ctx>,
    ) -> Result<()> {
        let r = self.call1(self.runtime_fn(hook), args)?.into_int_value();
        let bad =
            self.builder
                .build_int_compare(IntPredicate::NE, r, self.i32().const_zero(), "")?;
        self.builder
            .build_conditional_branch(bad, st.trap.expect("trap block"), next)?;
        Ok(())
    }

    /// Fuel check before a branch: continues in a fresh block unless fuel ran out.
    fn fuel_check(&self, st: &FnState<'ctx>) -> Result<()> {
        if !(self.opts.fuel || self.opts.sandbox.is_some()) || self.opts.aot {
            return Ok(());
        }
        let cont = self.ctx.append_basic_block(st.func, "fueled");
        self.check_hook(st, RT_CHECK_FUEL, &[], cont)?;
        self.builder.position_at_end(cont);
        Ok(())
    }

    // ---------------------------------------------------------------------------------
    // Functions
    // ---------------------------------------------------------------------------------

    fn lower_function(&self, func: &Function, air: &Module) -> Result<()> {
        let f = self.module.get_function(&func.name).unwrap();
        let b = &self.builder;
        let entry = self.ctx.append_basic_block(f, "entry");

        let mut blocks = HashMap::new();
        for block in &func.blocks {
            let bb = self.ctx.append_basic_block(f, &block.label);
            b.position_at_end(bb);
            let mut phis = Vec::with_capacity(block.params.len());
            for (name, ty) in &block.params {
                phis.push(b.build_phi(self.air_type(*ty), name)?);
            }
            blocks.insert(block.label.clone(), (bb, phis));
        }

        let checks = !self.opts.aot && (self.opts.fuel || self.opts.sandbox.is_some());
        let trap = checks.then(|| self.ctx.append_basic_block(f, "trap"));
        let fault = match (self.opts.sandbox, self.opts.aot) {
            (Some(_), false) => {
                let bb = self.ctx.append_basic_block(f, "fault");
                b.position_at_end(bb);
                let addr = b.build_phi(self.i64(), "addr")?;
                let size = b.build_phi(self.i64(), "size")?;
                Some((bb, addr, size))
            }
            _ => None,
        };

        let mut st = FnState {
            func: f,
            values: HashMap::new(),
            blocks,
            trap,
            fault,
        };
        for (i, (name, ty)) in func.params.iter().enumerate() {
            st.values
                .insert(name.clone(), (f.get_nth_param(i as u32).unwrap(), *ty));
        }

        b.position_at_end(entry);
        let first = st.blocks[&func.blocks[0].label].0;
        if self.opts.sandbox.is_some() && !self.opts.aot {
            self.check_hook(&st, RT_STACK_CHECK, &[], first)?;
        } else {
            b.build_unconditional_branch(first)?;
        }

        for block in &func.blocks {
            let (bb, phis) = st.blocks[&block.label].clone();
            b.position_at_end(bb);
            for ((name, ty), phi) in block.params.iter().zip(phis) {
                st.values.insert(name.clone(), (phi.as_basic_value(), *ty));
            }
            for inst in &block.instructions {
                self.lower_inst(&mut st, inst, air)?;
            }
            self.lower_terminator(&st, &block.terminator)?;
        }

        if let Some((bb, addr, size)) = st.fault {
            b.position_at_end(bb);
            if addr.count_incoming() == 0 {
                // No checked accesses: keep the block valid but unreachable.
                addr.as_instruction().erase_from_basic_block();
                size.as_instruction().erase_from_basic_block();
                b.build_unreachable()?;
            } else {
                self.call(
                    self.runtime_fn(RT_SANDBOX_FAULT),
                    &[addr.as_basic_value(), size.as_basic_value()],
                )?;
                b.build_unconditional_branch(st.trap.unwrap())?;
            }
        }
        if let Some(trap) = st.trap {
            b.position_at_end(trap);
            match func.ret_type {
                Some(r) => {
                    let zero = self.air_type(r).const_zero();
                    b.build_return(Some(&zero))?;
                }
                None => {
                    b.build_return(None)?;
                }
            }
        }
        Ok(())
    }

    fn val(&self, st: &FnState<'ctx>, name: &str) -> (BasicValueEnum<'ctx>, Type) {
        st.values[name]
    }

    fn int(&self, st: &FnState<'ctx>, name: &str) -> IntValue<'ctx> {
        st.values[name].0.into_int_value()
    }

    /// Jumps to `target`, passing `args` to its block parameters.
    fn jump_with_args(
        &self,
        st: &FnState<'ctx>,
        target: &str,
        args: &[String],
    ) -> Result<BasicBlock<'ctx>> {
        let (bb, phis) = &st.blocks[target];
        let here = self.builder.get_insert_block().unwrap();
        for (phi, arg) in phis.iter().zip(args) {
            phi.add_incoming(&[(&st.values[arg].0, here)]);
        }
        Ok(*bb)
    }

    /// Block a conditional branch should target for `target(args)`. Targets with
    /// parameters get their own edge block so the phis see a unique predecessor, even when
    /// both arms go to the same block.
    fn edge(&self, st: &FnState<'ctx>, target: &str, args: &[String]) -> Result<BasicBlock<'ctx>> {
        let (bb, phis) = &st.blocks[target];
        if phis.is_empty() {
            return Ok(*bb);
        }
        let saved = self.builder.get_insert_block().unwrap();
        let edge = self.ctx.append_basic_block(st.func, "edge");
        self.builder.position_at_end(edge);
        let dest = self.jump_with_args(st, target, args)?;
        self.builder.build_unconditional_branch(dest)?;
        self.builder.position_at_end(saved);
        Ok(edge)
    }

    fn lower_terminator(&self, st: &FnState<'ctx>, term: &Terminator) -> Result<()> {
        let b = &self.builder;
        match term {
            Terminator::Jmp { target, args, .. } => {
                self.fuel_check(st)?;
                let dest = self.jump_with_args(st, target, args)?;
                b.build_unconditional_branch(dest)?;
            }
            Terminator::Br {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                self.fuel_check(st)?;
                let c = self.int(st, cond);
                let c = b.build_int_compare(IntPredicate::NE, c, c.get_type().const_zero(), "")?;
                let t = self.edge(st, then_block, then_args)?;
                let e = self.edge(st, else_block, else_args)?;
                b.build_conditional_branch(c, t, e)?;
            }
            Terminator::Ret { val, .. } => match val {
                Some(v) => {
                    b.build_return(Some(&st.values[v].0))?;
                }
                None => {
                    b.build_return(None)?;
                }
            },
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------
    // Instructions
    // ---------------------------------------------------------------------------------

    fn lower_inst(&self, st: &mut FnState<'ctx>, inst: &Instruction, air: &Module) -> Result<()> {
        let b = &self.builder;
        match inst {
            Instruction::AssignConst { dst, val, ty, .. } => {
                let v: BasicValueEnum = match (val, ty) {
                    (Constant::Int(n), _) => self
                        .scalar_type(*ty)
                        .into_int_type()
                        .const_int(*n as u64, true)
                        .into(),
                    (Constant::Float(f), Type::F32) => {
                        self.ctx.f32_type().const_float(*f as f32 as f64).into()
                    }
                    (Constant::Float(f), _) => self.ctx.f64_type().const_float(*f).into(),
                };
                st.values.insert(dst.clone(), (v, *ty));
            }
            Instruction::Binary {
                op, dst, lhs, rhs, ..
            } => {
                let res = self.lower_binary(st, *op, lhs, rhs)?;
                st.values.insert(dst.clone(), res);
            }
            Instruction::Load { dst, ptr, ty, .. } => {
                let addr = self.int(st, ptr);
                let bytes = if ty.is_vector() {
                    vec_bits(*ty) / 8
                } else {
                    ty.byte_size() as u32
                };
                self.bounds_check(st, addr, self.c64(bytes as i64))?;
                let v = self.load(self.air_type(*ty), addr)?;
                st.values.insert(dst.clone(), (v, *ty));
            }
            Instruction::Store { ptr, val, .. } => {
                let addr = self.int(st, ptr);
                let (v, vty) = self.val(st, val);
                let bytes = if vty.is_vector() {
                    vec_bits(vty) / 8
                } else {
                    vty.byte_size() as u32
                };
                self.bounds_check(st, addr, self.c64(bytes as i64))?;
                self.store(v, addr)?;
            }
            Instruction::Call {
                dst, func, args, ..
            } => {
                let callee = self
                    .module
                    .get_function(func)
                    .ok_or_else(|| anyhow!("Unknown function '{func}' in call"))?;
                let argv: Vec<BasicMetadataValueEnum> =
                    args.iter().map(|a| st.values[a].0.into()).collect();
                let cs = b.build_call(callee, &argv, "")?;
                // Never turn AIR recursion into a loop: unbounded recursion must still hit
                // the stack limit rather than spin without fuel checks.
                cs.set_tail_call_kind(LLVMTailCallKind::LLVMTailCallKindNoTail);
                let is_extern = air.extern_functions.iter().any(|e| &e.name == func);
                if is_extern {
                    // Call the registered symbol as-is; do not fold or replace libm calls.
                    cs.add_attribute(AttributeLoc::Function, self.enum_attr("nobuiltin"));
                }
                if let Some(d) = dst {
                    let ret_ty = air
                        .functions
                        .iter()
                        .find(|f| &f.name == func)
                        .map(|f| f.ret_type)
                        .or_else(|| {
                            air.extern_functions
                                .iter()
                                .find(|f| &f.name == func)
                                .map(|f| f.ret_type)
                        })
                        .flatten()
                        .unwrap_or(Type::I32);
                    let v = cs
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| anyhow!("'{func}' returns no value"))?;
                    st.values.insert(d.clone(), (v, ret_ty));
                }
            }
            Instruction::Splat { dst, src, ty, .. } => {
                let (s, sty) = self.val(st, src);
                let v = self.splat(s, lane_count(*ty, sty))?;
                st.values.insert(dst.clone(), (self.to_canon(v, *ty)?, *ty));
            }
            Instruction::ExtractLane {
                dst, vec, lane, ty, ..
            } => {
                let (v, vty) = self.val(st, vec);
                let lanes = self.as_lanes(v, vty, *ty)?;
                let x = b.build_extract_element(lanes, self.c32(*lane), "")?;
                st.values.insert(dst.clone(), (x, *ty));
            }
            Instruction::Alloc { dst, size, .. } => {
                let (s, sty) = self.val(st, size);
                let s = if matches!(sty, Type::I64 | Type::Ptr) {
                    s.into_int_value()
                } else {
                    b.build_int_z_extend(s.into_int_value(), self.i64(), "")?
                };
                let p = self.call1(self.malloc_fn(), &[s.into()])?;
                st.values.insert(dst.clone(), (p, Type::Ptr));
            }
            Instruction::Free { ptr, .. } => {
                let p = self.val(st, ptr).0;
                self.call(self.free_fn(), &[p])?;
            }
            Instruction::Unary { op, dst, src, .. } => {
                let (s, sty) = self.val(st, src);
                let t = s.get_type();
                let r: BasicValueEnum = match (op, sty.is_float()) {
                    (UnaryOp::Neg, true) => b.build_float_neg(s.into_float_value(), "")?.into(),
                    (UnaryOp::Neg, false) => b.build_int_neg(s.into_int_value(), "")?.into(),
                    (UnaryOp::Abs, true) => self.intr("llvm.fabs", &[t], &[s])?,
                    // abs(MIN) = MIN, as Cranelift's smax(x, -x).
                    (UnaryOp::Abs, false) => {
                        self.intr("llvm.abs", &[t], &[s, self.i1().const_zero().into()])?
                    }
                    (UnaryOp::Sqrt, _) => self.intr("llvm.sqrt", &[t], &[s])?,
                };
                st.values.insert(dst.clone(), (r, sty));
            }
            Instruction::Cast {
                dst, src, ty, op, ..
            } => {
                let r = self.lower_cast(st, *op, src, *ty)?;
                st.values.insert(dst.clone(), (r, *ty));
            }
            Instruction::Select {
                dst,
                cond,
                then_val,
                else_val,
                ..
            } => {
                let c = self.int(st, cond);
                let c = b.build_int_compare(IntPredicate::NE, c, c.get_type().const_zero(), "")?;
                let (t, tty) = self.val(st, then_val);
                let e = self.val(st, else_val).0;
                let r = b.build_select(c, t, e, "")?;
                st.values.insert(dst.clone(), (r, tty));
            }
            Instruction::VectorReduce {
                op, dst, src, ty, ..
            } => {
                let (v, vty) = self.val(st, src);
                let r = self.reduce(*op, *ty, self.as_lanes(v, vty, *ty)?)?;
                st.values.insert(dst.clone(), (r, *ty));
            }
            Instruction::VBinary {
                op,
                dst,
                lhs,
                rhs,
                lane,
                ..
            } => {
                let (l, vty) = self.val(st, lhs);
                let r = self.val(st, rhs).0;
                let res = self.vbinary(*op, *lane, vty, l, r)?;
                st.values.insert(dst.clone(), (res, vty));
            }
            Instruction::VFma {
                dst,
                a,
                b: bb,
                c,
                lane,
                ..
            } => {
                let (av, vty) = self.val(st, a);
                let x = self.as_lanes(av, vty, *lane)?;
                let y = self.as_lanes(self.val(st, bb).0, vty, *lane)?;
                let z = self.as_lanes(self.val(st, c).0, vty, *lane)?;
                let r = self.intr(
                    "llvm.fma",
                    &[x.get_type().into()],
                    &[x.into(), y.into(), z.into()],
                )?;
                st.values.insert(dst.clone(), (self.to_canon(r, vty)?, vty));
            }
            Instruction::VCmp {
                op,
                dst,
                lhs,
                rhs,
                lane,
                ..
            } => {
                let (l, vty) = self.val(st, lhs);
                let r = self.val(st, rhs).0;
                let res = self.vcmp(*op, *lane, vty, l, r)?;
                st.values.insert(dst.clone(), (res, vty));
            }
            Instruction::VSelect {
                dst,
                mask,
                then_val,
                else_val,
                ..
            } => {
                let (m, vty) = self.val(st, mask);
                let (m, t, e) = (
                    m.into_vector_value(),
                    self.val(st, then_val).0.into_vector_value(),
                    self.val(st, else_val).0.into_vector_value(),
                );
                let mt = b.build_and(m, t, "")?;
                let nm = b.build_not(m, "")?;
                let me = b.build_and(nm, e, "")?;
                let r = b.build_or(mt, me, "")?;
                st.values.insert(dst.clone(), (r.into(), vty));
            }
            Instruction::VLen { dst, lane, .. } => {
                let n = lane_count(Type::Vx, *lane);
                st.values
                    .insert(dst.clone(), (self.c64(n as i64).into(), Type::I64));
            }
            Instruction::MaskedLoad {
                dst,
                ptr,
                count,
                ty,
                lane,
                ..
            } => {
                let addr = self.int(st, ptr);
                let n = self.clamp_count(self.int(st, count), lane_count(*ty, *lane))?;
                let bytes = b.build_int_mul(n, self.c64(lane.byte_size() as i64), "")?;
                self.bounds_check(st, addr, bytes)?;
                let vt = self.lane_vec(*ty, *lane);
                let mask = self.lane_mask(n, lane_count(*ty, *lane))?;
                let p = self.int_ptr(addr)?;
                let f = self.intrinsic("llvm.masked.load", &[vt.into(), p.get_type().into()]);
                let mut args: Vec<BasicValueEnum> = vec![p.into()];
                if f.count_params() == 4 {
                    args.push(self.c32(1).into());
                }
                args.push(mask.into());
                args.push(vt.const_zero().into());
                let v = self.call1(f, &args)?;
                st.values.insert(dst.clone(), (self.to_canon(v, *ty)?, *ty));
            }
            Instruction::MaskedStore {
                ptr,
                val,
                count,
                lane,
                ..
            } => {
                let addr = self.int(st, ptr);
                let (v, vty) = self.val(st, val);
                let n = self.clamp_count(self.int(st, count), lane_count(vty, *lane))?;
                let bytes = b.build_int_mul(n, self.c64(lane.byte_size() as i64), "")?;
                self.bounds_check(st, addr, bytes)?;
                let lanes = self.as_lanes(v, vty, *lane)?;
                let mask = self.lane_mask(n, lane_count(vty, *lane))?;
                let p = self.int_ptr(addr)?;
                let f = self.intrinsic(
                    "llvm.masked.store",
                    &[lanes.get_type().into(), p.get_type().into()],
                );
                let mut args: Vec<BasicValueEnum> = vec![lanes.into(), p.into()];
                if f.count_params() == 4 {
                    args.push(self.c32(1).into());
                }
                args.push(mask.into());
                self.call(f, &args)?;
            }
            Instruction::MatMul {
                pc,
                pa,
                pb,
                m,
                n,
                k,
                dtype,
                ..
            } => {
                let regs = [pc, pa, pb, m, n, k].map(|r| self.int(st, r));
                self.matmul(st, regs, *dtype)?;
            }
        }
        Ok(())
    }

    fn lower_binary(
        &self,
        st: &FnState<'ctx>,
        op: BinaryOp,
        lhs: &str,
        rhs: &str,
    ) -> Result<(BasicValueEnum<'ctx>, Type)> {
        let b = &self.builder;
        let (l, lty) = self.val(st, lhs);
        let (r, rty) = self.val(st, rhs);

        if lty.is_float() {
            let (x, y) = (l.into_float_value(), r.into_float_value());
            let pred = match op {
                BinaryOp::Eq => Some(FloatPredicate::OEQ),
                // Cranelift's NotEqual is unordered-or-not-equal.
                BinaryOp::Ne => Some(FloatPredicate::UNE),
                BinaryOp::Lt => Some(FloatPredicate::OLT),
                BinaryOp::Gt => Some(FloatPredicate::OGT),
                BinaryOp::Le => Some(FloatPredicate::OLE),
                BinaryOp::Ge => Some(FloatPredicate::OGE),
                _ => None,
            };
            if let Some(p) = pred {
                let c = b.build_float_compare(p, x, y, "")?;
                return Ok((self.zext_bool(c)?, Type::I32));
            }
            let t = l.get_type();
            let v: BasicValueEnum = match op {
                BinaryOp::Add => b.build_float_add(x, y, "")?.into(),
                BinaryOp::Sub => b.build_float_sub(x, y, "")?.into(),
                BinaryOp::Mul => b.build_float_mul(x, y, "")?.into(),
                BinaryOp::Div => b.build_float_div(x, y, "")?.into(),
                BinaryOp::Min => self.intr("llvm.minimum", &[t], &[l, r])?,
                BinaryOp::Max => self.intr("llvm.maximum", &[t], &[l, r])?,
                _ => return Err(anyhow!("Unsupported float op {op:?}")),
            };
            return Ok((v, lty));
        }

        let (x, y) = (l.into_int_value(), r.into_int_value());
        let int_ty = x.get_type();
        let bits = int_ty.get_bit_width();
        let cmp = |p| -> Result<(BasicValueEnum<'ctx>, Type)> {
            let c = b.build_int_compare(p, x, y, "")?;
            Ok((self.zext_bool(c)?, Type::I32))
        };
        let shift_amount = || b.build_and(y, int_ty.const_int(bits as u64 - 1, false), "");
        let res_ty = match (op, lty, rty) {
            (BinaryOp::Sub, Type::Ptr, Type::Ptr) => Type::I64,
            (_, Type::Ptr, _) | (_, _, Type::Ptr) => Type::Ptr,
            _ => lty,
        };
        let v: BasicValueEnum = match op {
            BinaryOp::Add => b.build_int_add(x, y, "")?.into(),
            BinaryOp::Sub => b.build_int_sub(x, y, "")?.into(),
            BinaryOp::Mul => b.build_int_mul(x, y, "")?.into(),
            BinaryOp::Div | BinaryOp::Rem => {
                // Division by zero and MIN / -1 give 0, as on Cranelift.
                let zero = int_ty.const_zero();
                let is_zero = b.build_int_compare(IntPredicate::EQ, y, zero, "")?;
                let is_m1 =
                    b.build_int_compare(IntPredicate::EQ, y, int_ty.const_all_ones(), "")?;
                let min = int_ty.const_int(1u64 << (bits - 1), false);
                let is_min = b.build_int_compare(IntPredicate::EQ, x, min, "")?;
                let ovf = b.build_and(is_m1, is_min, "")?;
                let bad = b.build_or(is_zero, ovf, "")?;
                let d = b
                    .build_select(bad, int_ty.const_int(1, false), y, "")?
                    .into_int_value();
                let q = if op == BinaryOp::Div {
                    b.build_int_signed_div(x, d, "")?
                } else {
                    b.build_int_signed_rem(x, d, "")?
                };
                b.build_select(bad, zero, q, "")?
            }
            BinaryOp::Udiv | BinaryOp::Urem => {
                let zero = int_ty.const_zero();
                let bad = b.build_int_compare(IntPredicate::EQ, y, zero, "")?;
                let d = b
                    .build_select(bad, int_ty.const_int(1, false), y, "")?
                    .into_int_value();
                let q = if op == BinaryOp::Udiv {
                    b.build_int_unsigned_div(x, d, "")?
                } else {
                    b.build_int_unsigned_rem(x, d, "")?
                };
                b.build_select(bad, zero, q, "")?
            }
            BinaryOp::And => b.build_and(x, y, "")?.into(),
            BinaryOp::Or => b.build_or(x, y, "")?.into(),
            BinaryOp::Xor => b.build_xor(x, y, "")?.into(),
            // Shift amounts are taken modulo the bit width, as on Cranelift.
            BinaryOp::Shl => b.build_left_shift(x, shift_amount()?, "")?.into(),
            BinaryOp::Shr => b.build_right_shift(x, shift_amount()?, true, "")?.into(),
            BinaryOp::Ushr => b.build_right_shift(x, shift_amount()?, false, "")?.into(),
            BinaryOp::Min => self.intr("llvm.smin", &[int_ty.into()], &[l, r])?,
            BinaryOp::Max => self.intr("llvm.smax", &[int_ty.into()], &[l, r])?,
            BinaryOp::Umin => self.intr("llvm.umin", &[int_ty.into()], &[l, r])?,
            BinaryOp::Umax => self.intr("llvm.umax", &[int_ty.into()], &[l, r])?,
            BinaryOp::Eq => return cmp(IntPredicate::EQ),
            BinaryOp::Ne => return cmp(IntPredicate::NE),
            BinaryOp::Lt => return cmp(IntPredicate::SLT),
            BinaryOp::Gt => return cmp(IntPredicate::SGT),
            BinaryOp::Le => return cmp(IntPredicate::SLE),
            BinaryOp::Ge => return cmp(IntPredicate::SGE),
            BinaryOp::Ult => return cmp(IntPredicate::ULT),
            BinaryOp::Ugt => return cmp(IntPredicate::UGT),
            BinaryOp::Ule => return cmp(IntPredicate::ULE),
            BinaryOp::Uge => return cmp(IntPredicate::UGE),
        };
        Ok((v, res_ty))
    }

    fn lower_cast(
        &self,
        st: &FnState<'ctx>,
        op: CastOp,
        src: &str,
        ty: Type,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let (s, sty) = self.val(st, src);
        if ty.is_vector() {
            // Only same-type bitcasts are valid for vectors: a no-op.
            return Ok(s);
        }
        let target = self.scalar_type(ty);
        Ok(match op {
            CastOp::Fext if sty == Type::F16 => self.f16_to_f32(s.into_int_value())?,
            CastOp::Fext if sty == Type::BF16 => self.bf16_to_f32(s.into_int_value())?,
            CastOp::Ftrunc if ty == Type::F16 => self.f32_to_f16(s)?,
            CastOp::Ftrunc if ty == Type::BF16 => self.f32_to_bf16(s)?,
            CastOp::Bitcast => self.bitcast(s, target)?,
            CastOp::Itof => b
                .build_signed_int_to_float(s.into_int_value(), target.into_float_type(), "")?
                .into(),
            CastOp::Ftoi => {
                // Saturating, NaN -> 0; i8/i16 saturate to i32 first, as on Cranelift.
                let via = if matches!(ty, Type::I8 | Type::I16) {
                    self.i32().into()
                } else {
                    target
                };
                let v = self.intr("llvm.fptosi.sat", &[via, s.get_type()], &[s])?;
                if via == target {
                    v
                } else {
                    b.build_int_truncate(v.into_int_value(), target.into_int_type(), "")?
                        .into()
                }
            }
            CastOp::Sext => b
                .build_int_s_extend(s.into_int_value(), target.into_int_type(), "")?
                .into(),
            CastOp::Zext => b
                .build_int_z_extend(s.into_int_value(), target.into_int_type(), "")?
                .into(),
            CastOp::Trunc => b
                .build_int_truncate(s.into_int_value(), target.into_int_type(), "")?
                .into(),
            CastOp::Fext => b
                .build_float_ext(s.into_float_value(), self.ctx.f64_type(), "")?
                .into(),
            CastOp::Ftrunc => b
                .build_float_trunc(s.into_float_value(), self.ctx.f32_type(), "")?
                .into(),
        })
    }

    /// Canonical recursive-halves reduction: `reduce(v) = op(reduce(lo), reduce(hi))`,
    /// built level by level by combining even and odd lanes.
    fn reduce(
        &self,
        op: VectorReduceOp,
        lane: Type,
        mut v: VectorValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let mut n = v.get_type().get_size();
        while n > 1 {
            let half = n / 2;
            let pick = |start: u32| {
                VectorType::const_vector(
                    &(0..half)
                        .map(|i| self.c32(2 * i + start))
                        .collect::<Vec<_>>(),
                )
            };
            let even = b.build_shuffle_vector(v, v, pick(0), "")?;
            let odd = b.build_shuffle_vector(v, v, pick(1), "")?;
            v = self.combine(op, lane, even, odd)?;
            n = half;
        }
        Ok(b.build_extract_element(v, self.c32(0), "")?)
    }

    fn combine(
        &self,
        op: VectorReduceOp,
        lane: Type,
        a: VectorValue<'ctx>,
        c: VectorValue<'ctx>,
    ) -> Result<VectorValue<'ctx>> {
        let b = &self.builder;
        let t: BasicTypeEnum = a.get_type().into();
        let (x, y) = (a.into(), c.into());
        Ok(match (op, lane.is_float()) {
            (VectorReduceOp::Sum, true) => b.build_float_add(a, c, "")?,
            (VectorReduceOp::Sum, false) => b.build_int_add(a, c, "")?,
            (VectorReduceOp::Max, true) => self
                .intr("llvm.maximum", &[t], &[x, y])?
                .into_vector_value(),
            (VectorReduceOp::Max, false) => {
                self.intr("llvm.smax", &[t], &[x, y])?.into_vector_value()
            }
            (VectorReduceOp::Min, true) => self
                .intr("llvm.minimum", &[t], &[x, y])?
                .into_vector_value(),
            (VectorReduceOp::Min, false) => {
                self.intr("llvm.smin", &[t], &[x, y])?.into_vector_value()
            }
        })
    }

    fn vbinary(
        &self,
        op: VBinOp,
        lane: Type,
        vty: Type,
        l: BasicValueEnum<'ctx>,
        r: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        if op.is_bitwise() {
            let (x, y) = (l.into_vector_value(), r.into_vector_value());
            let res = match op {
                VBinOp::And => b.build_and(x, y, "")?,
                VBinOp::Or => b.build_or(x, y, "")?,
                _ => b.build_xor(x, y, "")?,
            };
            return Ok(res.into());
        }
        let x = self.as_lanes(l, vty, lane)?;
        let y = self.as_lanes(r, vty, lane)?;
        let t: BasicTypeEnum = x.get_type().into();
        let (xv, yv) = (x.into(), y.into());
        let res: BasicValueEnum = match (op, lane.is_float()) {
            (VBinOp::Add, true) => b.build_float_add(x, y, "")?.into(),
            (VBinOp::Add, false) => b.build_int_add(x, y, "")?.into(),
            (VBinOp::Sub, true) => b.build_float_sub(x, y, "")?.into(),
            (VBinOp::Sub, false) => b.build_int_sub(x, y, "")?.into(),
            (VBinOp::Mul, true) => b.build_float_mul(x, y, "")?.into(),
            (VBinOp::Mul, false) => b.build_int_mul(x, y, "")?.into(),
            (VBinOp::Min, true) => self.intr("llvm.minimum", &[t], &[xv, yv])?,
            (VBinOp::Min, false) => self.intr("llvm.smin", &[t], &[xv, yv])?,
            (VBinOp::Max, true) => self.intr("llvm.maximum", &[t], &[xv, yv])?,
            (VBinOp::Max, false) => self.intr("llvm.smax", &[t], &[xv, yv])?,
            // The validator only allows float lanes for vdiv.
            (VBinOp::Div, _) => b.build_float_div(x, y, "")?.into(),
            (VBinOp::And | VBinOp::Or | VBinOp::Xor, _) => unreachable!(),
        };
        self.to_canon(res, vty)
    }

    fn vcmp(
        &self,
        op: VCmpOp,
        lane: Type,
        vty: Type,
        l: BasicValueEnum<'ctx>,
        r: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let x = self.as_lanes(l, vty, lane)?;
        let y = self.as_lanes(r, vty, lane)?;
        let bits = if lane.is_float() {
            let p = match op {
                VCmpOp::Eq => FloatPredicate::OEQ,
                VCmpOp::Ne => FloatPredicate::UNE,
                VCmpOp::Lt => FloatPredicate::OLT,
                VCmpOp::Gt => FloatPredicate::OGT,
                VCmpOp::Le => FloatPredicate::OLE,
                VCmpOp::Ge => FloatPredicate::OGE,
            };
            b.build_float_compare(p, x, y, "")?
        } else {
            let p = match op {
                VCmpOp::Eq => IntPredicate::EQ,
                VCmpOp::Ne => IntPredicate::NE,
                VCmpOp::Lt => IntPredicate::SLT,
                VCmpOp::Gt => IntPredicate::SGT,
                VCmpOp::Le => IntPredicate::SLE,
                VCmpOp::Ge => IntPredicate::SGE,
            };
            b.build_int_compare(p, x, y, "")?
        };
        // All-ones lanes where true.
        let mask_ty = self
            .int_of_bits(lane_bits(lane))
            .vec_type(lane_count(vty, lane));
        let m = b.build_int_s_extend(bits, mask_ty, "")?;
        self.to_canon(m, vty)
    }

    /// `min(max(count, 0), lanes)`.
    fn clamp_count(&self, count: IntValue<'ctx>, lanes: u32) -> Result<IntValue<'ctx>> {
        let t: BasicTypeEnum = self.i64().into();
        let n = self.intr("llvm.smax", &[t], &[count.into(), self.c64(0).into()])?;
        Ok(self
            .intr("llvm.smin", &[t], &[n, self.c64(lanes as i64).into()])?
            .into_int_value())
    }

    /// `<lanes x i1>` mask with the first `n` lanes set.
    fn lane_mask(&self, n: IntValue<'ctx>, lanes: u32) -> Result<VectorValue<'ctx>> {
        let iota =
            VectorType::const_vector(&(0..lanes).map(|i| self.c64(i as i64)).collect::<Vec<_>>());
        let ns = self.splat(n.into(), lanes)?;
        Ok(self
            .builder
            .build_int_compare(IntPredicate::ULT, iota, ns, "")?)
    }

    /// `mm`: same loop nest, accumulation order and fuel charge as the Cranelift backend.
    fn matmul(&self, st: &FnState<'ctx>, regs: [IntValue<'ctx>; 6], dtype: Type) -> Result<()> {
        let b = &self.builder;
        let [pc, pa, pb, m, n, k] = regs;
        let is_int = dtype == Type::I8;
        let acc_ty: BasicTypeEnum = if is_int {
            self.i32().into()
        } else {
            self.ctx.f32_type().into()
        };
        let esize = dtype.byte_size() as i64;
        let elem_ty: BasicTypeEnum = match dtype {
            Type::BF16 | Type::F16 => self.i16().into(),
            Type::I8 => self.ctx.i8_type().into(),
            _ => self.ctx.f32_type().into(),
        };
        let f = st.func;
        let blk = |name| self.ctx.append_basic_block(f, name);
        let (start, i_hdr, j_hdr, k_init, k_hdr, k_body, k_done, i_next, done) = (
            blk("mm.start"),
            blk("mm.i"),
            blk("mm.j"),
            blk("mm.kinit"),
            blk("mm.k"),
            blk("mm.kbody"),
            blk("mm.kdone"),
            blk("mm.inext"),
            blk("mm.done"),
        );
        let zero = self.c64(0);
        let pos = |v| b.build_int_compare(IntPredicate::SGT, v, zero, "");
        let any = b.build_and(pos(m)?, pos(n)?, "")?;
        let any = b.build_and(any, pos(k)?, "")?;
        b.build_conditional_branch(any, start, done)?;

        b.position_at_end(start);
        if self.opts.sandbox.is_some() && !self.opts.aot {
            let next = blk("mm.checked");
            let args = [pc, pa, pb, m, n, k, self.c64(esize)].map(|v| v.into());
            self.check_hook(st, RT_SANDBOX_CHECK_MM, &args, next)?;
            b.position_at_end(next);
        }
        if self.opts.fuel && !self.opts.aot {
            // units = m*n*k / 1024 + 1, in f64 and saturated so huge shapes cannot wrap.
            let f64t = self.ctx.f64_type();
            let to_f = |v| b.build_signed_int_to_float(v, f64t, "");
            let work = b.build_float_mul(to_f(m)?, to_f(n)?, "")?;
            let work = b.build_float_mul(work, to_f(k)?, "")?;
            let units = b.build_float_div(work, f64t.const_float(1024.0), "")?;
            let units = self
                .intr(
                    "llvm.fptosi.sat",
                    &[self.i64().into(), f64t.into()],
                    &[units.into()],
                )?
                .into_int_value();
            let units = b.build_int_add(units, self.c64(1), "")?;
            let next = blk("mm.fueled");
            self.check_hook(st, RT_CONSUME_FUEL, &[units.into()], next)?;
            b.position_at_end(next);
        }
        let pre = b.get_insert_block().unwrap();
        b.build_unconditional_branch(i_hdr)?;

        // for i in 0..m
        b.position_at_end(i_hdr);
        let i = b.build_phi(self.i64(), "i")?;
        i.add_incoming(&[(&zero, pre)]);
        let iv = i.as_basic_value().into_int_value();
        let more_i = b.build_int_compare(IntPredicate::SLT, iv, m, "")?;
        b.build_conditional_branch(more_i, j_hdr, done)?;

        // for j in 0..n
        b.position_at_end(j_hdr);
        let j = b.build_phi(self.i64(), "j")?;
        j.add_incoming(&[(&zero, i_hdr)]);
        let jv = j.as_basic_value().into_int_value();
        let more_j = b.build_int_compare(IntPredicate::SLT, jv, n, "")?;
        b.build_conditional_branch(more_j, k_init, i_next)?;

        b.position_at_end(i_next);
        let i2 = b.build_int_add(iv, self.c64(1), "")?;
        i.add_incoming(&[(&i2, i_next)]);
        b.build_unconditional_branch(i_hdr)?;

        let elem_ptr = |base, row, cols, col, es: i64| -> Result<IntValue<'ctx>> {
            let idx = b.build_int_mul(row, cols, "")?;
            let idx = b.build_int_add(idx, col, "")?;
            let off = b.build_int_mul(idx, self.c64(es), "")?;
            Ok(b.build_int_add(base, off, "")?)
        };

        b.position_at_end(k_init);
        let c_ptr = elem_ptr(pc, iv, n, jv, 4)?;
        let acc0 = self.load(acc_ty, c_ptr)?;
        b.build_unconditional_branch(k_hdr)?;

        // acc = C[i][j]; for kk in 0..k: acc += A[i][kk] * B[kk][j]
        b.position_at_end(k_hdr);
        let kk = b.build_phi(self.i64(), "kk")?;
        let acc = b.build_phi(acc_ty, "acc")?;
        kk.add_incoming(&[(&zero, k_init)]);
        acc.add_incoming(&[(&acc0, k_init)]);
        let kv = kk.as_basic_value().into_int_value();
        let more_k = b.build_int_compare(IntPredicate::SLT, kv, k, "")?;
        b.build_conditional_branch(more_k, k_body, k_done)?;

        b.position_at_end(k_body);
        let a_raw = self.load(elem_ty, elem_ptr(pa, iv, k, kv, esize)?)?;
        let b_raw = self.load(elem_ty, elem_ptr(pb, kv, n, jv, esize)?)?;
        let widen = |v: BasicValueEnum<'ctx>| -> Result<BasicValueEnum<'ctx>> {
            Ok(match dtype {
                Type::BF16 => self.bf16_to_f32(v.into_int_value())?,
                Type::F16 => self.f16_to_f32(v.into_int_value())?,
                Type::I8 => b
                    .build_int_s_extend(v.into_int_value(), self.i32(), "")?
                    .into(),
                _ => v,
            })
        };
        let (av, bv) = (widen(a_raw)?, widen(b_raw)?);
        let accv = acc.as_basic_value();
        let acc2: BasicValueEnum = if is_int {
            let p = b.build_int_mul(av.into_int_value(), bv.into_int_value(), "")?;
            b.build_int_add(accv.into_int_value(), p, "")?.into()
        } else {
            let p = b.build_float_mul(av.into_float_value(), bv.into_float_value(), "")?;
            b.build_float_add(accv.into_float_value(), p, "")?.into()
        };
        let kk2 = b.build_int_add(kv, self.c64(1), "")?;
        let body_end = b.get_insert_block().unwrap();
        kk.add_incoming(&[(&kk2, body_end)]);
        acc.add_incoming(&[(&acc2, body_end)]);
        b.build_unconditional_branch(k_hdr)?;

        b.position_at_end(k_done);
        self.store(accv, elem_ptr(pc, iv, n, jv, 4)?)?;
        let j2 = b.build_int_add(jv, self.c64(1), "")?;
        j.add_incoming(&[(&j2, k_done)]);
        b.build_unconditional_branch(j_hdr)?;

        b.position_at_end(done);
        Ok(())
    }

    // ---------------------------------------------------------------------------------
    // Trampolines
    // ---------------------------------------------------------------------------------

    /// `void tramp(u64 *args, u64 *ret)`: the host calling convention shared with the
    /// Cranelift backend (`lower_trampoline`).
    fn lower_trampoline(&self, func: &Function) -> Result<()> {
        let b = &self.builder;
        let ptr_t = self.ctx.ptr_type(AddressSpace::default());
        let ty = self
            .ctx
            .void_type()
            .fn_type(&[ptr_t.into(), ptr_t.into()], false);
        let t = self
            .module
            .add_function(&trampoline_name(&func.name), ty, Some(Linkage::External));
        self.add_target_attributes(t);
        b.position_at_end(self.ctx.append_basic_block(t, "entry"));
        let args_ptr = t.get_nth_param(0).unwrap().into_pointer_value();
        let ret_ptr = t.get_nth_param(1).unwrap().into_pointer_value();
        let mut argv: Vec<BasicMetadataValueEnum> = Vec::new();
        for (i, (_, pty)) in func.params.iter().enumerate() {
            let slot = unsafe { b.build_gep(self.i64(), args_ptr, &[self.c64(i as i64)], "")? };
            let raw = b.build_load(self.i64(), slot, "")?.into_int_value();
            let v: BasicValueEnum = match pty {
                Type::I64 | Type::Ptr => raw.into(),
                Type::I32 | Type::I16 | Type::I8 => b
                    .build_int_truncate(raw, self.scalar_type(*pty).into_int_type(), "")?
                    .into(),
                Type::F64 => b.build_bit_cast(raw, self.ctx.f64_type(), "")?,
                Type::F32 => {
                    let r32 = b.build_int_truncate(raw, self.i32(), "")?;
                    b.build_bit_cast(r32, self.ctx.f32_type(), "")?
                }
                other => return Err(anyhow!("Cannot pass {other} directly in scalar trampoline")),
            };
            argv.push(v.into());
        }
        let target = self.module.get_function(&func.name).unwrap();
        let cs = b.build_call(target, &argv, "")?;
        if let Some(r_ty) = func.ret_type {
            let res = cs.try_as_basic_value().basic().unwrap();
            let raw: IntValue = match r_ty {
                Type::I64 | Type::Ptr => res.into_int_value(),
                Type::I32 | Type::I16 | Type::I8 => {
                    b.build_int_z_extend(res.into_int_value(), self.i64(), "")?
                }
                Type::F64 => b.build_bit_cast(res, self.i64(), "")?.into_int_value(),
                Type::F32 => {
                    let r32 = b.build_bit_cast(res, self.i32(), "")?.into_int_value();
                    b.build_int_z_extend(r32, self.i64(), "")?
                }
                other => {
                    return Err(anyhow!(
                        "Cannot return {other} directly in scalar trampoline"
                    ))
                }
            };
            b.build_store(ret_ptr, raw)?;
        }
        b.build_return(None)?;
        Ok(())
    }
}
