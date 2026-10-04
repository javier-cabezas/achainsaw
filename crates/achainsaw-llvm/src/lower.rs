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
    PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};

/// Runtime hook symbols referenced by JIT code (see `achainsaw-codegen/src/jit.rs`).
pub const RT_MALLOC: &str = "__achainsaw_rt_malloc";
pub const RT_FREE: &str = "__achainsaw_rt_free";
pub const RT_FUEL_EXHAUSTED: &str = "__achainsaw_rt_fuel_exhausted";
pub const RT_CONSUME_FUEL: &str = "__achainsaw_rt_consume_fuel";
pub const RT_SANDBOX_FAULT: &str = "__achainsaw_rt_sandbox_fault";
pub const RT_STACK_CHECK: &str = "__achainsaw_rt_stack_check";
pub const RT_SANDBOX_CHECK_MM: &str = "__achainsaw_rt_sandbox_check_mm";
pub const RT_PAR_FOR: &str = "__achainsaw_rt_par_for";
pub const RT_FUEL_COUNTER: &str = "__achainsaw_rt_fuel_counter";

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

/// Matrix engines `mm` may use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatrixUnits {
    /// Intel AMX with BF16 (`tdpbf16ps`), INT8 (`tdpbssd`) and FP16 (`tdpfp16ps`) support.
    pub amx_bf16: bool,
    pub amx_int8: bool,
    pub amx_fp16: bool,
    /// Arm SME outer products into ZA (streaming mode).
    pub sme: bool,
}

/// Shape of `vx` for one compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxShape {
    /// A fixed width in bits: 128, 256 (AVX2) or 512 (AVX-512).
    Fixed(u32),
    /// SVE `<vscale x 128 bits>`. `vscale_range` bounds vscale: exact for the JIT, which
    /// knows the host's vector length, and `(1, 16)` for AOT code.
    Scalable { vscale_range: (u32, u32) },
}

impl Default for VxShape {
    fn default() -> Self {
        VxShape::Fixed(128)
    }
}

#[derive(Debug, Clone, Default)]
pub struct LowerOptions {
    /// Charge fuel at every branch and before `mm`.
    pub fuel: bool,
    /// Address of the engine's i64 fuel counter, decremented inline at every branch (also
    /// used for the halt checks of sandboxed code).
    pub fuel_counter: u64,
    /// Bounds-check memory accesses and stack depth (implies the halt checks at branches).
    pub sandbox: Option<SandboxBounds>,
    /// AOT objects: `alloc`/`free` call libc `malloc`/`free`, no runtime hooks, and no
    /// trampolines.
    pub aot: bool,
    /// Value of the `target-cpu` / `target-features` function attributes.
    pub target_cpu: String,
    pub target_features: String,
    /// Width and kind of `vx`.
    pub vx: VxShape,
    /// Widest vector register to use (`prefer-vector-width` / `min-legal-vector-width`), so
    /// x86 code uses zmm/ymm registers instead of splitting wide vectors.
    pub vector_width: Option<u32>,
    /// Matrix engines for `mm`; without one, `mm` uses vector FMAs at `vx` width.
    pub matrix: MatrixUnits,
}

fn lane_bits(lane: Type) -> u32 {
    lane.bit_width().expect("lane type has a width")
}

/// Applies an expression to a fixed or scalable vector operand.
macro_rules! vec1 {
    ($a:expr, |$x:ident| $e:expr) => {
        match $a {
            BasicValueEnum::VectorValue($x) => BasicValueEnum::from($e),
            BasicValueEnum::ScalableVectorValue($x) => BasicValueEnum::from($e),
            other => unreachable!("vector operand expected, found {other:?}"),
        }
    };
}

/// Applies an integer expression to a scalar or (fixed/scalable) vector operand.
macro_rules! anyi1 {
    ($a:expr, |$x:ident| $e:expr) => {
        match $a {
            BasicValueEnum::IntValue($x) => BasicValueEnum::from($e),
            BasicValueEnum::VectorValue($x) => BasicValueEnum::from($e),
            BasicValueEnum::ScalableVectorValue($x) => BasicValueEnum::from($e),
            other => unreachable!("integer operand expected, found {other:?}"),
        }
    };
}

/// Applies an integer expression to two scalar or vector operands of the same kind.
macro_rules! anyi2 {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {
        match ($a, $b) {
            (BasicValueEnum::IntValue($x), BasicValueEnum::IntValue($y)) => {
                BasicValueEnum::from($e)
            }
            (BasicValueEnum::VectorValue($x), BasicValueEnum::VectorValue($y)) => {
                BasicValueEnum::from($e)
            }
            (BasicValueEnum::ScalableVectorValue($x), BasicValueEnum::ScalableVectorValue($y)) => {
                BasicValueEnum::from($e)
            }
            (a, b) => unreachable!("integer operands expected, found {a:?} and {b:?}"),
        }
    };
}

/// Applies a float expression to two scalar or vector operands of the same kind.
macro_rules! anyf2 {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {
        match ($a, $b) {
            (BasicValueEnum::FloatValue($x), BasicValueEnum::FloatValue($y)) => {
                BasicValueEnum::from($e)
            }
            (BasicValueEnum::VectorValue($x), BasicValueEnum::VectorValue($y)) => {
                BasicValueEnum::from($e)
            }
            (BasicValueEnum::ScalableVectorValue($x), BasicValueEnum::ScalableVectorValue($y)) => {
                BasicValueEnum::from($e)
            }
            (a, b) => unreachable!("float operands expected, found {a:?} and {b:?}"),
        }
    };
}

/// Applies an expression to two vector operands of the same kind.
macro_rules! vec2 {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {
        match ($a, $b) {
            (BasicValueEnum::VectorValue($x), BasicValueEnum::VectorValue($y)) => {
                BasicValueEnum::from($e)
            }
            (BasicValueEnum::ScalableVectorValue($x), BasicValueEnum::ScalableVectorValue($y)) => {
                BasicValueEnum::from($e)
            }
            (a, b) => unreachable!("vector operands expected, found {a:?} and {b:?}"),
        }
    };
}

#[path = "matmul.rs"]
mod matmul;

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
        // `par` workers count fuel in their own counters.
        dynamic_fuel: air.uses_par(),
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
    let mut mm_dtypes = Vec::new();
    for inst in air
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
    {
        if let Instruction::MatMul { dtype, .. } = inst {
            if !mm_dtypes.contains(dtype) {
                mm_dtypes.push(*dtype);
            }
        }
    }
    for dtype in mm_dtypes {
        lw.build_mm_helper(dtype)?;
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
    /// Functions find their fuel counter on entry (`RT_FUEL_COUNTER`) instead of using
    /// `LowerOptions::fuel_counter` directly.
    dynamic_fuel: bool,
}

/// One AIR register: its LLVM value and AIR type.
type Values<'ctx> = HashMap<String, (BasicValueEnum<'ctx>, Type)>;

/// Per-function lowering state.
struct FnState<'ctx> {
    func: FunctionValue<'ctx>,
    entry: BasicBlock<'ctx>,
    values: Values<'ctx>,
    blocks: HashMap<String, (BasicBlock<'ctx>, Vec<PhiValue<'ctx>>)>,
    /// Returns zeroes once a runtime check fails; the host then reports the status.
    trap: Option<BasicBlock<'ctx>>,
    /// Reports a sandbox violation `(addr, size)` and jumps to `trap`.
    fault: Option<(BasicBlock<'ctx>, PhiValue<'ctx>, PhiValue<'ctx>)>,
    /// Address of this thread's fuel counter, found on entry (see `dynamic_fuel`).
    fuel_counter: Option<IntValue<'ctx>>,
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

    /// Whether `ty` is a scalable (SVE) vector in this compilation.
    fn scalable(&self, ty: Type) -> bool {
        ty == Type::Vx && matches!(self.opts.vx, VxShape::Scalable { .. })
    }

    /// Width of a vector type in bits; for a scalable `vx`, the width per vscale unit.
    fn vec_bits(&self, ty: Type) -> u32 {
        match ty {
            Type::V256 => 256,
            Type::V512 => 512,
            Type::Vx => match self.opts.vx {
                VxShape::Fixed(bits) => bits,
                VxShape::Scalable { .. } => 128,
            },
            _ => 128,
        }
    }

    /// Lane count, or lanes per vscale unit for a scalable `vx`.
    fn lanes_min(&self, ty: Type, lane: Type) -> u32 {
        self.vec_bits(ty) / lane_bits(lane)
    }

    fn vec_of(&self, elem: BasicTypeEnum<'ctx>, n: u32, scalable: bool) -> BasicTypeEnum<'ctx> {
        match (elem, scalable) {
            (BasicTypeEnum::IntType(t), false) => t.vec_type(n).into(),
            (BasicTypeEnum::IntType(t), true) => t.scalable_vec_type(n).into(),
            (BasicTypeEnum::FloatType(t), false) => t.vec_type(n).into(),
            (BasicTypeEnum::FloatType(t), true) => t.scalable_vec_type(n).into(),
            (other, _) => unreachable!("no vectors of {other:?}"),
        }
    }

    /// Canonical representation of an (untyped) AIR vector: i32 lanes.
    fn canon_vec(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        self.vec_of(self.i32().into(), self.vec_bits(ty) / 32, self.scalable(ty))
    }

    fn lane_vec(&self, ty: Type, lane: Type) -> BasicTypeEnum<'ctx> {
        self.vec_of(
            self.scalar_type(lane),
            self.lanes_min(ty, lane),
            self.scalable(ty),
        )
    }

    /// `i1` mask with one bit per `lane` of `ty`.
    fn mask_vec(&self, ty: Type, lane: Type) -> BasicTypeEnum<'ctx> {
        self.vec_of(
            self.i1().into(),
            self.lanes_min(ty, lane),
            self.scalable(ty),
        )
    }

    /// Number of `lane` lanes in `ty` as an i64, computed from vscale when scalable.
    fn lanes_value(&self, ty: Type, lane: Type) -> Result<IntValue<'ctx>> {
        let n = self.c64(self.lanes_min(ty, lane) as i64);
        if !self.scalable(ty) {
            return Ok(n);
        }
        let vscale = self
            .intr("llvm.vscale", &[self.i64().into()], &[])?
            .into_int_value();
        Ok(self.builder.build_int_mul(vscale, n, "")?)
    }

    /// Size of a vector of type `ty` in bytes, as an i64.
    fn vec_bytes(&self, ty: Type) -> Result<IntValue<'ctx>> {
        self.lanes_value(ty, Type::I8)
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
        if let Some(width) = self.opts.vector_width {
            for k in ["prefer-vector-width", "min-legal-vector-width"] {
                f.add_attribute(
                    AttributeLoc::Function,
                    self.ctx.create_string_attribute(k, &width.to_string()),
                );
            }
        }
        if let VxShape::Scalable {
            vscale_range: (min, max),
        } = self.opts.vx
        {
            f.add_attribute(
                AttributeLoc::Function,
                self.ctx.create_enum_attribute(
                    Attribute::get_named_enum_kind_id("vscale_range"),
                    ((min as u64) << 32) | max as u64,
                ),
            );
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
        decl(RT_FUEL_EXHAUSTED, i32t.fn_type(&[], false));
        decl(RT_CONSUME_FUEL, i32t.fn_type(&[i64t.into()], false));
        decl(
            RT_SANDBOX_FAULT,
            void.fn_type(&[i64t.into(), i64t.into()], false),
        );
        decl(RT_STACK_CHECK, i32t.fn_type(&[], false));
        decl(RT_SANDBOX_CHECK_MM, i32t.fn_type(&[i64t.into(); 7], false));
        decl(RT_PAR_FOR, i32t.fn_type(&[i64t.into(); 4], false));
        decl(RT_FUEL_COUNTER, i64t.fn_type(&[], false));
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

    fn as_lanes(
        &self,
        v: BasicValueEnum<'ctx>,
        ty: Type,
        lane: Type,
    ) -> Result<BasicValueEnum<'ctx>> {
        self.bitcast(v, self.lane_vec(ty, lane))
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

    /// Splats `x` (of AIR type `lane`) into every lane of a vector of type `ty`.
    fn splat(&self, x: BasicValueEnum<'ctx>, lane: Type, ty: Type) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let n = self.lanes_min(ty, lane);
        let mask = self.vec_of(self.i32().into(), n, self.scalable(ty));
        Ok(match self.lane_vec(ty, lane) {
            BasicTypeEnum::VectorType(vt) => {
                let one = b.build_insert_element(vt.get_poison(), x, self.c32(0), "")?;
                let zeros = mask.into_vector_type().const_zero();
                b.build_shuffle_vector(one, vt.get_poison(), zeros, "")?
                    .into()
            }
            BasicTypeEnum::ScalableVectorType(vt) => {
                let one = b.build_insert_element(vt.get_poison(), x, self.c32(0), "")?;
                let zeros = mask.into_scalable_vector_type().const_zero();
                b.build_shuffle_vector(one, vt.get_poison(), zeros, "")?
                    .into()
            }
            other => unreachable!("{other:?}"),
        })
    }

    // ---------------------------------------------------------------------------------
    // f16 / bf16 (bit-exact ports of the Cranelift sequences)
    // ---------------------------------------------------------------------------------

    /// Lane count and scalability of a vector value; `None` for scalars.
    fn shape_of(v: BasicValueEnum<'ctx>) -> Option<(u32, bool)> {
        match v {
            BasicValueEnum::VectorValue(x) => Some((x.get_type().get_size(), false)),
            BasicValueEnum::ScalableVectorValue(x) => Some((x.get_type().get_size(), true)),
            _ => None,
        }
    }

    /// `elem`, or a vector of `elem` with the given shape.
    fn shaped(&self, elem: BasicTypeEnum<'ctx>, shape: Option<(u32, bool)>) -> BasicTypeEnum<'ctx> {
        match shape {
            None => elem,
            Some((n, scalable)) => self.vec_of(elem, n, scalable),
        }
    }

    /// Constant `c` (a scalar constant), splatted to `shape`.
    fn splat_const(
        &self,
        c: BasicValueEnum<'ctx>,
        shape: Option<(u32, bool)>,
    ) -> Result<BasicValueEnum<'ctx>> {
        Ok(match shape {
            None => c,
            Some((n, false)) => VectorType::const_vector(&vec![c; n as usize]).into(),
            Some((n, true)) => {
                let b = &self.builder;
                let vt = self
                    .vec_of(c.get_type(), n, true)
                    .into_scalable_vector_type();
                let one = b.build_insert_element(vt.get_poison(), c, self.c32(0), "")?;
                let zeros = self.i32().scalable_vec_type(n).const_zero();
                b.build_shuffle_vector(one, vt.get_poison(), zeros, "")?
                    .into()
            }
        })
    }

    /// Zero-extends integer lanes (or a scalar) to i32.
    fn zext_to_i32(&self, v: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let t = self.shaped(self.i32().into(), Self::shape_of(v));
        Ok(match v {
            BasicValueEnum::IntValue(x) => b.build_int_z_extend(x, t.into_int_type(), "")?.into(),
            BasicValueEnum::VectorValue(x) => {
                b.build_int_z_extend(x, t.into_vector_type(), "")?.into()
            }
            BasicValueEnum::ScalableVectorValue(x) => b
                .build_int_z_extend(x, t.into_scalable_vector_type(), "")?
                .into(),
            other => unreachable!("{other:?}"),
        })
    }

    /// binary16 bits (i16, or i16 lanes) -> f32. Exact; NaN payloads are preserved. The same
    /// integer sequence as Cranelift's `f16_to_f32`, so both backends agree bit for bit.
    fn f16_to_f32(&self, h: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let shape = Self::shape_of(h);
        let k = |v: u32| self.splat_const(self.c32(v).into(), shape);
        let x = self.zext_to_i32(h)?;
        let sign = anyi2!(x, k(0x8000)?, |p, q| b.build_and(p, q, "")?);
        let sign = anyi2!(sign, k(16)?, |p, q| b.build_left_shift(p, q, "")?);
        let exp = anyi2!(x, k(10)?, |p, q| b.build_right_shift(p, q, false, "")?);
        let exp = anyi2!(exp, k(0x1f)?, |p, q| b.build_and(p, q, "")?);
        let mant = anyi2!(x, k(0x3ff)?, |p, q| b.build_and(p, q, "")?);
        let mant13 = anyi2!(mant, k(13)?, |p, q| b.build_left_shift(p, q, "")?);
        let exp32 = anyi2!(exp, k(112)?, |p, q| b.build_int_add(p, q, "")?);
        let exp32 = anyi2!(exp32, k(23)?, |p, q| b.build_left_shift(p, q, "")?);
        let normal = anyi2!(exp32, mant13, |p, q| b.build_or(p, q, "")?);
        let infnan = anyi2!(mant13, k(0x7f80_0000)?, |p, q| b.build_or(p, q, "")?);
        // Zero/subnormal: mant * 2^-24, exact in f32.
        let f32t = self.shaped(self.ctx.f32_type().into(), shape);
        let mant_f: BasicValueEnum = match mant {
            BasicValueEnum::IntValue(m) => b
                .build_unsigned_int_to_float(m, f32t.into_float_type(), "")?
                .into(),
            BasicValueEnum::VectorValue(m) => b
                .build_unsigned_int_to_float(m, f32t.into_vector_type(), "")?
                .into(),
            BasicValueEnum::ScalableVectorValue(m) => b
                .build_unsigned_int_to_float(m, f32t.into_scalable_vector_type(), "")?
                .into(),
            other => unreachable!("{other:?}"),
        };
        let scale = self.splat_const(
            self.ctx
                .f32_type()
                .const_float(f32::from_bits(0x3380_0000) as f64)
                .into(),
            shape,
        )?;
        let sub_f = anyf2!(mant_f, scale, |p, q| b.build_float_mul(p, q, "")?);
        let sub = self.bitcast(sub_f, x.get_type())?;
        let is_sub = anyi2!(exp, k(0)?, |p, q| b.build_int_compare(
            IntPredicate::EQ,
            p,
            q,
            ""
        )?);
        let is_max = anyi2!(exp, k(31)?, |p, q| b.build_int_compare(
            IntPredicate::EQ,
            p,
            q,
            ""
        )?);
        let mag = anyi1!(is_max, |c| b.build_select(c, infnan, normal, "")?);
        let mag = anyi1!(is_sub, |c| b.build_select(c, sub, mag, "")?);
        let bits = anyi2!(mag, sign, |p, q| b.build_or(p, q, "")?);
        self.bitcast(bits, f32t)
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

    /// bfloat16 bits (i16, or i16 lanes) -> f32. Exact.
    fn bf16_to_f32(&self, h: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let shape = Self::shape_of(h);
        let x = self.zext_to_i32(h)?;
        let sixteen = self.splat_const(self.c32(16).into(), shape)?;
        let x = anyi2!(x, sixteen, |p, q| b.build_left_shift(p, q, "")?);
        self.bitcast(x, self.shaped(self.ctx.f32_type().into(), shape))
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
        // Decrement the engine's counter in place; only when it reaches zero ask the runtime
        // whether to unwind (budget spent, or a failure was recorded).
        let b = &self.builder;
        let addr = st
            .fuel_counter
            .unwrap_or_else(|| self.c64(self.opts.fuel_counter as i64));
        let fuel = self.load(self.i64().into(), addr)?.into_int_value();
        let left = b.build_int_sub(fuel, self.c64(1), "fuel")?;
        self.store(left.into(), addr)?;
        let out = b.build_int_compare(IntPredicate::SLE, left, self.c64(0), "")?;
        let out = self
            .intr(
                "llvm.expect",
                &[self.i1().into()],
                &[out.into(), self.i1().const_zero().into()],
            )?
            .into_int_value();
        let slow = self.ctx.append_basic_block(st.func, "fuel.out");
        let cont = self.ctx.append_basic_block(st.func, "fueled");
        b.build_conditional_branch(out, slow, cont)?;
        b.position_at_end(slow);
        self.check_hook(st, RT_FUEL_EXHAUSTED, &[], cont)?;
        b.position_at_end(cont);
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
            entry,
            values: HashMap::new(),
            blocks,
            trap,
            fault,
            fuel_counter: None,
        };
        for (i, (name, ty)) in func.params.iter().enumerate() {
            st.values
                .insert(name.clone(), (f.get_nth_param(i as u32).unwrap(), *ty));
        }

        b.position_at_end(entry);
        let has_branches = func
            .blocks
            .iter()
            .any(|b| !matches!(b.terminator, Terminator::Ret { .. }));
        if checks && self.dynamic_fuel && has_branches {
            // This thread's counter, or the engine's when no engine call is active.
            let active = self
                .call1(self.runtime_fn(RT_FUEL_COUNTER), &[])?
                .into_int_value();
            let none = b.build_int_compare(IntPredicate::EQ, active, self.c64(0), "")?;
            let engine = self.c64(self.opts.fuel_counter as i64);
            st.fuel_counter = Some(
                b.build_select(none, engine, active, "fuel.counter")?
                    .into_int_value(),
            );
        }
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
                let bytes = self.access_bytes(*ty)?;
                self.bounds_check(st, addr, bytes)?;
                let v = self.load(self.air_type(*ty), addr)?;
                st.values.insert(dst.clone(), (v, *ty));
            }
            Instruction::Store { ptr, val, .. } => {
                let addr = self.int(st, ptr);
                let (v, vty) = self.val(st, val);
                let bytes = self.access_bytes(vty)?;
                self.bounds_check(st, addr, bytes)?;
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
                let v = self.splat(s, sty, *ty)?;
                st.values.insert(dst.clone(), (self.to_canon(v, *ty)?, *ty));
            }
            Instruction::ExtractLane {
                dst, vec, lane, ty, ..
            } => {
                let (v, vty) = self.val(st, vec);
                let lanes = self.as_lanes(v, vty, *ty)?;
                let x = vec1!(lanes, |l| b.build_extract_element(
                    l,
                    self.c32(*lane),
                    ""
                )?);
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
                let r = self.reduce(st, *op, *ty, vty, self.as_lanes(v, vty, *ty)?)?;
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
                let r = self.intr("llvm.fma", &[x.get_type()], &[x, y, z])?;
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
                let (t, e) = (self.val(st, then_val).0, self.val(st, else_val).0);
                let mt = vec2!(m, t, |x, y| b.build_and(x, y, "")?);
                let nm = vec1!(m, |x| b.build_not(x, "")?);
                let me = vec2!(nm, e, |x, y| b.build_and(x, y, "")?);
                let r = vec2!(mt, me, |x, y| b.build_or(x, y, "")?);
                st.values.insert(dst.clone(), (r, vty));
            }
            Instruction::VLen { dst, lane, .. } => {
                let n = self.lanes_value(Type::Vx, *lane)?;
                st.values.insert(dst.clone(), (n.into(), Type::I64));
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
                let n = self.clamp_count(self.int(st, count), self.lanes_value(*ty, *lane)?)?;
                let bytes = b.build_int_mul(n, self.c64(lane.byte_size() as i64), "")?;
                self.bounds_check(st, addr, bytes)?;
                let vt = self.lane_vec(*ty, *lane);
                let mask = self.lane_mask(n, *ty, *lane)?;
                let p = self.int_ptr(addr)?;
                let f = self.intrinsic("llvm.masked.load", &[vt, p.get_type().into()]);
                let mut args: Vec<BasicValueEnum> = vec![p.into()];
                if f.count_params() == 4 {
                    args.push(self.c32(1).into());
                }
                args.push(mask);
                args.push(vt.const_zero());
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
                let n = self.clamp_count(self.int(st, count), self.lanes_value(vty, *lane)?)?;
                let bytes = b.build_int_mul(n, self.c64(lane.byte_size() as i64), "")?;
                self.bounds_check(st, addr, bytes)?;
                let lanes = self.as_lanes(v, vty, *lane)?;
                let mask = self.lane_mask(n, vty, *lane)?;
                let p = self.int_ptr(addr)?;
                let f = self.intrinsic(
                    "llvm.masked.store",
                    &[lanes.get_type(), p.get_type().into()],
                );
                let mut args: Vec<BasicValueEnum> = vec![lanes, p.into()];
                if f.count_params() == 4 {
                    args.push(self.c32(1).into());
                }
                args.push(mask);
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
            Instruction::Par {
                count, func, args, ..
            } => {
                let n = self.int(st, count);
                if self.opts.aot {
                    self.serial_par(st, n, func, args)?;
                } else {
                    self.par(st, n, func, args)?;
                }
            }
        }
        Ok(())
    }

    /// `par n, f(args)`: packs the arguments in trampoline layout (slot 0 is the index,
    /// set by the runtime) and hands `f`'s trampoline to `RT_PAR_FOR`.
    fn par(
        &self,
        st: &FnState<'ctx>,
        n: IntValue<'ctx>,
        func: &str,
        args: &[String],
    ) -> Result<()> {
        let b = &self.builder;
        let slots = 1 + args.len() as u32;
        // In the entry block so a `par` inside a loop reuses one buffer.
        let entry_b = self.ctx.create_builder();
        match st.entry.get_terminator() {
            Some(t) => entry_b.position_before(&t),
            None => entry_b.position_at_end(st.entry),
        }
        let buf = entry_b.build_alloca(self.i64().array_type(slots), "par.args")?;
        for (j, arg) in args.iter().enumerate() {
            let (v, ty) = self.val(st, arg);
            let raw: IntValue = match ty {
                Type::I32 | Type::I16 | Type::I8 => {
                    b.build_int_z_extend(v.into_int_value(), self.i64(), "")?
                }
                Type::F64 => b.build_bit_cast(v, self.i64(), "")?.into_int_value(),
                Type::F32 => {
                    let bits = b.build_bit_cast(v, self.i32(), "")?.into_int_value();
                    b.build_int_z_extend(bits, self.i64(), "")?
                }
                _ => v.into_int_value(),
            };
            let slot = unsafe { b.build_gep(self.i64(), buf, &[self.c64(j as i64 + 1)], "")? };
            b.build_store(slot, raw)?;
        }
        let tramp = self.trampoline_fn(func);
        let tramp =
            b.build_ptr_to_int(tramp.as_global_value().as_pointer_value(), self.i64(), "")?;
        let buf = b.build_ptr_to_int(buf, self.i64(), "")?;
        let hook_args = [tramp, buf, self.c64(slots as i64), n].map(|v| v.into());
        if st.trap.is_some() {
            let next = self.ctx.append_basic_block(st.func, "par.done");
            self.check_hook(st, RT_PAR_FOR, &hook_args, next)?;
            b.position_at_end(next);
        } else {
            self.call(self.runtime_fn(RT_PAR_FOR), &hook_args)?;
        }
        Ok(())
    }

    /// `par n, f(args)` as a counted loop calling `f(i, args)` in order (AOT code, which has
    /// no runtime to run it in parallel; any order is a valid execution of `par`).
    fn serial_par(
        &self,
        st: &FnState<'ctx>,
        n: IntValue<'ctx>,
        func: &str,
        args: &[String],
    ) -> Result<()> {
        let b = &self.builder;
        let callee = self
            .module
            .get_function(func)
            .ok_or_else(|| anyhow!("Unknown function '{func}' in par"))?;
        let pre = b.get_insert_block().unwrap();
        let header = self.ctx.append_basic_block(st.func, "par.header");
        let body = self.ctx.append_basic_block(st.func, "par.body");
        let done = self.ctx.append_basic_block(st.func, "par.done");
        b.build_unconditional_branch(header)?;

        b.position_at_end(header);
        let i = b.build_phi(self.i64(), "par.i")?;
        i.add_incoming(&[(&self.c64(0), pre)]);
        let iv = i.as_basic_value().into_int_value();
        let more = b.build_int_compare(IntPredicate::SLT, iv, n, "")?;
        b.build_conditional_branch(more, body, done)?;

        b.position_at_end(body);
        let mut argv: Vec<BasicMetadataValueEnum> = vec![iv.into()];
        argv.extend(
            args.iter()
                .map(|a| BasicMetadataValueEnum::from(st.values[a].0)),
        );
        let cs = b.build_call(callee, &argv, "")?;
        cs.set_tail_call_kind(LLVMTailCallKind::LLVMTailCallKindNoTail);
        let next = b.build_int_add(iv, self.c64(1), "")?;
        i.add_incoming(&[(&next, b.get_insert_block().unwrap())]);
        b.build_unconditional_branch(header)?;

        b.position_at_end(done);
        Ok(())
    }

    /// Declaration of `func`'s scalar trampoline (defined by `lower_trampoline`).
    fn trampoline_fn(&self, func: &str) -> FunctionValue<'ctx> {
        let name = trampoline_name(func);
        self.module.get_function(&name).unwrap_or_else(|| {
            let ptr_t = self.ctx.ptr_type(AddressSpace::default());
            let ty = self
                .ctx
                .void_type()
                .fn_type(&[ptr_t.into(), ptr_t.into()], false);
            self.module.add_function(&name, ty, Some(Linkage::External))
        })
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
            CastOp::Fext if sty == Type::F16 => self.f16_to_f32(s)?,
            CastOp::Fext if sty == Type::BF16 => self.bf16_to_f32(s)?,
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

    /// Bytes a plain `ld`/`st` of `ty` touches.
    fn access_bytes(&self, ty: Type) -> Result<IntValue<'ctx>> {
        if ty.is_vector() {
            self.vec_bytes(ty)
        } else {
            Ok(self.c64(ty.byte_size() as i64))
        }
    }

    /// Canonical recursive-halves reduction: `reduce(v) = op(reduce(lo), reduce(hi))`,
    /// which combines adjacent lanes first. Fixed vectors combine even and odd lanes level
    /// by level with shuffles; scalable vectors run the same tree through memory.
    fn reduce(
        &self,
        st: &FnState<'ctx>,
        op: VectorReduceOp,
        lane: Type,
        vty: Type,
        v: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let mut v = match v {
            BasicValueEnum::VectorValue(v) => v,
            _ => return self.reduce_in_memory(st, op, lane, vty, v),
        };
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
            v = self
                .combine(op, lane, even.into(), odd.into())?
                .into_vector_value();
            n = half;
        }
        Ok(b.build_extract_element(v, self.c32(0), "")?)
    }

    /// Reduction of a scalable vector: spills it to a stack slot and repeatedly combines
    /// adjacent pairs in place (`buf[i] = op(buf[2i], buf[2i+1])`) until one lane is left.
    /// For power-of-two lane counts (every SVE vector length in use) this is exactly the
    /// recursive-halves tree; an odd lane out is carried to the next level unchanged.
    fn reduce_in_memory(
        &self,
        st: &FnState<'ctx>,
        op: VectorReduceOp,
        lane: Type,
        vty: Type,
        v: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let elem = self.scalar_type(lane);
        let i64t = self.i64();
        // The slot lives in the entry block so reductions inside loops reuse it.
        let entry_b = self.ctx.create_builder();
        match st.entry.get_terminator() {
            Some(t) => entry_b.position_before(&t),
            None => entry_b.position_at_end(st.entry),
        }
        let buf = entry_b.build_alloca(v.get_type(), "reduce.buf")?;
        b.build_store(buf, v)?;
        let n0 = self.lanes_value(vty, lane)?;
        let at = |i: IntValue<'ctx>| unsafe { b.build_gep(elem, buf, &[i], "") };

        let f = st.func;
        let (lvl, pairs, body, next_lvl, done) = (
            self.ctx.append_basic_block(f, "reduce.level"),
            self.ctx.append_basic_block(f, "reduce.pairs"),
            self.ctx.append_basic_block(f, "reduce.body"),
            self.ctx.append_basic_block(f, "reduce.next"),
            self.ctx.append_basic_block(f, "reduce.done"),
        );
        let pre = b.get_insert_block().unwrap();
        b.build_unconditional_branch(lvl)?;

        // while n > 1
        b.position_at_end(lvl);
        let n = b.build_phi(i64t, "n")?;
        n.add_incoming(&[(&n0, pre)]);
        let nv = n.as_basic_value().into_int_value();
        let more = b.build_int_compare(IntPredicate::UGT, nv, self.c64(1), "")?;
        let half = b.build_int_add(nv, self.c64(1), "")?;
        let half = b.build_right_shift(half, self.c64(1), false, "")?;
        b.build_conditional_branch(more, pairs, done)?;

        // for i in 0..ceil(n/2)
        b.position_at_end(pairs);
        let i = b.build_phi(i64t, "i")?;
        i.add_incoming(&[(&self.c64(0), lvl)]);
        let iv = i.as_basic_value().into_int_value();
        let more_i = b.build_int_compare(IntPredicate::ULT, iv, half, "")?;
        b.build_conditional_branch(more_i, body, next_lvl)?;

        b.position_at_end(body);
        let j = b.build_int_add(iv, iv, "")?;
        let j1 = b.build_int_add(j, self.c64(1), "")?;
        let x = b.build_load(elem, at(j)?, "")?;
        // With an odd lane count the last lane has no partner and is carried over as-is;
        // reload buf[j] instead of reading past the vector, then keep x.
        let y_idx_ok = b.build_int_compare(IntPredicate::ULT, j1, nv, "")?;
        let safe_j1 = b.build_select(y_idx_ok, j1, j, "")?.into_int_value();
        let y = b.build_load(elem, at(safe_j1)?, "")?;
        let r = self.combine(op, lane, x, y)?;
        let r = b.build_select(y_idx_ok, r, x, "")?;
        b.build_store(at(iv)?, r)?;
        let i2 = b.build_int_add(iv, self.c64(1), "")?;
        i.add_incoming(&[(&i2, body)]);
        b.build_unconditional_branch(pairs)?;

        b.position_at_end(next_lvl);
        n.add_incoming(&[(&half, next_lvl)]);
        b.build_unconditional_branch(lvl)?;

        b.position_at_end(done);
        Ok(b.build_load(elem, buf, "")?)
    }

    /// One reduction step on scalars or vectors of `lane` values.
    fn combine(
        &self,
        op: VectorReduceOp,
        lane: Type,
        a: BasicValueEnum<'ctx>,
        c: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        let t = a.get_type();
        Ok(match (op, lane.is_float(), a, c) {
            (
                VectorReduceOp::Sum,
                true,
                BasicValueEnum::FloatValue(x),
                BasicValueEnum::FloatValue(y),
            ) => b.build_float_add(x, y, "")?.into(),
            (
                VectorReduceOp::Sum,
                false,
                BasicValueEnum::IntValue(x),
                BasicValueEnum::IntValue(y),
            ) => b.build_int_add(x, y, "")?.into(),
            (VectorReduceOp::Sum, true, _, _) => vec2!(a, c, |x, y| b.build_float_add(x, y, "")?),
            (VectorReduceOp::Sum, false, _, _) => vec2!(a, c, |x, y| b.build_int_add(x, y, "")?),
            (VectorReduceOp::Max, true, _, _) => self.intr("llvm.maximum", &[t], &[a, c])?,
            (VectorReduceOp::Max, false, _, _) => self.intr("llvm.smax", &[t], &[a, c])?,
            (VectorReduceOp::Min, true, _, _) => self.intr("llvm.minimum", &[t], &[a, c])?,
            (VectorReduceOp::Min, false, _, _) => self.intr("llvm.smin", &[t], &[a, c])?,
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
            return Ok(match op {
                VBinOp::And => vec2!(l, r, |x, y| b.build_and(x, y, "")?),
                VBinOp::Or => vec2!(l, r, |x, y| b.build_or(x, y, "")?),
                _ => vec2!(l, r, |x, y| b.build_xor(x, y, "")?),
            });
        }
        let x = self.as_lanes(l, vty, lane)?;
        let y = self.as_lanes(r, vty, lane)?;
        let t = x.get_type();
        let res = match (op, lane.is_float()) {
            (VBinOp::Add, true) => vec2!(x, y, |a, c| b.build_float_add(a, c, "")?),
            (VBinOp::Add, false) => vec2!(x, y, |a, c| b.build_int_add(a, c, "")?),
            (VBinOp::Sub, true) => vec2!(x, y, |a, c| b.build_float_sub(a, c, "")?),
            (VBinOp::Sub, false) => vec2!(x, y, |a, c| b.build_int_sub(a, c, "")?),
            (VBinOp::Mul, true) => vec2!(x, y, |a, c| b.build_float_mul(a, c, "")?),
            (VBinOp::Mul, false) => vec2!(x, y, |a, c| b.build_int_mul(a, c, "")?),
            (VBinOp::Min, true) => self.intr("llvm.minimum", &[t], &[x, y])?,
            (VBinOp::Min, false) => self.intr("llvm.smin", &[t], &[x, y])?,
            (VBinOp::Max, true) => self.intr("llvm.maximum", &[t], &[x, y])?,
            (VBinOp::Max, false) => self.intr("llvm.smax", &[t], &[x, y])?,
            // The validator only allows float lanes for vdiv.
            (VBinOp::Div, _) => vec2!(x, y, |a, c| b.build_float_div(a, c, "")?),
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
            vec2!(x, y, |a, c| b.build_float_compare(p, a, c, "")?)
        } else {
            let p = match op {
                VCmpOp::Eq => IntPredicate::EQ,
                VCmpOp::Ne => IntPredicate::NE,
                VCmpOp::Lt => IntPredicate::SLT,
                VCmpOp::Gt => IntPredicate::SGT,
                VCmpOp::Le => IntPredicate::SLE,
                VCmpOp::Ge => IntPredicate::SGE,
            };
            vec2!(x, y, |a, c| b.build_int_compare(p, a, c, "")?)
        };
        // All-ones lanes where true.
        let mask_ty = self.vec_of(
            self.int_of_bits(lane_bits(lane)).into(),
            self.lanes_min(vty, lane),
            self.scalable(vty),
        );
        let m: BasicValueEnum = match bits {
            BasicValueEnum::VectorValue(m) => b
                .build_int_s_extend(m, mask_ty.into_vector_type(), "")?
                .into(),
            BasicValueEnum::ScalableVectorValue(m) => b
                .build_int_s_extend(m, mask_ty.into_scalable_vector_type(), "")?
                .into(),
            other => unreachable!("{other:?}"),
        };
        self.to_canon(m, vty)
    }

    /// `min(max(count, 0), lanes)`.
    fn clamp_count(&self, count: IntValue<'ctx>, lanes: IntValue<'ctx>) -> Result<IntValue<'ctx>> {
        let t: BasicTypeEnum = self.i64().into();
        let n = self.intr("llvm.smax", &[t], &[count.into(), self.c64(0).into()])?;
        Ok(self
            .intr("llvm.smin", &[t], &[n, lanes.into()])?
            .into_int_value())
    }

    /// Mask with the first `n` lanes of `ty` set (`llvm.get.active.lane.mask`, which
    /// becomes `whilelo` on SVE and a k-mask on AVX-512).
    fn lane_mask(&self, n: IntValue<'ctx>, ty: Type, lane: Type) -> Result<BasicValueEnum<'ctx>> {
        self.intr(
            "llvm.get.active.lane.mask",
            &[self.mask_vec(ty, lane), self.i64().into()],
            &[self.c64(0).into(), n.into()],
        )
    }

    /// `mm`: the same checks and fuel charge as the Cranelift backend, then a call to the
    /// dtype's kernel (see `matmul.rs`). Non-positive dimensions are a no-op.
    fn matmul(&self, st: &FnState<'ctx>, regs: [IntValue<'ctx>; 6], dtype: Type) -> Result<()> {
        let b = &self.builder;
        let [pc, pa, pb, m, n, k] = regs;
        let esize = dtype.byte_size() as i64;
        let f = st.func;
        let blk = |name| self.ctx.append_basic_block(f, name);
        let (start, done) = (blk("mm.start"), blk("mm.done"));
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
        let helper = self
            .module
            .get_function(&Self::mm_helper_name(dtype))
            .expect("mm helper defined");
        let cs = b.build_call(helper, &regs.map(|r| r.into()), "")?;
        cs.set_tail_call_kind(LLVMTailCallKind::LLVMTailCallKindNoTail);
        b.build_unconditional_branch(done)?;

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
        let t = self.trampoline_fn(&func.name);
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
