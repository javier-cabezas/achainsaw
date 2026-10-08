//! `mm` kernels for the LLVM backend.
//!
//! `mm pc, pa, pb, m, n, k:dtype` lowers to a call to a per-dtype helper,
//! `__achainsaw_mm_<dtype>(pc, pa, pb, m, n, k)`, made only after the caller has run the
//! sandbox check and charged fuel, so C is untouched when either fails. The helper uses the
//! best kernel the target has:
//! - **AMX** (x86 with `amx-bf16`, `amx-int8` or `amx-fp16`): 16x16 tile dot products.
//! - **SME** (AArch64 with `sme`): ZA-tile outer products in streaming mode.
//! - **Vector FMA** otherwise: each `vx`-wide strip of a C row accumulates
//!   `broadcast(A[i][kk]) * B[kk][strip]` over `kk`, with masked tails.
//!
//! Float results agree with the scalar reference within rounding error (fused or widened
//! products round differently); i8 results are exact. bf16 uses AMX or SME only with
//! `LowerOptions::fast_math`: AMX's `tdpbf16ps` and SME's `bfmopa` (as BFDOT, without
//! FEAT_EBF16's extended behavior) treat bf16 subnormal inputs as zero, while AIR widens every
//! bf16 value exactly. AMX-FP16, SME's widening f16 and f32 outer products, and the int8
//! engines are exact, so they are always used.

use super::*;

/// Kernel family used for one dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MmKernel {
    Fma,
    Sme,
    Amx,
}

impl<'a, 'ctx> ModuleLowerer<'a, 'ctx> {
    pub(super) fn mm_helper_name(dtype: Type) -> String {
        format!("__achainsaw_mm_{dtype}")
    }

    pub(super) fn mm_kernel(&self, dtype: Type) -> MmKernel {
        let units = self.opts.matrix;
        // The bf16 matrix engines flush subnormal inputs (see the module docs).
        let exact = dtype != Type::BF16 || self.opts.fast_math;
        let amx = match dtype {
            Type::BF16 => units.amx_bf16,
            Type::I8 => units.amx_int8,
            Type::F16 => units.amx_fp16,
            _ => false,
        };
        if amx && exact {
            MmKernel::Amx
        } else if units.sme && exact {
            MmKernel::Sme
        } else {
            MmKernel::Fma
        }
    }

    /// Defines the `mm` helper for `dtype`.
    pub(super) fn build_mm_helper(&self, dtype: Type) -> Result<()> {
        let i64t = self.i64();
        let ty = self.ctx.void_type().fn_type(&[i64t.into(); 6], false);
        let f = self
            .module
            .add_function(&Self::mm_helper_name(dtype), ty, Some(Linkage::Internal));
        self.add_target_attributes(f);
        let entry = self.ctx.append_basic_block(f, "entry");
        self.builder.position_at_end(entry);
        let p = |i| f.get_nth_param(i).unwrap().into_int_value();
        let regs = [p(0), p(1), p(2), p(3), p(4), p(5)];
        match self.mm_kernel(dtype) {
            MmKernel::Fma => self.mm_fma(f, dtype, regs)?,
            MmKernel::Sme => self.mm_sme(f, dtype, regs)?,
            MmKernel::Amx => self.mm_amx(f, dtype, regs)?,
        }
        self.builder.build_return(None)?;
        Ok(())
    }

    // ---------------------------------------------------------------------------------
    // Building blocks
    // ---------------------------------------------------------------------------------

    /// Emits `for iv in (start..end).step_by(step)` (signed compare) carrying `init` through
    /// the iterations. `body` gets the induction variable and the carried values and returns
    /// the next carried values; the builder ends up after the loop, and the result is the
    /// carried values on exit.
    pub(super) fn for_loop(
        &self,
        f: FunctionValue<'ctx>,
        name: &str,
        (start, end, step): (IntValue<'ctx>, IntValue<'ctx>, IntValue<'ctx>),
        init: &[BasicValueEnum<'ctx>],
        body: impl FnOnce(IntValue<'ctx>, &[BasicValueEnum<'ctx>]) -> Result<Vec<BasicValueEnum<'ctx>>>,
    ) -> Result<Vec<BasicValueEnum<'ctx>>> {
        let b = &self.builder;
        let pre = b.get_insert_block().unwrap();
        let hdr = self.ctx.append_basic_block(f, &format!("{name}.hdr"));
        let body_bb = self.ctx.append_basic_block(f, &format!("{name}.body"));
        let exit = self.ctx.append_basic_block(f, &format!("{name}.exit"));
        b.build_unconditional_branch(hdr)?;

        b.position_at_end(hdr);
        let iv = b.build_phi(start.get_type(), name)?;
        iv.add_incoming(&[(&start, pre)]);
        let carried: Vec<PhiValue> = init
            .iter()
            .map(|v| {
                let phi = b.build_phi(v.get_type(), "")?;
                phi.add_incoming(&[(v, pre)]);
                Ok(phi)
            })
            .collect::<Result<_>>()?;
        let ivv = iv.as_basic_value().into_int_value();
        let more = b.build_int_compare(IntPredicate::SLT, ivv, end, "")?;
        b.build_conditional_branch(more, body_bb, exit)?;

        b.position_at_end(body_bb);
        let current: Vec<BasicValueEnum> = carried.iter().map(|p| p.as_basic_value()).collect();
        let next = body(ivv, &current)?;
        let latch = b.get_insert_block().unwrap();
        let iv2 = b.build_int_add(ivv, step, "")?;
        iv.add_incoming(&[(&iv2, latch)]);
        for (phi, v) in carried.iter().zip(&next) {
            phi.add_incoming(&[(v, latch)]);
        }
        b.build_unconditional_branch(hdr)?;

        b.position_at_end(exit);
        Ok(current)
    }

    /// `base + (row * cols + col) * esize`.
    pub(super) fn elem_addr(
        &self,
        base: IntValue<'ctx>,
        row: IntValue<'ctx>,
        cols: IntValue<'ctx>,
        col: IntValue<'ctx>,
        esize: i64,
    ) -> Result<IntValue<'ctx>> {
        let b = &self.builder;
        let idx = b.build_int_mul(row, cols, "")?;
        let idx = b.build_int_add(idx, col, "")?;
        let off = b.build_int_mul(idx, self.c64(esize), "")?;
        Ok(b.build_int_add(base, off, "")?)
    }

    /// Masked vector load (masked-off lanes are zero and never accessed).
    pub(super) fn masked_load(
        &self,
        ty: BasicTypeEnum<'ctx>,
        addr: IntValue<'ctx>,
        mask: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let p = self.int_ptr(addr)?;
        let f = self.intrinsic("llvm.masked.load", &[ty, p.get_type().into()]);
        let mut args: Vec<BasicValueEnum> = vec![p.into()];
        if f.count_params() == 4 {
            args.push(self.c32(1).into());
        }
        args.push(mask);
        args.push(ty.const_zero());
        self.call1(f, &args)
    }

    /// Masked vector store of the enabled lanes of `v`.
    pub(super) fn masked_store(
        &self,
        v: BasicValueEnum<'ctx>,
        addr: IntValue<'ctx>,
        mask: BasicValueEnum<'ctx>,
    ) -> Result<()> {
        let p = self.int_ptr(addr)?;
        let f = self.intrinsic("llvm.masked.store", &[v.get_type(), p.get_type().into()]);
        let mut args: Vec<BasicValueEnum> = vec![v, p.into()];
        if f.count_params() == 4 {
            args.push(self.c32(1).into());
        }
        args.push(mask);
        self.call(f, &args)?;
        Ok(())
    }

    /// Storage type of one `mm` element of `dtype`.
    pub(super) fn mm_elem_type(&self, dtype: Type) -> BasicTypeEnum<'ctx> {
        match dtype {
            Type::BF16 | Type::F16 => self.i16().into(),
            Type::I8 => self.ctx.i8_type().into(),
            _ => self.ctx.f32_type().into(),
        }
    }

    /// Widens `mm` elements (scalar or lanes) to the accumulator type: f32, or i32 for i8.
    pub(super) fn mm_widen(
        &self,
        dtype: Type,
        v: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        let b = &self.builder;
        Ok(match dtype {
            Type::BF16 => self.bf16_to_f32(v)?,
            Type::F16 => self.f16_to_f32(v)?,
            Type::I8 => {
                let t = self.shaped(self.i32().into(), Self::shape_of(v));
                match v {
                    BasicValueEnum::IntValue(x) => {
                        b.build_int_s_extend(x, t.into_int_type(), "")?.into()
                    }
                    BasicValueEnum::VectorValue(x) => {
                        b.build_int_s_extend(x, t.into_vector_type(), "")?.into()
                    }
                    BasicValueEnum::ScalableVectorValue(x) => b
                        .build_int_s_extend(x, t.into_scalable_vector_type(), "")?
                        .into(),
                    other => unreachable!("{other:?}"),
                }
            }
            _ => v,
        })
    }

    // ---------------------------------------------------------------------------------
    // Vector FMA kernel (every target)
    // ---------------------------------------------------------------------------------

    /// Vector FMA `mm`. Rows of C are processed `ROWS` at a time (then one at a time for the
    /// remainder); for each `vx`-wide strip of columns, the block's accumulators start as C
    /// and add `broadcast(A[r][kk]) * B[kk][strip]` for every `kk`, so each B strip is loaded
    /// and widened once per row block. Per element the sum runs over `kk` in order, like the
    /// scalar reference, but fused where the target has FMA. Masked loads/stores cover the
    /// last strip.
    fn mm_fma(&self, f: FunctionValue<'ctx>, dtype: Type, regs: [IntValue<'ctx>; 6]) -> Result<()> {
        const ROWS: i64 = 4;
        let b = &self.builder;
        let [_, _, _, m, _, _] = regs;
        let full = b.build_and(m, self.c64(!(ROWS - 1)), "m.blocks")?;
        self.for_loop(
            f,
            "mm.i4",
            (self.c64(0), full, self.c64(ROWS)),
            &[],
            |i, _| {
                self.mm_fma_rows(f, dtype, regs, i, ROWS as usize)?;
                Ok(vec![])
            },
        )?;
        self.for_loop(f, "mm.i1", (full, m, self.c64(1)), &[], |i, _| {
            self.mm_fma_rows(f, dtype, regs, i, 1)?;
            Ok(vec![])
        })?;
        Ok(())
    }

    /// Rows `i..i + rows` of C for the vector FMA kernel.
    fn mm_fma_rows(
        &self,
        f: FunctionValue<'ctx>,
        dtype: Type,
        regs: [IntValue<'ctx>; 6],
        i: IntValue<'ctx>,
        rows: usize,
    ) -> Result<()> {
        let b = &self.builder;
        let [pc, pa, pb, _, n, k] = regs;
        let is_int = dtype == Type::I8;
        let acc_lane = if is_int { Type::I32 } else { Type::F32 };
        let esize = dtype.byte_size() as i64;
        let lanes = self.lanes_min(Type::Vx, Type::I32);
        let scalable = self.scalable(Type::Vx);
        let acc_ty = self.lane_vec(Type::Vx, acc_lane);
        let elem_ty = self.vec_of(self.mm_elem_type(dtype), lanes, scalable);
        let mask_ty = self.vec_of(self.i1().into(), lanes, scalable);
        let vl = self.lanes_value(Type::Vx, Type::I32)?;
        let (zero, one) = (self.c64(0), self.c64(1));
        let row = |r: usize| b.build_int_add(i, self.c64(r as i64), "");

        self.for_loop(f, "mm.j", (zero, n, vl), &[], |j, _| {
            let left = b.build_int_sub(n, j, "")?;
            let cnt = self
                .intr("llvm.umin", &[self.i64().into()], &[left.into(), vl.into()])?
                .into_int_value();
            let mask = self.intr(
                "llvm.get.active.lane.mask",
                &[mask_ty, self.i64().into()],
                &[zero.into(), cnt.into()],
            )?;
            let mut c_addrs = Vec::with_capacity(rows);
            let mut init = Vec::with_capacity(rows);
            for r in 0..rows {
                let addr = self.elem_addr(pc, row(r)?, n, j, 4)?;
                init.push(self.masked_load(acc_ty, addr, mask)?);
                c_addrs.push(addr);
            }
            let accs = self.for_loop(f, "mm.k", (zero, k, one), &init, |kk, accs| {
                let b_addr = self.elem_addr(pb, kk, n, j, esize)?;
                let bv = self.masked_load(elem_ty, b_addr, mask)?;
                let bv = self.mm_widen(dtype, bv)?;
                let mut next = Vec::with_capacity(rows);
                for (r, acc) in accs.iter().enumerate() {
                    let a_addr = self.elem_addr(pa, row(r)?, k, kk, esize)?;
                    let a = self.load(self.mm_elem_type(dtype), a_addr)?;
                    let a = self.splat(self.mm_widen(dtype, a)?, acc_lane, Type::Vx)?;
                    next.push(if is_int {
                        let prod = vec2!(a, bv, |x, y| b.build_int_mul(x, y, "")?);
                        vec2!(*acc, prod, |x, y| b.build_int_add(x, y, "")?)
                    } else {
                        // Fused where the target has FMA, multiply + add elsewhere (a libcall
                        // per element otherwise); `mm` only promises rounding-level agreement.
                        self.intr("llvm.fmuladd", &[acc_ty], &[a, bv, *acc])?
                    });
                }
                Ok(next)
            })?;
            for (acc, addr) in accs.into_iter().zip(c_addrs) {
                self.masked_store(acc, addr, mask)?;
            }
            Ok(vec![])
        })?;
        Ok(())
    }

    // ---------------------------------------------------------------------------------
    // SME kernel (AArch64 with FEAT_SME)
    // ---------------------------------------------------------------------------------

    /// k elements packed into one 32-bit ZA element: 1 for f32 (`fmopa`), 2 for bf16/f16
    /// (widening `bfmopa`/`fmopa`), 4 for i8 (`smopa`).
    fn sme_group(dtype: Type) -> u32 {
        match dtype {
            Type::F32 => 1,
            Type::I8 => 4,
            _ => 2,
        }
    }

    /// SME `mm` in streaming mode, one `S x S` block of C at a time (S = 32-bit lanes in a
    /// streaming vector):
    /// 1. load the C block into ZA tile 0;
    /// 2. for each chunk of k, load the A rows into tile 1 (zero-padded past k), so each
    ///    vertical slice of tile 1 is a column of A, i.e. the `zn` operand;
    /// 3. per slice, load the matching B rows (zero past k), interleave them into the
    ///    pair/quad layout widening outer products expect, and accumulate `zn (x) zm`;
    /// 4. store the C block.
    ///
    /// Predicates cover partial blocks at the m and n edges.
    fn mm_sme(&self, f: FunctionValue<'ctx>, dtype: Type, regs: [IntValue<'ctx>; 6]) -> Result<()> {
        let b = &self.builder;
        for attr in ["aarch64_pstate_sm_enabled", "aarch64_new_za"] {
            f.add_attribute(
                AttributeLoc::Function,
                self.ctx.create_string_attribute(attr, ""),
            );
        }
        // vscale in streaming mode follows the streaming vector length, which need not equal
        // the SVE length the module's `vscale_range` describes.
        let vscale_range = Attribute::get_named_enum_kind_id("vscale_range");
        f.remove_enum_attribute(AttributeLoc::Function, vscale_range);
        f.add_attribute(
            AttributeLoc::Function,
            self.ctx.create_enum_attribute(vscale_range, (1 << 32) | 16),
        );
        let [pc, pa, pb, m, n, k] = regs;
        let g = Self::sme_group(dtype);
        let esize = dtype.byte_size() as i64;
        let i64t: BasicTypeEnum = self.i64().into();
        let (zero, one) = (self.c64(0), self.c64(1));

        // Types: 32-bit containers, `dtype` elements (4*g per vscale), and their predicates.
        let i32v = self.vec_of(self.i32().into(), 4, true);
        let p32 = self.vec_of(self.i1().into(), 4, true);
        let elems = 4 * g;
        let pel = self.vec_of(self.i1().into(), elems, true);
        let load_ty = self.vec_of(self.mm_elem_type(dtype), elems, true);
        let op_ty: BasicTypeEnum = match dtype {
            Type::F32 => self.vec_of(self.ctx.f32_type().into(), 4, true),
            Type::BF16 => self.vec_of(self.ctx.bf16_type().into(), 8, true),
            Type::F16 => self.vec_of(self.ctx.f16_type().into(), 8, true),
            _ => self.vec_of(self.ctx.i8_type().into(), 16, true),
        };
        let mopa = match dtype {
            Type::F32 => "llvm.aarch64.sme.mopa",
            Type::I8 => "llvm.aarch64.sme.smopa.wide",
            _ => "llvm.aarch64.sme.mopa.wide",
        };

        let vscale = self.intr("llvm.vscale", &[i64t], &[])?.into_int_value();
        let s = b.build_int_mul(vscale, self.c64(4), "svl.s")?;
        let kb = b.build_int_mul(s, self.c64(g as i64), "k.block")?;
        let umin = |x: IntValue<'ctx>, y: IntValue<'ctx>| -> Result<IntValue<'ctx>> {
            Ok(self
                .intr("llvm.umin", &[i64t], &[x.into(), y.into()])?
                .into_int_value())
        };
        let mask = |ty: BasicTypeEnum<'ctx>, cnt: IntValue<'ctx>| {
            self.intr(
                "llvm.get.active.lane.mask",
                &[ty, i64t],
                &[zero.into(), cnt.into()],
            )
        };
        let tile = |t: u64| self.i32().const_int(t, false).into();
        // ZA slice indices go through an empty inline-asm copy so they always reach the
        // instruction in a register: LLVM 22 miscompiles constant slice indices (e.g. after
        // fully unrolling a row loop), addressing the slice from an uninitialized w12.
        let opaque_ty = self.i32().fn_type(&[self.i32().into()], false);
        let opaque = self.ctx.create_inline_asm(
            opaque_ty,
            String::new(),
            "=r,0".to_string(),
            false,
            false,
            None,
            false,
        );
        let as_i32 = |v: IntValue<'ctx>| -> Result<BasicValueEnum<'ctx>> {
            let v = b.build_int_truncate(v, self.i32(), "")?;
            let cs = b.build_indirect_call(opaque_ty, opaque, &[v.into()], "slice")?;
            Ok(cs.try_as_basic_value().basic().expect("i32 result"))
        };
        let ptrue32 = mask(p32, s)?;

        self.for_loop(f, "sme.i", (zero, m, s), &[], |i0, _| {
            let rows = umin(b.build_int_sub(m, i0, "")?, s)?;
            let g64 = self.c64(g as i64);
            let pn = mask(pel, b.build_int_mul(rows, g64, "")?)?;
            self.for_loop(f, "sme.j", (zero, n, s), &[], |j0, _| {
                let cols = umin(b.build_int_sub(n, j0, "")?, s)?;
                let pcol = mask(p32, cols)?;
                let pm = mask(pel, b.build_int_mul(cols, g64, "")?)?;

                // 1. C block -> ZA tile 0.
                let st1 = |name: &str, r: IntValue<'ctx>| -> Result<()> {
                    let addr = self.elem_addr(pc, b.build_int_add(i0, r, "")?, n, j0, 4)?;
                    let ptr = self.int_ptr(addr)?;
                    self.call(
                        self.intrinsic(name, &[]),
                        &[pcol, ptr.into(), tile(0), as_i32(r)?],
                    )?;
                    Ok(())
                };
                self.for_loop(f, "sme.cload", (zero, rows, one), &[], |r, _| {
                    st1("llvm.aarch64.sme.ld1w.horiz", r)?;
                    Ok(vec![])
                })?;

                self.for_loop(f, "sme.k", (zero, k, kb), &[], |k0, _| {
                    let kc = umin(b.build_int_sub(k, k0, "")?, kb)?;
                    // 2. A rows (k0..k0+kc, zero beyond) -> ZA tile 1, as 32-bit containers.
                    let pk = mask(pel, kc)?;
                    self.for_loop(f, "sme.aload", (zero, rows, one), &[], |r, _| {
                        let row = b.build_int_add(i0, r, "")?;
                        let addr = self.elem_addr(pa, row, k, k0, esize)?;
                        let z = self.masked_load(load_ty, addr, pk)?;
                        let z = self.bitcast(z, i32v)?;
                        self.call(
                            self.intrinsic("llvm.aarch64.sme.write.horiz", &[i32v]),
                            &[tile(1), as_i32(r)?, ptrue32, z],
                        )?;
                        Ok(vec![])
                    })?;
                    // 3. One outer product per group of g k-values.
                    let gm1 = self.c64(g as i64 - 1);
                    let slices =
                        b.build_int_unsigned_div(b.build_int_add(kc, gm1, "")?, g64, "")?;
                    self.for_loop(f, "sme.p", (zero, slices, one), &[], |p, _| {
                        let zn = self.call1(
                            self.intrinsic("llvm.aarch64.sme.read.vert", &[i32v]),
                            &[i32v.const_zero(), ptrue32, tile(1), as_i32(p)?],
                        )?;
                        let zn = self.bitcast(zn, op_ty)?;
                        let kbase = b.build_int_add(k0, b.build_int_mul(p, g64, "")?, "")?;
                        let mut rows_b = Vec::with_capacity(g as usize);
                        for t in 0..g {
                            let kk = b.build_int_add(kbase, self.c64(t as i64), "")?;
                            let valid = b.build_int_compare(IntPredicate::SLT, kk, k, "")?;
                            let cnt = b.build_select(valid, cols, zero, "")?.into_int_value();
                            let addr = self.elem_addr(pb, kk, n, j0, esize)?;
                            rows_b.push(self.masked_load(load_ty, addr, mask(pel, cnt)?)?);
                        }
                        let zip = |x: BasicValueEnum<'ctx>, y: BasicValueEnum<'ctx>| {
                            self.intr("llvm.aarch64.sve.zip1", &[load_ty], &[x, y])
                        };
                        let zm = match g {
                            1 => rows_b[0],
                            2 => zip(rows_b[0], rows_b[1])?,
                            _ => zip(zip(rows_b[0], rows_b[2])?, zip(rows_b[1], rows_b[3])?)?,
                        };
                        let zm = self.bitcast(zm, op_ty)?;
                        self.call(self.intrinsic(mopa, &[op_ty]), &[tile(0), pn, pm, zn, zm])?;
                        Ok(vec![])
                    })?;
                    Ok(vec![])
                })?;

                // 4. ZA tile 0 -> C block.
                self.for_loop(f, "sme.cstore", (zero, rows, one), &[], |r, _| {
                    st1("llvm.aarch64.sme.st1w.horiz", r)?;
                    Ok(vec![])
                })?;
                Ok(vec![])
            })?;
            Ok(vec![])
        })?;
        Ok(())
    }

    // ---------------------------------------------------------------------------------
    // AMX kernel (x86 with AMX-BF16 / AMX-INT8 / AMX-FP16)
    // ---------------------------------------------------------------------------------

    /// Calls an intrinsic whose operands or result include `x86_amx`, which inkwell cannot
    /// represent; values are passed as raw LLVM handles.
    fn raw_call(
        &self,
        name: &str,
        args: &[inkwell::llvm_sys::prelude::LLVMValueRef],
    ) -> inkwell::llvm_sys::prelude::LLVMValueRef {
        use inkwell::types::AsTypeRef;
        use inkwell::values::AsValueRef;
        let f = self.intrinsic(name, &[]);
        let mut args = args.to_vec();
        unsafe {
            inkwell::llvm_sys::core::LLVMBuildCall2(
                self.builder.as_mut_ptr(),
                f.get_type().as_type_ref(),
                f.as_value_ref(),
                args.as_mut_ptr(),
                args.len() as u32,
                c"".as_ptr(),
            )
        }
    }

    /// AMX `mm` on 16x16 blocks of C. Each block goes through zero-padded 1 KiB stack
    /// buffers so every tile has the full constant shape (16 rows x 64 bytes): C, a
    /// 16 x KB chunk of A (KB = 32 bf16/f16 or 64 i8 values), and the matching KB x 16
    /// chunk of B repacked in the VNNI layout the dot-product instructions expect (each
    /// 64-byte row holds, per column, the 2 or 4 consecutive k values it pairs with).
    /// Tiles never live across loop iterations: each k step loads C, A and B, runs one
    /// `tdpbf16ps`/`tdpfp16ps`/`tdpbssd`, and stores C back.
    ///
    /// Compile-tested only (no AMX hardware or emulator was available); see the PR notes.
    fn mm_amx(&self, f: FunctionValue<'ctx>, dtype: Type, regs: [IntValue<'ctx>; 6]) -> Result<()> {
        use inkwell::values::AsValueRef;
        let b = &self.builder;
        let [pc, pa, pb, m, n, k] = regs;
        let i64t: BasicTypeEnum = self.i64().into();
        let (zero, one) = (self.c64(0), self.c64(1));
        let esize = dtype.byte_size() as i64;
        let g: i64 = if dtype == Type::I8 { 4 } else { 2 };
        let kb = self.c64(64 / esize);
        let dot = match dtype {
            Type::BF16 => "llvm.x86.tdpbf16ps.internal",
            Type::F16 => "llvm.x86.tdpfp16ps.internal",
            _ => "llvm.x86.tdpbssd.internal",
        };
        let sixteen = self.c64(16);

        let buf_ty = self.ctx.i8_type().array_type(1024);
        let alloc = |name: &str| -> Result<PointerValue<'ctx>> {
            let p = b.build_alloca(buf_ty, name)?;
            p.as_instruction()
                .unwrap()
                .set_alignment(64)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(p)
        };
        let (cbuf, abuf, bbuf) = (alloc("amx.c")?, alloc("amx.a")?, alloc("amx.b")?);
        let as_int = |p: PointerValue<'ctx>| b.build_ptr_to_int(p, self.i64(), "");
        let (cbuf_i, abuf_i, bbuf_i) = (as_int(cbuf)?, as_int(abuf)?, as_int(bbuf)?);
        let umin = |x: IntValue<'ctx>, y: IntValue<'ctx>| -> Result<IntValue<'ctx>> {
            Ok(self
                .intr("llvm.umin", &[i64t], &[x.into(), y.into()])?
                .into_int_value())
        };
        let clear = |p: PointerValue<'ctx>| -> Result<()> {
            b.build_memset(p, 64, self.ctx.i8_type().const_zero(), self.c64(1024))?;
            Ok(())
        };
        let i16c = |v: u64| self.i16().const_int(v, false).as_value_ref();
        let stride = self.c64(64).as_value_ref();
        let tile_load = |p: PointerValue<'ctx>| {
            self.raw_call(
                "llvm.x86.tileloadd64.internal",
                &[i16c(16), i16c(64), p.as_value_ref(), stride],
            )
        };

        self.for_loop(f, "amx.i", (zero, m, sixteen), &[], |i0, _| {
            let rows = umin(b.build_int_sub(m, i0, "")?, sixteen)?;
            self.for_loop(f, "amx.j", (zero, n, sixteen), &[], |j0, _| {
                let cols = umin(b.build_int_sub(n, j0, "")?, sixteen)?;
                let row_bytes = b.build_int_mul(cols, self.c64(4), "")?;
                // C block -> cbuf (zero-padded).
                clear(cbuf)?;
                self.for_loop(f, "amx.cin", (zero, rows, one), &[], |r, _| {
                    let src = self.elem_addr(pc, b.build_int_add(i0, r, "")?, n, j0, 4)?;
                    let dst = self.elem_addr(cbuf_i, r, sixteen, zero, 4)?;
                    b.build_memcpy(self.int_ptr(dst)?, 1, self.int_ptr(src)?, 1, row_bytes)?;
                    Ok(vec![])
                })?;

                self.for_loop(f, "amx.k", (zero, k, kb), &[], |k0, _| {
                    let kc = umin(b.build_int_sub(k, k0, "")?, kb)?;
                    // A rows (kc values each) -> abuf.
                    clear(abuf)?;
                    let a_bytes = b.build_int_mul(kc, self.c64(esize), "")?;
                    self.for_loop(f, "amx.ain", (zero, rows, one), &[], |r, _| {
                        let src = self.elem_addr(pa, b.build_int_add(i0, r, "")?, k, k0, esize)?;
                        let dst =
                            b.build_int_add(abuf_i, b.build_int_mul(r, self.c64(64), "")?, "")?;
                        b.build_memcpy(self.int_ptr(dst)?, 1, self.int_ptr(src)?, 1, a_bytes)?;
                        Ok(vec![])
                    })?;
                    // B chunk -> bbuf in VNNI layout: element (kk, c) goes to row kk / g,
                    // byte offset (c * g + kk % g) * esize.
                    clear(bbuf)?;
                    let elem = self.mm_elem_type(dtype);
                    self.for_loop(f, "amx.bk", (zero, kc, one), &[], |kk, _| {
                        let row = b.build_int_add(k0, kk, "")?;
                        let vrow = b.build_int_unsigned_div(kk, self.c64(g), "")?;
                        let lane = b.build_int_unsigned_rem(kk, self.c64(g), "")?;
                        self.for_loop(f, "amx.bc", (zero, cols, one), &[], |c, _| {
                            let src =
                                self.elem_addr(pb, row, n, b.build_int_add(j0, c, "")?, esize)?;
                            let v = self.load(elem, src)?;
                            let idx =
                                b.build_int_add(b.build_int_mul(c, self.c64(g), "")?, lane, "")?;
                            let off = b.build_int_add(
                                b.build_int_mul(vrow, self.c64(64), "")?,
                                b.build_int_mul(idx, self.c64(esize), "")?,
                                "",
                            )?;
                            self.store(v, b.build_int_add(bbuf_i, off, "")?)?;
                            Ok(vec![])
                        })?;
                        Ok(vec![])
                    })?;
                    // C += A * B on full 16-row, 64-byte tiles.
                    let c_t = tile_load(cbuf);
                    let a_t = tile_load(abuf);
                    let b_t = tile_load(bbuf);
                    let c_t = self.raw_call(dot, &[i16c(16), i16c(64), i16c(64), c_t, a_t, b_t]);
                    self.raw_call(
                        "llvm.x86.tilestored64.internal",
                        &[i16c(16), i16c(64), cbuf.as_value_ref(), stride, c_t],
                    );
                    Ok(vec![])
                })?;

                // cbuf -> C block.
                self.for_loop(f, "amx.cout", (zero, rows, one), &[], |r, _| {
                    let dst = self.elem_addr(pc, b.build_int_add(i0, r, "")?, n, j0, 4)?;
                    let src = self.elem_addr(cbuf_i, r, sixteen, zero, 4)?;
                    b.build_memcpy(self.int_ptr(dst)?, 1, self.int_ptr(src)?, 1, row_bytes)?;
                    Ok(vec![])
                })?;
                Ok(vec![])
            })?;
            Ok(vec![])
        })?;
        Ok(())
    }
}
