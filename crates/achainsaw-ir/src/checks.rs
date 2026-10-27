//! Loops that pay for their checks once, on entry. JIT code checks fuel at every branch, and
//! sandboxed code checks every memory access; each check leaves the loop through a call, so
//! LLVM does not vectorize a loop that has one. [`hoist_loop_checks`] gives a loop whose trip
//! count is known on entry a copy without those checks, and a dispatch block in front of both
//! that computes, in AIR, the fuel the whole loop takes and the bytes each of its indexed
//! accesses covers. The backends then charge that fuel at once and check those ranges against
//! the sandbox (`FunctionChecks::dispatch`); when both succeed the copy runs, and otherwise the
//! original loop, which fails exactly where it did. Results, the memory touched and the fuel
//! used are the same either way.
//!
//! A loop qualifies when one block outside it enters it; it calls nothing (calls, `alloc`,
//! `free`, `mm`, `par`); its blocks, apart from the edges back to the header, form a DAG in
//! which every way around takes the same number of branches (an if/else whose ways are equally
//! long, so an iteration's fuel does not depend on the data); and one block that every
//! iteration runs leaves it, by a branch on a test that holds while the loop goes on. The test
//! is `lt v, n`, `gt n, v`, or `gt r, k` / `ge r, k` on `r = sub n, v` (the masked-loop guards
//! of `crate::loops`), possibly several joined by `and`, where `v` is an induction variable
//! plus a constant: a header parameter, a counter or a pointer, that grows by a positive
//! constant or a `vl` on every way back. `n` and `k` are invariant: defined before the loop, a
//! constant or `vl`, or a header parameter every way back passes unchanged. The copy needs no
//! checks for accesses through an invariant pointer indexed by a counter plus a constant,
//! through a pointer induction variable plus a constant, or at an invariant address. The
//! module must have unique register names; the result is meant for code generation only (the
//! dispatch block's condition is the backend's to compute).

use crate::ast::*;
use crate::cfg::Cfg;
use crate::loops::{
    add_exit_args, all_names, live_outs, natural_loop, retarget, route_live_outs, targets,
    MAX_LOOP_INSTRUCTIONS,
};
use crate::opt::{infer_reg_types, Signatures};
use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// What the backends do differently for the loops [`hoist_loop_checks`] rewrote, by function.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopChecks {
    pub functions: HashMap<String, FunctionChecks>,
}

impl LoopChecks {
    pub fn function(&self, name: &str) -> Option<&FunctionChecks> {
        self.functions.get(name)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FunctionChecks {
    /// Blocks whose branches take no fuel: the dispatch blocks and the copies.
    pub free: HashSet<String>,
    /// Accesses of the copies, by block and instruction index, that a dispatch range covers:
    /// they need no sandbox check.
    pub covered: HashSet<(String, usize)>,
    /// Dispatch blocks: each ends with `br ok, copy(..), loop(..)`, where `ok` is a constant 0
    /// that the backend replaces with whether the ranges lie in the sandbox and the fuel could
    /// be charged.
    pub dispatch: HashMap<String, Dispatch>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch {
    /// i64: the fuel the loop takes, its exit included (more than any budget when it does not
    /// end).
    pub units: String,
    /// Byte ranges holding every covered access.
    pub ranges: Vec<Range>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    /// ptr: the first byte.
    pub lo: String,
    /// i64: the length; 0 is empty, and one above the sandbox's size (-1 when too large to
    /// compute) never fits.
    pub span: String,
    /// A store's range, which must also miss the sandbox's read-only `data`.
    pub writes: bool,
}

/// More fuel than any budget: a loop that would not end charges it.
pub const ENDLESS_UNITS: i64 = 1 << 62;
/// Trip counts above this charge `ENDLESS_UNITS`, so `trips * blocks` cannot overflow.
const MAX_TRIPS: i64 = 1 << 48;
/// Spans above this many bytes do not fit any sandbox.
const MAX_SPAN: i64 = 1 << 40;
/// Steps above this many elements leave the dispatch's ranges unchecked: the copy never runs.
const MAX_STEP: i64 = 1 << 20;
/// Loops of more blocks are left alone, so `trips * cost` cannot overflow.
const MAX_BLOCKS: usize = 1 << 10;

/// Rewrites every qualifying loop in `module` (see the module docs).
pub fn hoist_loop_checks(module: &mut Module) -> LoopChecks {
    let mut sigs: Signatures = HashMap::new();
    for ext in &module.extern_functions {
        sigs.insert(ext.name.clone(), ext.ret_type.into_iter().collect());
    }
    for func in &module.functions {
        sigs.insert(func.name.clone(), func.rets.clone());
    }
    let mut checks = LoopChecks::default();
    for f in &mut module.functions {
        crate::loops::hoist_stack_buffers(f);
        let cfg = Cfg::build(f);
        let headers: Vec<String> = (1..f.blocks.len())
            .filter(|&h| cfg.reachable[h] && !natural_loop(&cfg, h).is_empty())
            .map(|h| f.blocks[h].label.clone())
            .collect();
        let mut fc = FunctionChecks::default();
        let mut serial = 0;
        for header in headers {
            hoist_loop(f, &header, &sigs, &mut serial, &mut fc);
        }
        if !fc.dispatch.is_empty() {
            checks.functions.insert(f.name.clone(), fc);
        }
    }
    checks
}

/// How an induction variable grows per iteration: a constant, or a register holding a `vl`.
#[derive(Clone, PartialEq)]
enum Step {
    Const(i64),
    Reg(String),
}

/// A header parameter that grows by a positive `step` on every way back to the header: a
/// counter (i64), whose steps count elements of the accesses it indexes, or a pointer, whose
/// steps are bytes.
struct Iv {
    param: usize,
    step: Step,
    ptr: bool,
}

/// A test that holds while the trips go on, on `v + o` for header parameter `param`'s value
/// `v` in the iteration.
struct Test {
    param: usize,
    o: i64,
    kind: TestKind,
}

enum TestKind {
    /// `v < n`.
    Below { n: String },
    /// `n - v > k`, `k` a constant or `vl - 1` (`ge r, vl`).
    Above { n: String, k: Bound },
}

#[derive(Clone)]
enum Bound {
    Const(i64),
    VlMinus1(String),
}

/// A register's definition, if it is a constant or a `vl`, which the dispatch block can
/// compute again wherever it is defined.
fn constant_like(defs: &HashMap<&str, &Instruction>, r: &str) -> bool {
    matches!(
        defs.get(r),
        Some(Instruction::AssignConst {
            val: Constant::Int(_),
            ..
        }) | Some(Instruction::VLen { .. })
    )
}

fn int_const(defs: &HashMap<&str, &Instruction>, r: &str) -> Option<i64> {
    match defs.get(r)? {
        Instruction::AssignConst {
            val: Constant::Int(n),
            ..
        } => Some(*n),
        _ => None,
    }
}

/// The tests `cond` (defined in the loop) joins, on registers that are an induction variable
/// plus a constant (`counter`), or `None`.
fn parse_tests(
    cond: &str,
    defs: &HashMap<&str, &Instruction>,
    in_loop: &HashSet<&str>,
    invariant: &dyn Fn(&str) -> bool,
    counter: &dyn Fn(&str) -> Option<(usize, i64)>,
) -> Option<Vec<Test>> {
    if !in_loop.contains(cond) {
        return None;
    }
    let Instruction::Binary { op, lhs, rhs, .. } = defs.get(cond)? else {
        return None;
    };
    // `r = sub n, v` defined in the loop: its `n` and `v`.
    let rest = |r: &str| -> Option<(String, (usize, i64))> {
        if !in_loop.contains(r) {
            return None;
        }
        match defs.get(r)? {
            Instruction::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
                ..
            } if invariant(lhs) => Some((lhs.clone(), counter(rhs)?)),
            _ => None,
        }
    };
    let bound = |k: &str, minus: i64| -> Option<Bound> {
        if let Some(c) = int_const(defs, k) {
            let c = c.checked_sub(minus)?;
            return (c >= -(1 << 62)).then_some(Bound::Const(c));
        }
        match defs.get(k)? {
            Instruction::VLen { .. } if minus == 1 => Some(Bound::VlMinus1(k.to_string())),
            _ => None,
        }
    };
    let below = |n: &str, v: &str| -> Option<Vec<Test>> {
        let (param, o) = counter(v)?;
        Some(vec![Test {
            param,
            o,
            kind: TestKind::Below { n: n.to_string() },
        }])
    };
    match op {
        BinaryOp::And => {
            let mut tests = parse_tests(lhs, defs, in_loop, invariant, counter)?;
            tests.extend(parse_tests(rhs, defs, in_loop, invariant, counter)?);
            Some(tests)
        }
        BinaryOp::Lt if invariant(rhs) && counter(lhs).is_some() => below(rhs, lhs),
        BinaryOp::Gt if invariant(lhs) && counter(rhs).is_some() => below(lhs, rhs),
        // `r > k`, `k < r`, `r >= k`, `k <= r`.
        BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le => {
            let (r, k) = match op {
                BinaryOp::Gt | BinaryOp::Ge => (lhs, rhs),
                _ => (rhs, lhs),
            };
            let minus = matches!(op, BinaryOp::Ge | BinaryOp::Le) as i64;
            let (n, (param, o)) = rest(r)?;
            Some(vec![Test {
                param,
                o,
                kind: TestKind::Above {
                    n,
                    k: bound(k, minus)?,
                },
            }])
        }
        _ => None,
    }
}

/// The argument lists of `t`'s edges into `target`.
fn edge_args<'t>(t: &'t Terminator, target: &str) -> Vec<&'t Vec<String>> {
    t.edges()
        .into_iter()
        .filter(|(to, _)| *to == target)
        .map(|(_, args)| args)
        .collect()
}

/// Emits the dispatch block's instructions, naming registers `{prefix}_d{n}`.
struct Emitter<'a> {
    prefix: &'a str,
    next: usize,
    span: crate::diag::Span,
    out: Vec<Instruction>,
    /// Registers computed again in the dispatch block, by the loop's name.
    again: HashMap<String, String>,
}

impl Emitter<'_> {
    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("{}_d{}", self.prefix, self.next)
    }

    fn cst(&mut self, v: i64, ty: Type) -> String {
        let dst = self.fresh();
        self.out.push(Instruction::AssignConst {
            dst: dst.clone(),
            val: Constant::Int(v),
            ty,
            span: self.span,
        });
        dst
    }

    fn bin(&mut self, op: BinaryOp, lhs: &str, rhs: &str) -> String {
        let dst = self.fresh();
        self.out.push(Instruction::Binary {
            op,
            dst: dst.clone(),
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
            span: self.span,
        });
        dst
    }

    fn bin_c(&mut self, op: BinaryOp, lhs: &str, rhs: i64) -> String {
        let c = self.cst(rhs, Type::I64);
        self.bin(op, lhs, &c)
    }

    fn select(&mut self, cond: &str, a: &str, b: &str) -> String {
        let dst = self.fresh();
        self.out.push(Instruction::Select {
            dst: dst.clone(),
            cond: cond.to_string(),
            then_val: a.to_string(),
            else_val: b.to_string(),
            span: self.span,
        });
        dst
    }

    /// `r` as an i64: pointers through a `bitcast`, so the arithmetic is valid AIR.
    fn int(&mut self, r: &str, ty: Type) -> String {
        if ty != Type::Ptr {
            return r.to_string();
        }
        let dst = self.fresh();
        self.out.push(Instruction::Cast {
            op: CastOp::Bitcast,
            dst: dst.clone(),
            src: r.to_string(),
            ty: Type::I64,
            span: self.span,
        });
        dst
    }

    fn vl(&mut self, lane: Type) -> String {
        let dst = self.fresh();
        self.out.push(Instruction::VLen {
            dst: dst.clone(),
            lane,
            span: self.span,
        });
        dst
    }

    /// `r` in the dispatch block: itself when defined outside the loop, or a constant or `vl`
    /// the loop defines, computed again.
    fn reg(
        &mut self,
        r: &str,
        in_loop: &HashSet<&str>,
        defs: &HashMap<&str, &Instruction>,
    ) -> String {
        if !in_loop.contains(r) {
            return r.to_string();
        }
        if let Some(d) = self.again.get(r) {
            return d.clone();
        }
        let d = match defs[r] {
            Instruction::AssignConst {
                val: Constant::Int(n),
                ty,
                ..
            } => self.cst(*n, *ty),
            Instruction::VLen { lane, .. } => self.vl(*lane),
            _ => unreachable!("only constants and vl are computed again"),
        };
        self.again.insert(r.to_string(), d.clone());
        d
    }

    /// Trips `(d - 1) / s + 1` for a positive `d`; a shift for a constant power-of-two step,
    /// which saves the division (and Cranelift its check for a zero divisor).
    fn ceil_div(&mut self, d: &str, s: &str, step: &Step) -> String {
        let d1 = self.bin_c(BinaryOp::Sub, d, 1);
        let q = match step {
            Step::Const(c) if (*c as u64).is_power_of_two() => {
                self.bin_c(BinaryOp::Ushr, &d1, c.trailing_zeros() as i64)
            }
            _ => self.bin(BinaryOp::Udiv, &d1, s),
        };
        self.bin_c(BinaryOp::Add, &q, 1)
    }
}

/// Bytes of one access of type `ty`: a constant, or `vl i8` for `vx`.
fn access_bytes(e: &mut Emitter, ty: Type) -> String {
    match ty {
        Type::Vx => e.vl(Type::I8),
        t => e.cst(t.byte_size() as i64, Type::I64),
    }
}

fn hoist_loop(
    f: &mut Function,
    header: &str,
    sigs: &Signatures,
    serial: &mut usize,
    fc: &mut FunctionChecks,
) -> bool {
    let cfg = Cfg::build(f);
    let Some(h) = f.blocks.iter().position(|b| b.label == header) else {
        return false;
    };
    let body = natural_loop(&cfg, h);
    if body.is_empty() || body.len() > MAX_BLOCKS {
        return false;
    }
    let in_body: HashSet<usize> = body.iter().copied().collect();
    let index: HashMap<&str, usize> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.as_str(), i))
        .collect();

    // One block leaves the loop, by a branch whose other way stays in it.
    let mut exiting = None;
    for &b in &body {
        let to = targets(&f.blocks[b].terminator);
        let out = to
            .iter()
            .filter(|t| !in_body.contains(&index[t.as_str()]))
            .count();
        if out == 0 {
            continue;
        }
        if exiting.is_some() || out != 1 || to.len() != 2 {
            return false;
        }
        exiting = Some(b);
    }
    let Some(ex) = exiting else {
        return false;
    };
    let Terminator::Br {
        cond,
        then_block,
        else_block,
        ..
    } = &f.blocks[ex].terminator
    else {
        return false;
    };
    if !in_body.contains(&index[then_block.as_str()]) {
        return false;
    }
    let exit = else_block.clone();

    // The loop's edges, apart from those back to the header, form a DAG from the header in
    // which every way to a block takes the same number of branches (`dist`), and every way
    // back to the header the same (`cost`): the fuel of an iteration does not depend on the
    // way it takes.
    let forward = |b: usize| -> Vec<usize> {
        targets(&f.blocks[b].terminator)
            .into_iter()
            .map(|t| index[t.as_str()])
            .filter(|&t| t != h && in_body.contains(&t))
            .collect()
    };
    let mut indegree: HashMap<usize, usize> = body.iter().map(|&b| (b, 0)).collect();
    for &b in &body {
        for t in forward(b) {
            *indegree.get_mut(&t).unwrap() += 1;
        }
    }
    let mut order = Vec::with_capacity(body.len());
    let mut ready = vec![h];
    while let Some(b) = ready.pop() {
        order.push(b);
        for t in forward(b) {
            let d = indegree.get_mut(&t).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.push(t);
            }
        }
    }
    if order.len() != body.len() {
        return false;
    }
    let mut dist: HashMap<usize, usize> = HashMap::from([(h, 0)]);
    for &b in &order {
        let d = dist[&b] + 1;
        for t in forward(b) {
            if *dist.entry(t).or_insert(d) != d {
                return false;
            }
        }
    }
    let latches: Vec<usize> = body
        .iter()
        .copied()
        .filter(|&b| !edge_args(&f.blocks[b].terminator, header).is_empty())
        .collect();
    let cost = dist[&latches[0]] + 1;
    // Every iteration runs the exiting block's test.
    if latches
        .iter()
        .any(|&l| dist[&l] + 1 != cost || !cfg.dominates(ex, l))
    {
        return false;
    }
    // Blocks that may run on the last iteration too, the one that leaves: the exiting block
    // and those before it.
    let mut before_exit: HashSet<usize> = HashSet::from([ex]);
    for &b in order.iter().rev() {
        if forward(b).iter().any(|t| before_exit.contains(t)) {
            before_exit.insert(b);
        }
    }

    let size: usize = body.iter().map(|&b| f.blocks[b].instructions.len()).sum();
    let calls_out = body
        .iter()
        .flat_map(|&b| &f.blocks[b].instructions)
        .any(|i| {
            matches!(
                i,
                Instruction::Call { .. }
                    | Instruction::CallIndirect { .. }
                    | Instruction::Alloc { .. }
                    | Instruction::Free { .. }
                    | Instruction::MatMul { .. }
                    | Instruction::Par { .. }
            )
        });
    if size > MAX_LOOP_INSTRUCTIONS || calls_out {
        return false;
    }

    // One block outside enters the loop, by one edge.
    let outside: Vec<usize> = cfg.preds[h]
        .iter()
        .copied()
        .filter(|p| cfg.reachable[*p] && !in_body.contains(p))
        .collect();
    let [entry] = outside[..] else {
        return false;
    };
    let edges = targets(&f.blocks[entry].terminator)
        .into_iter()
        .filter(|t| t.as_str() == header)
        .count();
    if edges != 1 {
        return false;
    }

    // Registers defined in the loop, and the definition of every register.
    let mut in_loop: HashSet<&str> = HashSet::new();
    let mut defs: HashMap<&str, &Instruction> = HashMap::new();
    for (bi, block) in f.blocks.iter().enumerate() {
        for inst in &block.instructions {
            for d in inst.dsts() {
                defs.insert(d, inst);
                if in_body.contains(&bi) {
                    in_loop.insert(d);
                }
            }
        }
        if in_body.contains(&bi) {
            in_loop.extend(block.params.iter().map(|(p, _)| p.as_str()));
        }
    }
    let header_block = &f.blocks[h];

    // Registers equal to an i64 or pointer header parameter plus a constant, in the iteration
    // that computes them; a block parameter only when every edge into it agrees.
    let mut offset: HashMap<String, (usize, i64)> = header_block
        .params
        .iter()
        .enumerate()
        .filter(|(_, (_, t))| matches!(t, Type::I64 | Type::Ptr))
        .map(|(k, (p, _))| (p.clone(), (k, 0)))
        .collect();
    for &b in &order {
        let block = &f.blocks[b];
        if b != h {
            for (pi, (p, _)) in block.params.iter().enumerate() {
                let mut agreed: Option<Option<(usize, i64)>> = None;
                for &pred in &cfg.preds[b] {
                    for args in edge_args(&f.blocks[pred].terminator, &block.label) {
                        let o = offset.get(args[pi].as_str()).copied();
                        agreed = match agreed {
                            None => Some(o),
                            Some(prev) if prev == o => Some(o),
                            Some(_) => Some(None),
                        };
                    }
                }
                if let Some(Some(o)) = agreed {
                    offset.insert(p.clone(), o);
                }
            }
        }
        for inst in &block.instructions {
            if let Instruction::Binary {
                op, dst, lhs, rhs, ..
            } = inst
            {
                let shifted = match (op, offset.get(lhs.as_str()), offset.get(rhs.as_str())) {
                    (BinaryOp::Add, Some(&(k, o)), None) => {
                        int_const(&defs, rhs).and_then(|c| Some((k, o.checked_add(c)?)))
                    }
                    (BinaryOp::Add, None, Some(&(k, o))) => {
                        int_const(&defs, lhs).and_then(|c| Some((k, o.checked_add(c)?)))
                    }
                    (BinaryOp::Sub, Some(&(k, o)), None) => {
                        int_const(&defs, rhs).and_then(|c| Some((k, o.checked_sub(c)?)))
                    }
                    _ => None,
                };
                if let Some(o) = shifted {
                    offset.insert(dst.clone(), o);
                }
            }
        }
    }

    // What every way back passes for each parameter: itself (`fixed`, invariant), or itself
    // plus a positive step (an induction variable).
    let step_of = |a: &str, k: usize| -> Option<Step> {
        match offset.get(a) {
            Some(&(kk, s)) if kk == k && (0..=1 << 32).contains(&s) => Some(Step::Const(s)),
            Some(_) => None,
            None => match defs.get(a) {
                Some(Instruction::Binary {
                    op: BinaryOp::Add,
                    lhs,
                    rhs,
                    ..
                }) => {
                    let w = match (offset.get(lhs.as_str()), offset.get(rhs.as_str())) {
                        (Some(&(kk, 0)), None) if kk == k => rhs,
                        (None, Some(&(kk, 0))) if kk == k => lhs,
                        _ => return None,
                    };
                    match defs.get(w.as_str()) {
                        Some(Instruction::VLen { .. }) => Some(Step::Reg(w.clone())),
                        _ => None,
                    }
                }
                _ => None,
            },
        }
    };
    let mut ivs: Vec<Iv> = Vec::new();
    let mut fixed: Vec<usize> = Vec::new();
    for (k, (p, ty)) in header_block.params.iter().enumerate() {
        if !offset.contains_key(p.as_str()) {
            continue;
        }
        let mut step: Option<Option<Step>> = None;
        for &l in &latches {
            for args in edge_args(&f.blocks[l].terminator, header) {
                let s = step_of(&args[k], k);
                step = match step {
                    None => Some(s),
                    Some(prev) if prev == s => Some(s),
                    Some(_) => Some(None),
                };
            }
        }
        match step.flatten() {
            Some(Step::Const(0)) => fixed.push(k),
            Some(step) => ivs.push(Iv {
                param: k,
                step,
                ptr: *ty == Type::Ptr,
            }),
            None => {}
        }
    }
    let iv_of = |k: usize| ivs.iter().position(|iv| iv.param == k);
    let fixed_names: HashSet<&str> = fixed
        .iter()
        .map(|&k| header_block.params[k].0.as_str())
        .collect();
    let invariant =
        |r: &str| !in_loop.contains(r) || constant_like(&defs, r) || fixed_names.contains(r);
    let counter = |r: &str| -> Option<(usize, i64)> {
        let &(k, o) = offset.get(r)?;
        iv_of(k).map(|_| (k, o))
    };
    let Some(tests) = parse_tests(cond, &defs, &in_loop, &invariant, &counter) else {
        return false;
    };

    // Types are needed for stores and live-out registers.
    let types_cell: std::cell::OnceCell<HashMap<String, Type>> = std::cell::OnceCell::new();
    let types = || types_cell.get_or_init(|| infer_reg_types(f, sigs));

    // Accesses whose addresses the dispatch block can bound: through an invariant pointer at
    // a counter plus a constant, through a pointer induction variable plus a constant, or at
    // an invariant address.
    enum Place {
        Fixed {
            ptr: String,
        },
        Indexed {
            ptr: String,
            iv: usize,
            o: i64,
            unit: i64,
        },
        Bumped {
            iv: usize,
            o: i64,
        },
    }
    struct Access {
        block: usize,
        inst: usize,
        place: Place,
        ty: Type,
        writes: bool,
    }
    let mut accesses: Vec<Access> = Vec::new();
    for &b in &body {
        for (i, inst) in f.blocks[b].instructions.iter().enumerate() {
            let (ptr, index, ty, implied) = match inst {
                Instruction::Load { ptr, index, ty, .. } => (ptr, index, *ty, *ty),
                Instruction::MaskedLoad {
                    ptr,
                    index,
                    ty,
                    lane,
                    ..
                } => (ptr, index, *ty, *lane),
                Instruction::Store {
                    ptr, index, val, ..
                } => {
                    let Some(&ty) = types().get(val.as_str()) else {
                        continue;
                    };
                    (ptr, index, ty, ty)
                }
                Instruction::MaskedStore {
                    ptr,
                    index,
                    val,
                    lane,
                    ..
                } => {
                    let Some(&ty) = types().get(val.as_str()) else {
                        continue;
                    };
                    (ptr, index, ty, *lane)
                }
                _ => continue,
            };
            let ivs_at = |r: &str| offset.get(r).and_then(|&(k, o)| Some((iv_of(k)?, o)));
            let place = match index {
                None if invariant(ptr) => Place::Fixed { ptr: ptr.clone() },
                None => match ivs_at(ptr) {
                    Some((iv, o)) if ivs[iv].ptr => Place::Bumped { iv, o },
                    _ => continue,
                },
                Some(ix) if invariant(ptr) => match ivs_at(&ix.reg) {
                    Some((iv, o)) if !ivs[iv].ptr => Place::Indexed {
                        ptr: ptr.clone(),
                        iv,
                        o,
                        unit: ix.unit_or(implied).byte_size() as i64,
                    },
                    _ => continue,
                },
                Some(_) => continue,
            };
            accesses.push(Access {
                block: b,
                inst: i,
                place,
                ty,
                writes: matches!(
                    inst,
                    Instruction::Store { .. } | Instruction::MaskedStore { .. }
                ),
            });
        }
    }

    let exit_idx = index[exit.as_str()];
    let Some((live_out, live_types)) = live_outs(f, &cfg, &in_body, &in_loop, exit_idx, &types)
    else {
        return false;
    };

    // A prefix no name in the function starts with.
    let names = all_names(f);
    let prefix = loop {
        *serial += 1;
        let p = format!("__lc{serial}");
        if !names.iter().any(|n| n.starts_with(&p)) {
            break p;
        }
    };
    let span = f.blocks[h].span;
    let mut e = Emitter {
        prefix: &prefix,
        next: 0,
        span,
        out: Vec::new(),
        again: HashMap::new(),
    };

    // The dispatch block takes the header's parameters: the induction variables' first
    // values, and the fixed ones' values throughout.
    let dparams: Vec<(String, Type)> = f.blocks[h]
        .params
        .iter()
        .map(|(p, t)| (format!("{prefix}_p_{p}"), *t))
        .collect();
    for &k in &fixed {
        e.again
            .insert(f.blocks[h].params[k].0.clone(), dparams[k].0.clone());
    }
    let steps: Vec<String> = ivs
        .iter()
        .map(|iv| match &iv.step {
            Step::Const(c) => e.cst(*c, Type::I64),
            Step::Reg(w) => e.reg(w, &in_loop, &defs),
        })
        .collect();

    // Trips: the fewest any test allows, on `v0 = x0 + o` for the variable's first value
    // x0. `v < n` holds for `(n - v0 - 1) / s + 1` trips when v0 < n, unless v would wrap
    // past i64::MAX and the loop never end; `n - v > k` holds for `(n - v0 - k - 1) / s + 1`
    // when `n - v0 > k` (wrapping), since it falls by `s` without wrapping while it is above
    // k.
    let zero = e.cst(0, Type::I64);
    let endless = e.cst(ENDLESS_UNITS, Type::I64);
    let mut trips: Option<String> = None;
    for test in &tests {
        let iv = iv_of(test.param).expect("tests are on induction variables");
        let (s, step) = (&steps[iv], &ivs[iv].step);
        let (x0, ty) = &dparams[test.param];
        let x0 = e.int(x0, *ty);
        let v0 = match test.o {
            0 => x0,
            o => e.bin_c(BinaryOp::Add, &x0, o),
        };
        let t = match &test.kind {
            TestKind::Below { n } => {
                let ty = types().get(n.as_str()).copied().unwrap_or(Type::I64);
                let n = e.reg(n, &in_loop, &defs);
                let n = e.int(&n, ty);
                let some = e.bin(BinaryOp::Lt, &v0, &n);
                let d = e.bin(BinaryOp::Sub, &n, &v0);
                let t = e.ceil_div(&d, s, step);
                let t1 = e.bin_c(BinaryOp::Sub, &t, 1);
                let last = e.bin(BinaryOp::Mul, &t1, s);
                let last = e.bin(BinaryOp::Add, &v0, &last);
                let max = e.cst(i64::MAX, Type::I64);
                let room = e.bin(BinaryOp::Sub, &max, s);
                let wraps = e.bin(BinaryOp::Gt, &last, &room);
                let t = e.select(&wraps, &endless, &t);
                e.select(&some, &t, &zero)
            }
            TestKind::Above { n, k } => {
                let ty = types().get(n.as_str()).copied().unwrap_or(Type::I64);
                let n = e.reg(n, &in_loop, &defs);
                let n = e.int(&n, ty);
                let k = match k {
                    Bound::Const(c) => e.cst(*c, Type::I64),
                    Bound::VlMinus1(w) => {
                        let w = e.reg(w, &in_loop, &defs);
                        e.bin_c(BinaryOp::Sub, &w, 1)
                    }
                };
                let r0 = e.bin(BinaryOp::Sub, &n, &v0);
                let some = e.bin(BinaryOp::Gt, &r0, &k);
                let d = e.bin(BinaryOp::Sub, &r0, &k);
                let t = e.ceil_div(&d, s, step);
                e.select(&some, &t, &zero)
            }
        };
        trips = Some(match trips {
            None => t,
            Some(prev) => e.bin(BinaryOp::Umin, &prev, &t),
        });
    }
    let trips = trips.expect("at least one test");

    // Fuel: `cost` units per trip, and the branches of the last iteration up to and including
    // the exiting block's.
    let many = e.bin_c(BinaryOp::Ugt, &trips, MAX_TRIPS);
    let per_trip = e.bin_c(BinaryOp::Mul, &trips, cost as i64);
    let all = e.bin_c(BinaryOp::Add, &per_trip, dist[&ex] as i64 + 1);
    let units = e.select(&many, &endless, &all);

    // Ranges. Blocks up to the exiting one run once more than the others, on the last
    // iteration, so their accesses reach `trips` steps past the first, and the others'
    // `trips - 1` (none when there are no trips); counting every block as if it ran on every
    // trip covers the accesses of those that run on some. Too many trips, or too long a step,
    // cannot fit: with trips <= MAX_SPAN and steps <= MAX_STEP the products stay below 2^64.
    let mut ranges = Vec::new();
    let no_trips = e.bin(BinaryOp::Eq, &trips, &zero);
    let many = e.bin_c(BinaryOp::Ugt, &trips, MAX_SPAN);
    let too_far = e.cst(-1, Type::I64);
    let trips1 = e.bin_c(BinaryOp::Sub, &trips, 1);
    let unfit: Vec<String> = steps
        .iter()
        .map(|s| {
            let long = e.bin_c(BinaryOp::Ugt, s, MAX_STEP);
            e.bin(BinaryOp::Or, &many, &long)
        })
        .collect();
    for a in &accesses {
        let last_too = before_exit.contains(&a.block);
        let bytes = access_bytes(&mut e, a.ty);
        let strided = |e: &mut Emitter, lo: String, iv: usize, unit: i64| {
            let n = if last_too { &trips } else { &trips1 };
            let dist = e.bin(BinaryOp::Mul, n, &steps[iv]);
            let wide = e.bin_c(BinaryOp::Ugt, &dist, MAX_SPAN);
            let bad = e.bin(BinaryOp::Or, &unfit[iv], &wide);
            let dist = e.bin_c(BinaryOp::Mul, &dist, unit);
            let span = e.bin(BinaryOp::Add, &dist, &bytes);
            (lo, e.select(&bad, &too_far, &span))
        };
        let (lo, span) = match &a.place {
            Place::Fixed { ptr } => (e.reg(ptr, &in_loop, &defs), bytes.clone()),
            Place::Indexed { ptr, iv, o, unit } => {
                let first = e.bin_c(BinaryOp::Add, &dparams[ivs[*iv].param].0, *o);
                let off = e.bin_c(BinaryOp::Mul, &first, *unit);
                let ptr = e.reg(ptr, &in_loop, &defs);
                let lo = e.bin(BinaryOp::Add, &ptr, &off);
                strided(&mut e, lo, *iv, *unit)
            }
            Place::Bumped { iv, o } => {
                let lo = e.bin_c(BinaryOp::Add, &dparams[ivs[*iv].param].0, *o);
                strided(&mut e, lo, *iv, 1)
            }
        };
        let span = if last_too {
            span
        } else {
            e.select(&no_trips, &zero, &span)
        };
        ranges.push(Range {
            lo,
            span,
            writes: a.writes,
        });
    }
    let ok = e.fresh();
    e.out.push(Instruction::AssignConst {
        dst: ok.clone(),
        val: Constant::Int(0),
        ty: Type::I32,
        span,
    });
    let dispatch_insts = e.out;

    // The copy, entered from the dispatch block, with the loop's own registers renamed.
    let labels: HashSet<&str> = body.iter().map(|&b| f.blocks[b].label.as_str()).collect();
    let reg = |name: &str| -> String {
        if in_loop.contains(name) {
            format!("{prefix}_{name}")
        } else {
            name.to_string()
        }
    };
    let label = |l: &str| -> String {
        if labels.contains(l) {
            format!("{prefix}_{l}")
        } else {
            l.to_string()
        }
    };
    let exit_params: Vec<(String, String)> = live_out
        .iter()
        .map(|v| (v.clone(), format!("{prefix}_x_{v}")))
        .collect();
    let mut copy: Vec<Block> = Vec::new();
    for &b in &body {
        let src = &f.blocks[b];
        let mut instructions = Vec::with_capacity(src.instructions.len());
        for inst in &src.instructions {
            let mut inst = inst.clone();
            for d in inst.dsts_mut() {
                *d = reg(d);
            }
            for r in inst.operands_mut() {
                *r = reg(r);
            }
            instructions.push(inst);
        }
        let mut terminator = src.terminator.clone();
        retarget(&mut terminator, &label);
        for r in crate::names::terminator_operands_mut(&mut terminator) {
            *r = reg(r);
        }
        add_exit_args(&mut terminator, &exit, &exit_params, |v| reg(v));
        copy.push(Block {
            label: label(&src.label),
            params: src.params.iter().map(|(p, t)| (reg(p), *t)).collect(),
            instructions,
            terminator,
            span: src.span,
        });
    }
    let dispatch_label = format!("{prefix}_dispatch");
    let dargs: Vec<String> = dparams.iter().map(|(p, _)| p.clone()).collect();
    let dispatch = Block {
        label: dispatch_label.clone(),
        params: dparams,
        instructions: dispatch_insts,
        terminator: Terminator::Br {
            cond: ok,
            then_block: label(header),
            then_args: dargs.clone(),
            else_block: header.to_string(),
            else_args: dargs,
            span,
        },
        span,
    };

    for a in &accesses {
        fc.covered.insert((label(&f.blocks[a.block].label), a.inst));
    }
    fc.free.insert(dispatch_label.clone());
    fc.free.extend(copy.iter().map(|b| b.label.clone()));
    fc.dispatch
        .insert(dispatch_label.clone(), Dispatch { units, ranges });

    // The entry goes to the dispatch block; the exit takes the loop's live-out registers.
    retarget(&mut f.blocks[entry].terminator, &|l: &str| {
        if l == header {
            dispatch_label.clone()
        } else {
            l.to_string()
        }
    });
    route_live_outs(f, &in_body, exit_idx, &exit_params, &live_types);
    let at = body.iter().min().copied().unwrap_or(h);
    let mut inserted = vec![dispatch];
    inserted.extend(copy);
    f.blocks.splice(at..at, inserted);
    true
}
