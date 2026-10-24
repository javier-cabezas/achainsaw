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
//! A loop qualifies when its blocks form a chain from the header, whose branch is the only way
//! out, through blocks that each jump to the next and back to the header; it calls nothing
//! (calls, `alloc`, `free`, `mm`, `par`), and one block outside it enters it. Its header tests
//! one counter `i`, a header parameter that grows by a positive constant or a `vl` per
//! iteration, as `lt i, n`, `gt n, i`, or `gt r, k` / `ge r, k` on `r = sub n, i` (the
//! masked-loop guards of `crate::loops`), possibly several such tests joined by `and`, with
//! `n` and `k` invariant. The module must have unique register names; the result is meant for
//! code generation only (the dispatch block's condition is the backend's to compute).

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
    /// Byte ranges `(lo, span)` (ptr, i64) holding every covered access; a span of 0 is
    /// empty, and one above the sandbox's size (-1 when too large to compute) never fits.
    pub ranges: Vec<(String, String)>,
}

/// More fuel than any budget: a loop that would not end charges it.
pub const ENDLESS_UNITS: i64 = 1 << 62;
/// Trip counts above this charge `ENDLESS_UNITS`, so `trips * blocks` cannot overflow.
const MAX_TRIPS: i64 = 1 << 48;
/// Spans above this many bytes do not fit any sandbox.
const MAX_SPAN: i64 = 1 << 40;
/// Steps above this many elements leave the dispatch's ranges unchecked: the copy never runs.
const MAX_STEP: i64 = 1 << 20;
/// Longer chains of blocks are left alone, so `trips * blocks` cannot overflow.
const MAX_CHAIN: usize = 1 << 10;

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

/// How the counter grows per iteration: a constant, or a register holding a `vl`.
#[derive(Clone)]
enum Step {
    Const(i64),
    Reg(String),
}

/// A header test that holds while the trips go on.
enum Test {
    /// `i < n`.
    Below { n: String },
    /// `n - i > k`, `k` a constant or `vl - 1` (`ge r, vl`).
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

/// The tests `cond` (defined in the header) joins, all on header parameter `iv`, or `None`.
fn parse_tests(
    cond: &str,
    defs: &HashMap<&str, &Instruction>,
    header_defs: &HashSet<&str>,
    invariant: &dyn Fn(&str) -> bool,
    iv: &mut Option<String>,
    header_params: &HashSet<&str>,
) -> Option<Vec<Test>> {
    if !header_defs.contains(cond) {
        return None;
    }
    let Instruction::Binary { op, lhs, rhs, .. } = defs.get(cond)? else {
        return None;
    };
    let mut counter = |r: &str| -> bool {
        if !header_params.contains(r) {
            return false;
        }
        match iv {
            Some(p) => p == r,
            None => {
                *iv = Some(r.to_string());
                true
            }
        }
    };
    // `r = sub n, i` defined in the header: its `n`.
    let rest = |r: &str, counter: &mut dyn FnMut(&str) -> bool| -> Option<String> {
        if !header_defs.contains(r) {
            return None;
        }
        match defs.get(r)? {
            Instruction::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
                ..
            } if invariant(lhs) && counter(rhs) => Some(lhs.clone()),
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
    match op {
        BinaryOp::And => {
            let mut tests = parse_tests(lhs, defs, header_defs, invariant, iv, header_params)?;
            tests.extend(parse_tests(
                rhs,
                defs,
                header_defs,
                invariant,
                iv,
                header_params,
            )?);
            Some(tests)
        }
        BinaryOp::Lt if invariant(rhs) && counter(lhs) => {
            Some(vec![Test::Below { n: rhs.clone() }])
        }
        BinaryOp::Gt if invariant(lhs) && counter(rhs) => {
            Some(vec![Test::Below { n: lhs.clone() }])
        }
        // `r > k`, `k < r`, `r >= k`, `k <= r`.
        BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le => {
            let (r, k) = match op {
                BinaryOp::Gt | BinaryOp::Ge => (lhs, rhs),
                _ => (rhs, lhs),
            };
            let minus = matches!(op, BinaryOp::Ge | BinaryOp::Le) as i64;
            let n = rest(r, &mut counter)?;
            Some(vec![Test::Above {
                n,
                k: bound(k, minus)?,
            }])
        }
        _ => None,
    }
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
    if body.is_empty() {
        return false;
    }
    let in_body: HashSet<usize> = body.iter().copied().collect();
    let index: HashMap<&str, usize> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.as_str(), i))
        .collect();

    // The header's branch goes on into the chain and leaves to `exit`.
    let Terminator::Br {
        cond,
        then_block,
        then_args,
        else_block,
        ..
    } = &f.blocks[h].terminator
    else {
        return false;
    };
    if !in_body.contains(&index[then_block.as_str()])
        || in_body.contains(&index[else_block.as_str()])
    {
        return false;
    }
    let exit = else_block.clone();

    // The chain: header, then blocks that each jump to the next, the last back to the header.
    let mut chain = vec![h];
    let mut next = index[then_block.as_str()];
    while next != h {
        if !in_body.contains(&next) || chain.contains(&next) {
            return false;
        }
        chain.push(next);
        let Terminator::Jmp { target, .. } = &f.blocks[next].terminator else {
            return false;
        };
        next = index[target.as_str()];
    }
    if chain.len() != body.len() || chain.len() > MAX_CHAIN {
        return false;
    }
    let size: usize = chain.iter().map(|&b| f.blocks[b].instructions.len()).sum();
    let calls_out = chain
        .iter()
        .flat_map(|&b| &f.blocks[b].instructions)
        .any(|i| {
            matches!(
                i,
                Instruction::Call { .. }
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
    let header_params: HashSet<&str> = header_block
        .params
        .iter()
        .map(|(p, _)| p.as_str())
        .collect();
    let header_defs: HashSet<&str> = header_block
        .instructions
        .iter()
        .flat_map(|i| i.dsts())
        .collect();
    let invariant = |r: &str| !in_loop.contains(r) || constant_like(&defs, r);

    let mut iv: Option<String> = None;
    let Some(tests) = parse_tests(
        cond,
        &defs,
        &header_defs,
        &invariant,
        &mut iv,
        &header_params,
    ) else {
        return false;
    };
    let Some(iv) = iv else {
        return false;
    };
    let k = header_block
        .params
        .iter()
        .position(|(p, _)| *p == iv)
        .expect("header parameter");
    if header_block.params[k].1 != Type::I64 {
        return false;
    }

    // Registers equal to the counter plus a constant, along the chain.
    let mut offset: HashMap<String, i64> = HashMap::from([(iv.clone(), 0)]);
    let mut incoming = then_args.clone();
    for (pos, &b) in chain.iter().enumerate() {
        let block = &f.blocks[b];
        if pos > 0 {
            for ((p, _), a) in block.params.iter().zip(&incoming) {
                if let Some(&o) = offset.get(a.as_str()) {
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
                    (BinaryOp::Add, Some(&o), None) => {
                        int_const(&defs, rhs).and_then(|c| o.checked_add(c))
                    }
                    (BinaryOp::Add, None, Some(&o)) => {
                        int_const(&defs, lhs).and_then(|c| o.checked_add(c))
                    }
                    (BinaryOp::Sub, Some(&o), None) => {
                        int_const(&defs, rhs).and_then(|c| o.checked_sub(c))
                    }
                    _ => None,
                };
                if let Some(o) = shifted {
                    offset.insert(dst.clone(), o);
                }
            }
        }
        if let Terminator::Jmp { args, .. } = &block.terminator {
            incoming = args.clone();
        }
    }
    // The next counter, which the last block passes back to the header.
    let next_iv = match chain.len() {
        1 => &then_args[k],
        _ => &incoming[k],
    };
    let step = match offset.get(next_iv.as_str()) {
        Some(&s) if s > 0 && s <= 1 << 32 => Step::Const(s),
        Some(_) => return false,
        None => match defs.get(next_iv.as_str()) {
            Some(Instruction::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
                ..
            }) => {
                let w = match (offset.get(lhs.as_str()), offset.get(rhs.as_str())) {
                    (Some(0), None) => rhs,
                    (None, Some(0)) => lhs,
                    _ => return false,
                };
                match defs.get(w.as_str()) {
                    Some(Instruction::VLen { .. }) => Step::Reg(w.clone()),
                    _ => return false,
                }
            }
            _ => return false,
        },
    };

    // Types are needed for stores and live-out registers.
    let types_cell: std::cell::OnceCell<HashMap<String, Type>> = std::cell::OnceCell::new();
    let types = || types_cell.get_or_init(|| infer_reg_types(f, sigs));

    // Indexed accesses of invariant pointers by the counter plus a constant, and accesses at
    // invariant addresses: (chain position, instruction, pointer, offset and unit bytes of
    // the index, type of the access).
    struct Access {
        pos: usize,
        inst: usize,
        ptr: String,
        index: Option<(i64, i64)>,
        ty: Type,
    }
    let mut accesses: Vec<Access> = Vec::new();
    for (pos, &b) in chain.iter().enumerate() {
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
            if in_loop.contains(ptr.as_str()) {
                continue;
            }
            let index = match index {
                None => None,
                Some(ix) => match offset.get(ix.reg.as_str()) {
                    Some(&o) => Some((o, ix.unit_or(implied).byte_size() as i64)),
                    None => continue,
                },
            };
            accesses.push(Access {
                pos,
                inst: i,
                ptr: ptr.clone(),
                index,
                ty,
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

    // The dispatch block takes the header's parameters; `i0` is the counter's first value.
    let dparams: Vec<(String, Type)> = f.blocks[h]
        .params
        .iter()
        .map(|(p, t)| (format!("{prefix}_p_{p}"), *t))
        .collect();
    let i0 = dparams[k].0.clone();
    let s = match &step {
        Step::Const(c) => e.cst(*c, Type::I64),
        Step::Reg(w) => e.reg(w, &in_loop, &defs),
    };

    // Trips: the fewest any test allows. `i < n` holds for `(n - i0 - 1) / s + 1` trips when
    // i0 < n, unless the counter would wrap past i64::MAX and the loop never end;
    // `n - i > k` holds for `(n - i0 - k - 1) / s + 1` when `n - i0 > k` (wrapping), since it
    // falls by `s` without wrapping while it is above k.
    let zero = e.cst(0, Type::I64);
    let endless = e.cst(ENDLESS_UNITS, Type::I64);
    let mut trips: Option<String> = None;
    for test in &tests {
        let t = match test {
            Test::Below { n } => {
                let n = e.reg(n, &in_loop, &defs);
                let some = e.bin(BinaryOp::Lt, &i0, &n);
                let d = e.bin(BinaryOp::Sub, &n, &i0);
                let t = e.ceil_div(&d, &s, &step);
                let t1 = e.bin_c(BinaryOp::Sub, &t, 1);
                let last = e.bin(BinaryOp::Mul, &t1, &s);
                let last = e.bin(BinaryOp::Add, &i0, &last);
                let max = e.cst(i64::MAX, Type::I64);
                let room = e.bin(BinaryOp::Sub, &max, &s);
                let wraps = e.bin(BinaryOp::Gt, &last, &room);
                let t = e.select(&wraps, &endless, &t);
                e.select(&some, &t, &zero)
            }
            Test::Above { n, k } => {
                let n = e.reg(n, &in_loop, &defs);
                let k = match k {
                    Bound::Const(c) => e.cst(*c, Type::I64),
                    Bound::VlMinus1(w) => {
                        let w = e.reg(w, &in_loop, &defs);
                        e.bin_c(BinaryOp::Sub, &w, 1)
                    }
                };
                let r0 = e.bin(BinaryOp::Sub, &n, &i0);
                let some = e.bin(BinaryOp::Gt, &r0, &k);
                let d = e.bin(BinaryOp::Sub, &r0, &k);
                let t = e.ceil_div(&d, &s, &step);
                e.select(&some, &t, &zero)
            }
        };
        trips = Some(match trips {
            None => t,
            Some(prev) => e.bin(BinaryOp::Umin, &prev, &t),
        });
    }
    let trips = trips.expect("at least one test");

    // Fuel: one unit per block per trip, and the header's last branch.
    let many = e.bin_c(BinaryOp::Ugt, &trips, MAX_TRIPS);
    let per_trip = e.bin_c(BinaryOp::Mul, &trips, chain.len() as i64);
    let all = e.bin_c(BinaryOp::Add, &per_trip, 1);
    let units = e.select(&many, &endless, &all);

    // Ranges. The header runs once more than the other blocks, on the last test, so its
    // accesses reach `trips * s` elements past the first, and the others' `(trips - 1) * s`
    // (none when there are no trips). Too many trips, or too long a step, cannot fit: with
    // trips <= MAX_SPAN and s <= MAX_STEP the products stay below 2^64.
    let mut ranges = Vec::new();
    let no_trips = e.bin(BinaryOp::Eq, &trips, &zero);
    let many = e.bin_c(BinaryOp::Ugt, &trips, MAX_SPAN);
    let long = e.bin_c(BinaryOp::Ugt, &s, MAX_STEP);
    let unfit = e.bin(BinaryOp::Or, &many, &long);
    let too_far = e.cst(-1, Type::I64);
    let trips1 = e.bin_c(BinaryOp::Sub, &trips, 1);
    for a in &accesses {
        let bytes = access_bytes(&mut e, a.ty);
        let (lo, span) = match a.index {
            None => (a.ptr.clone(), bytes),
            Some((o, unit)) => {
                let first = e.bin_c(BinaryOp::Add, &i0, o);
                let off = e.bin_c(BinaryOp::Mul, &first, unit);
                let lo = e.bin(BinaryOp::Add, &a.ptr, &off);
                let steps = if a.pos == 0 { &trips } else { &trips1 };
                let dist = e.bin(BinaryOp::Mul, steps, &s);
                let wide = e.bin_c(BinaryOp::Ugt, &dist, MAX_SPAN);
                let bad = e.bin(BinaryOp::Or, &unfit, &wide);
                let dist = e.bin_c(BinaryOp::Mul, &dist, unit);
                let span = e.bin(BinaryOp::Add, &dist, &bytes);
                (lo, e.select(&bad, &too_far, &span))
            }
        };
        let span = if a.pos == 0 {
            span
        } else {
            e.select(&no_trips, &zero, &span)
        };
        ranges.push((lo, span));
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
    let labels: HashSet<&str> = chain.iter().map(|&b| f.blocks[b].label.as_str()).collect();
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
    for &b in &chain {
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
        fc.covered
            .insert((label(&f.blocks[chain[a.pos]].label), a.inst));
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
