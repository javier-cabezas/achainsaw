//! Full-vector loops for masked ones. The simplest correct vector loop, and the one models
//! write, masks every access: `r = sub n, i`, `v = ldm x[i]:vx, r:f32`, ..., `stm y[i], v, r:f32`.
//! A masked access costs a branch or a mask per iteration where a plain one costs nothing.
//! When every iteration with `n - i` at least a vector's lanes would load and store whole
//! vectors anyway, [`split_masked_loops`] puts a copy of the loop in front of it. The copy runs
//! while a guard checked on every iteration shows `n - i` is at least the lanes of each masked
//! access that depends on it, and plain `ld`/`st` replace those accesses there. When the guard
//! fails, the original loop runs the remaining iterations, tail included. Each iteration does
//! what it did before, so results are unchanged, on every backend; only fuel use differs (the
//! guard is a block of its own).
//!
//! A count is known to cover the lanes when it is `sub n, p` (`p` a parameter of the loop
//! header, `n` defined outside the loop, guarded as `n - p >= lanes`), a `vl` of a lane type
//! no wider than the access's, or the `min` of two such counts. Loops whose blocks leave to
//! more than one block, or longer than [`MAX_LOOP_INSTRUCTIONS`], are left alone. The module
//! must be valid, with unique register names (`crate::names::unique_names`).

use crate::ast::*;
use crate::cfg::Cfg;
use crate::opt::{infer_reg_types, Signatures};
use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// Larger loops are not copied, to bound code size.
pub const MAX_LOOP_INSTRUCTIONS: usize = 400;

/// Splits every eligible loop in `module`; returns how many it split.
pub fn split_masked_loops(module: &mut Module) -> usize {
    let mut sigs: Signatures = HashMap::new();
    for ext in &module.extern_functions {
        sigs.insert(ext.name.clone(), ext.ret_type.into_iter().collect());
    }
    for func in &module.functions {
        sigs.insert(func.name.clone(), func.rets.clone());
    }
    module
        .functions
        .iter_mut()
        .map(|f| split_function(f, &sigs))
        .sum()
}

/// The lanes a masked access covers: `vl lane` lanes of `vx`, or a fixed count.
#[derive(Clone, Copy, PartialEq)]
enum Lanes {
    Vx(Type),
    Fixed(i64),
}

fn lanes_of(ty: Type, lane: Type) -> Lanes {
    if ty == Type::Vx {
        Lanes::Vx(lane)
    } else {
        Lanes::Fixed((ty.byte_size() / lane.byte_size()) as i64)
    }
}

/// A guard term: `n - p >= lanes`, or `p < n` without lanes.
#[derive(Clone, PartialEq)]
struct Fact {
    n: String,
    p: String,
    lanes: Option<Lanes>,
}

/// True if some loop of `f` holds a masked access (worth a closer look).
pub fn has_split_candidates(f: &Function) -> bool {
    !candidate_headers(f).is_empty()
}

/// The headers of loops that hold a masked access.
fn candidate_headers(f: &Function) -> Vec<String> {
    let masked = |b: &Block| {
        b.instructions.iter().any(|i| {
            matches!(
                i,
                Instruction::MaskedLoad { .. } | Instruction::MaskedStore { .. }
            )
        })
    };
    if !f.blocks.iter().any(masked) {
        return Vec::new();
    }
    let cfg = Cfg::build(f);
    (1..f.blocks.len())
        .filter(|&h| cfg.reachable[h])
        .filter(|&h| natural_loop(&cfg, h).iter().any(|&b| masked(&f.blocks[b])))
        .map(|h| f.blocks[h].label.clone())
        .collect()
}

fn split_function(f: &mut Function, sigs: &Signatures) -> usize {
    let headers: Vec<String> = candidate_headers(f);
    let mut serial = 0;
    let mut count = 0;
    for header in headers {
        if split_loop(f, &header, sigs, &mut serial) {
            count += 1;
        }
    }
    count
}

/// The blocks of the natural loop of block `h` (empty if `h` heads no loop).
pub(crate) fn natural_loop(cfg: &Cfg, h: usize) -> Vec<usize> {
    let tails: Vec<usize> = cfg.preds[h]
        .iter()
        .copied()
        .filter(|&t| cfg.reachable[t] && cfg.dominates(h, t))
        .collect();
    if tails.is_empty() {
        return Vec::new();
    }
    let mut body: HashSet<usize> = HashSet::from([h]);
    let mut stack = tails;
    while let Some(b) = stack.pop() {
        if body.insert(b) {
            stack.extend(cfg.preds[b].iter().copied().filter(|&p| cfg.reachable[p]));
        }
    }
    let mut blocks: Vec<usize> = body.into_iter().collect();
    blocks.sort();
    blocks
}

pub(crate) fn targets(t: &Terminator) -> Vec<&String> {
    match t {
        Terminator::Jmp { target, .. } => vec![target],
        Terminator::Br {
            then_block,
            else_block,
            ..
        } => vec![then_block, else_block],
        Terminator::Ret { .. } => vec![],
    }
}

/// Facts that make `count >= lanes`, if there are any.
fn covering_facts(
    count: &str,
    lanes: Lanes,
    defs: &HashMap<&str, &Instruction>,
    header_params: &HashSet<&str>,
    in_loop: &HashSet<&str>,
) -> Option<Vec<Fact>> {
    match defs.get(count)? {
        Instruction::Binary {
            op: BinaryOp::Sub,
            lhs,
            rhs,
            ..
        } if header_params.contains(rhs.as_str()) && !in_loop.contains(lhs.as_str()) => {
            Some(vec![Fact {
                n: lhs.clone(),
                p: rhs.clone(),
                lanes: Some(lanes),
            }])
        }
        Instruction::Binary {
            op: BinaryOp::Min,
            lhs,
            rhs,
            ..
        } => {
            let mut facts = covering_facts(lhs, lanes, defs, header_params, in_loop)?;
            facts.extend(covering_facts(rhs, lanes, defs, header_params, in_loop)?);
            Some(facts)
        }
        // `vl t` lanes of vx cover a vx access of lanes at least as wide, and a v128 one.
        Instruction::VLen { lane: t, .. } => {
            let covers = match lanes {
                Lanes::Vx(lane) => t.byte_size() <= lane.byte_size(),
                Lanes::Fixed(k) => (16 / t.byte_size()) as i64 >= k,
            };
            covers.then(Vec::new)
        }
        _ => None,
    }
}

fn split_loop(f: &mut Function, header: &str, sigs: &Signatures, serial: &mut usize) -> bool {
    let cfg = Cfg::build(f);
    let Some(h) = f.blocks.iter().position(|b| b.label == header) else {
        return false;
    };
    if h == 0 || !cfg.reachable[h] {
        return false;
    }
    let body = natural_loop(&cfg, h);
    if body.is_empty() {
        return false;
    }
    let size: usize = body.iter().map(|&b| f.blocks[b].instructions.len()).sum();
    if size > MAX_LOOP_INSTRUCTIONS {
        return false;
    }
    let in_body: HashSet<usize> = body.iter().copied().collect();
    let labels: HashSet<&str> = body.iter().map(|&b| f.blocks[b].label.as_str()).collect();

    // One block outside the loop that it leaves to.
    let mut exits: HashSet<&str> = HashSet::new();
    for &b in &body {
        for t in targets(&f.blocks[b].terminator) {
            if !labels.contains(t.as_str()) {
                exits.insert(t.as_str());
            }
        }
    }
    let [exit] = exits.into_iter().collect::<Vec<_>>()[..] else {
        return false;
    };
    let exit = exit.to_string();

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
    let header_params: HashSet<&str> = f.blocks[h].params.iter().map(|(p, _)| p.as_str()).collect();

    // Types are needed for stores and live-out registers only, and cost a pass.
    let types_cell: std::cell::OnceCell<HashMap<String, Type>> = std::cell::OnceCell::new();
    let types = || types_cell.get_or_init(|| infer_reg_types(f, sigs));

    // The masked accesses whose counts the guard covers, by block and instruction.
    let mut facts: Vec<Fact> = Vec::new();
    let mut plain: HashSet<(usize, usize)> = HashSet::new();
    for &b in &body {
        for (k, inst) in f.blocks[b].instructions.iter().enumerate() {
            let (count, lanes) = match inst {
                Instruction::MaskedLoad {
                    count, ty, lane, ..
                } => (count, lanes_of(*ty, *lane)),
                Instruction::MaskedStore {
                    val, count, lane, ..
                } => {
                    let Some(&ty) = types().get(val.as_str()) else {
                        continue;
                    };
                    (count, lanes_of(ty, *lane))
                }
                _ => continue,
            };
            let Some(needed) = covering_facts(count, lanes, &defs, &header_params, &in_loop) else {
                continue;
            };
            // A count that is only a `vl` needs no guard, and is not this loop's to split.
            if needed.is_empty() {
                continue;
            }
            for fact in needed {
                if !facts.contains(&fact) {
                    facts.push(fact);
                }
            }
            plain.insert((b, k));
        }
    }
    if plain.is_empty() {
        return false;
    }

    // The header's own test, when the guard decides it (`lt p, n`, or `gt rest, 0` on a
    // guarded `rest = sub n, p`): the copy then goes straight to the branch it takes.
    let taken = header_test(&f.blocks[h], &defs, &header_params, &in_loop, &mut facts);

    let exit_idx = f
        .blocks
        .iter()
        .position(|b| b.label == exit)
        .expect("exit block");
    let Some((live_out, live_types)) = live_outs(f, &cfg, &in_body, &in_loop, exit_idx, &types)
    else {
        return false;
    };

    // A prefix no name in the function starts with.
    let names = all_names(f);
    let prefix = loop {
        *serial += 1;
        let p = format!("__ts{serial}");
        if !names.iter().any(|n| n.starts_with(&p)) {
            break p;
        }
    };
    let reg = |name: &str| -> String {
        if in_loop.contains(name) {
            format!("{prefix}_{name}")
        } else {
            name.to_string()
        }
    };
    let guard_label = format!("{prefix}_guard");
    let label = |l: &str| -> String {
        if l == header {
            guard_label.clone()
        } else if labels.contains(l) {
            format!("{prefix}_{l}")
        } else {
            l.to_string()
        }
    };
    let exit_params: Vec<(String, String)> = live_out
        .iter()
        .map(|v| (v.clone(), format!("{prefix}_x_{v}")))
        .collect();

    // The copy, with plain accesses, entered and re-entered through the guard.
    let mut copy: Vec<Block> = Vec::new();
    for &b in &body {
        let src = &f.blocks[b];
        let mut instructions = Vec::with_capacity(src.instructions.len());
        for (k, inst) in src.instructions.iter().enumerate() {
            let mut inst = if plain.contains(&(b, k)) {
                unmasked(inst)
            } else {
                inst.clone()
            };
            for d in inst.dsts_mut() {
                *d = reg(d);
            }
            for r in inst.operands_mut() {
                *r = reg(r);
            }
            instructions.push(inst);
        }
        let mut terminator = src.terminator.clone();
        if let (
            true,
            Some(then),
            Terminator::Br {
                then_block,
                then_args,
                else_block,
                else_args,
                span,
                ..
            },
        ) = (b == h, taken, &terminator)
        {
            let (target, args) = if then {
                (then_block, then_args)
            } else {
                (else_block, else_args)
            };
            terminator = Terminator::Jmp {
                target: target.clone(),
                args: args.clone(),
                span: *span,
            };
        }
        retarget(&mut terminator, &label);
        for r in crate::names::terminator_operands_mut(&mut terminator) {
            *r = reg(r);
        }
        add_exit_args(&mut terminator, &exit, &exit_params, |v| reg(v));
        copy.push(Block {
            label: format!("{prefix}_{}", src.label),
            params: src.params.iter().map(|(p, t)| (reg(p), *t)).collect(),
            instructions,
            terminator,
            span: src.span,
        });
    }
    let guard = guard_block(&f.blocks[h], &guard_label, &prefix, &facts, header);

    // Entries go to the guard.
    for (bi, block) in f.blocks.iter_mut().enumerate() {
        if !in_body.contains(&bi) {
            retarget(&mut block.terminator, &|l: &str| {
                if l == header {
                    guard_label.clone()
                } else {
                    l.to_string()
                }
            });
        }
    }
    route_live_outs(f, &in_body, exit_idx, &exit_params, &live_types);
    let at = body.iter().min().copied().unwrap_or(h);
    let mut inserted = vec![guard];
    inserted.extend(copy);
    f.blocks.splice(at..at, inserted);
    true
}

/// Registers the loop (blocks `in_body`, defining `in_loop`) defines and later code reads,
/// with their types: the exit block takes them as parameters when a copy of the loop also
/// defines them. `None` when it cannot, since another way into it does not come from code it
/// dominates, where those parameters are in scope.
pub(crate) fn live_outs<'a>(
    f: &Function,
    cfg: &Cfg,
    in_body: &HashSet<usize>,
    in_loop: &HashSet<&str>,
    exit_idx: usize,
    types: &dyn Fn() -> &'a HashMap<String, Type>,
) -> Option<(Vec<String>, Vec<Type>)> {
    let mut live_out: Vec<String> = Vec::new();
    for (bi, block) in f.blocks.iter().enumerate() {
        if in_body.contains(&bi) {
            continue;
        }
        let mut reads: Vec<&String> = block
            .instructions
            .iter()
            .flat_map(|i| i.operands())
            .collect();
        reads.extend(terminator_reads(&block.terminator));
        for r in reads {
            if in_loop.contains(r.as_str()) && !live_out.contains(r) {
                live_out.push(r.clone());
            }
        }
    }
    if !live_out.is_empty() {
        let outside = cfg.preds[exit_idx]
            .iter()
            .filter(|p| !in_body.contains(p) && cfg.reachable[**p]);
        if outside.clone().any(|&p| !cfg.dominates(exit_idx, p))
            || live_out.iter().any(|v| !types().contains_key(v))
        {
            return None;
        }
    }
    let live_types = live_out.iter().map(|v| types()[v.as_str()]).collect();
    Some((live_out, live_types))
}

/// Gives the exit block `exit_params` (live-out register, parameter), passed by the edges of
/// the loop (`in_body`) as the registers themselves; code outside the loop reads the
/// parameters instead. Copies of the loop pass their own registers (`add_exit_args`).
pub(crate) fn route_live_outs(
    f: &mut Function,
    in_body: &HashSet<usize>,
    exit_idx: usize,
    exit_params: &[(String, String)],
    live_types: &[Type],
) {
    if exit_params.is_empty() {
        return;
    }
    let exit = f.blocks[exit_idx].label.clone();
    let subst: HashMap<String, String> = exit_params.iter().cloned().collect();
    for (bi, block) in f.blocks.iter_mut().enumerate() {
        if in_body.contains(&bi) {
            add_exit_args(&mut block.terminator, &exit, exit_params, |v| v.to_string());
            continue;
        }
        for inst in &mut block.instructions {
            for r in inst.operands_mut() {
                if let Some(n) = subst.get(r.as_str()) {
                    *r = n.clone();
                }
            }
        }
        for r in crate::names::terminator_operands_mut(&mut block.terminator) {
            if let Some(n) = subst.get(r.as_str()) {
                *r = n.clone();
            }
        }
        // Other ways into the exit block come from code it dominates, which now reads the
        // parameters.
        add_exit_args(&mut block.terminator, &exit, exit_params, |v| {
            subst[v].clone()
        });
    }
    let exit_block = &mut f.blocks[exit_idx];
    for ((_, p), ty) in exit_params.iter().zip(live_types) {
        exit_block.params.push((p.clone(), *ty));
    }
}

/// Which way the header's branch goes when the guard holds, if the guard decides it; adds the
/// `p < n` term that `lt p, n` needs.
fn header_test(
    header: &Block,
    defs: &HashMap<&str, &Instruction>,
    header_params: &HashSet<&str>,
    in_loop: &HashSet<&str>,
    facts: &mut Vec<Fact>,
) -> Option<bool> {
    let Terminator::Br { cond, .. } = &header.terminator else {
        return None;
    };
    let Some(Instruction::Binary { op, lhs, rhs, .. }) = defs.get(cond.as_str()) else {
        return None;
    };
    let index = |r: &str| header_params.contains(r);
    let invariant = |r: &str| !in_loop.contains(r);
    let int = |r: &str| match defs.get(r) {
        Some(Instruction::AssignConst {
            val: Constant::Int(k),
            ..
        }) => Some(*k),
        _ => None,
    };
    // `rest` is `sub n, p` with `n - p >= lanes >= 1` in the guard.
    let guarded_rest = |r: &str| match defs.get(r) {
        Some(Instruction::Binary {
            op: BinaryOp::Sub,
            lhs: n,
            rhs: p,
            ..
        }) => facts
            .iter()
            .any(|f| f.lanes.is_some() && f.n == *n && f.p == *p),
        _ => false,
    };
    let (p, n, when_less) = match op {
        BinaryOp::Lt if index(lhs) && invariant(rhs) => (lhs, rhs, true),
        BinaryOp::Gt if index(rhs) && invariant(lhs) => (rhs, lhs, true),
        BinaryOp::Ge if index(lhs) && invariant(rhs) => (lhs, rhs, false),
        BinaryOp::Le if index(rhs) && invariant(lhs) => (rhs, lhs, false),
        BinaryOp::Gt if guarded_rest(lhs) && int(rhs).is_some_and(|k| k <= 0) => return Some(true),
        BinaryOp::Ge if guarded_rest(lhs) && int(rhs).is_some_and(|k| k <= 1) => return Some(true),
        BinaryOp::Le if guarded_rest(lhs) && int(rhs).is_some_and(|k| k <= 0) => {
            return Some(false)
        }
        BinaryOp::Lt if guarded_rest(lhs) && int(rhs).is_some_and(|k| k <= 1) => {
            return Some(false)
        }
        _ => return None,
    };
    let fact = Fact {
        n: n.clone(),
        p: p.clone(),
        lanes: None,
    };
    if !facts.contains(&fact) {
        facts.push(fact);
    }
    Some(when_less)
}

/// `guard(params): rest = sub n, p; ok = ge rest, lanes; ...; br ok, copy(params), header(params)`.
fn guard_block(h: &Block, label: &str, prefix: &str, facts: &[Fact], header: &str) -> Block {
    let span = h.span;
    let params: Vec<(String, Type)> = h
        .params
        .iter()
        .map(|(p, t)| (format!("{prefix}_g_{p}"), *t))
        .collect();
    let param = |p: &str| -> String {
        if h.params.iter().any(|(q, _)| q == p) {
            format!("{prefix}_g_{p}")
        } else {
            p.to_string()
        }
    };
    let mut instructions = Vec::new();
    let mut ok: Option<String> = None;
    for (k, fact) in facts.iter().enumerate() {
        let rest = format!("{prefix}_g{k}_rest");
        let lanes = format!("{prefix}_g{k}_lanes");
        let cond = format!("{prefix}_g{k}_ok");
        let Some(fact_lanes) = fact.lanes else {
            instructions.push(Instruction::Binary {
                op: BinaryOp::Lt,
                dst: cond.clone(),
                lhs: param(&fact.p),
                rhs: param(&fact.n),
                span,
            });
            ok = Some(conjoin(
                &mut instructions,
                ok,
                cond,
                &format!("{prefix}_g{k}_all"),
                span,
            ));
            continue;
        };
        instructions.push(Instruction::Binary {
            op: BinaryOp::Sub,
            dst: rest.clone(),
            lhs: param(&fact.n),
            rhs: param(&fact.p),
            span,
        });
        instructions.push(match fact_lanes {
            Lanes::Vx(lane) => Instruction::VLen {
                dst: lanes.clone(),
                lane,
                span,
            },
            Lanes::Fixed(k) => Instruction::AssignConst {
                dst: lanes.clone(),
                val: Constant::Int(k),
                ty: Type::I64,
                span,
            },
        });
        instructions.push(Instruction::Binary {
            op: BinaryOp::Ge,
            dst: cond.clone(),
            lhs: rest,
            rhs: lanes,
            span,
        });
        ok = Some(conjoin(
            &mut instructions,
            ok,
            cond,
            &format!("{prefix}_g{k}_all"),
            span,
        ));
    }
    let args: Vec<String> = params.iter().map(|(p, _)| p.clone()).collect();
    Block {
        label: label.to_string(),
        params,
        instructions,
        terminator: Terminator::Br {
            cond: ok.expect("at least one fact"),
            then_block: format!("{prefix}_{header}"),
            then_args: args.clone(),
            else_block: header.to_string(),
            else_args: args,
            span,
        },
        span,
    }
}

/// `cond`, and-ed with the conditions so far (`prev`) into `both`.
fn conjoin(
    instructions: &mut Vec<Instruction>,
    prev: Option<String>,
    cond: String,
    both: &str,
    span: crate::diag::Span,
) -> String {
    let Some(prev) = prev else {
        return cond;
    };
    instructions.push(Instruction::Binary {
        op: BinaryOp::And,
        dst: both.to_string(),
        lhs: prev,
        rhs: cond,
        span,
    });
    both.to_string()
}

/// A masked access as a plain one (its count covers the lanes).
fn unmasked(inst: &Instruction) -> Instruction {
    let unit = |index: &Option<Index>, lane: Type| {
        index.as_ref().map(|ix| Index {
            reg: ix.reg.clone(),
            unit: Some(ix.unit_or(lane)),
        })
    };
    match inst {
        Instruction::MaskedLoad {
            dst,
            ptr,
            index,
            ty,
            lane,
            span,
            ..
        } => Instruction::Load {
            dst: dst.clone(),
            ptr: ptr.clone(),
            index: unit(index, *lane),
            ty: *ty,
            span: *span,
        },
        Instruction::MaskedStore {
            ptr,
            index,
            val,
            lane,
            span,
            ..
        } => Instruction::Store {
            ptr: ptr.clone(),
            index: unit(index, *lane),
            val: val.clone(),
            span: *span,
        },
        other => other.clone(),
    }
}

pub(crate) fn retarget(t: &mut Terminator, map: &dyn Fn(&str) -> String) {
    match t {
        Terminator::Jmp { target, .. } => *target = map(target),
        Terminator::Br {
            then_block,
            else_block,
            ..
        } => {
            *then_block = map(then_block);
            *else_block = map(else_block);
        }
        Terminator::Ret { .. } => {}
    }
}

/// Appends `arg(v)` for each live-out register `v` to the edges of `t` into `exit`.
pub(crate) fn add_exit_args(
    t: &mut Terminator,
    exit: &str,
    params: &[(String, String)],
    arg: impl Fn(&str) -> String,
) {
    if params.is_empty() {
        return;
    }
    let extra = || params.iter().map(|(v, _)| arg(v)).collect::<Vec<_>>();
    match t {
        Terminator::Jmp { target, args, .. } if target == exit => args.extend(extra()),
        Terminator::Br {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => {
            if then_block == exit {
                then_args.extend(extra());
            }
            if else_block == exit {
                else_args.extend(extra());
            }
        }
        _ => {}
    }
}

fn terminator_reads(t: &Terminator) -> Vec<&String> {
    match t {
        Terminator::Jmp { args, .. } => args.iter().collect(),
        Terminator::Br {
            cond,
            then_args,
            else_args,
            ..
        } => std::iter::once(cond)
            .chain(then_args)
            .chain(else_args)
            .collect(),
        Terminator::Ret { vals, .. } => vals.iter().collect(),
    }
}

pub(crate) fn all_names(f: &Function) -> HashSet<String> {
    let mut names: HashSet<String> = f.params.iter().map(|(p, _)| p.clone()).collect();
    for b in &f.blocks {
        names.insert(b.label.clone());
        names.extend(b.params.iter().map(|(p, _)| p.clone()));
        for i in &b.instructions {
            names.extend(i.dsts().into_iter().map(String::from));
        }
    }
    names
}
