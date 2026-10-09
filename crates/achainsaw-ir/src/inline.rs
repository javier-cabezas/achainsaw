//! Inlining of `inline fn`s. Every call to one is replaced by a copy of its body: the
//! calling block ends with a jump into the copy, whose `ret a, b` become jumps to a new
//! block that receives the results as its parameters and continues after the call. The
//! copy's entry is then merged into the calling block, and a copy with a single `ret` into
//! the continuation (its results replacing the call's destinations), so a straight-line
//! inline function adds no branches, and so no fuel checks, where it is used. Code
//! generation (both backends, JIT and AOT) inlines before lowering, so inline functions have
//! no call overhead; the optimizer does it too, so `achainsaw opt` shows the result. The
//! inline functions themselves stay in the module, callable as usual.

use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// `module` with every call to an `inline fn` expanded (borrowed when there is none).
pub fn inline_module(module: &Module) -> std::borrow::Cow<'_, Module> {
    if !module.functions.iter().any(|f| f.inline) {
        return std::borrow::Cow::Borrowed(module);
    }
    let mut m = module.clone();
    inline_in_place(&mut m);
    std::borrow::Cow::Owned(m)
}

/// Expands every call to an `inline fn` in `module` and returns how many it expanded. The
/// module must be valid (so inline functions are not recursive); the result is valid too.
pub fn inline_in_place(module: &mut Module) -> usize {
    let bodies: HashMap<String, Function> = module
        .functions
        .iter()
        .filter(|f| f.inline)
        .map(|f| (f.name.clone(), f.clone()))
        .collect();
    if bodies.is_empty() {
        return 0;
    }
    module
        .functions
        .iter_mut()
        .map(|f| inline_function(f, &bodies))
        .sum()
}

fn is_inline_call(inst: &Instruction, bodies: &HashMap<String, Function>) -> bool {
    matches!(inst, Instruction::Call { func, .. } if bodies.contains_key(func))
}

/// Every register and label name in `f`, to pick prefixes that collide with none of them.
fn names_in(f: &Function) -> HashSet<String> {
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

fn inline_function(f: &mut Function, bodies: &HashMap<String, Function>) -> usize {
    let mut names = names_in(f);
    let mut serial = 0;
    let mut count = 0;
    let mut bi = 0;
    while bi < f.blocks.len() {
        let Some(k) = f.blocks[bi]
            .instructions
            .iter()
            .position(|i| is_inline_call(i, bodies))
        else {
            bi += 1;
            continue;
        };
        // A prefix no name in the function starts with.
        let prefix = loop {
            serial += 1;
            let p = format!("__in{serial}");
            if !names.iter().any(|n| n.starts_with(&p)) {
                break p;
            }
        };
        let block = &mut f.blocks[bi];
        let rest = block.instructions.split_off(k + 1);
        let Some(Instruction::Call {
            dsts,
            func,
            args,
            span,
        }) = block.instructions.pop()
        else {
            unreachable!("position found an inline call");
        };
        let callee = &bodies[&func];
        let copy = Expansion::new(&prefix, callee, &args);
        let cont = format!("{prefix}ret");
        let mut new_blocks: Vec<Block> =
            callee.blocks.iter().map(|b| copy.block(b, &cont)).collect();
        // The results arrive as the continuation's parameters: the call's destinations, or
        // fresh names when the call discards them.
        let params = callee
            .rets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let name = dsts
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("{prefix}r{i}"));
                (name, *t)
            })
            .collect();
        let entry = copy.label(&callee.blocks[0].label);
        let terminator = std::mem::replace(
            &mut block.terminator,
            Terminator::Jmp {
                target: entry,
                args: Vec::new(),
                span,
            },
        );
        new_blocks.push(Block {
            label: cont,
            params,
            instructions: rest,
            terminator,
            span: block.span,
        });
        for b in &new_blocks {
            names.insert(b.label.clone());
            names.extend(b.params.iter().map(|(p, _)| p.clone()));
            for i in &b.instructions {
                names.extend(i.dsts().into_iter().map(String::from));
            }
        }
        f.blocks.splice(bi + 1..bi + 1, new_blocks);
        merge_entry(f, bi);
        merge_single_return(f, &prefix);
        count += 1;
        // The merged block may now hold inline calls from the callee or the code after the
        // call: look at it again (each pass removes one call, and inlining is not recursive).
    }
    count
}

/// Merges the callee copy's entry (the block after `bi`, entered only from `bi` since AIR
/// never branches to an entry, and without parameters) into block `bi`.
fn merge_entry(f: &mut Function, bi: usize) {
    let entry = f.blocks.remove(bi + 1);
    let b = &mut f.blocks[bi];
    b.instructions.extend(entry.instructions);
    b.terminator = entry.terminator;
}

/// When the copy has a single `ret` (one jump to the continuation `{prefix}ret`), merges the
/// continuation into that block: the continuation's parameters, the call's results, become
/// the returned registers wherever they are used.
fn merge_single_return(f: &mut Function, prefix: &str) {
    let cont = format!("{prefix}ret");
    let jumps: Vec<usize> = f
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(&b.terminator, Terminator::Jmp { target, .. } if *target == cont))
        .map(|(i, _)| i)
        .collect();
    let [from] = jumps[..] else {
        return;
    };
    let Some(ci) = f.blocks.iter().position(|b| b.label == cont) else {
        return;
    };
    let Terminator::Jmp { args, .. } = &f.blocks[from].terminator else {
        unreachable!()
    };
    let subst: HashMap<String, String> = f.blocks[ci]
        .params
        .iter()
        .map(|(p, _)| p.clone())
        .zip(args.iter().cloned())
        .collect();
    let cont_block = f.blocks.remove(ci);
    let from = if ci < from { from - 1 } else { from };
    let b = &mut f.blocks[from];
    b.instructions.extend(cont_block.instructions);
    b.terminator = cont_block.terminator;
    if !subst.is_empty() {
        substitute(f, &subst);
    }
}

/// Replaces every use of the registers in `subst`, in every block of `f`.
fn substitute(f: &mut Function, subst: &HashMap<String, String>) {
    let fix = |r: &mut String| {
        if let Some(n) = subst.get(r.as_str()) {
            *r = n.clone();
        }
    };
    for b in &mut f.blocks {
        for i in &mut b.instructions {
            for r in i.operands_mut() {
                fix(r);
            }
        }
        match &mut b.terminator {
            Terminator::Jmp { args, .. } => args.iter_mut().for_each(fix),
            Terminator::Br {
                cond,
                then_args,
                else_args,
                ..
            } => {
                fix(cond);
                then_args.iter_mut().for_each(fix);
                else_args.iter_mut().for_each(fix);
            }
            Terminator::Ret { vals, .. } => vals.iter_mut().for_each(fix),
        }
    }
}

/// Renaming for one copy of a callee: its parameters become the call's arguments, every
/// other register and every label gets the copy's prefix.
struct Expansion<'a> {
    prefix: &'a str,
    args: HashMap<&'a str, &'a str>,
}

impl<'a> Expansion<'a> {
    fn new(prefix: &'a str, callee: &'a Function, args: &'a [String]) -> Self {
        let args = callee
            .params
            .iter()
            .map(|(p, _)| p.as_str())
            .zip(args.iter().map(String::as_str))
            .collect();
        Self { prefix, args }
    }

    fn reg(&self, name: &str) -> String {
        match self.args.get(name) {
            Some(arg) => (*arg).to_string(),
            None => format!("{}_{name}", self.prefix),
        }
    }

    fn label(&self, label: &str) -> String {
        format!("{}_{label}", self.prefix)
    }

    fn block(&self, b: &Block, cont: &str) -> Block {
        let instructions = b
            .instructions
            .iter()
            .map(|i| {
                let mut i = i.clone();
                for r in i.dsts_mut() {
                    *r = self.reg(r);
                }
                for r in i.operands_mut() {
                    *r = self.reg(r);
                }
                i
            })
            .collect();
        let regs = |v: &[String]| v.iter().map(|r| self.reg(r)).collect::<Vec<_>>();
        let terminator = match &b.terminator {
            Terminator::Jmp { target, args, span } => Terminator::Jmp {
                target: self.label(target),
                args: regs(args),
                span: *span,
            },
            Terminator::Br {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
                span,
            } => Terminator::Br {
                cond: self.reg(cond),
                then_block: self.label(then_block),
                then_args: regs(then_args),
                else_block: self.label(else_block),
                else_args: regs(else_args),
                span: *span,
            },
            Terminator::Ret { vals, span } => Terminator::Jmp {
                target: cont.to_string(),
                args: regs(vals),
                span: *span,
            },
        };
        Block {
            label: self.label(&b.label),
            params: b.params.iter().map(|(p, t)| (self.reg(p), *t)).collect(),
            instructions,
            terminator,
            span: b.span,
        }
    }
}
