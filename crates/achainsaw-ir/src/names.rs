//! Shadowed register names. AIR scopes a register by dominance: a definition is visible in
//! the code it dominates, and a later definition of the same name, later in the block or in a
//! block it dominates, shadows it there (`loop(i:i64)` and `body(i:i64)` are two registers).
//! The validator checks that every use can see only one definition (`ERR_AMBIGUOUS_REGISTER`
//! otherwise). [`unique_names`] then renames the shadowing definitions apart, so the inliner,
//! the optimizer and code generation see each name defined once. Both inlining entry points
//! call it, so every path to code generation does.

use crate::ast::*;
use crate::cfg::Cfg;
use std::collections::{HashMap, HashSet};

/// True if some register of `f` is defined more than once.
pub fn has_shadowing(f: &Function) -> bool {
    let mut seen: HashSet<&str> = HashSet::new();
    let params = f.params.iter().map(|(p, _)| p.as_str());
    let block_params = f
        .blocks
        .iter()
        .flat_map(|b| b.params.iter().map(|(p, _)| p.as_str()));
    let dsts = f
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter().flat_map(|i| i.dsts()));
    let mut all = params.chain(block_params).chain(dsts);
    all.any(|name| !seen.insert(name))
}

/// Renames every shadowing definition in `module` (and its uses) to a fresh name; returns
/// whether anything changed. The module must be valid.
pub fn unique_names(module: &mut Module) -> bool {
    let mut changed = false;
    for f in &mut module.functions {
        changed |= unique_function_names(f);
    }
    changed
}

/// [`unique_names`] for one function.
pub fn unique_function_names(f: &mut Function) -> bool {
    if !has_shadowing(f) {
        return false;
    }
    rename_function(f);
    true
}

fn rename_function(f: &mut Function) {
    let cfg = Cfg::build(f);
    let mut taken: HashSet<String> = f.params.iter().map(|(p, _)| p.clone()).collect();
    for b in &f.blocks {
        taken.insert(b.label.clone());
        taken.extend(b.params.iter().map(|(p, _)| p.clone()));
        for i in &b.instructions {
            taken.extend(i.dsts().into_iter().map(String::from));
        }
    }
    // Names that already belong to a definition: a second definition gets a fresh name.
    let mut claimed: HashSet<String> = f.params.iter().map(|(p, _)| p.clone()).collect();
    let base: HashMap<String, String> = f
        .params
        .iter()
        .map(|(p, _)| (p.clone(), p.clone()))
        .collect();
    let mut define = |name: &mut String, scope: &mut HashMap<String, String>| {
        let original = name.clone();
        if !claimed.insert(original.clone()) {
            let mut k = 1;
            *name = loop {
                let candidate = format!("{original}_{k}");
                if taken.insert(candidate.clone()) {
                    break candidate;
                }
                k += 1;
            };
            claimed.insert(name.clone());
        }
        scope.insert(original, name.clone());
    };

    let mut order: Vec<usize> = cfg.rpo.clone();
    order.extend((0..f.blocks.len()).filter(|&i| !cfg.reachable[i]));
    let mut final_scopes: Vec<Option<HashMap<String, String>>> = vec![None; f.blocks.len()];
    for b_idx in order {
        let mut scope = if b_idx != 0 && cfg.reachable[b_idx] {
            final_scopes[cfg.idom[b_idx]]
                .clone()
                .unwrap_or_else(|| base.clone())
        } else {
            base.clone()
        };
        let block = &mut f.blocks[b_idx];
        for (p, _) in &mut block.params {
            define(p, &mut scope);
        }
        for inst in &mut block.instructions {
            for r in inst.operands_mut() {
                resolve(r, &scope);
            }
            for d in inst.dsts_mut() {
                define(d, &mut scope);
            }
        }
        for r in terminator_operands_mut(&mut block.terminator) {
            resolve(r, &scope);
        }
        final_scopes[b_idx] = Some(scope);
    }
}

fn resolve(r: &mut String, scope: &HashMap<String, String>) {
    if let Some(name) = scope.get(r.as_str()) {
        r.clone_from(name);
    }
}

/// The registers a terminator reads.
pub(crate) fn terminator_operands_mut(t: &mut Terminator) -> Vec<&mut String> {
    t.operands_mut()
}
