//! Function handles (`fnref`) and indirect calls (`call_indirect`).
//!
//! A module's handles index its functions in order: `fnref f` is 1 + `f`'s index, so 0 is
//! never a handle. Each function's entry in the table both backends build holds its address
//! and a signature id (its parameter and result types, numbered per module); a call checks
//! the handle's range and the id against its own signature's before calling, and otherwise
//! calls nothing, gives zeros and reports `ERR_BAD_INDIRECT_CALL` (AOT objects, which have
//! no runtime, trap instead).

use crate::ast::{Instruction, Module};
use crate::types::Type;
use std::collections::HashMap;

/// A function's parameter and result types.
pub type Signature = (Vec<Type>, Vec<Type>);

/// The handle table of a module: its functions in handle order and their signature ids.
#[derive(Debug, Clone, Default)]
pub struct HandleTable {
    /// (function, signature id), entry `h - 1` for handle `h`.
    pub entries: Vec<(String, u32)>,
    ids: HashMap<Signature, u32>,
}

impl HandleTable {
    pub fn new(module: &Module) -> Self {
        let mut ids: HashMap<Signature, u32> = HashMap::new();
        let entries = module
            .functions
            .iter()
            .map(|f| {
                let sig = (f.params.iter().map(|(_, t)| *t).collect(), f.rets.clone());
                let next = ids.len() as u32;
                (f.name.clone(), *ids.entry(sig).or_insert(next))
            })
            .collect();
        Self { entries, ids }
    }

    /// The handle of function `name`.
    pub fn handle(&self, name: &str) -> Option<i64> {
        self.entries
            .iter()
            .position(|(f, _)| f == name)
            .map(|i| i as i64 + 1)
    }

    /// The id a call with these types checks for: no function's id when none has them.
    pub fn signature_id(&self, params: &[Type], rets: &[Type]) -> u32 {
        self.ids
            .get(&(params.to_vec(), rets.to_vec()))
            .copied()
            .unwrap_or(u32::MAX)
    }
}

/// Whether the module takes handles or calls through them (it then needs its table).
pub fn uses_handles(module: &Module) -> bool {
    module.functions.iter().any(|f| {
        f.blocks.iter().flat_map(|b| &b.instructions).any(|i| {
            matches!(
                i,
                Instruction::FnRef { .. } | Instruction::CallIndirect { .. }
            )
        })
    })
}

/// Replaces each `call_indirect` whose handle comes from a `fnref` in the same function, with
/// a matching signature, by a direct `call` (which inlining and the backends then treat as
/// any call). Needs unique register names. Returns how many calls it replaced.
pub fn devirtualize(module: &mut Module) -> usize {
    let sigs: HashMap<String, Signature> = module
        .functions
        .iter()
        .map(|f| {
            (
                f.name.clone(),
                (f.params.iter().map(|(_, t)| *t).collect(), f.rets.clone()),
            )
        })
        .collect();
    let rets: crate::opt::Signatures = sigs
        .iter()
        .map(|(n, (_, r))| (n.clone(), r.clone()))
        .collect();
    let mut replaced = 0;
    for f in &mut module.functions {
        let refs: HashMap<String, String> = f
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| match i {
                Instruction::FnRef { dst, func, .. } => Some((dst.clone(), func.clone())),
                _ => None,
            })
            .collect();
        if refs.is_empty() {
            continue;
        }
        let types = crate::opt::infer_reg_types(f, &rets);
        for b in &mut f.blocks {
            for inst in &mut b.instructions {
                let Instruction::CallIndirect {
                    dsts,
                    handle,
                    args,
                    rets: want,
                    span,
                } = inst
                else {
                    continue;
                };
                let Some(target) = refs.get(handle.as_str()) else {
                    continue;
                };
                let (params, have) = &sigs[target];
                let arg_types: Option<Vec<Type>> =
                    args.iter().map(|a| types.get(a).copied()).collect();
                if arg_types.as_ref() != Some(params) || have != want {
                    continue;
                }
                *inst = Instruction::Call {
                    dsts: std::mem::take(dsts),
                    func: target.clone(),
                    args: std::mem::take(args),
                    span: *span,
                };
                replaced += 1;
            }
        }
    }
    replaced
}
