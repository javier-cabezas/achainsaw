//! Control-flow graph analysis for AIR functions: reachability, reverse post-order,
//! immediate dominators (Cooper-Harvey-Kennedy) and DFS retreating edges.

use crate::ast::{Function, Terminator};
use std::collections::{HashMap, HashSet};

pub const UNDEF: usize = usize::MAX;

#[derive(Debug, Clone)]
pub struct Cfg {
    /// Successor block indices per block (edges to unknown labels are ignored).
    pub succs: Vec<Vec<usize>>,
    /// Predecessor block indices per block.
    pub preds: Vec<Vec<usize>>,
    /// Reverse post-order over blocks reachable from the entry block (index 0).
    pub rpo: Vec<usize>,
    pub reachable: Vec<bool>,
    /// Immediate dominator per block. `idom[0] == 0`; `UNDEF` for unreachable blocks.
    pub idom: Vec<usize>,
    /// Edges `(from, to)` found by DFS that point at a block still on the DFS stack.
    /// Every cycle in the CFG contains at least one of them, so metering these edges
    /// bounds the running time of any loop.
    pub retreating: HashSet<(usize, usize)>,
}

impl Cfg {
    pub fn build(func: &Function) -> Self {
        let n = func.blocks.len();
        let index: HashMap<&str, usize> = func
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.label.as_str(), i))
            .collect();

        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, block) in func.blocks.iter().enumerate() {
            let targets: Vec<&str> = match &block.terminator {
                Terminator::Jmp { target, .. } => vec![target.as_str()],
                Terminator::Br {
                    then_block,
                    else_block,
                    ..
                } => vec![then_block.as_str(), else_block.as_str()],
                Terminator::Ret { .. } => vec![],
            };
            for t in targets {
                if let Some(&j) = index.get(t) {
                    succs[i].push(j);
                    preds[j].push(i);
                }
            }
        }

        // Iterative DFS computing post-order and retreating edges.
        let mut state = vec![0u8; n]; // 0 = unvisited, 1 = on stack, 2 = done
        let mut post: Vec<usize> = Vec::new();
        let mut retreating = HashSet::new();
        if n > 0 {
            let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
            state[0] = 1;
            while let Some(top) = stack.last_mut() {
                let node = top.0;
                if top.1 < succs[node].len() {
                    let succ = succs[node][top.1];
                    top.1 += 1;
                    match state[succ] {
                        0 => {
                            state[succ] = 1;
                            stack.push((succ, 0));
                        }
                        1 => {
                            retreating.insert((node, succ));
                        }
                        _ => {}
                    }
                } else {
                    state[node] = 2;
                    post.push(node);
                    stack.pop();
                }
            }
        }

        let rpo: Vec<usize> = post.iter().rev().copied().collect();
        let mut reachable = vec![false; n];
        let mut rpo_index = vec![UNDEF; n];
        for (pos, &b) in rpo.iter().enumerate() {
            reachable[b] = true;
            rpo_index[b] = pos;
        }

        // Cooper-Harvey-Kennedy iterative dominators.
        let mut idom = vec![UNDEF; n];
        if n > 0 {
            idom[0] = 0;
            let mut changed = true;
            while changed {
                changed = false;
                for &b in rpo.iter().skip(1) {
                    let mut new_idom = UNDEF;
                    for &p in &preds[b] {
                        if idom[p] == UNDEF {
                            continue;
                        }
                        new_idom = if new_idom == UNDEF {
                            p
                        } else {
                            let (mut a, mut c) = (p, new_idom);
                            while a != c {
                                while rpo_index[a] > rpo_index[c] {
                                    a = idom[a];
                                }
                                while rpo_index[c] > rpo_index[a] {
                                    c = idom[c];
                                }
                            }
                            a
                        };
                    }
                    if idom[b] != new_idom {
                        idom[b] = new_idom;
                        changed = true;
                    }
                }
            }
        }

        Self {
            succs,
            preds,
            rpo,
            reachable,
            idom,
            retreating,
        }
    }

    /// True if block `a` dominates block `b` (both must be reachable).
    pub fn dominates(&self, a: usize, mut b: usize) -> bool {
        if !self.reachable[a] || !self.reachable[b] {
            return false;
        }
        loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = self.idom[b];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn parse(src: &str) -> Function {
        let mut p = Parser::new(src).unwrap();
        p.parse_module().unwrap().functions.remove(0)
    }

    #[test]
    fn diamond_dominators_and_loop_edges() {
        let f = parse(
            r#"
fn f(c:i32)->i32
  b0:
    jmp b1
  b1:
    br c, b2, b3
  b2:
    jmp b4
  b3:
    jmp b4
  b4:
    br c, b1, b5
  b5:
    ret c
"#,
        );
        let cfg = Cfg::build(&f);
        assert_eq!(cfg.idom[4], 1); // b4's idom is b1 (join of the diamond)
        assert_eq!(cfg.idom[5], 4);
        assert!(cfg.dominates(1, 5));
        assert!(!cfg.dominates(2, 4));
        assert!(cfg.retreating.contains(&(4, 1)));
        assert_eq!(cfg.retreating.len(), 1);
    }

    #[test]
    fn unreachable_blocks_are_flagged() {
        let f = parse("fn f()\n  b0:\n    ret\n  b1:\n    ret\n");
        let cfg = Cfg::build(&f);
        assert!(cfg.reachable[0]);
        assert!(!cfg.reachable[1]);
        assert_eq!(cfg.idom[1], UNDEF);
    }
}
