//! Cross-loop reuse refers to optimizing the pattern of when a program accesses data items or
//! cache lines in one loop nest and then again in a later part of the program.
//!
//! Our approach is to identify loops that read the same view more than once, and mark the first
//! for caching.
//!
//! Two `wax.for` loops in one block, over the same bounds, both loading the same partition view.
//! The first pass can stage what it reads into thread-private registers so the second reads them
//! back instead of re-loading.
//!
//! We use a pliron [`Analysis`] rather than a pre-scan inside the lowering, for two reasons.
//! It is cached per operation by the `AnalysisManager`, so asking twice costs once. Additionally
//! the conclusion is recorded in the IR as an attribute, so that the lowering itself does not
//! decide anything, it simply reads the decision.
use crate::attr::Attribute;
use crate::opcode::Opcode;
use crate::source::WaxSource;
use crate::types::{ScalarType, Type};
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::pass::{Analysis, AnalysisManager};
use pliron::result::Result;

/// Set on a `wax.for` that should STAGE a view into registers; the value is the chunk
/// count, which is also the cache length.
pub const ATTR_SMEM_STAGE: &str = "tile_smem_stage";
/// Set on the `wax.for` that CONSUMES what the staging loop cached.
pub const ATTR_SMEM_CONSUME: &str = "tile_smem_consume";

/// Bytes per thread the staging would hold live, set alongside [`ATTR_SMEM_STAGE`].
///
/// We want `wax` passes to be applicable to backend neutral. As such passes should not define affordability.
/// For this pass the concept of affordability is the register budget.
/// We compute the cost. The backend applies its budget.
pub const ATTR_SMEM_BYTES: &str = "tile_smem_bytes";

/// A staging/consuming pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    pub staging: Ptr<Operation>,
    pub consuming: Ptr<Operation>,
    /// Bytes per thread the staged cache would occupy. See [`ATTR_SMEM_BYTES`].
    pub bytes_per_thread: i32,
    pub num_chunks: i32,
}

/// Every loop pair in the function worth caching.
pub struct CrossLoopReuse {
    pub plans: Vec<Plan>,
}

impl Analysis for CrossLoopReuse {
    fn name(&self) -> &str {
        "wax-cross-loop-reuse"
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, _analyses: &mut AnalysisManager) -> Result<Self> {
        Ok(CrossLoopReuse {
            plans: find_plans(ctx, op),
        })
    }
}

/// The plans in the entry block of the func.
pub fn find_plans(ctx: &Context, func: Ptr<Operation>) -> Vec<Plan> {
    let src = DialectView { ctx };
    let Some(block) = src
        .regions(func)
        .first()
        .and_then(|r| src.region_blocks(*r).first().copied())
    else {
        return Vec::new();
    };

    let fors: Vec<Ptr<Operation>> = src
        .block_ops(block)
        .into_iter()
        .filter(|&o| src.opcode(o) == Opcode::For)
        .collect();

    let mut plans = Vec::new();
    for a in 0..fors.len() {
        for b in (a + 1)..fors.len() {
            let (first, second) = (fors[a], fors[b]);
            let Some(num_chunks) = matching_bounds(&src, first, second) else {
                continue;
            };
            let read_by_first = views_read(&src, first);
            if read_by_first.is_empty() {
                continue;
            }
            let read_by_second = views_read(&src, second);
            let Some(shared) = read_by_first.iter().find(|v| read_by_second.contains(v)) else {
                continue;
            };
            let Some(elem_bytes) = view_elem_bytes(&src, *shared) else {
                continue;
            };
            plans.push(Plan {
                staging: first,
                consuming: second,
                num_chunks,
                bytes_per_thread: num_chunks.saturating_mul(elem_bytes),
            });
        }
    }
    plans
}

/// Record on the IR, so the lowering reads the decision directly.
///
/// Returns how many pairs were marked.
pub fn annotate(ctx: &Context, func: Ptr<Operation>) -> usize {
    let plans = find_plans(ctx, func);
    for p in &plans {
        set_int_attr(ctx, p.staging, ATTR_SMEM_STAGE, p.num_chunks as i64);
        set_int_attr(ctx, p.staging, ATTR_SMEM_BYTES, p.bytes_per_thread as i64);
        set_int_attr(ctx, p.consuming, ATTR_SMEM_CONSUME, 1);
    }
    plans.len()
}

/// Loop bounds are considered matching if both start at a constant 0 and share an upper bound.
/// The difference is the chunk count.
// TODO: Investigate if we can capture more patterns.
fn matching_bounds(
    src: &DialectView<'_>,
    first: Ptr<Operation>,
    second: Ptr<Operation>,
) -> Option<i32> {
    let (o1, o2) = (src.operands(first), src.operands(second));
    if o1.len() < 2 || o2.len() < 2 {
        return None;
    }
    // Lower bounds. Both const zero, or the same resolved (non-const) zero.
    match (const_i32(src, o1[0]), const_i32(src, o2[0])) {
        (Some(0), Some(0)) => {}
        // TODO: incorrect. Both this and the above arm are mutually exclusive.
        _ if o1[0] == o2[0] && const_i32(src, o1[0]) == Some(0) => {}
        _ => return None,
    }
    // Upper bounds. The same constant, or the same resolved value.
    let n = match (const_i32(src, o1[1]), const_i32(src, o2[1])) {
        (Some(a), Some(b)) if a == b => a,
        // TODO: incorrect. Both this and the above arm are mutually exclusive.
        (Some(a), None) | (None, Some(a)) if o1[1] == o2[1] => a,
        _ => return None,
    };
    (n > 0).then_some(n)
}

/// Partition views loaded inside `for_op`'s body.
fn views_read(src: &DialectView<'_>, for_op: Ptr<Operation>) -> Vec<pliron::value::Value> {
    let mut views = Vec::new();
    for r in src.regions(for_op) {
        for b in src.region_blocks(r) {
            for op in src.block_ops(b) {
                if src.opcode(op) != Opcode::LoadViewTko {
                    continue;
                }
                let opds = src.operands(op);
                let Some(&pv) = opds.first() else { continue };
                if !matches!(src.value_type(pv), Type::PartitionView(_)) {
                    continue;
                }
                views.push(pv);
            }
        }
    }
    views
}

/// Bytes held by the view of one thread.
fn view_elem_bytes(src: &DialectView<'_>, v: pliron::value::Value) -> Option<i32> {
    match src.value_type(v) {
        Type::PartitionView(pv) => Some(pv.tensor_view.element_type.byte_width() as i32),
        _ => None,
    }
}

/// Extract a constant i32 from the value, if it has one.
fn const_i32(src: &DialectView<'_>, v: pliron::value::Value) -> Option<i32> {
    use pliron::value::DefiningEntity;
    let DefiningEntity::Op(op) = v.defining_entity() else {
        return None;
    };
    if src.opcode(op) != Opcode::Constant {
        return None;
    }
    src.attributes(op).into_iter().find_map(|(k, a)| match a {
        Attribute::DenseElements(de)
            if k == "value"
                && de.shape.is_empty()                              // scalar
                && de.element_type == Type::Scalar(ScalarType::I32) // correct type
                && de.data.len() >= 4                               // (possibly redundant) size check
                =>
        {
            Some(i32::from_le_bytes(de.data[..4].try_into().unwrap()))
        }
        _ => None,
    })
}

/// Set an integer attribute on the operation.
fn set_int_attr(ctx: &Context, op: Ptr<Operation>, key: &str, v: i64) {
    use crate::dialect::attr_mirror::WaxAttrs;
    use crate::dialect::ops::ATTR_KEY_TILE_ATTRS;
    let mut attrs = op
        .deref(ctx)
        .attributes
        .get::<WaxAttrs>(&ATTR_KEY_TILE_ATTRS.try_into().unwrap())
        .map(|a| a.0.clone())
        .unwrap_or_default();
    attrs.retain(|(k, _)| k != key);
    attrs.push((
        key.to_string(),
        Attribute::Integer(v, Type::Scalar(ScalarType::I32)),
    ));
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_TILE_ATTRS.try_into().unwrap(), WaxAttrs(attrs));
}

/// Dialect read through [`WaxSource`] so the pass sees ops the same way lowering does.
struct DialectView<'a> {
    ctx: &'a Context,
}

impl WaxSource for DialectView<'_> {
    type Op = Ptr<Operation>;
    type Val = pliron::value::Value;
    type Block = Ptr<pliron::basic_block::BasicBlock>;
    type Region = Ptr<pliron::region::Region>;

    fn opcode(&self, op: Self::Op) -> Opcode {
        crate::dialect::source::opcode_of(self.ctx, op)
    }
    fn operands(&self, op: Self::Op) -> Vec<Self::Val> {
        op.deref(self.ctx).operands().collect()
    }
    fn result_types(&self, op: Self::Op) -> Vec<Type> {
        use pliron::r#type::Typed;
        let o = op.deref(self.ctx);
        (0..o.get_num_results())
            .filter_map(|i| {
                crate::dialect::types::from_pliron(self.ctx, o.get_result(i).get_type(self.ctx))
            })
            .collect()
    }
    fn attributes(&self, op: Self::Op) -> Vec<(String, Attribute)> {
        use crate::dialect::attr_mirror::WaxAttrs;
        use crate::dialect::ops::ATTR_KEY_TILE_ATTRS;
        op.deref(self.ctx)
            .attributes
            .get::<WaxAttrs>(&ATTR_KEY_TILE_ATTRS.try_into().unwrap())
            .map(|a| a.0.clone())
            .unwrap_or_default()
    }
    fn regions(&self, op: Self::Op) -> Vec<Self::Region> {
        op.deref(self.ctx).regions().collect()
    }
    fn op_result(&self, op: Self::Op, i: u32) -> Option<Self::Val> {
        let o = op.deref(self.ctx);
        ((i as usize) < o.get_num_results()).then(|| o.get_result(i as usize))
    }
    fn value_type(&self, v: Self::Val) -> Type {
        use pliron::r#type::Typed;
        crate::dialect::types::from_pliron(self.ctx, v.get_type(self.ctx)).unwrap_or(Type::Token)
    }
    fn region_blocks(&self, r: Self::Region) -> Vec<Self::Block> {
        use pliron::linked_list::ContainsLinkedList;
        r.deref(self.ctx).iter(self.ctx).collect()
    }
    fn block_ops(&self, b: Self::Block) -> Vec<Self::Op> {
        use pliron::linked_list::ContainsLinkedList;
        b.deref(self.ctx).iter(self.ctx).collect()
    }
    fn block_args(&self, b: Self::Block) -> Vec<(Self::Val, Type)> {
        let blk = b.deref(self.ctx);
        blk.arguments().map(|v| (v, self.value_type(v))).collect()
    }
}
