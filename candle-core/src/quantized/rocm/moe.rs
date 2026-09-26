//! Indexed MoE: grouped quantized matmul for prefill, matvec for small batches.
//!
//! The weights are a single `(num_experts, n, k)` quantized tensor; `ids` picks
//! which expert each of the `batch * topk` routed tokens goes to.
//! For small batches, `indexed_moe_forward_*_q8_1` in
//! `candle-kernels/src/quantized.cu` uses the same
//! `vec_dot_q*_q8_1` inner loop [`super::mmvq`] uses, with the expert index
//! folded into the weight pointer, so it inherits MMVQ's `q8_1` activation
//! requantization wholesale (see [`super::q8_1`]).
//!
//! Q5K/Q6K prefill with at least eight routed columns per expert instead
//! groups pairs on the GPU and runs shared MMQ tiles with indirect input/output
//! columns. Its Q5 minimum correction retains the vector path's quantized sums.
//! Only expert counts/status cross to the host; weights/activations stay resident.
//!
//! Vector path mirrors `quantized/cuda.rs::indexed_moe_forward_fused_q8_1_input`.

use super::kernels::{arg, launch_err, MATRIX_ROW_PADDING, WARP_SIZE};
use super::q8_1::{buffer_bytes, pad, quantize_q8_1};
use super::QRocmStorage;
use crate::backend::BackendDevice;
use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{
    kernels, RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory,
};
use crate::{Layout, Result, Shape, Tensor};

/// Immutable GPU-packed token/expert assignments reusable across projections.
/// Captures the assignments at construction; later mutation of the source IDs
/// does not alter this routing. Weight stacks must have the same expert count
/// and device/stream. This object is local to one routing decision, not a model
/// state snapshot or a cache keyed by Tensor identity.
pub struct GroupedMoeRouting {
    pub(crate) ids: Tensor, // Retained for layout/device validation, never re-read for routing.
    packed: PackedRouting,
}

impl std::fmt::Debug for GroupedMoeRouting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedMoeRouting")
            .field("num_experts", &self.packed.num_experts)
            .field("batch", &self.packed.batch)
            .field("topk", &self.packed.topk)
            .finish_non_exhaustive()
    }
}

impl GroupedMoeRouting {
    pub fn new(ids: &Tensor, num_experts: usize) -> Result<Self> {
        let packed = match &*ids.storage() {
            crate::Storage::Rocm(storage) => {
                PackedRouting::new(storage, ids.layout(), num_experts)?
            }
            _ => crate::bail!("grouped MoE routing requires ROCm IDs"),
        };
        Ok(Self {
            ids: ids.clone(),
            packed,
        })
    }
}

struct PackedRouting {
    device: crate::rocm_backend::RocmDevice,
    num_experts: usize,
    batch: usize,
    topk: usize,
    max_count: usize,
    pairs: crate::rocm_backend::SendSyncDeviceMemory<u32>,
    counts: crate::rocm_backend::SendSyncDeviceMemory<u32>,
}

/// `nwarps` the kernel is written for. It is a compile-time constant inside
/// `indexed_moe_forward`, sizing the `tmp_shared[nwarps - 1][WARP_SIZE]`
/// inter-warp reduction buffer, so `blockDim.y` has to be exactly this.
const NWARPS: usize = 4;

// Only for buffers whose producers overwrite every element before any read.
// Test builds can seed these allocations with dirty bytes to verify that contract.
fn work_buffer<T>(dev: &RocmDevice, len: usize) -> Result<SendSyncDeviceMemory<T>> {
    let buffer = dev.alloc::<T>(len)?;
    #[cfg(test)]
    let buffer = tests::initialize_work_buffer(buffer)?;
    Ok(buffer)
}

/// Kernel entry point for `dtype`, or `None` when there is none.
///
/// Note the casing: these spell the K-quants `q4k`, not `q4_K` as the MMVQ
/// family does. Taken verbatim from `quantized.cu`.
fn kernel_name(dtype: GgmlDType) -> Option<&'static str> {
    let name = match dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        GgmlDType::Q4_0 => "indexed_moe_forward_q4_0_q8_1",
        GgmlDType::Q4_1 => "indexed_moe_forward_q4_1_q8_1",
        GgmlDType::Q5_0 => "indexed_moe_forward_q5_0_q8_1",
        GgmlDType::Q5_1 => "indexed_moe_forward_q5_1_q8_1",
        _ => return None,
    };
    Some(name)
}

/// The same kernels with task-major launch geometry; see `indexed_moe_forward`
/// in `quantized.cu`. Identical values, different DRAM traffic.
fn task_major_kernel_name(dtype: GgmlDType) -> Option<&'static str> {
    let name = match dtype {
        GgmlDType::Q2K => "indexed_moe_forward_task_major_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_task_major_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_task_major_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_task_major_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_task_major_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_task_major_q8_0_q8_1",
        GgmlDType::Q4_0 => "indexed_moe_forward_task_major_q4_0_q8_1",
        GgmlDType::Q4_1 => "indexed_moe_forward_task_major_q4_1_q8_1",
        GgmlDType::Q5_0 => "indexed_moe_forward_task_major_q5_0_q8_1",
        GgmlDType::Q5_1 => "indexed_moe_forward_task_major_q5_1_q8_1",
        _ => return None,
    };
    Some(name)
}

/// Task-major geometry from 16 tokens up (batched decode), where tokens'
/// routed pairs sharing expert rows through the cache pays: measured on
/// gfx1151 at the LFM2.5 shapes, 1.15x at 16 and 1.26-1.30x at 32, but a
/// 4-17% loss at 2-4 tokens, where there is little to share.
/// `CANDLE_ROCM_MOE_ROW_MAJOR=1` keeps the original geometry, for A/B runs.
const TASK_MAJOR_MIN_BATCH: usize = 16;

fn use_task_major(d: &Dims) -> bool {
    static ROW_MAJOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let row_major = *ROW_MAJOR.get_or_init(|| {
        std::env::var("CANDLE_ROCM_MOE_ROW_MAJOR").is_ok_and(|v| v != "0" && !v.is_empty())
    });
    d.batch >= TASK_MAJOR_MIN_BATCH && !row_major
}

/// The shapes the kernel launch is derived from, once validated.
struct Dims {
    num_experts: usize,
    n: usize,
    k: usize,
    batch: usize,
    /// `1` when every expert of a token shares one activation row, `topk` when
    /// each routed pair carries its own.
    input_dim1: usize,
    topk: usize,
}

fn dims(self_shape: &Shape, input_l: &Layout, ids_l: &Layout) -> Result<Dims> {
    let (num_experts, n, k) = self_shape.dims3()?;
    let (batch, input_dim1, input_k) = input_l.shape().dims3()?;
    let (ids_batch, topk) = ids_l.shape().dims2()?;
    if input_k != k {
        crate::bail!(
            "indexed_moe_forward: weights are {self_shape:?} but the input has k={input_k}"
        )
    }
    if ids_batch != batch {
        crate::bail!("indexed_moe_forward: input batch {batch} but ids batch {ids_batch}")
    }
    if input_dim1 != 1 && input_dim1 != topk {
        crate::bail!("indexed_moe_forward: input dim 1 is {input_dim1}, expected 1 or topk {topk}")
    }
    if num_experts == 0 || topk == 0 || batch == 0 || n == 0 || k == 0 {
        crate::bail!(
            "indexed_moe_forward: empty shape {self_shape:?} / {:?}",
            ids_l.shape()
        )
    }
    Ok(Dims {
        num_experts,
        n,
        k,
        batch,
        input_dim1,
        topk,
    })
}

/// `q` is `(num_experts, n, k)`, `input` is `(batch, topk or 1, k)` f32 and
/// `ids` is `(batch, topk)` u32. Returns `(batch, topk, n)` f32.
pub(super) fn forward(
    q: &QRocmStorage,
    self_shape: &Shape,
    input: &RocmStorage,
    input_l: &Layout,
    ids: &RocmStorage,
    ids_l: &Layout,
) -> Result<(RocmStorage, Shape)> {
    forward_impl(q, self_shape, input, input_l, ids, ids_l, None, None, None)
}

pub(super) fn forward_prepared(
    q: &QRocmStorage,
    self_shape: &Shape,
    input: &RocmStorage,
    input_l: &Layout,
    ids: &RocmStorage,
    ids_l: &Layout,
    routing: &GroupedMoeRouting,
) -> Result<(RocmStorage, Shape)> {
    forward_impl(
        q,
        self_shape,
        input,
        input_l,
        ids,
        ids_l,
        Some(true),
        Some(&routing.packed),
        None,
    )
}

pub(super) fn supports(q: &QRocmStorage, shape: &Shape, batch: usize, topk: usize) -> bool {
    let Ok((num_experts, n, k)) = shape.dims3() else {
        return false;
    };
    num_experts > 0
        && num_experts <= 65535
        && n > 0
        && topk > 0
        && batch
            .checked_mul(topk)
            .is_some_and(|pairs| pairs <= i32::MAX as usize)
        && use_grouped(
            q.dtype,
            &Dims {
                num_experts,
                n,
                k,
                batch,
                input_dim1: 1,
                topk,
            },
        )
}

#[allow(clippy::too_many_arguments)]
fn forward_impl(
    q: &QRocmStorage,
    self_shape: &Shape,
    input: &RocmStorage,
    input_l: &Layout,
    ids: &RocmStorage,
    ids_l: &Layout,
    grouped_override: Option<bool>,
    prepared: Option<&PackedRouting>,
    task_major_override: Option<bool>,
) -> Result<(RocmStorage, Shape)> {
    if !q.device.same_device(&input.device) || !q.device.same_device(&ids.device) {
        crate::bail!(
            "indexed_moe_forward: weights, input and ids must share the same ROCm device/stream"
        )
    }
    let name = match kernel_name(q.dtype) {
        Some(name) => name,
        None => crate::bail!(
            "indexed_moe_forward is not implemented for {:?} on ROCm; \
             it needs one of q2k, q3k, q4k, q5k, q6k or q8_0",
            q.dtype
        ),
    };
    let d = dims(self_shape, input_l, ids_l)?;
    if !d.k.is_multiple_of(q.dtype.block_size()) {
        crate::bail!(
            "indexed_moe_forward: k={} is not a multiple of the {:?} block size {}",
            d.k,
            q.dtype,
            q.dtype.block_size()
        )
    }
    // Both MMVQ and MMQ use signed 32-bit local indexing. Reject shapes
    // outside that contract before casting dimensions or allocating scratch.
    for factors in [
        [d.num_experts, d.n, d.k],
        [d.batch, d.topk, d.n],
        [d.batch, d.input_dim1, pad(d.k, MATRIX_ROW_PADDING)],
    ] {
        let count = factors.iter().try_fold(1usize, |n, x| n.checked_mul(*x));
        if !matches!(count, Some(n) if n <= i32::MAX as usize) {
            crate::bail!("indexed_moe_forward: shape exceeds 32-bit kernel indexing")
        }
    }
    // The kernel strides between experts by `n * k / block_size` blocks with no
    // bound of its own, so a short payload would read past the allocation.
    let data_elems = q.len / q.dtype.type_size() * q.dtype.block_size();
    if data_elems < d.num_experts * d.n * d.k {
        crate::bail!(
            "indexed_moe_forward: weights hold {data_elems} elems, need {}",
            d.num_experts * d.n * d.k
        )
    }

    let (y, y_offset) = match (&input.slice, input_l.contiguous_offsets()) {
        (RocmStorageSlice::F32(y), Some((o1, o2))) if o2 - o1 == d.batch * d.input_dim1 * d.k => {
            (y, o1)
        }
        (RocmStorageSlice::F32(_), _) => {
            crate::bail!("indexed_moe_forward expects a contiguous input, got {input_l:?}")
        }
        (slice, _) => crate::bail!(
            "indexed_moe_forward expects an f32 input, got {:?}",
            slice.dtype()
        ),
    };
    let (ids_mem, ids_offset) = match (&ids.slice, ids_l.contiguous_offsets()) {
        (RocmStorageSlice::U32(m), Some((o1, o2))) if o2 - o1 == d.batch * d.topk => (m, o1),
        (RocmStorageSlice::U32(_), _) => {
            crate::bail!("indexed_moe_forward expects contiguous u32 ids, got {ids_l:?}")
        }
        (slice, _) => crate::bail!(
            "indexed_moe_forward expects u32 ids, got {:?}",
            slice.dtype()
        ),
    };

    let dev = &q.device;
    let total_rows = d.batch * d.input_dim1;
    let k_padded = pad(d.k, MATRIX_ROW_PADDING);
    // quantize_q8_1 writes all qs and both half headers in every 32-value
    // block, including the zero-padded tail of each 512-column row stride.
    let input_q8_1 = work_buffer::<u8>(dev, buffer_bytes(d.k, total_rows))?;
    quantize_q8_1(y, y_offset, &input_q8_1, d.k, total_rows, dev)?;

    // Vector kernels assign one output per (row, token, slot); grouped kernels
    // assign every valid routed pair's rows, including partial tiles. Neither
    // accumulates into out. Launch/validation errors return before exposing it.
    let out = work_buffer::<f32>(dev, d.batch * d.topk * d.n)?;
    let grouped = grouped_override.unwrap_or_else(|| use_grouped(q.dtype, &d));
    if grouped {
        let owned;
        let routing = match prepared {
            Some(p) => p,
            None => {
                owned = PackedRouting::new(ids, ids_l, d.num_experts)?;
                &owned
            }
        };
        grouped_forward(q, &d, &input_q8_1, routing, &out)?;
        return Ok((
            RocmStorage {
                slice: RocmStorageSlice::F32(out),
                device: dev.clone(),
            },
            (d.batch, d.topk, d.n).into(),
        ));
    }
    let task_major = task_major_override.unwrap_or_else(|| use_task_major(&d));
    let name = if task_major {
        task_major_kernel_name(q.dtype).expect("every indexed dtype has a task-major entry")
    } else {
        name
    };
    let func = dev.get_or_load_func(name, &kernels::QUANTIZED)?;

    let w_ptr = q.data.as_ptr();
    let y_ptr = input_q8_1.as_ptr();
    // SAFETY: `ids_offset` is within the buffer — `contiguous_offsets` returned
    // it against this layout and the u32 storage backing it.
    let ids_ptr = unsafe { ids_mem.ptr_at(ids_offset) };
    let out_ptr = out.as_ptr();
    let n_i = d.n as i32;
    let k_i = d.k as i32;
    let batch_i = d.batch as i32;
    let topk_i = d.topk as i32;
    let k_padded_i = k_padded as i32;
    let input_dim1_i = d.input_dim1 as i32;
    let mut args = vec![
        arg(&w_ptr),
        arg(&y_ptr),
        arg(&ids_ptr),
        arg(&out_ptr),
        arg(&n_i),
        arg(&k_i),
        arg(&batch_i),
        arg(&topk_i),
        arg(&k_padded_i),
        arg(&input_dim1_i),
    ];
    // One block per (output row, batch, routed expert). Row-major: `blockIdx.x`
    // is the row and the kernel flattens `(blockIdx.y, blockIdx.z)` into the
    // task id it indexes `ids` with. Task-major: `blockIdx.x` is the task id and
    // `blockIdx.y` the row, so co-scheduled blocks share weight rows.
    let grid = if task_major {
        Dim3::new_2d((d.batch * d.topk) as u32, d.n as u32)
    } else {
        Dim3::new_3d(d.n as u32, d.batch as u32, d.topk as u32)
    };
    func.launch(
        grid,
        Dim3::new_2d(WARP_SIZE as u32, NWARPS as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(name, e))?;

    Ok((
        RocmStorage {
            slice: RocmStorageSlice::F32(out),
            device: dev.clone(),
        },
        (d.batch, d.topk, d.n).into(),
    ))
}

// Keep decode and short/sparse suffixes on the vector path. Eight routed
// columns per expert is a conservative starting point; this is a dispatch
// rule, never a retry after a grouped-kernel failure.
fn use_grouped(dtype: GgmlDType, d: &Dims) -> bool {
    matches!(dtype, GgmlDType::Q5K | GgmlDType::Q6K)
        && d.batch > 1
        && d.batch * d.topk / d.num_experts >= 8
        && super::mmq::supports(dtype, d.k)
}

impl PackedRouting {
    fn new(ids: &RocmStorage, ids_l: &Layout, num_experts: usize) -> Result<Self> {
        let (batch, topk) = ids_l.shape().dims2()?;
        let total_pairs = batch
            .checked_mul(topk)
            .filter(|&n| n > 0 && n <= i32::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("invalid grouped MoE route dimensions".into()))?;
        if num_experts == 0 || num_experts > 65535 {
            crate::bail!("invalid grouped MoE expert count")
        }
        let (ids_mem, ids_offset) = match (&ids.slice, ids_l.contiguous_offsets()) {
            (RocmStorageSlice::U32(m), Some((start, end))) if end - start == total_pairs => {
                (m, start)
            }
            _ => crate::bail!("grouped MoE routing requires contiguous u32 IDs"),
        };
        let dev = &ids.device;
        let scratch_len = num_experts
            .checked_mul(total_pairs)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("grouped MoE routing scratch overflow".into()))?;
        let pairs = dev.alloc_zeros::<u32>(scratch_len)?;
        let counts = dev.alloc_zeros::<u32>(num_experts + 1)?;
        // Input/ID offsets have been checked against their contiguous layouts.
        let ids_ptr = unsafe { ids_mem.ptr_at(ids_offset) };
        let pairs_ptr = pairs.as_ptr();
        let counts_ptr = counts.as_ptr();
        let total_i = total_pairs as i32;
        let experts_i = num_experts as i32;
        let mut args = vec![
            arg(&ids_ptr),
            arg(&pairs_ptr),
            arg(&counts_ptr),
            arg(&total_i),
            arg(&experts_i),
        ];
        let func = dev.get_or_load_func("moe_group_pairs", &kernels::QUANTIZED)?;
        func.launch(
            Dim3::new_1d(num_experts as u32),
            Dim3::new_1d(256),
            0,
            Some(dev.stream()),
            &mut args,
        )
        .map_err(|e| launch_err("moe_group_pairs", e))?;
        // A small metadata readback (33 u32 for LFM2.5), not the routing tensor.
        // clone_dtoh synchronizes the owning stream. This makes invalid IDs an
        // explicit error before any weight reads and bounds the column grid to
        // the busiest expert, including duplicate routes within the same token.
        let host_counts = dev.clone_dtoh(&counts)?;
        if host_counts[num_experts] != 0 {
            crate::bail!("grouped MoE: expert id is outside the weight stack")
        }
        let accounted: u64 = host_counts[..num_experts].iter().map(|n| *n as u64).sum();
        if accounted != total_pairs as u64 {
            crate::bail!("grouped MoE routing counts do not account for every pair")
        }
        let max_count = *host_counts[..num_experts].iter().max().unwrap() as usize;
        if max_count == 0 {
            crate::bail!("grouped MoE produced no routed pairs")
        }
        Ok(Self {
            device: dev.clone(),
            num_experts,
            batch,
            topk,
            max_count,
            pairs,
            counts,
        })
    }
}

fn grouped_forward(
    q: &QRocmStorage,
    d: &Dims,
    input_q8: &crate::rocm_backend::SendSyncDeviceMemory<u8>,
    routing: &PackedRouting,
    out: &crate::rocm_backend::SendSyncDeviceMemory<f32>,
) -> Result<()> {
    let name = match q.dtype {
        GgmlDType::Q5K => "grouped_mul_mat_q5_K",
        GgmlDType::Q6K => "grouped_mul_mat_q6_K",
        _ => crate::bail!("grouped MoE requires Q5K or Q6K weights"),
    };
    if !super::mmq::supports(q.dtype, d.k) || d.num_experts > 65535 {
        crate::bail!("unsupported grouped MoE dimensions")
    }
    let dev = &q.device;
    let plan = super::mmq::plan(q.dtype, dev.mmq_tiles())
        .ok_or_else(|| crate::Error::Msg("missing grouped MoE tile geometry".into()))?;
    if routing.num_experts != d.num_experts
        || routing.batch != d.batch
        || routing.topk != d.topk
        || !routing.device.same_device(dev)
    {
        crate::bail!(
            "grouped MoE routing does not match weight experts, input shape or device/stream"
        )
    }
    let total_pairs = d.batch * d.topk;
    let total_i = total_pairs as i32;
    let max_count = routing.max_count;
    if max_count.div_ceil(plan.mmq_x) > 65535 {
        crate::bail!("grouped MoE column grid exceeds HIP limit")
    }
    let pairs_ptr = routing.pairs.as_ptr();
    let counts_ptr = routing.counts.as_ptr();
    let w_ptr = q.data.as_ptr();
    let y_ptr = input_q8.as_ptr();
    let out_ptr = out.as_ptr();
    let n_i = d.n as i32;
    let k_i = d.k as i32;
    let k_padded_i = pad(d.k, MATRIX_ROW_PADDING) as i32;
    let topk_i = d.topk as i32;
    let input_dim1_i = d.input_dim1 as i32;
    let mut args = vec![
        arg(&w_ptr),
        arg(&y_ptr),
        arg(&out_ptr),
        arg(&pairs_ptr),
        arg(&counts_ptr),
        arg(&n_i),
        arg(&k_i),
        arg(&k_padded_i),
        arg(&total_i),
        arg(&topk_i),
        arg(&input_dim1_i),
    ];
    let func = dev.get_or_load_func(name, &kernels::QUANTIZED)?;
    func.launch(
        Dim3::new_3d(
            d.n.div_ceil(plan.mmq_y) as u32,
            max_count.div_ceil(plan.mmq_x) as u32,
            d.num_experts as u32,
        ),
        Dim3::new_2d(WARP_SIZE as u32, plan.nwarps as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(name, e))?;
    Ok(())
}

#[cfg(test)]
fn forward_for_test(
    w: &crate::quantized::QTensor,
    x: &crate::Tensor,
    ids: &crate::Tensor,
    grouped: bool,
) -> Result<crate::Tensor> {
    match (&w.storage, &*x.storage(), &*ids.storage()) {
        (
            crate::quantized::QStorage::Rocm(q),
            crate::Storage::Rocm(x_s),
            crate::Storage::Rocm(ids_s),
        ) => {
            let (out, shape) = forward_impl(
                q,
                w.shape(),
                x_s,
                x.layout(),
                ids_s,
                ids.layout(),
                Some(grouped),
                None,
                None,
            )?;
            Ok(crate::tensor::from_storage(
                crate::Storage::Rocm(out),
                shape,
                crate::op::BackpropOp::none(),
                false,
            ))
        }
        _ => crate::bail!("test expects ROCm weights and tensors"),
    }
}

/// The vector path with an explicit launch geometry.
#[cfg(test)]
fn forward_vector_for_test(
    w: &crate::quantized::QTensor,
    x: &crate::Tensor,
    ids: &crate::Tensor,
    task_major: bool,
) -> Result<crate::Tensor> {
    match (&w.storage, &*x.storage(), &*ids.storage()) {
        (
            crate::quantized::QStorage::Rocm(q),
            crate::Storage::Rocm(x_s),
            crate::Storage::Rocm(ids_s),
        ) => {
            let (out, shape) = forward_impl(
                q,
                w.shape(),
                x_s,
                x.layout(),
                ids_s,
                ids.layout(),
                Some(false),
                None,
                Some(task_major),
            )?;
            Ok(crate::tensor::from_storage(
                crate::Storage::Rocm(out),
                shape,
                crate::op::BackpropOp::none(),
                false,
            ))
        }
        _ => crate::bail!("test expects ROCm weights and tensors"),
    }
}

#[cfg(test)]
mod tests;
