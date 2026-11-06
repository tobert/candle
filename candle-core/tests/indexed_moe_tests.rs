// Indexed MoE mat-vec parity: `QTensor::indexed_moe_forward` (the SYCL fused
// integer kernel on that backend) must agree with a CPU reference computed on
// the dequantized routed rows. Allocation is a few MB (4 experts x 32 x 256),
// safe beside a resident GPU workload. The CPU arm is the reference; only the
// SYCL arm is new.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};

// `#[macro_export]` places `test_device!` at the crate root.
use candle_core::test_device;

/// Live bytes on a SYCL device's queue; `None` on other backends.
fn live_bytes(device: &Device) -> Option<usize> {
    #[cfg(feature = "sycl")]
    if device.is_sycl() {
        let Device::Sycl(d) = device else {
            unreachable!()
        };
        return Some(d.queue().live_bytes());
    }
    let _ = device;
    None
}

fn reference(
    w_stack: &Tensor, // (num_experts, n, k) dequantized
    input: &Tensor,   // (batch, k) or (batch, topk, k)
    ids: &Tensor,     // (batch, topk) u32
) -> Result<Tensor> {
    let (batch, topk) = ids.dims2()?;
    let k = *input.dims().last().unwrap();
    // (batch, 1, k) broadcasts to (batch, topk, k); a rank-3 input already has
    // one row per task.
    let inp = if input.dims().len() == 2 {
        input
            .unsqueeze(1)?
            .broadcast_as((batch, topk, k))?
            .contiguous()?
    } else {
        input.clone()
    };
    let flat = ids.flatten_all()?;
    let routed = w_stack.index_select(&flat, 0)?; // (b*t, n, k)
    let n = routed.dims()[1];
    routed
        .matmul(&inp.reshape((batch * topk, k, 1))?)?
        .reshape((batch, topk, n))
}

fn indexed_moe_q8_0(dev: &Device) -> Result<()> {
    run_case(dev, GgmlDType::Q8_0, 3, 2, false)?;
    // The dense fallback is a different path (gather + GEMM, no kernel), and
    // batch > 1 with a broadcast activation is where the row expansion lives.
    // CUDA and ROCm reject F32/F16 weight stacks, so those cases run where
    // the path exists (CPU, Metal and SYCL).
    if !(dev.is_cuda() || dev.is_rocm()) {
        run_case(dev, GgmlDType::F32, 3, 2, false)?;
        run_case(dev, GgmlDType::F16, 2, 1, false)?;
    }
    // 128 tasks >= the grouped-prefill threshold (64): exercises the
    // dequantize-once + gather + per-expert GEMM + scatter path on SYCL.
    // Q5K at batch 3 first: the integer mat-vec's own quantization-error
    // baseline for this dtype, so the grouped number can be compared against
    // the path that shares its weights.
    run_case(dev, GgmlDType::Q5K, 3, 2, false)?;
    run_case(dev, GgmlDType::Q5K, 32, 4, false)?;
    run_case(dev, GgmlDType::Q6K, 32, 4, false)?;
    run_case(dev, GgmlDType::Q8_0, 32, 4, false)?;
    // Rank-3 activation: one row per (batch, topk) task, the down-projection
    // shape. The rank-2 cases above broadcast one row across a task's experts.
    run_case(dev, GgmlDType::Q5K, 3, 2, true)?;
    run_case(dev, GgmlDType::Q5K, 32, 4, true)?;
    bad_expert_id_is_refused(dev)?;
    bad_expert_id_survives_a_pool_flush(dev)?;
    bad_expert_id_surfaces_at_the_host_read(dev)?;
    grouped_call_reuses_the_dequant_cache(dev)?;
    Ok(())
}

fn run_case(
    dev: &Device,
    dtype: GgmlDType,
    batch: usize,
    topk: usize,
    per_task: bool,
) -> Result<()> {
    let cpu = &Device::Cpu;
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);

    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let input = if per_task {
        Tensor::rand(-1f32, 1f32, (batch, topk, k), cpu)?
    } else {
        Tensor::rand(-1f32, 1f32, (batch, k), cpu)?
    };
    let mut ids_v = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for t in 0..topk {
            ids_v.push(((b + t * 2) % num_experts) as u32);
        }
    }
    let ids = Tensor::from_vec(ids_v, (batch, topk), cpu)?;

    let expected = reference(&w, &input, &ids)?;

    // Quantize on CPU, then rebuild the QTensor on the device under test from
    // the same raw block bytes (`QTensor` has no to_device).
    let w_q_cpu = QTensor::quantize(&w, dtype)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?,
        shape,
    )?;

    // A broadcast activation is rank 2 only on SYCL; CUDA and ROCm take it as
    // (batch, 1, k), the shape their kernels' `dims` require.
    let input_dev = if per_task || dev.is_sycl() {
        input.to_device(dev)?
    } else {
        input.unsqueeze(1)?.to_device(dev)?
    };
    let got = w_q.indexed_moe_forward(&input_dev, &ids.to_device(dev)?);

    let got = match got {
        Ok(g) => g,
        // CPU and Metal have no indexed-MoE path: the CPU reference above is
        // the test. CUDA, ROCm and SYCL implement it, so an error there is a
        // failure, never a skip.
        Err(_) if dev.is_cpu() || dev.is_metal() => return Ok(()),
        Err(e) => return Err(e),
    };

    let got_v = got.to_device(cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let exp_v = expected.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = got_v
        .iter()
        .zip(exp_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Quantized dtypes carry block-scale error on a k-term dot against the
    // f32 reference — measured 0.44 at k=256 for Q5_K (the mat-vec path,
    // whose integer dots are exact against the dequantized blocks; the error
    // is the quantization itself, not the path). The grouped prefill adds
    // f16 rounding of both operands on top (0.5–0.70 observed on random
    // draws at k=256), so the quantized tolerance is 1.0: above that tail and
    // far below a wrong-expert error (O(n)). Dense dtypes are exact to f32
    // rounding.
    let tol = if dtype == GgmlDType::F32 { 1e-3 } else { 1.0 };
    assert!(
        max_diff < tol,
        "indexed_moe_forward {dtype:?} batch={batch} topk={topk} per_task={per_task} \
         mismatch on {dev:?}: max |Δ| = {max_diff}"
    );
    Ok(())
}

/// A bad expert id must be refused, never read out of bounds. CUDA and ROCm
/// validate ids while packing their routing on the host; this covers the SYCL
/// decode path, where ids are produced on device and the kernel itself guards
/// them, reporting through the queue at the next synchronize.
fn bad_expert_id_is_refused(dev: &Device) -> Result<()> {
    if !dev.is_sycl() {
        return Ok(());
    }
    let cpu = &Device::Cpu;
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(bytes),
            dev,
            GgmlDType::Q5K,
        )?,
        w_q_cpu.shape().clone(),
    )?;
    let input = Tensor::rand(-1f32, 1f32, (1, k), cpu)?.to_device(dev)?;
    let ids = Tensor::from_vec(vec![0u32, num_experts as u32], (1, 2), cpu)?.to_device(dev)?;
    let result = match w_q.indexed_moe_forward(&input, &ids) {
        Ok(_) => dev.synchronize(),
        Err(e) => Err(e),
    };
    let err = result.expect_err("an out-of-range expert id must be refused");
    let msg = err.to_string();
    assert!(msg.contains("expert"), "unexpected error: {msg}");
    // The guard is cleared when reported: the queue must work again.
    dev.synchronize()?;
    Ok(())
}

/// The queue's internal pool flushes (64 declined buffers, or 256 MiB) must
/// not consume a guard error: only `Device::synchronize` and device-to-host
/// copies read and clear it.
fn bad_expert_id_survives_a_pool_flush(dev: &Device) -> Result<()> {
    if !dev.is_sycl() {
        return Ok(());
    }
    let cpu = &Device::Cpu;
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(bytes),
            dev,
            GgmlDType::Q5K,
        )?,
        w_q_cpu.shape().clone(),
    )?;
    let input = Tensor::rand(-1f32, 1f32, (1, k), cpu)?.to_device(dev)?;
    let ids = Tensor::from_vec(vec![0u32, num_experts as u32], (1, 2), cpu)?.to_device(dev)?;
    // Trips the device-side guard; the call itself succeeds.
    let _ = w_q.indexed_moe_forward(&input, &ids)?;
    // 80 same-size tensors: the pool keeps at most 16 per exact size, so the
    // rest reach the deferred freer and trip its 64-buffer flush.
    let held: Vec<Tensor> = (0..80)
        .map(|_| Tensor::zeros((n, k), DType::F32, dev))
        .collect::<Result<_>>()?;
    drop(held);
    let err = dev
        .synchronize()
        .expect_err("the internal flush must not consume the guard error");
    assert!(
        err.to_string().contains("expert"),
        "unexpected error: {err}"
    );
    // The guard is cleared when reported: the queue must work again.
    dev.synchronize()?;
    Ok(())
}

/// lfm2d's decode loop never calls `Device::synchronize`: the first host-
/// visible point after a MoE layer is the logits read. A tripped guard must
/// surface there, as an error, not as numbers computed with expert 0.
fn bad_expert_id_surfaces_at_the_host_read(dev: &Device) -> Result<()> {
    if !dev.is_sycl() {
        return Ok(());
    }
    let cpu = &Device::Cpu;
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(bytes),
            dev,
            GgmlDType::Q5K,
        )?,
        w_q_cpu.shape().clone(),
    )?;
    let input = Tensor::rand(-1f32, 1f32, (1, k), cpu)?.to_device(dev)?;
    let ids = Tensor::from_vec(vec![0u32, num_experts as u32], (1, 2), cpu)?.to_device(dev)?;
    // Trips the device-side guard; the call itself succeeds.
    let out = w_q.indexed_moe_forward(&input, &ids)?;
    let err = out
        .to_device(cpu)
        .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        .expect_err("the device-to-host read must report the tripped guard");
    assert!(
        err.to_string().contains("expert"),
        "unexpected error: {err}"
    );
    // Reported once: the next read and sync are clean.
    out.to_device(cpu)?;
    dev.synchronize()?;
    Ok(())
}

/// The grouped path caches the f16 expert stack per weight; a repeat call on
/// the same QTensor must hit it and return the same answer, and a decode-shaped
/// call in between (a different request on the same storage) must not disturb
/// it. Also covers the public dequantize staying owned: the QTensor here is not
/// dropped, but the cache entry must not be handed out as an alias.
fn grouped_call_reuses_the_dequant_cache(dev: &Device) -> Result<()> {
    if !dev.is_sycl() {
        return Ok(());
    }
    let cpu = &Device::Cpu;
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(bytes),
            dev,
            GgmlDType::Q5K,
        )?,
        w_q_cpu.shape().clone(),
    )?;
    let (batch, topk) = (32usize, 4usize);
    let input = Tensor::rand(-1f32, 1f32, (batch, k), cpu)?.to_device(dev)?;
    let ids = Tensor::from_vec(
        (0..batch * topk)
            .map(|i| (i % num_experts) as u32)
            .collect::<Vec<_>>(),
        (batch, topk),
        cpu,
    )?
    .to_device(dev)?;
    let live = || live_bytes(dev).expect("a SYCL device has a live-byte account");
    let before = live();
    let first = w_q
        .indexed_moe_forward(&input, &ids)?
        .to_device(cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let after_first = live();
    // The grouped path caches one f16 dense copy per expert; nothing else it
    // allocates stays live once the output is on the CPU.
    assert_eq!(
        after_first - before,
        num_experts * n * k * 2,
        "the grouped call must cache one f16 copy per expert"
    );
    // Decode-shaped request on the same weight (mat-vec, no dense copy).
    let small = Tensor::rand(-1f32, 1f32, (3usize, k), cpu)?.to_device(dev)?;
    let small_ids = Tensor::from_vec(vec![0u32, 1, 2, 3, 0, 1], (3, 2), cpu)?.to_device(dev)?;
    let _ = w_q.indexed_moe_forward(&small, &small_ids)?;
    drop(small);
    drop(small_ids);
    assert_eq!(
        live(),
        after_first,
        "the decode-shaped call touched the dequant cache"
    );
    let second = w_q
        .indexed_moe_forward(&input, &ids)?
        .to_device(cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(
        live(),
        after_first,
        "the repeat grouped call must hit the cached dequant, not allocate another"
    );
    let max_diff = first
        .iter()
        .zip(second.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff == 0.0,
        "a repeated grouped call must reuse the cached dequant: max |Δ| = {max_diff}"
    );
    Ok(())
}

test_device!(
    indexed_moe_q8_0,
    indexed_moe_q8_0_cpu,
    indexed_moe_q8_0_cuda,
    indexed_moe_q8_0_metal,
    indexed_moe_q8_0_rocm,
    indexed_moe_q8_0_sycl
);
