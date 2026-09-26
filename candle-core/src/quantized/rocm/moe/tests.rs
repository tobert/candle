//! Hardware correctness tests for [`super::forward`].
//!
//! The reference is the CPU backend: dequantize the expert stack once, then run
//! the routing with plain tensor ops. That keeps the reference independent of
//! every ROCm code path — it does not even touch the GPU — so a shared bug in
//! the launch geometry cannot hide.

use crate::quantized::{GgmlDType, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{Device, Result, Tensor};

thread_local! {
    static WORK_BUFFER_PATTERN: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

// Runs only in tests. Production buffers receive no initialization; tests can
// inject deterministic old contents without depending on allocator reuse.
pub(super) fn initialize_work_buffer<T>(
    mut buffer: crate::rocm_backend::SendSyncDeviceMemory<T>,
) -> Result<crate::rocm_backend::SendSyncDeviceMemory<T>> {
    if let Some(pattern) = WORK_BUFFER_PATTERN.get() {
        buffer.memset(pattern).map_err(crate::Error::wrap)?;
    }
    Ok(buffer)
}

fn with_work_buffer_pattern<T>(pattern: i32, f: impl FnOnce() -> Result<T>) -> Result<T> {
    struct Reset(Option<i32>);
    impl Drop for Reset {
        fn drop(&mut self) {
            WORK_BUFFER_PATTERN.set(self.0);
        }
    }
    let _reset = Reset(WORK_BUFFER_PATTERN.replace(Some(pattern)));
    f()
}

#[test]
#[ignore = "requires actual ROCm hardware"]
fn expert_buffers_overwrite_dirty_storage_rocm() -> Result<()> {
    let device = Device::new_rocm(0)?;
    // Directly witness the test hook, so dropping it cannot silently turn this
    // into a test that only ever sees clean allocations.
    let Device::Rocm(dev) = &device else {
        unreachable!()
    };
    let probe = with_work_buffer_pattern(0x5a, || super::work_buffer::<u8>(dev, 73))?;
    assert_eq!(dev.clone_dtoh(&probe)?, vec![0x5a; 73]);

    for dtype in MOE_DTYPES {
        let mut cases = vec![(1, 1, 1, 256, false), (3, 4, 17, 768, false)];
        if matches!(dtype, GgmlDType::Q5K | GgmlDType::Q6K) {
            cases.extend([(65, 4, 131, 768, true), (33, 4, 1, 256, true)]);
        }
        for (batch, topk, n, k, grouped) in cases {
            let weights = QTensor::quantize_onto(
                &Tensor::from_vec(ramp(5 * n * k, 61.), (5, n, k), &Device::Cpu)?,
                dtype,
                &device,
            )?;
            for input_dim1 in [1, topk] {
                let input = Tensor::from_vec(
                    ramp((batch + 1) * input_dim1 * k, 43.),
                    (batch + 1, input_dim1, k),
                    &device,
                )?
                .narrow(0, 1, batch)?;
                // Repeated expert IDs (all one expert for the grouped singleton
                // output case), unused experts, and a nonzero ID view offset.
                let ids = Tensor::from_vec(
                    (0..(batch + 1) * topk)
                        .map(|i| {
                            if n == 1 {
                                3u32
                            } else {
                                ((i / 2 + 1) % 4) as u32
                            }
                        })
                        .collect(),
                    (batch + 1, topk),
                    &device,
                )?
                .narrow(0, 1, batch)?;
                let run = || {
                    super::forward_for_test(&weights, &input, &ids, grouped)?
                        .flatten_all()?
                        .to_vec1::<f32>()
                };
                let expected = with_work_buffer_pattern(0, run)?;
                assert!(expected.iter().all(|v| v.is_finite()));
                for pattern in [0xff, 0x5a] {
                    let got = with_work_buffer_pattern(pattern, run)?;
                    assert_eq!(got, expected, "dtype={dtype:?}, grouped={grouped}, batch={batch}, topk={topk}, input_dim1={input_dim1}");
                }
            }
        }
    }
    Ok(())
}

/// `RocmDevice::new` fails on machines without a GPU; those runs skip.
macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Every dtype with an `indexed_moe_forward_*_q8_1` kernel.
const MOE_DTYPES: [GgmlDType; 10] = [
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8_0,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
];

/// Deterministic, spread over a couple of octaves so quantization has something
/// to lose. Coprime divisors keep the weight and activation patterns from
/// lining up into an artificially easy dot product.
fn ramp(len: usize, div: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32 / div).sin()).collect()
}

/// The op, on the CPU, from the dequantized experts.
///
/// `out[b][j] = w[ids[b][j]] @ x[b][if input_dim1 == 1 { 0 } else { j }]`.
fn reference(
    w: &Tensor, // (num_experts, n, k), dequantized
    x: &Tensor, // (batch, input_dim1, k)
    ids: &[u32],
    topk: usize,
) -> Result<Tensor> {
    let (batch, input_dim1, _) = x.dims3()?;
    let mut rows = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for j in 0..topk {
            let expert = ids[b * topk + j] as usize;
            let we = w.get(expert)?; // (n, k)
            let xr = x.get(b)?.get(if input_dim1 == 1 { 0 } else { j })?; // (k,)
            rows.push(we.matmul(&xr.unsqueeze(1)?)?.squeeze(1)?);
        }
    }
    Tensor::stack(&rows, 0)?.reshape((batch, topk, ()))
}

/// Run one shape on the GPU and check it against [`reference`].
///
/// The tolerance is relative to the largest magnitude in the reference row
/// rather than per element: the kernel requantizes the activations to `q8_1`,
/// seven mantissa bits, and the residual accumulates over `k` terms, so a dot
/// product that lands near zero has an error set by the row's scale and not by
/// its own value. Same reasoning and same 2% figure as the MMVQ tests.
#[allow(clippy::too_many_arguments)]
fn check(
    device: &Device,
    dtype: GgmlDType,
    num_experts: usize,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize,
    input_dim1: usize,
) -> Result<()> {
    let w = Tensor::from_vec(
        ramp(num_experts * n * k, 61.),
        (num_experts, n, k),
        &Device::Cpu,
    )?;
    // The reference dequantizes the *same* rounding the GPU sees, so the only
    // difference left between the two is the kernel's `q8_1` activations.
    let dequantized = QTensor::quantize(&w, dtype)?.dequantize(&Device::Cpu)?;
    let qw = QTensor::quantize(&w.to_device(device)?, dtype)?;

    let x_data = ramp(batch * input_dim1 * k, 43.);
    let x_cpu = Tensor::from_vec(x_data.clone(), (batch, input_dim1, k), &Device::Cpu)?;
    let x = Tensor::from_vec(x_data, (batch, input_dim1, k), device)?;

    // A spread of experts per token, wrapping so every expert gets used. The
    // `+ 3` matters: routing the first pair to expert 0 makes the expert stride
    // a no-op there, which is exactly how the q8_0 kernel's wrong stride hid.
    let ids: Vec<u32> = (0..batch * topk)
        .map(|i| ((i * 5 + 3) % num_experts) as u32)
        .collect();
    let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), device)?;

    let got = qw.indexed_moe_forward(&x, &ids_t)?;
    assert_eq!(got.dims(), [batch, topk, n], "{dtype:?}");
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let want = reference(&dequantized, &x_cpu, &ids, topk)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 0.02 * scale,
            "{dtype:?} experts={num_experts} n={n} k={k} batch={batch} topk={topk} \
             input_dim1={input_dim1}: element {i} is {g}, reference {w}"
        );
    }
    Ok(())
}

/// One shared activation row per token, i.e. the gate/up projection: the same
/// hidden state goes to all `topk` experts.
#[test]
fn indexed_moe_shared_input_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        check(&device, dtype, 4, 96, 256, 3, 2, 1)?;
    }
    Ok(())
}

/// One activation row per routed pair, i.e. the down projection.
#[test]
fn indexed_moe_per_expert_input_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        check(&device, dtype, 4, 96, 256, 3, 2, 2)?;
    }
    Ok(())
}

/// `k` past one `MATRIX_ROW_PADDING` stride, so a wrong padded row stride in
/// the `q8_1` activation buffer shows up rather than cancelling out.
#[test]
fn indexed_moe_unpadded_k_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        check(&device, dtype, 3, 64, 768, 2, 2, 1)?;
    }
    Ok(())
}

/// A single token routed to a single expert — the smallest launch, and the one
/// where a grid of `(n, 1, 1)` would still pass if the batch/topk flattening
/// were wrong. Paired with a wide expert stack so a stride error moves the
/// result well outside tolerance.
#[test]
fn indexed_moe_single_token_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        check(&device, dtype, 8, 128, 256, 1, 1, 1)?;
    }
    Ok(())
}

#[test]
fn indexed_moe_rejects_a_dtype_without_a_kernel_rocm() -> Result<()> {
    let device = rocm_device!();
    let w = Tensor::zeros((2, 32, 256), crate::DType::F32, &device)?;
    let qw = QTensor::quantize(&w, GgmlDType::F16)?;
    let x = Tensor::zeros((1, 1, 256), crate::DType::F32, &device)?;
    let ids = Tensor::from_vec(vec![0u32], (1, 1), &device)?;
    let err = qw.indexed_moe_forward(&x, &ids).unwrap_err().to_string();
    assert!(err.contains("F16"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn indexed_moe_rejects_a_mismatched_batch_rocm() -> Result<()> {
    let device = rocm_device!();
    let w = Tensor::zeros((2, 32, 256), crate::DType::F32, &device)?;
    let qw = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let x = Tensor::zeros((3, 1, 256), crate::DType::F32, &device)?;
    let ids = Tensor::from_vec(vec![0u32, 1], (2, 1), &device)?;
    let err = qw.indexed_moe_forward(&x, &ids).unwrap_err().to_string();
    assert!(err.contains("batch"), "unexpected error: {err}");
    Ok(())
}

// Explicit hardware tests: absence of a GPU is an error, not a passing skip.
#[test]
#[ignore = "requires ROCm GPU"]
fn grouped_moe_matches_cpu_and_individual_routes_rocm() -> Result<()> {
    let device = Device::new_rocm(0)?;
    // Partial output/input tiles, padded K stride, duplicate routes, one hot
    // expert and unused experts. Nonzero input/ID offsets catch pointer bugs.
    for dtype in [GgmlDType::Q5K, GgmlDType::Q6K] {
        for (batch, topk, input_dim1, hot) in
            [(65, 4, 1, false), (33, 4, 4, false), (67, 2, 2, true)]
        {
            let (experts, n, k) = (5, 131, 768);
            let dense =
                Tensor::from_vec(ramp(experts * n * k, 61.), (experts, n, k), &Device::Cpu)?;
            let weights = QTensor::quantize_onto(&dense, dtype, &device)?;
            let w_cpu = QTensor::quantize(&dense, dtype)?.dequantize(&Device::Cpu)?;
            let input = Tensor::from_vec(
                ramp((batch + 1) * input_dim1 * k, 43.),
                (batch + 1, input_dim1, k),
                &device,
            )?
            .narrow(0, 1, batch)?;
            let ids: Vec<u32> = (0..(batch + 1) * topk)
                .map(|i| if hot { 3 } else { ((i * 7 + i / 3) % 4) as u32 })
                .collect();
            let ids_t =
                Tensor::from_vec(ids.clone(), (batch + 1, topk), &device)?.narrow(0, 1, batch)?;
            let got = super::forward_for_test(&weights, &input, &ids_t, true)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = reference(&w_cpu, &input.to_device(&Device::Cpu)?, &ids[topk..], topk)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // Existing q8_1 activation accuracy contract, evaluated per row.
            for (actual, expected) in got.chunks(n).zip(want.chunks(n)) {
                let scale = expected.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                for (a, b) in actual.iter().zip(expected) {
                    assert!(
                        (a - b).abs() <= 0.02 * scale,
                        "{dtype:?}: grouped {a} CPU {b}"
                    );
                }
            }
            // A single-route matvec uses identical activation quantization but
            // a different accumulation order: compare more tightly than CPU.
            for pair in [0, topk + 1, batch * topk - 1] {
                let x = input
                    .narrow(0, pair / topk, 1)?
                    .narrow(1, if input_dim1 == 1 { 0 } else { pair % topk }, 1)?
                    .contiguous()?;
                let id = Tensor::from_vec(vec![ids[topk + pair]], (1, 1), &device)?;
                let old = weights
                    .indexed_moe_forward(&x, &id)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let scale = old.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                for (a, b) in got[pair * n..(pair + 1) * n].iter().zip(&old) {
                    assert!(
                        (a - b).abs() <= 2e-5 * scale,
                        "{dtype:?}: grouped {a} matvec {b}"
                    );
                }
            }
            let again = super::forward_for_test(&weights, &input, &ids_t, true)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(got, again, "route packing must not affect arithmetic");
            let dispatched = weights
                .indexed_moe_forward(&input, &ids_t)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                got, dispatched,
                "public dispatch must reach grouped arithmetic"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires ROCm GPU"]
fn grouped_moe_rejects_invalid_expert_ids_rocm() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let dense = Tensor::zeros((2, 32, 256), crate::DType::F32, &device)?;
    let w = QTensor::quantize(&dense, GgmlDType::Q5K)?;
    let x = Tensor::zeros((16, 1, 256), crate::DType::F32, &device)?;
    for invalid in [2, u32::MAX] {
        let mut ids = vec![0u32; 32];
        ids[31] = invalid;
        let ids = Tensor::from_vec(ids, (16, 2), &device)?;
        let error = super::forward_for_test(&w, &x, &ids, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("expert id"), "{error}");
    }
    Ok(())
}

#[test]
fn grouped_moe_dispatch_keeps_decode_and_sparse_suffixes_on_matvec() {
    let mut d = super::Dims {
        num_experts: 32,
        n: 3584,
        k: 2048,
        batch: 1,
        topk: 4,
        input_dim1: 1,
    };
    assert!(!super::use_grouped(GgmlDType::Q5K, &d));
    d.batch = 63;
    assert!(!super::use_grouped(GgmlDType::Q5K, &d));
    d.batch = 64;
    assert!(super::use_grouped(GgmlDType::Q5K, &d));
    assert!(super::use_grouped(GgmlDType::Q6K, &d));
    assert!(!super::use_grouped(GgmlDType::Q4K, &d));
    d.k = 257;
    assert!(!super::use_grouped(GgmlDType::Q5K, &d));
}

/// Wall-clock dispatch comparison, not a profiler. Both paths include their
/// own activation quantization, allocations, routing and synchronization.
#[test]
#[ignore = "manual timing run on ROCm GPU"]
fn bench_grouped_moe_lfm25_rocm() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for (dtype, n, k, slots) in [
        (GgmlDType::Q5K, 3584, 2048, 1),
        (GgmlDType::Q6K, 2048, 1792, 4),
    ] {
        let dense = Tensor::from_vec(ramp(32 * n * k, 61.), (32, n, k), &Device::Cpu)?;
        let weights = QTensor::quantize_onto(&dense, dtype, &dev)?;
        for batch in [32, 64, 128] {
            let input = Tensor::from_vec(ramp(batch * slots * k, 43.), (batch, slots, k), &dev)?;
            let ids = Tensor::from_vec(
                (0..batch * 4)
                    .map(|i| ((i * 7 + i / 5) % 32) as u32)
                    .collect::<Vec<_>>(),
                (batch, 4),
                &dev,
            )?;
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..7 {
                // Alternate ordering; first pair warms kernels/allocator.
                for grouped in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    dev.synchronize()?;
                    let start = std::time::Instant::now();
                    let out = super::forward_for_test(&weights, &input, &ids, grouped)?;
                    dev.synchronize()?;
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    std::hint::black_box(out);
                    if round > 0 {
                        times[usize::from(grouped)].push(elapsed);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let old = (times[0][2] + times[0][3]) / 2.;
            let new = (times[1][2] + times[1][3]) / 2.;
            println!("GROUPED_TIMING dtype={dtype:?} n={n} k={k} batch={batch} matvec_ms={old:.4} grouped_ms={new:.4} speedup={:.3}",old/new);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires ROCm GPU"]
fn grouped_moe_rejects_invalid_layouts_rocm() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    let dense = Tensor::zeros((2, 32, 256), crate::DType::F32, &dev)?;
    let w = QTensor::quantize(&dense, GgmlDType::Q5K)?;
    let x = Tensor::zeros((16, 1, 256), crate::DType::F32, &dev)?;
    let ids = Tensor::zeros((16, 2), crate::DType::U32, &dev)?;
    let strided = Tensor::zeros((16, 1, 512), crate::DType::F32, &dev)?.narrow(2, 0, 256)?;
    let strided_ids = Tensor::zeros((16, 4), crate::DType::U32, &dev)?.narrow(1, 0, 2)?;
    for (input, routes, expected) in [
        (strided, ids.clone(), "contiguous input"),
        (x.clone(), strided_ids, "contiguous u32 ids"),
        (x.to_dtype(crate::DType::F16)?, ids.clone(), "f32 input"),
        (x.clone(), ids.to_dtype(crate::DType::I64)?, "u32 ids"),
        (x.narrow(0, 0, 0)?, ids.narrow(0, 0, 0)?, "empty shape"),
    ] {
        let err = super::forward_for_test(&w, &input, &routes, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{err}");
    }
    // Same physical GPU but a different Candle device owns a different stream.
    let other = Device::new_rocm(0)?;
    let other_x = Tensor::zeros((16, 1, 256), crate::DType::F32, &other)?;
    let err = w
        .indexed_moe_forward(&other_x, &ids)
        .unwrap_err()
        .to_string();
    assert!(err.contains("same ROCm device/stream"), "{err}");
    Ok(())
}

// Isolate Q5's minimum term: zero quantized weights, nonzero per-block minima,
// and activations whose original sum differs strongly from d*sum(q8).
#[test]
#[ignore = "requires ROCm GPU"]
fn grouped_moe_q5_minimum_uses_quantized_activation_sum_rocm() -> Result<()> {
    use crate::quantized::k_quants::BlockQ5K;
    use half::f16;
    let dev = Device::new_rocm(0)?;
    let Device::Rocm(gpu) = &dev else {
        unreachable!()
    };
    let block = BlockQ5K {
        d: f16::from_f32(0.5),
        dmin: f16::from_f32(0.25),
        // sc=1 for all groups; minima=1..8 in the GGML packed 6-bit layout.
        scales: [1, 1, 1, 1, 1, 2, 3, 4, 0x51, 0x61, 0x71, 0x81],
        qh: [0; 32],
        qs: [0; 128],
    };
    let w = QTensor::new(
        crate::quantized::rocm::load_quantized(gpu, &[block])?,
        (1, 1, 256),
    )?;
    for sign in [-1f32, 0., 1.] {
        let mut row = Vec::new();
        let mut expected = 0f32;
        let mut original_sum_result = 0f32;
        for group in 1..=8 {
            let amplitude = group as f32;
            let scale = f16::from_f32(amplitude / 127.).to_f32();
            expected -= 0.25 * amplitude * sign * 127. * scale;
            for i in 0..32 {
                let value = sign * amplitude * if i == 0 { 1. } else { 0.49 / 127. };
                row.push(value);
                original_sum_result -= 0.25 * amplitude * value;
            }
        }
        if sign != 0. {
            assert!((expected - original_sum_result).abs() > 1.);
        }
        let x = Tensor::from_vec(row.repeat(16), (16, 1, 256), &dev)?;
        let ids = Tensor::zeros((16, 1), crate::DType::U32, &dev)?;
        let got = super::forward_for_test(&w, &x, &ids, true)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for value in got {
            assert!((value-expected).abs()<=2e-5*expected.abs().max(1.),"got {value}, quantized-sum reference {expected}, original-sum result {original_sum_result}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires actual ROCm hardware; device failure is an error"]
fn prepared_routing_reuses_packing_and_captures_ids_rocm() -> Result<()> {
    use super::GroupedMoeRouting;
    let device = Device::new_rocm(0)?;
    let (experts, batch, topk, n, k) = (4, 32, 3, 96, 256);
    let ids_data = (0..(batch + 1) * topk)
        .map(|i| ((i * 7 + i / 3) % experts) as u32)
        .collect::<Vec<_>>();
    let ids = Tensor::from_vec(ids_data, (batch + 1, topk), &device)?.narrow(0, 1, batch)?;
    let routing = GroupedMoeRouting::new(&ids, experts)?;
    let pairs_address = routing.packed.pairs.as_ptr();
    let counts_address = routing.packed.counts.as_ptr();
    let mut cases = Vec::new();
    for (dtype, slots) in [(GgmlDType::Q5K, 1), (GgmlDType::Q6K, topk)] {
        let dense = Tensor::from_vec(ramp(experts * n * k, 61.), (experts, n, k), &Device::Cpu)?;
        let weights = QTensor::quantize_onto(&dense, dtype, &device)?;
        let input = Tensor::from_vec(
            ramp((batch + 1) * slots * k, 43.),
            (batch + 1, slots, k),
            &device,
        )?
        .narrow(0, 1, batch)?;
        assert!(weights.supports_grouped_moe(batch, topk));
        let expected = weights
            .indexed_moe_forward(&input, &ids)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = weights
            .grouped_moe_forward(&input, &routing)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(expected, actual);
        cases.push((weights, input, expected));
    }
    // Prepared routing is a snapshot of assignments, not a memo indexed by a
    // mutable Tensor identity. Changing the source IDs must not change it.
    ids.slice_set(
        &Tensor::zeros((batch, topk), crate::DType::U32, &device)?,
        0,
        0,
    )?;
    for (weights, input, expected) in cases {
        assert_eq!(
            expected,
            weights
                .grouped_moe_forward(&input, &routing)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        assert_ne!(
            expected,
            weights
                .indexed_moe_forward(&input, &ids)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
    }
    assert_eq!(pairs_address, routing.packed.pairs.as_ptr());
    assert_eq!(counts_address, routing.packed.counts.as_ptr());
    Ok(())
}

#[test]
#[ignore = "requires actual ROCm hardware; device failure is an error"]
fn prepared_routing_rejects_bad_ids_weights_and_devices_rocm() -> Result<()> {
    use super::GroupedMoeRouting;
    let device = Device::new_rocm(0)?;
    let ids = Tensor::zeros((32, 2), crate::DType::U32, &device)?;
    assert!(GroupedMoeRouting::new(&ids, 0).is_err());
    assert!(GroupedMoeRouting::new(&ids.to_dtype(crate::DType::F32)?, 4).is_err());
    assert!(GroupedMoeRouting::new(&Tensor::full(4u32, (32, 2), &device)?, 4).is_err());
    assert!(GroupedMoeRouting::new(&ids.transpose(0, 1)?, 4).is_err());
    let routing = GroupedMoeRouting::new(&ids, 4)?;
    let dense = Tensor::zeros((2, 96, 256), crate::DType::F32, &Device::Cpu)?;
    let weights = QTensor::quantize_onto(&dense, GgmlDType::Q5K, &device)?;
    let input = Tensor::zeros((32, 1, 256), crate::DType::F32, &device)?;
    assert!(weights.grouped_moe_forward(&input, &routing).is_err());
    let other_device = Device::new_rocm(0)?;
    let foreign = GroupedMoeRouting::new(
        &Tensor::zeros((32, 2), crate::DType::U32, &other_device)?,
        2,
    )?;
    assert!(weights.grouped_moe_forward(&input, &foreign).is_err());
    let correct = GroupedMoeRouting::new(&ids, 2)?;
    assert!(weights
        .grouped_moe_forward(&input.narrow(0, 0, 31)?, &correct)
        .is_err());
    assert!(!weights.supports_grouped_moe(1, 2));
    Ok(())
}

/// Task-major launch geometry computes every output with the same block and
/// the same reduction as the row-major one, so the bits must match; the CPU
/// comparisons above cover the values themselves. Shared (gate/up) and
/// per-expert (down) inputs, a lone token and decode-sized batches.
#[test]
fn indexed_moe_task_major_is_bit_identical_to_row_major_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        for (batch, topk, input_dim1) in [(1, 1, 1), (3, 2, 1), (3, 2, 2), (17, 4, 1), (32, 4, 4)] {
            let (experts, n, k) = (8, 96, 256);
            let w = Tensor::from_vec(ramp(experts * n * k, 61.), (experts, n, k), &Device::Cpu)?;
            let qw = QTensor::quantize(&w.to_device(&device)?, dtype)?;
            let x = Tensor::from_vec(
                ramp(batch * input_dim1 * k, 43.),
                (batch, input_dim1, k),
                &device,
            )?;
            let ids = Tensor::from_vec(
                (0..batch * topk)
                    .map(|i| ((i * 5 + 3) % experts) as u32)
                    .collect::<Vec<_>>(),
                (batch, topk),
                &device,
            )?;
            let row = super::forward_vector_for_test(&qw, &x, &ids, false)?;
            let task = super::forward_vector_for_test(&qw, &x, &ids, true)?;
            assert_eq!(row.dims(), task.dims());
            assert_eq!(
                row.flatten_all()?.to_vec1::<f32>()?,
                task.flatten_all()?.to_vec1::<f32>()?,
                "{dtype:?} batch={batch} topk={topk} input_dim1={input_dim1}"
            );
        }
    }
    Ok(())
}

#[test]
fn task_major_geometry_starts_at_sixteen_tokens() {
    let mut d = super::Dims {
        num_experts: 32,
        n: 3584,
        k: 2048,
        batch: 1,
        topk: 4,
        input_dim1: 1,
    };
    assert!(!super::use_task_major(&d));
    d.batch = 15;
    assert!(!super::use_task_major(&d));
    d.batch = 16;
    assert!(super::use_task_major(&d));
}

/// Decode-sized batches at the LFM2.5 expert shapes, row- vs task-major. Wall
/// clock with synchronisation, activation quantization included.
#[test]
#[ignore = "manual timing run on ROCm GPU"]
fn bench_task_major_moe_lfm25_rocm() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for (dtype, n, k, slots) in [
        (GgmlDType::Q5K, 3584, 2048, 1),
        (GgmlDType::Q6K, 2048, 1792, 4),
    ] {
        let dense = Tensor::from_vec(ramp(32 * n * k, 61.), (32, n, k), &Device::Cpu)?;
        let weights = QTensor::quantize_onto(&dense, dtype, &dev)?;
        for batch in [1, 2, 4, 8, 16, 32] {
            let input = Tensor::from_vec(ramp(batch * slots * k, 43.), (batch, slots, k), &dev)?;
            // Distinct experts within a token, spread across tokens.
            let ids = Tensor::from_vec(
                (0..batch * 4)
                    .map(|i| ((i / 4 * 11 + (i % 4) * 8 + i / 20) % 32) as u32)
                    .collect::<Vec<_>>(),
                (batch, 4),
                &dev,
            )?;
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..21 {
                for task_major in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    dev.synchronize()?;
                    let start = std::time::Instant::now();
                    let out = super::forward_vector_for_test(&weights, &input, &ids, task_major)?;
                    dev.synchronize()?;
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    std::hint::black_box(out);
                    if round > 0 {
                        times[usize::from(task_major)].push(elapsed);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let (row, task) = (times[0][10], times[1][10]);
            println!("TASK_MAJOR_TIMING dtype={dtype:?} n={n} k={k} batch={batch} row_major_ms={row:.4} task_major_ms={task:.4} speedup={:.3}", row / task);
        }
    }
    Ok(())
}
