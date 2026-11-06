#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, test_utils::to_vec3_round, Device, IndexOp, Result, Tensor};

fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = candle_nn::ops::softmax(&tensor.log()?, 0)?;
    let t1 = candle_nn::ops::softmax(&tensor.log()?, 1)?;
    let t2 = candle_nn::ops::softmax(&tensor.log()?, 2)?;
    assert_eq!(
        to_vec3_round(&t0, 4)?,
        &[
            // 3/5, 1/2, 4/11
            [[0.6, 0.5, 0.3636], [0.1111, 0.7143, 0.5294]],
            // 2/5, 1/2, 7/11
            [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t1, 4)?,
        &[
            // 3/4, 1/6, 4/13
            [[0.75, 0.1667, 0.3077], [0.25, 0.8333, 0.6923]],
            // 2/10, 1/3, 7/15
            [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    let t2 = candle_nn::ops::softmax_last_dim(&tensor.log()?)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    Ok(())
}

fn rms_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn rms_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    let diff = ((&t - &t2)?)
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    // The SYCL fused path is a CPU shim; the "slow" path runs on-device
    // primitives whose reduction accumulates differently.
    let tol = if device.is_sycl() { 2e-3 } else { 1e-5 };
    assert!(diff < tol, "rms_norml diff {diff}");
    if !device.is_sycl() {
        assert_eq!(to_vec3_round(&t, 2)?, to_vec3_round(&t2, 2)?);
    }
    Ok(())
}

fn rms_norm_large_magnitude(device: &Device) -> Result<()> {
    let (rows, hidden) = (4usize, 6912usize);
    let data: Vec<f32> = (0..rows * hidden)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * ((i as f32 * 0.17).sin().abs() * 7e9)
        })
        .collect();
    let tensor = Tensor::from_vec(data, (rows, hidden), device)?;
    let alpha = Tensor::ones(hidden, candle::DType::F32, device)?;

    let fused = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let slow = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;

    let fused_v = fused.flatten_all()?.to_vec1::<f32>()?;
    let slow_v = slow.flatten_all()?.to_vec1::<f32>()?;
    for &v in &fused_v {
        assert!(
            v.is_finite(),
            "rms_norm produced a non-finite value for large-magnitude input"
        );
    }
    for &v in &slow_v {
        assert!(
            v.is_finite(),
            "rms_norm_slow produced a non-finite value for large-magnitude input"
        );
    }
    let diff = fused_v
        .iter()
        .zip(slow_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < 5e-3,
        "rms_norm and rms_norm_slow disagree: max |Δ| = {diff}"
    );
    Ok(())
}

fn layer_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn layer_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

#[test]
fn softmax_numerical_stability() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = Tensor::new(&[1234f32, 0.], dev)?;
    let softmax = candle_nn::ops::softmax(&xs, 0)?;
    assert_eq!(softmax.to_vec1::<f32>()?, &[1f32, 0.]);
    Ok(())
}

fn ropei(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope_i(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope_thd(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &cos, &sin)?.transpose(1, 2)?
    };
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(0..1)?, &cos, &sin)?
    };
    let rope2 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(1..2)?, &cos2, &sin2)?
    };

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &both_cos, &both_sin)?
    };
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

type RopeFn = fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>;

/// The three rope layouts, each with the axis of `x` that holds the sequence.
fn rope_variants() -> [(&'static str, RopeFn, usize); 3] {
    [
        ("rope_i", candle_nn::rotary_emb::rope_i, 2),
        ("rope", candle_nn::rotary_emb::rope, 2),
        ("rope_thd", candle_nn::rotary_emb::rope_thd, 1),
    ]
}

fn rope_input(
    seq_dim: usize,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    device: &Device,
) -> Result<Tensor> {
    let shape = if seq_dim == 1 {
        (b, t, h, d)
    } else {
        (b, h, t, d)
    };
    Tensor::randn(0f32, 1f32, shape, device)
}

/// `rope`, `rope_i` and `rope_thd` take a cos/sin table with more rows than the
/// sequence (only a shorter one is an error) and read its first `t` rows, per
/// batch for a batched table. Every device must give the CPU's numbers for the
/// table narrowed to exactly `t` rows.
fn rope_long_table(device: &Device) -> Result<()> {
    let (b, h, d) = (2, 3, 8);
    let half = d / 2;
    let extra = 4;
    for (name, op, seq_dim) in rope_variants() {
        // t == 1 is the decode shape; the CPU kernels take a separate branch there.
        for t in [1usize, 5] {
            let x = rope_input(seq_dim, b, h, t, d, device)?;
            let x_cpu = x.to_device(&Device::Cpu)?;
            let rows = t + extra;
            for batched in [false, true] {
                let shape: Vec<usize> = if batched {
                    vec![b, rows, half]
                } else {
                    vec![rows, half]
                };
                let cos = Tensor::randn(0f32, 1f32, shape.as_slice(), device)?;
                let sin = Tensor::randn(0f32, 1f32, shape.as_slice(), device)?;
                let row_dim = if batched { 1 } else { 0 };
                let narrow = |cs: &Tensor| -> Result<Tensor> {
                    cs.to_device(&Device::Cpu)?
                        .narrow(row_dim, 0, t)?
                        .contiguous()
                };
                let want = op(&x_cpu, &narrow(&cos)?, &narrow(&sin)?)?;
                let got = op(&x, &cos, &sin)?.to_device(&Device::Cpu)?;
                let diff = (got - &want)?.abs()?.max_all()?.to_vec0::<f32>()?;
                let tol = if device.is_cpu() { 0.0 } else { 1e-5 };
                assert!(
                    diff <= tol,
                    "{name} t {t} batched {batched}: max diff {diff} against the narrowed table"
                );
                if batched {
                    // An oracle off the batched path: batch i alone, through the
                    // unbatched path, on its own table's first t rows.
                    let (cos_cpu, sin_cpu) = (narrow(&cos)?, narrow(&sin)?);
                    let per_batch = (0..b)
                        .map(|i| op(&x_cpu.narrow(0, i, 1)?, &cos_cpu.get(i)?, &sin_cpu.get(i)?))
                        .collect::<Result<Vec<_>>>()?;
                    let oracle = Tensor::cat(&per_batch, 0)?;
                    let diff = (&want - oracle)?.abs()?.max_all()?.to_vec0::<f32>()?;
                    assert!(
                        diff == 0.0,
                        "{name} t {t}: batched differs from per-batch by {diff}"
                    );
                    // A caller's own cut is a non-contiguous view; it is taken.
                    let view_cos = cos.narrow(1, 0, t)?;
                    let view_sin = sin.narrow(1, 0, t)?;
                    let got = op(&x, &view_cos, &view_sin)?.to_device(&Device::Cpu)?;
                    let diff = (got - &want)?.abs()?.max_all()?.to_vec0::<f32>()?;
                    assert!(
                        diff <= tol,
                        "{name} t {t}: a narrowed view differs by {diff}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// A table shorter than the sequence, with the wrong head dim or batch count,
/// mixing batched and unbatched, or in another dtype than `x` is refused on
/// every device, never read out of bounds.
fn rope_bad_table(device: &Device) -> Result<()> {
    let (b, h, t, d) = (2, 3, 5, 8);
    let half = d / 2;
    for (name, op, seq_dim) in rope_variants() {
        let x = rope_input(seq_dim, b, h, t, d, device)?;
        for batched in [false, true] {
            let shape = |rows: usize, cols: usize| -> Vec<usize> {
                if batched {
                    vec![b, rows, cols]
                } else {
                    vec![rows, cols]
                }
            };
            let table = |rows: usize, cols: usize| -> Result<Tensor> {
                Tensor::randn(0f32, 1f32, shape(rows, cols).as_slice(), device)
            };
            let ok = table(t, half)?;
            let short = table(t - 1, half)?;
            assert!(
                op(&x, &short, &ok).is_err() && op(&x, &ok, &short).is_err(),
                "{name} batched {batched}: a {t}-step input took a {}-row table",
                t - 1
            );
            let wide = table(t, half + 1)?;
            assert!(
                op(&x, &wide, &ok).is_err() && op(&x, &ok, &wide).is_err(),
                "{name} batched {batched}: head dim {d} took a table {} wide",
                half + 1
            );
            let f16 = table(t, half)?.to_dtype(candle::DType::F16)?;
            assert!(
                op(&x, &f16, &f16).is_err(),
                "{name} batched {batched}: an f32 input took an f16 table"
            );
        }
        // Batched and unbatched halves of one pair: the kernels would read
        // batch 0's rows for every batch.
        let flat = Tensor::randn(0f32, 1f32, (t, half), device)?;
        let batched = Tensor::randn(0f32, 1f32, (b, t, half), device)?;
        assert!(
            op(&x, &batched, &flat).is_err() && op(&x, &flat, &batched).is_err(),
            "{name}: a mixed batched/unbatched cos/sin pair was taken"
        );
        let one = Tensor::randn(0f32, 1f32, (1, t, half), device)?;
        assert!(
            op(&x, &one, &one).is_err(),
            "{name}: a 1-batch table took a {b}-batch input"
        );
    }
    Ok(())
}

fn sigmoid(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let s1 = candle_nn::ops::sigmoid(&tensor)?;
    let s2 = (1. / (1. + tensor.neg()?.exp()?)?)?;
    let diff = (s1 - s2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    // SYCL's `exp` need not match the fused sigmoid bit-for-bit; every
    // other backend computes both sides the same way.
    let tol = if device.is_sycl() { 1e-6 } else { 0.0 };
    assert!(diff <= tol, "sigmoid diff {diff}");
    Ok(())
}

test_device!(
    ropei,
    ropei_cpu,
    ropei_gpu,
    ropei_metal,
    ropei_rocm,
    ropei_sycl
);
test_device!(rope, rope_cpu, rope_gpu, rope_metal, rope_rocm, rope_sycl);
test_device!(
    rope_thd,
    rope_thd_cpu,
    rope_thd_gpu,
    rope_thd_metal,
    rope_thd_rocm,
    rope_thd_sycl
);
test_device!(
    rope_long_table,
    rope_long_table_cpu,
    rope_long_table_gpu,
    rope_long_table_metal,
    rope_long_table_rocm,
    rope_long_table_sycl
);
test_device!(
    rope_bad_table,
    rope_bad_table_cpu,
    rope_bad_table_gpu,
    rope_bad_table_metal,
    rope_bad_table_rocm,
    rope_bad_table_sycl
);
test_device!(
    softmax,
    softmax_cpu,
    softmax_gpu,
    softmax_metal,
    softmax_rocm,
    softmax_sycl
);
test_device!(
    rms_norm,
    rms_norm_cpu,
    rms_norm_gpu,
    rms_norm_metal,
    rms_norm_rocm,
    rms_norm_sycl
);
test_device!(
    rms_norml,
    rms_norml_cpu,
    rms_norml_gpu,
    rms_norml_metal,
    rms_norml_rocm,
    rms_norml_sycl
);
test_device!(
    rms_norm_large_magnitude,
    rms_norm_large_magnitude_cpu,
    rms_norm_large_magnitude_gpu,
    rms_norm_large_magnitude_metal,
    rms_norm_large_magnitude_rocm,
    rms_norm_large_magnitude_sycl
);
test_device!(layer_norm, ln_cpu, ln_gpu, ln_metal, ln_rocm, ln_sycl);
test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal, lnl_rocm, lnl_sycl);
test_device!(
    sigmoid,
    sigmoid_cpu,
    sigmoid_gpu,
    sigmoid_metal,
    sigmoid_rocm,
    sigmoid_sycl
);
