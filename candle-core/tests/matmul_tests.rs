use candle_core::{test_device, DType, Device, IndexOp, Result, Tensor};

fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3::<f32>()?, &expected);

    // Also perform the matmul on contiguous transposed versions.
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(!a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), &[6, 1, 2]);

    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(!b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), &[6, 1, 3]);

    assert_eq!(a_tt.matmul(&b)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a_tt.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    Ok(())
}

fn matmul_bf16(device: &Device) -> Result<()> {
    if !device.supports_bf16() {
        return Ok(());
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);
    Ok(())
}

fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), &[3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            // With cuda, we see errors of up to ~1e-12.
            assert!(sum_diff2.to_vec0::<f32>()? < 1e-6)
        }
    }
    Ok(())
}

fn zero_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::zeros((2, 0), DType::F32, device)?;
    let rhs = Tensor::zeros((0, 3), DType::F32, device)?;
    let output = lhs.matmul(&rhs)?;
    assert_eq!(output.dims(), &[2, 3]);
    assert_eq!(output.to_vec2::<f32>()?, &[[0., 0., 0.], [0., 0., 0.]]);

    let lhs = Tensor::zeros((2, 3, 4), DType::F32, device)?
        .transpose(1, 2)?
        .narrow(1, 0, 0)?;
    let rhs = Tensor::zeros((2, 4, 3), DType::F32, device)?
        .transpose(1, 2)?
        .narrow(2, 0, 0)?;
    assert!(!lhs.is_contiguous());
    assert!(!rhs.is_contiguous());
    assert_eq!(lhs.dims(), &[2, 0, 3]);
    assert_eq!(rhs.dims(), &[2, 3, 0]);
    assert_eq!(lhs.matmul(&rhs)?.dims(), &[2, 0, 0]);
    Ok(())
}

fn assert_matmul_error(
    device: &Device,
    lhs: (&[usize], DType),
    rhs: (&[usize], DType),
    expected: &str,
) -> Result<()> {
    let lhs = Tensor::zeros(lhs.0, lhs.1, device)?;
    let rhs = Tensor::zeros(rhs.0, rhs.1, device)?;
    let err = lhs.matmul(&rhs).unwrap_err();
    assert!(
        err.to_string().starts_with(expected),
        "unexpected error: {err}"
    );
    Ok(())
}

fn zero_matmul_validation(device: &Device) -> Result<()> {
    use DType::{F16, F32};

    let shape_error = "shape mismatch in matmul";
    assert_matmul_error(device, (&[0, 2], F32), (&[3, 4], F16), shape_error)?;
    assert_matmul_error(device, (&[2, 3], F32), (&[4, 0], F32), shape_error)?;
    assert_matmul_error(device, (&[0, 2, 3], F32), (&[1, 3, 4], F32), shape_error)?;
    assert_matmul_error(
        device,
        (&[0, 2], F32),
        (&[2, 3], F16),
        "dtype mismatch in matmul",
    )?;
    Ok(())
}

fn zero_matmul_device_validation(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let lhs = Tensor::zeros((0, 2), DType::F32, &Device::Cpu)?;
    let rhs = Tensor::zeros((2, 3), DType::F16, device)?;
    let err = lhs.matmul(&rhs).unwrap_err();
    assert!(
        err.to_string().starts_with("device mismatch in matmul"),
        "unexpected error: {err}"
    );
    Ok(())
}

// A rank-2 rhs is broadcast over the batch dims only, which `broadcast_matmul` folds into a
// single 2D matmul instead of copying the rhs. Check the folded path against the per-batch
// products it stands for, contiguous lhs (folded) and non-contiguous lhs (fallback) alike.
fn broadcast_matmul_rank2_rhs(device: &Device) -> Result<()> {
    let rhs = Tensor::randn(0f32, 1f32, (5, 2), device)?;
    for lhs in [
        Tensor::randn(0f32, 1f32, (3, 4, 5), device)?,
        Tensor::randn(0f32, 1f32, (3, 6, 4, 5), device)?,
        Tensor::randn(0f32, 1f32, (1, 1, 5), device)?,
        Tensor::randn(0f32, 1f32, (3, 5, 4), device)?.transpose(1, 2)?,
    ] {
        let out = lhs.broadcast_matmul(&rhs)?;
        let mut dims = lhs.dims().to_vec();
        let n = dims.len();
        dims[n - 1] = 2;
        assert_eq!(out.dims(), dims.as_slice());
        // Same product, computed the way the doc comment describes it.
        let batch: usize = lhs.dims()[..n - 2].iter().product();
        let (m, k) = (lhs.dims()[n - 2], lhs.dims()[n - 1]);
        let flat = lhs.reshape((batch, m, k))?;
        let out = out.reshape((batch, m, 2))?;
        for b in 0..batch {
            let diff = (out.i(b)? - flat.i(b)?.matmul(&rhs)?)?.sqr()?.sum_all()?;
            assert!(diff.to_vec0::<f32>()? < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn tensor_dot() -> Result<()> {
    let lhs = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
    let rhs = Tensor::new(&[4., 5., 6.], &Device::Cpu)?;
    let expected = Tensor::new(32., &Device::Cpu)?;
    let dot_ret = lhs.dot(&rhs)?;
    candle_core::test_utils::assert_tensor_eq(&dot_ret, &expected)?;
    Ok(())
}

#[test]
fn tensor_mv() -> Result<()> {
    let mat = Tensor::new(&[[1., 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let vec = Tensor::new(&[1., 1., 1.], &Device::Cpu)?;
    let expected = Tensor::new(&[6., 15.], &Device::Cpu)?;
    let mv_ret = mat.mv(&vec)?;
    candle_core::test_utils::assert_tensor_eq(&mv_ret, &expected)?;
    Ok(())
}

// https://github.com/huggingface/candle/issues/1948
fn squeeze_mm(device: &Device) -> Result<()> {
    let seq_len = 8_usize;
    let a = Tensor::zeros((1, seq_len, 16), DType::F32, device)?;
    let x = a.i((.., seq_len - 1, ..))?;
    let w = Tensor::zeros((32, 16), DType::F32, device)?.t()?;
    let x = x.matmul(&w)?;
    assert_eq!(x.dims(), &[1, 32]);
    Ok(())
}

// https://github.com/huggingface/candle/issues/1992
fn mm_layout(device: &Device) -> Result<()> {
    let a = Tensor::arange(0f32, 16f32, device)?.reshape((1, 1, 4, 4))?;
    let b = Tensor::arange(0f32, 8f32, device)?.reshape((1, 1, 4, 2))?;
    let mm1 = a.matmul(&b)?;
    // Forces the layout to be:
    // shape: [1, 1, 4, 2], stride: [8, 2, 2, 1], start_offset: 0
    // This is still a contiguous matrix but matmul checks are only the two last dimensions have
    // non 1 sizes but matmul check may be reluctant to handle it.
    let b = b.transpose(1, 2)?.force_contiguous()?.transpose(1, 2)?;
    let mm2 = a.matmul(&b)?;
    let diff = (mm1 - mm2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

test_device!(matmul, matmul_cpu, matmul_gpu, matmul_metal, matmul_rocm);
test_device!(
    matmul_bf16,
    matmul_bf16_cpu,
    matmul_bf16_gpu,
    matmul_bf16_metal,
    matmul_bf16_rocm
);
test_device!(
    broadcast_matmul,
    broadcast_matmul_cpu,
    broadcast_matmul_gpu,
    broadcast_matmul_metal,
    broadcast_matmul_rocm
);
test_device!(
    squeeze_mm,
    squeeze_mm_cpu,
    squeeze_mm_gpu,
    squeeze_mm_metal,
    squeeze_mm_rocm
);
test_device!(
    mm_layout,
    mm_layout_cpu,
    mm_layout_gpu,
    mm_layout_metal,
    mm_layout_rocm
);
test_device!(
    zero_matmul,
    zero_matmul_cpu,
    zero_matmul_gpu,
    zero_matmul_metal,
    zero_matmul_rocm
);
test_device!(
    zero_matmul_validation,
    zero_matmul_validation_cpu,
    zero_matmul_validation_gpu,
    zero_matmul_validation_metal,
    zero_matmul_validation_rocm
);
test_device!(
    zero_matmul_device_validation,
    zero_matmul_device_validation_cpu,
    zero_matmul_device_validation_gpu,
    zero_matmul_device_validation_metal,
    zero_matmul_device_validation_rocm
);
test_device!(
    broadcast_matmul_rank2_rhs,
    broadcast_matmul_rank2_rhs_cpu,
    broadcast_matmul_rank2_rhs_gpu,
    broadcast_matmul_rank2_rhs_metal,
    broadcast_matmul_rank2_rhs_rocm
);

/// A batched matmul with one side broadcast over the batch, against each batch
/// computed alone from contiguous copies. The layouts are the ones the CPU
/// backend used to fold into a single GEMM without checking that the folded
/// rows (or columns) are evenly strided: a `(b, 1, k)` lhs whose size-one dim
/// carries stride 1 (what `(b, k, 1).transpose(1, 2).contiguous()` gives, and
/// is_contiguous accepts), a transposed lhs, and a broadcast lhs with m > 1.
fn broadcast_matmul_folds(device: &Device) -> Result<()> {
    let ramp = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * s).sin()).collect() };
    let check = |lhs: &Tensor, rhs: &Tensor, what: &str| -> Result<()> {
        let got = lhs.matmul(rhs)?;
        let b = got.dim(0)?;
        for i in 0..b {
            let want = lhs.i(i)?.contiguous()?.matmul(&rhs.i(i)?.contiguous()?)?;
            let (g, w) = (got.i(i)?.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
            for (a, e) in g.iter().zip(&w) {
                assert!((a - e).abs() < 1e-5, "{what}: batch {i}: {g:?} vs {w:?}");
            }
        }
        Ok(())
    };
    let (b, m, n, k) = (3, 4, 5, 8);
    let w = Tensor::from_vec(ramp(n * k, 0.7), (n, k), device)?;
    // (b, 1, k) with stride 1 on its size-one dim, against a broadcast rhs.
    let x = Tensor::from_vec(ramp(b * k, 0.3), (b, k, 1), device)?
        .transpose(1, 2)?
        .contiguous()?;
    assert!(x.is_contiguous());
    check(&x, &w.broadcast_left(b)?.t()?, "size-one lhs, broadcast rhs")?;
    // A transposed lhs (b, m, k) against a broadcast rhs.
    let xt = Tensor::from_vec(ramp(b * m * k, 0.2), (b, k, m), device)?.transpose(1, 2)?;
    check(&xt, &w.broadcast_left(b)?.t()?, "transposed lhs, broadcast rhs")?;
    // A broadcast lhs with m > 1 against a batched rhs, plain and transposed.
    let a = Tensor::from_vec(ramp(m * k, 0.9), (m, k), device)?.broadcast_left(b)?;
    let r = Tensor::from_vec(ramp(b * k * n, 0.4), (b, k, n), device)?;
    check(&a, &r, "broadcast lhs m>1")?;
    let rt = Tensor::from_vec(ramp(b * n * k, 0.4), (b, n, k), device)?.transpose(1, 2)?;
    check(&a, &rt, "broadcast lhs m>1, transposed rhs")?;
    // And m == 1 with a (b, k, 1) rhs whose size-one dim carries stride 1.
    let a1 = Tensor::from_vec(ramp(k, 0.9), (1, k), device)?.broadcast_left(b)?;
    let r1 = Tensor::from_vec(ramp(b * k, 0.5), (b, 1, k), device)?
        .transpose(1, 2)?
        .contiguous()?;
    check(&a1, &r1, "broadcast lhs m=1, size-one rhs")?;
    Ok(())
}
test_device!(
    broadcast_matmul_folds,
    broadcast_matmul_folds_cpu,
    broadcast_matmul_folds_gpu,
    broadcast_matmul_folds_metal,
    broadcast_matmul_folds_rocm
);
