use candle::{DType, Device, Result, Tensor};
use candle_nn::lfm2::{moe_combine, moe_route};

fn reference_route(x: &Tensor, bias: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let scores = candle_nn::ops::sigmoid(x)?;
    let ids = scores
        .broadcast_add(bias)?
        .arg_sort_last_dim(false)?
        .narrow(1, 0, topk)?
        .contiguous()?;
    let w = scores.gather(&ids, 1)?;
    let w = w.broadcast_div(&(w.sum_keepdim(1)? + 1e-6)?)?;
    Ok((ids, w))
}

fn exercise(d: &Device) -> Result<()> {
    for (rows, experts, topk) in [(1, 32, 4), (17, 32, 4), (331, 32, 4), (3, 32, 2), (3, 7, 2)] {
        for tied in [false, true] {
            // Padded rows and nonzero offsets exercise normalization of views.
            let data: Vec<f32> = (0..(rows + 1) * (experts + 2))
                .map(|i| {
                    if tied {
                        (i % 3) as f32
                    } else {
                        (i as f32 * 0.317).sin() * 10.
                    }
                })
                .collect();
            let x = Tensor::from_vec(data, (rows + 1, experts + 2), d)?
                .narrow(0, 1, rows)?
                .narrow(1, 1, experts)?;
            let bias = Tensor::from_vec(
                (0..experts + 1)
                    .map(|i| if tied { 0. } else { (i % 5) as f32 * 0.4 })
                    .collect(),
                experts + 1,
                d,
            )?
            .narrow(0, 1, experts)?;
            let want = reference_route(&x, &bias, topk)?;
            let got = moe_route(&x, &bias, topk)?;
            assert_eq!(
                got.0.to_vec2::<u32>()?,
                want.0.to_vec2::<u32>()?,
                "expert IDs including tied ordering"
            );
            assert_eq!(
                got.1.to_vec2::<f32>()?,
                want.1.to_vec2::<f32>()?,
                "unbiased weights and reduction order"
            );
        }
    }
    for value in [-1000f32, 0., 1000.] {
        let x = Tensor::full(value, (2, 32), d)?;
        let bias = Tensor::zeros(32, DType::F32, d)?;
        let want = reference_route(&x, &bias, 4)?;
        let got = moe_route(&x, &bias, 4)?;
        assert_eq!(got.0.to_vec2::<u32>()?, want.0.to_vec2::<u32>()?);
        assert_eq!(got.1.to_vec2::<f32>()?, want.1.to_vec2::<f32>()?);
    }
    for (batch, slots, h) in [(1, 4, 2048), (17, 4, 33), (3, 2, 7)] {
        let x = Tensor::from_vec(
            (0..(batch + 1) * slots * (h + 2))
                .map(|i| (i as f32 * 0.271).sin())
                .collect(),
            (batch + 1, slots, h + 2),
            d,
        )?
        .narrow(0, 1, batch)?
        .narrow(2, 1, h)?;
        let w = Tensor::from_vec(
            (0..(batch + 1) * (slots + 1))
                .map(|i| (i as f32 * 0.417).cos())
                .collect(),
            (batch + 1, slots + 1),
            d,
        )?
        .narrow(0, 1, batch)?
        .narrow(1, 1, slots)?;
        let want = x.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
        assert_eq!(
            moe_combine(&x, &w)?.to_vec2::<f32>()?,
            want.to_vec2::<f32>()?
        );
    }
    // Cancellation exposes reassociation: GPU's existing tree is (a+c)+(b+d).
    // This differs from both a sequential sum and a contracted multiply-add.
    let x = Tensor::new(&[[[1e20f32], [1.], [-1e20], [1.]]], d)?;
    let w = Tensor::ones((1, 4), DType::F32, d)?;
    let want = x.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
    assert_eq!(
        moe_combine(&x, &w)?.to_vec2::<f32>()?,
        want.to_vec2::<f32>()?
    );
    let eps = f32::EPSILON;
    let x = Tensor::new(&[[[1. + eps], [0.], [-1.], [0.]]], d)?;
    let w = Tensor::new(&[[1. - eps, 0., 1., 0.]], d)?;
    assert_eq!(moe_combine(&x, &w)?.to_vec2::<f32>()?, vec![vec![0.]]);
    let x = Tensor::arange(0f32, 24., d)?.reshape((2, 4, 3))?;
    let w = Tensor::arange(0f32, 8., d)?.reshape((4, 2))?.t()?;
    assert!(!w.is_contiguous());
    assert_eq!(
        moe_combine(&x, &w)?.to_vec2::<f32>()?,
        x.broadcast_mul(&w.unsqueeze(2)?)?
            .sum(1)?
            .to_vec2::<f32>()?
    );
    Ok(())
}

#[test]
fn moe_ops_cpu() -> Result<()> {
    exercise(&Device::Cpu)
}

#[test]
fn moe_ops_reject_invalid_inputs() -> Result<()> {
    let d = &Device::Cpu;
    let x = Tensor::zeros((2, 32), DType::F32, d)?;
    let bias = Tensor::zeros(32, DType::F32, d)?;
    for k in [0, 33] {
        assert!(moe_route(&x, &bias, k).is_err());
    }
    assert!(moe_route(&x, &bias.narrow(0, 0, 31)?, 4).is_err());
    assert!(moe_route(&x.to_dtype(DType::F64)?, &bias, 4).is_err());
    assert!(moe_route(&Tensor::zeros((0, 32), DType::F32, d)?, &bias, 4).is_err());
    let x = Tensor::zeros((2, 4, 7), DType::F32, d)?;
    assert!(moe_combine(&x, &Tensor::zeros((2, 3), DType::F32, d)?).is_err());
    assert!(moe_combine(&x, &Tensor::zeros((2, 4), DType::F64, d)?).is_err());
    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires actual ROCm hardware"]
fn moe_ops_rocm() -> Result<()> {
    exercise(&Device::new_rocm(0)?)
}
