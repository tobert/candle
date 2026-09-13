use candle::{Device, Result, Tensor};
use candle_nn::lfm2::{short_conv_step, swiglu};
fn exercise(d: &Device) -> Result<()> {
    let h = 33;
    let packed = Tensor::from_vec(
        (0..2 * 3 * h)
            .map(|i| (i as f32 / 29.).sin())
            .collect::<Vec<_>>(),
        (2, 1, 3 * h),
        d,
    )?;
    for k in [1, 3, 5] {
        let weight = Tensor::from_vec(
            (0..h * k)
                .map(|i| (i as f32 / 17.).cos())
                .collect::<Vec<_>>(),
            (h, k),
            d,
        )?;
        let state = Tensor::from_vec(
            (0..2 * h * k).map(|i| i as f32 / 113.).collect::<Vec<_>>(),
            (2, h, k),
            d,
        )?;
        let old = state.flatten_all()?.to_vec1::<f32>()?;
        // Compare to the old Tensor decomposition on the same device, including
        // ROCm. This witnesses the prefill/decode rounding boundary directly.
        let bcx = packed.transpose(1, 2)?;
        let bx = (bcx.narrow(1, 0, h)? * bcx.narrow(1, 2 * h, h)?)?;
        let combined = Tensor::cat(&[&state, &bx], 2)?;
        let mut sum = Tensor::zeros_like(&bx)?;
        for tap in 0..k {
            sum = (sum
                + combined
                    .narrow(2, tap + 1, 1)?
                    .broadcast_mul(&weight.narrow(1, tap, 1)?.unsqueeze(0)?)?)?;
        }
        let tensor_reference = (bcx.narrow(1, h, h)? * sum)?
            .transpose(1, 2)?
            .contiguous()?;

        let (out, next) = short_conv_step(&packed, &weight, &state)?;
        assert_eq!(
            out.flatten_all()?.to_vec1::<f32>()?,
            tensor_reference.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(short_conv_step(&packed, &weight, &state.narrow(1, 0, h - 1)?).is_err());

        let input = packed.flatten_all()?.to_vec1::<f32>()?;
        let w = weight.flatten_all()?.to_vec1::<f32>()?;
        let mut want = Vec::new();
        let mut next_want = old.clone();
        for b in 0..2 {
            for c in 0..h {
                let bx = input[b * 3 * h + c] * input[b * 3 * h + 2 * h + c];
                let mut sum = 0f32;
                for j in 0..k {
                    let x = if j + 1 < k {
                        old[(b * h + c) * k + j + 1]
                    } else {
                        bx
                    };
                    let product = x * w[c * k + j];
                    sum += product;
                    next_want[(b * h + c) * k + j] = x;
                }
                want.push(sum * input[b * 3 * h + h + c]);
            }
        }
        assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, want);
        assert_eq!(next.flatten_all()?.to_vec1::<f32>()?, next_want);
        assert_eq!(state.flatten_all()?.to_vec1::<f32>()?, old);
        // Returned state is a view with a nonzero offset on ROCm. It must be
        // usable as the next input, not just readable back by the test.
        let next_cpu = next.to_device(&Device::Cpu)?;
        let want_next = short_conv_step(
            &packed.to_device(&Device::Cpu)?,
            &weight.to_device(&Device::Cpu)?,
            &next_cpu,
        )?
        .0;
        let actual_next = short_conv_step(&packed, &weight, &next)?.0;
        assert_eq!(
            actual_next.flatten_all()?.to_vec1::<f32>()?,
            want_next.flatten_all()?.to_vec1::<f32>()?
        );

        let (again, _) = short_conv_step(&packed, &weight, &state)?;
        assert_eq!(again.flatten_all()?.to_vec1::<f32>()?, want);
    }
    let input = Tensor::from_vec(
        (0..3 * 2 * 257)
            .map(|i| i as f32 / 131. - 5.)
            .collect::<Vec<_>>(),
        (3, 514),
        d,
    )?;
    let want = (input.narrow(1, 0, 257)?.silu()? * input.narrow(1, 257, 257)?)?;
    let out = swiglu(&input)?;
    assert_eq!(
        out.flatten_all()?.to_vec1::<f32>()?,
        want.flatten_all()?.to_vec1::<f32>()?
    );
    assert!(swiglu(&Tensor::zeros((2, 3), candle::DType::F32, d)?).is_err());
    Ok(())
}
#[test]
fn fused_lfm2_cpu() -> Result<()> {
    exercise(&Device::Cpu)
}
#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires actual ROCm hardware"]
fn fused_lfm2_rocm() -> Result<()> {
    exercise(&Device::new_rocm(0)?)
}
