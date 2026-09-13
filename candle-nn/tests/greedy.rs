use candle::{Device, Result, Tensor};
use candle_nn::sampling::GreedySampler;

fn exercise(device: &Device) -> Result<()> {
    // Cross reduction tiles, repeated prompt IDs, sign-aware penalty, and ties.
    let mut values = vec![-10f32; 128000];
    values[3] = 1.02;
    values[70000] = 1.;
    values[90000] = 1.;
    let mut s = GreedySampler::new(device, values.len(), &[3, 3], 1.05)?;
    let logits = Tensor::from_vec(values.clone(), (1, values.len()), device)?;
    assert_eq!(s.sample(&logits)?, 70000);
    assert_eq!(s.sample(&logits)?, 90000);
    assert_eq!(s.sample(&logits)?, 3);
    assert_eq!(
        logits.flatten_all()?.to_vec1::<f32>()?,
        values,
        "logits must be immutable"
    );
    let mut a = GreedySampler::new(device, 2, &[0, 0], 1.05)?;
    let negative = Tensor::new(&[-1f32, -1.02], device)?;
    assert_eq!(a.sample(&negative)?, 1);
    // A failed selection must not accept any token into the history.
    let mut a = GreedySampler::new(device, 3, &[], 1.05)?;
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(a
            .sample(&Tensor::new(&[1.02f32, bad, 1.], device)?)
            .is_err());
    }
    assert_eq!(a.sample(&Tensor::new(&[1.02f32, 1., 0.], device)?)?, 0);
    assert!(a.sample(&Tensor::new(&[1f32, 2.], device)?).is_err());
    for penalty in [0f32, -1., f32::NAN, f32::INFINITY] {
        assert!(GreedySampler::new(device, 3, &[], penalty).is_err());
    }
    assert!(GreedySampler::new(device, 3, &[3], 1.05).is_err());
    assert!(GreedySampler::new(device, 0, &[], 1.05).is_err());
    let strided = Tensor::zeros((3, 2), candle::DType::F32, device)?.transpose(0, 1)?;
    let mut stride_sampler = GreedySampler::new(device, 6, &[], 1.05)?;
    assert!(stride_sampler.sample(&strided).is_err());
    // Preserve the original raw-finite policy, including IEEE overflow after
    // a penalty. The daemon restricts penalties to 1..=2.
    let mut extreme = GreedySampler::new(device, 2, &[0, 1], 2.)?;
    assert_eq!(
        extreme.sample(&Tensor::new(&[-f32::MAX, -f32::MAX], device)?)?,
        0
    );
    let offset = Tensor::new(&[99f32, 0., 1., 1., 99.], device)?.narrow(0, 1, 3)?;
    let mut off = GreedySampler::new(device, 3, &[], 1.)?;
    assert_eq!(off.sample(&offset)?, 1);
    assert_eq!(off.sample(&offset)?, 1);
    for n in [1, 255, 256, 1023, 1024, 1025, 4097] {
        let mut values = vec![-5f32; n];
        values[n - 1] = -1.;
        let mut s = GreedySampler::new(device, n, &[], 1.05)?;
        assert_eq!(
            s.sample(&Tensor::from_vec(values.clone(), n, device)?)?,
            (n - 1) as u32
        );
        values[n - 1] = f32::NAN;
        assert!(s.sample(&Tensor::from_vec(values, n, device)?).is_err());
    }
    // An independent branch starts from prompt history, not another's output.
    let mut again = GreedySampler::new(device, 128000, &[3, 3], 1.05)?;
    assert_eq!(again.sample(&logits)?, 70000);
    Ok(())
}
#[test]
fn greedy_contract_cpu() -> Result<()> {
    exercise(&Device::Cpu)
}
#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires real ROCm; initialization errors must fail"]
fn greedy_contract_rocm() -> Result<()> {
    exercise(&Device::new_rocm(0)?)
}
