use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Result, Tensor,
};
#[test]
fn concatenate_packed_rows_preserves_expert_boundaries_and_quantization() -> Result<()> {
    for dtype in [GgmlDType::F32, GgmlDType::Q5K, GgmlDType::Q6K] {
        let a = Tensor::from_vec(
            (0..2 * 3 * 256)
                .map(|i| (i as f32 / 97.).sin())
                .collect::<Vec<_>>(),
            (2, 3, 256),
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            (0..2 * 5 * 256)
                .map(|i| (i as f32 / 71.).cos())
                .collect::<Vec<_>>(),
            (2, 5, 256),
            &Device::Cpu,
        )?;
        let a = QTensor::quantize(&a, dtype)?;
        let b = QTensor::quantize(&b, dtype)?;
        let frozen = a.data()?.into_owned();
        let packed = QTensor::cat(&[&a, &b], 1)?;
        assert_eq!(packed.dtype(), dtype);
        assert_eq!(packed.shape().dims(), &[2, 8, 256]);
        let want = Tensor::cat(
            &[a.dequantize(&Device::Cpu)?, b.dequantize(&Device::Cpu)?],
            1,
        )?;
        assert_eq!(
            packed
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(a.data()?.as_ref(), frozen);
        assert!(QTensor::cat(&[&a, &b], 0).is_err());
        assert!(QTensor::cat(&[&a, &a], 2).is_err());
    }
    let x = Tensor::zeros((2, 256), candle_core::DType::F32, &Device::Cpu)?;
    let a = QTensor::quantize(&x, GgmlDType::Q5K)?;
    let b = QTensor::quantize(&x, GgmlDType::Q6K)?;
    assert!(QTensor::cat(&[&a, &b], 0).is_err());
    assert!(QTensor::cat(&[], 0).is_err());
    Ok(())
}
