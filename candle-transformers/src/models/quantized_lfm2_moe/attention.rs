use candle::{bail, Result, Tensor};

/// Fold each KV head's query heads into GEMM's row dimension. K and V remain
/// shared (including their capacity-padded batch strides), so attention never
/// expands the cache from KV-head count to query-head count on CPU/ROCm.
pub(super) fn forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    neg_inf: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, seq, width) = q.dims4()?;
    let (kb, kv_heads, total, kw) = k.dims4()?;
    if batch != kb
        || width != kw
        || kv_heads == 0
        || heads == 0
        || seq == 0
        || width == 0
        || total == 0
        || !heads.is_multiple_of(kv_heads)
        || k.dims() != v.dims()
    {
        bail!("invalid LFM2 grouped attention dimensions")
    }
    let repeats = heads / kv_heads;
    if !q.device().is_cpu() && !q.device().is_rocm() {
        let k = crate::utils::repeat_kv(k.clone(), repeats)?;
        let v = crate::utils::repeat_kv(v.clone(), repeats)?;
        let scores = (q.matmul(&k.t()?)? / (width as f64).sqrt())?;
        return probabilities(scores, mask, neg_inf)?.matmul(&v.contiguous()?);
    }
    let grouped_q = q.reshape((batch, kv_heads, repeats * seq, width))?;
    let scores = (grouped_q.matmul(&k.t()?)? / (width as f64).sqrt())?
        .reshape((batch, heads, seq, total))?;
    let probabilities =
        probabilities(scores, mask, neg_inf)?.reshape((batch, kv_heads, repeats * seq, total))?;
    probabilities.matmul(v)?.reshape((batch, heads, seq, width))
}

fn probabilities(scores: Tensor, mask: Option<&Tensor>, neg_inf: &Tensor) -> Result<Tensor> {
    let scores = match mask {
        Some(mask) => super::masked_fill(&scores, &mask.broadcast_as(scores.shape())?, neg_inf)?,
        None => scores,
    };
    candle_nn::ops::softmax_last_dim(&scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{build_causal_mask, repeat_kv};
    use candle::{DType, Device};

    fn ramp(shape: (usize, usize, usize, usize), offset: f32, device: &Device) -> Result<Tensor> {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        Tensor::from_vec(
            (0..n).map(|i| ((i as f32 + offset) / 17.).sin()).collect(),
            shape,
            device,
        )
    }

    fn check(device: &Device) -> Result<()> {
        for (batch, heads, repeats, seq, past, width) in [
            (1, 2, 4, 1, 7, 8),
            (2, 2, 3, 3, 5, 8),
            (1, 2, 1, 5, 0, 8),
            (1, 8, 4, 1, 331, 64),
            (1, 8, 4, 17, 128, 64),
        ] {
            let total = past + seq;
            let q = ramp((batch, heads * repeats, seq, width), 3., device)?;
            // Padded cache capacity and a nonzero offset must not be compacted
            // into repeated heads merely to satisfy GEMM batch-stride handling.
            let k = ramp((batch, heads, total + 11, width), 7., device)?.narrow(2, 2, total)?;
            let v = ramp((batch, heads, total + 11, width), 11., device)?.narrow(2, 2, total)?;
            let mask = if seq > 1 {
                Some(build_causal_mask(seq, past, device)?)
            } else {
                None
            };
            let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
            let actual = forward(&q, &k, &v, mask.as_ref(), &neg_inf)?;
            let rk = repeat_kv(k.clone(), repeats)?;
            let rv = repeat_kv(v.clone(), repeats)?;
            let scores = (q.matmul(&rk.t()?)? / (width as f64).sqrt())?;
            let scores = match &mask {
                Some(m) => {
                    super::super::masked_fill(&scores, &m.broadcast_as(scores.shape())?, &neg_inf)?
                }
                None => scores,
            };
            let reference = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&rv.contiguous()?)?;
            assert_eq!(actual.dims(), reference.dims());
            let a = actual.flatten_all()?.to_vec1::<f32>()?;
            let r = reference.flatten_all()?.to_vec1::<f32>()?;
            for (a, b) in a.iter().zip(&r) {
                assert!(
                    (a - b).abs() < 2e-5,
                    "grouped attention {a} vs repeated {b}"
                );
            }
            // Independent scalar witness for the multi-batch/multi-token case.
            if batch == 2 {
                let qv = q.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                let kv = k.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                let vv = v.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                for b in 0..batch {
                    for h in 0..heads * repeats {
                        for t in 0..seq {
                            let visible = past + t + 1;
                            let scores = (0..visible)
                                .map(|j| {
                                    (0..width)
                                        .map(|d| {
                                            qv[((b * heads * repeats + h) * seq + t) * width + d]
                                                as f64
                                                * kv[((b * heads + h / repeats) * total + j)
                                                    * width
                                                    + d]
                                                    as f64
                                        })
                                        .sum::<f64>()
                                        / (width as f64).sqrt()
                                })
                                .collect::<Vec<_>>();
                            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let sum = scores.iter().map(|x| (x - max).exp()).sum::<f64>();
                            for d in 0..width {
                                let expected = scores
                                    .iter()
                                    .enumerate()
                                    .map(|(j, x)| {
                                        (x - max).exp() / sum
                                            * vv[((b * heads + h / repeats) * total + j) * width
                                                + d]
                                                as f64
                                    })
                                    .sum::<f64>();
                                assert!(
                                    (a[((b * heads * repeats + h) * seq + t) * width + d] as f64
                                        - expected)
                                        .abs()
                                        < 2e-5
                                );
                            }
                        }
                    }
                }
            }
        }
        let q = Tensor::zeros((1, 3, 1, 8), DType::F32, device)?;
        let kv = Tensor::zeros((1, 2, 1, 8), DType::F32, device)?;
        let neg = Tensor::new(f32::NEG_INFINITY, device)?;
        assert!(forward(&q, &kv, &kv, None, &neg).is_err());
        Ok(())
    }

    #[test]
    fn grouped_attention_matches_repeated_heads_and_scalar_reference() -> Result<()> {
        check(&Device::Cpu)
    }

    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires actual ROCm hardware; device failure is an error"]
    fn rocm_grouped_attention_matches_repeated_heads() -> Result<()> {
        check(&Device::new_rocm(0)?)
    }
}
