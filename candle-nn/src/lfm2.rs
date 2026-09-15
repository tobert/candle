//! Fused inference operations for LFM2's convolution, SwiGLU and MoE routing.
//! The specialized decode update follows the same decomposition as vLLM's
//! ShortConv/causal_conv1d_update, but writes fresh state for snapshot safety.
use candle::{bail, DType, Result, Tensor, D};
#[cfg(feature = "rocm")]
mod moe_rocm;
#[cfg(feature = "rocm")]
mod rocm;

/// Select experts using sigmoid(logits) + bias, then normalize the selected
/// unbiased sigmoid scores. Input is (tokens, experts); outputs are contiguous
/// (tokens, topk) U32 IDs and F32 weights. This is an inference-only operation.
/// ROCm fuses the 32-expert/top-4 case, preserving its bitonic tie ordering and
/// reduction tree. Other shapes/backends use the explicit tensor decomposition.
pub fn moe_route(logits: &Tensor, bias: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let (rows, experts) = logits.dims2()?;
    if rows == 0
        || topk == 0
        || topk > experts
        || bias.dims1()? != experts
        || logits.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !logits.device().same_device(bias.device())
    {
        bail!("invalid LFM2 MoE routing inputs")
    }
    #[cfg(feature = "rocm")]
    if matches!(logits.device(), candle::Device::Rocm(_)) && experts == 32 && topk == 4 {
        return moe_rocm::route(&logits.contiguous()?, &bias.contiguous()?);
    }
    let scores = crate::ops::sigmoid(logits)?;
    let ids = scores
        .broadcast_add(bias)?
        .arg_sort_last_dim(false)?
        .narrow(1, 0, topk)?
        .contiguous()?;
    let weights = scores.gather(&ids, 1)?;
    let weights = weights.broadcast_div(&(weights.sum_keepdim(1)? + 1e-6)?)?;
    Ok((ids, weights))
}

/// Weight and sum expert outputs (tokens, topk, hidden) into (tokens, hidden).
/// ROCm top-4 fuses multiplication and reduction without the intermediate tensor,
/// retaining separate F32 rounding and the original reduction tree. Inference only.
pub fn moe_combine(outputs: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (rows, topk, hidden) = outputs.dims3()?;
    if rows == 0
        || topk == 0
        || hidden == 0
        || weights.dims() != [rows, topk]
        || outputs.dtype() != DType::F32
        || weights.dtype() != DType::F32
        || !outputs.device().same_device(weights.device())
    {
        bail!("invalid LFM2 MoE combine inputs")
    }
    #[cfg(feature = "rocm")]
    if matches!(outputs.device(), candle::Device::Rocm(_)) && topk == 4 {
        return outputs
            .contiguous()?
            .apply_op2_no_bwd(&weights.contiguous()?, &moe_rocm::Combine);
    }
    outputs.broadcast_mul(&weights.unsqueeze(2)?)?.sum(1)
}

/// One gated convolution step. Input is (batch,1,3*hidden) in B,C,x order,
/// weights (hidden,k), previous state (batch,hidden,k). Both returned tensors
/// are immutable views of fresh storage on ROCm; previous state is untouched.
pub fn short_conv_step(
    input: &Tensor,
    weight: &Tensor,
    previous: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, s, three_h) = input.dims3()?;
    let (h, k) = weight.dims2()?;
    if s != 1
        || h == 0
        || k == 0
        || b == 0
        || three_h != 3 * h
        || previous.dims() != [b, h, k]
        || [input, weight, previous]
            .iter()
            .any(|t| t.dtype() != DType::F32 || !t.device().same_device(input.device()))
    {
        bail!("invalid gated short convolution step inputs")
    }
    #[cfg(feature = "rocm")]
    if matches!(input.device(), candle::Device::Rocm(_)) {
        let out = input.contiguous()?.apply_op3_no_bwd(
            &weight.contiguous()?,
            &previous.contiguous()?,
            &rocm::Step { b, h, k },
        )?;
        return Ok((
            out.narrow(0, 0, b * h)?.reshape((b, 1, h))?,
            out.narrow(0, b * h, b * h * k)?.reshape((b, h, k))?,
        ));
    }
    let bcx = input.transpose(1, 2)?;
    let bx = (bcx.narrow(1, 0, h)? * bcx.narrow(1, 2 * h, h)?)?;
    let combined = Tensor::cat(&[previous, &bx], 2)?;
    let mut sum = Tensor::zeros_like(&bx)?;
    for tap in 0..k {
        sum = (sum
            + combined
                .narrow(2, tap + 1, 1)?
                .broadcast_mul(&weight.narrow(1, tap, 1)?.unsqueeze(0)?)?)?;
    }
    let next = combined.narrow(2, 1, k)?.contiguous()?.copy()?;
    Ok((
        (bcx.narrow(1, h, h)? * sum)?
            .transpose(1, 2)?
            .contiguous()?,
        next,
    ))
}
/// Apply SiLU to the first half of each row and multiply by its second half.
pub fn swiglu(input: &Tensor) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    if input.elem_count() == 0
        || width == 0
        || !width.is_multiple_of(2)
        || input.dtype() != DType::F32
    {
        bail!("SwiGLU expects nonempty even-width f32 rows")
    }
    #[cfg(feature = "rocm")]
    if matches!(input.device(), candle::Device::Rocm(_)) {
        return input.contiguous()?.apply_op1_no_bwd(&rocm::SwiGlu);
    }
    crate::ops::swiglu(input)
}
