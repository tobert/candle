//! Fused inference operations for LFM2's gated short convolution and SwiGLU.
//! The specialized decode update follows the same decomposition as vLLM's
//! ShortConv/causal_conv1d_update, but writes fresh state for snapshot safety.
use candle::{bail, DType, Result, Tensor, D};
#[cfg(feature = "rocm")]
mod rocm;

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
