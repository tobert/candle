//! Stateful greedy selection with full-history sign-aware repetition penalty.
//!
//! Like vLLM's token-presence penalty masks and llama.cpp's backend argmax,
//! ROCm performs the vocabulary scan on device. The history belongs to one
//! evaluation; no Clone implementation can accidentally alias mutable state.
use candle::{bail, DType, Device, Result, Tensor};

pub struct GreedySampler {
    device: Device,
    vocab: usize,
    penalty: f32,
    state: State,
}
enum State {
    Host(Vec<bool>),
    #[cfg(feature = "rocm")]
    Rocm(Box<super::greedy_rocm::Sampler>),
}
impl GreedySampler {
    pub fn new(device: &Device, vocab: usize, history: &[u32], penalty: f32) -> Result<Self> {
        if vocab == 0 || vocab >= u32::MAX as usize || !penalty.is_finite() || penalty <= 0. {
            bail!("invalid greedy vocabulary size or repetition penalty")
        }
        let mut seen = vec![false; vocab];
        for &id in history {
            *seen
                .get_mut(id as usize)
                .ok_or_else(|| candle::Error::Msg("history token outside vocabulary".into()))? =
                true;
        }
        let state = match device {
            #[cfg(feature = "rocm")]
            Device::Rocm(dev) => {
                State::Rocm(Box::new(super::greedy_rocm::Sampler::new(dev, &seen)?))
            }
            // Explicit portable implementation for CPU/CUDA/Metal. ROCm never
            // retries a failed device operation through this host path.
            _ => State::Host(seen),
        };
        Ok(Self {
            device: device.clone(),
            vocab,
            penalty,
            state,
        })
    }
    /// Select and accept one token. Exact ties choose the smallest token ID.
    /// Raw nonfinite logits are errors and leave penalty history unchanged.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        if !logits.device().same_device(&self.device)
            || logits.dtype() != DType::F32
            || logits.elem_count() != self.vocab
            || !logits.is_contiguous()
        {
            bail!("greedy selection expects contiguous f32 logits on its device with matching vocabulary")
        }
        match &mut self.state {
            #[cfg(feature = "rocm")]
            State::Rocm(s) => s.sample(logits, self.penalty),
            State::Host(seen) => {
                let values = logits.flatten_all()?.to_vec1::<f32>()?;
                if values.iter().any(|v| !v.is_finite()) {
                    bail!("nonfinite model logits")
                }
                let adjusted = |i: usize| {
                    let v = values[i];
                    if !seen[i] {
                        v
                    } else if v < 0. {
                        v * self.penalty
                    } else {
                        v / self.penalty
                    }
                };
                let mut best = 0;
                for i in 1..values.len() {
                    if adjusted(i) > adjusted(best) {
                        best = i;
                    }
                }
                seen[best] = true;
                Ok(best as u32)
            }
        }
    }
}
