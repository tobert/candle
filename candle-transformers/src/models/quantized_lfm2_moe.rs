//! Quantized LFM2 MoE causal decoder (including LFM2.5-8B-A1B).
//!
//! Weights are immutable. State clones share immutable tensor storage; every
//! append allocates replacement tensors. No slice_set/scatter_set is permitted
//! on state storage. This is copy-on-write at the tensor replacement boundary,
//! not an append-optimized allocator. A failed forward leaves its input state
//! unchanged. Snapshots are process-local and tied to the loaded model instance.
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{bail, DType, Device, IndexOp, Module, Result, Tensor, D};
use std::sync::Arc;

fn route(logits: &Tensor, bias: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let (_, experts) = logits.dims2()?;
    if topk == 0 || topk > experts || bias.dims1()? != experts {
        bail!("invalid LFM2 MoE routing dimensions")
    }
    let scores = candle_nn::ops::sigmoid(logits)?;
    let ids = scores
        .broadcast_add(bias)?
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, topk)?
        .contiguous()?;
    let weights = scores.gather(&ids, D::Minus1)?;
    let weights = weights.broadcast_div(&(weights.sum_keepdim(D::Minus1)? + 1e-6)?)?;
    Ok((ids, weights))
}

/// Direct causal depthwise convolution, including cached multi-token suffixes.
/// State holds the last k gated inputs (not projected outputs).
fn causal_conv(bx: &Tensor, weight: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    let (batch, hidden, seq) = bx.dims3()?;
    let (channels, k) = weight.dims2()?;
    if k == 0 || seq == 0 || channels != hidden {
        bail!("invalid causal convolution shape")
    }
    let previous = match state {
        Some(s) => {
            if s.dims() != [batch, hidden, k] {
                bail!("invalid convolution state shape")
            }
            s.clone()
        }
        None => Tensor::zeros((batch, hidden, k), bx.dtype(), bx.device())?,
    };
    let combined = Tensor::cat(&[&previous, bx], 2)?;
    let mut out = Tensor::zeros_like(bx)?;
    for tap in 0..k {
        let values = combined.narrow(2, tap + 1, seq)?;
        let w = weight.narrow(1, tap, 1)?.unsqueeze(0)?;
        out = (out + values.broadcast_mul(&w)?)?;
    }
    // contiguous materializes this strided tail and does not retain the large
    // combined allocation. copy covers the single-channel contiguous-view case.
    let next = combined.narrow(2, seq, k)?.contiguous()?.copy()?;
    Ok((out, next))
}

#[derive(Debug, Clone)]
struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}
impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2
            .forward(&(candle_nn::ops::silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?)
    }
}

// CPU is an explicit reference implementation, not a GPU fallback.
#[derive(Debug)]
enum Experts {
    Quantized(Arc<QTensor>),
    Cpu(Tensor),
}
impl Experts {
    fn new(w: QTensor, device: &Device) -> Result<Self> {
        if device.is_cpu() {
            Ok(Self::Cpu(w.dequantize(device)?))
        } else {
            Ok(Self::Quantized(Arc::new(w)))
        }
    }
    fn forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(w) => w.indexed_moe_forward(x, ids),
            Self::Cpu(w) => {
                let (batch, slots) = ids.dims2()?;
                let (_, n, k) = w.dims3()?;
                let weights = w.index_select(&ids.flatten_all()?, 0)?;
                let x = x.broadcast_as((batch, slots, k))?.contiguous()?.reshape((
                    batch * slots,
                    k,
                    1,
                ))?;
                weights.matmul(&x)?.reshape((batch, slots, n))
            }
        }
    }
}
// Merge packed gate/up rows once at load, as in vLLM's w13 projection.
// Keep an explicit mixed-format implementation for GGUFs whose two matrices
// use different block types; never requantize weights to force a merge.
#[derive(Debug)]
enum ExpertGateUp {
    Merged(Experts),
    Separate { gate: Experts, up: Experts },
}
impl ExpertGateUp {
    fn new(gate: QTensor, up: QTensor, device: &Device) -> Result<Self> {
        if gate.dtype() == up.dtype() {
            Ok(Self::Merged(Experts::new(
                QTensor::cat(&[&gate, &up], 1)?,
                device,
            )?))
        } else {
            Ok(Self::Separate {
                gate: Experts::new(gate, device)?,
                up: Experts::new(up, device)?,
            })
        }
    }
    fn forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Merged(proj) => candle_nn::lfm2::swiglu(&proj.forward(x, ids)?),
            Self::Separate { gate, up } => {
                candle_nn::ops::silu(&gate.forward(x, ids)?)? * up.forward(x, ids)?
            }
        }
    }
}
#[derive(Debug)]
struct Moe {
    gate: QMatMul,
    bias: Tensor,
    gate_up: ExpertGateUp,
    down: Experts,
    topk: usize,
}
impl Moe {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat = x.reshape((b * s, h))?;
        let (ids, weights) = route(&self.gate.forward(&flat)?, &self.bias, self.topk)?;
        let x = flat.unsqueeze(1)?;
        let activated = self.gate_up.forward(&x, &ids)?;
        self.down
            .forward(&activated, &ids)?
            .broadcast_mul(&weights.unsqueeze(2)?)?
            .sum(1)?
            .reshape((b, s, h))
    }
}
#[derive(Debug)]
enum FeedForward {
    Dense(Mlp),
    Moe(Moe),
}
impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b, _n, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&y)
    }
}

#[derive(Debug)]
struct ShortConv {
    input: QMatMul,
    output: QMatMul,
    weight: Tensor,
}
impl ShortConv {
    fn forward(&self, x: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        let h = x.dim(2)?;
        let projected = self.input.forward(x)?;
        if x.dim(1)? == 1 {
            let previous = match state.as_ref() {
                Some(s) => s.clone(),
                None => Tensor::zeros((x.dim(0)?, h, self.weight.dim(1)?), x.dtype(), x.device())?,
            };
            let (out, next) =
                candle_nn::lfm2::short_conv_step(&projected, &self.weight, &previous)?;
            *state = Some(next);
            return self.output.forward(&out);
        }
        let bcx = projected.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, h)?;
        let c = bcx.narrow(1, h, h)?;
        let bx = (b * bcx.narrow(1, 2 * h, h)?)?;
        let (conv, next) = causal_conv(&bx, &self.weight, state.as_ref())?;
        *state = Some(next);
        self.output
            .forward(&(c * conv)?.transpose(1, 2)?.contiguous()?)
    }
}
#[derive(Debug)]
enum Operator {
    Attention(Box<AttentionLayer>),
    Conv(ShortConv),
}
#[derive(Debug)]
struct Layer {
    norm: RmsNorm,
    ffn_norm: RmsNorm,
    operator: Operator,
    ffn: FeedForward,
}

/// An opaque, branchable snapshot of the entire causal hybrid state.
/// Cloning shares immutable storage. It cannot be used with another model,
/// even when that model happens to have the same shape.
#[derive(Debug, Clone)]
pub struct State {
    owner: Arc<()>,
    len: usize,
    kv: Vec<Option<(Tensor, Tensor)>>,
    conv: Vec<Option<Tensor>>,
}
impl State {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug)]
pub struct Model {
    embedding: QMatMul,
    output: QMatMul,
    norm: RmsNorm,
    layers: Vec<Layer>,
    owner: Arc<()>,
    device: Device,
    context: usize,
    vocab: usize,
    hidden: usize,
}

impl Model {
    pub fn new_state(&self) -> State {
        State {
            owner: self.owner.clone(),
            len: 0,
            kv: vec![None; self.layers.len()],
            conv: vec![None; self.layers.len()],
        }
    }
    pub fn context_length(&self) -> usize {
        self.context
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Batch-one forward, returning next-token logits. Caller bounds prefill
    /// chunks; all validation happens before committing the new state.
    pub fn forward(&self, tokens: &[u32], state: &mut State) -> Result<Tensor> {
        if !Arc::ptr_eq(&state.owner, &self.owner) {
            bail!("snapshot belongs to another model")
        }
        let seq = tokens.len();
        let end = state
            .len
            .checked_add(seq)
            .ok_or_else(|| candle::Error::Msg("context length overflow".into()))?;
        if seq == 0 || end > self.context {
            bail!("empty input or context limit exceeded")
        }
        if tokens.iter().any(|&t| t as usize >= self.vocab) {
            bail!("token outside model vocabulary")
        }
        let mut next = state.clone();
        let ids = Tensor::from_slice(tokens, (1, seq), &self.device)?;
        let mask = if seq > 1 {
            Some(crate::utils::build_causal_mask(
                seq,
                state.len,
                &self.device,
            )?)
        } else {
            None
        };
        let mut x = self.embedding.embedding(&ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm.forward(&x)?;
            let y = match &layer.operator {
                Operator::Attention(a) => {
                    a.forward(&normed, mask.as_ref(), state.len, &mut next.kv[i])?
                }
                Operator::Conv(c) => c.forward(&normed, &mut next.conv[i])?,
            };
            x = (x + y)?;
            x = (&x + layer.ffn.forward(&layer.ffn_norm.forward(&x)?)?)?;
        }
        let x = self.norm.forward(&x)?.i((.., seq - 1, ..))?.contiguous()?;
        let logits = self.output.forward(&x)?;
        next.len = end;
        *state = next;
        Ok(logits)
    }

    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let get = |key: &str| {
            ct.metadata
                .get(key)
                .ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata {key}")))
        };
        if get("general.architecture")?.to_string()? != "lfm2moe" {
            bail!("expected lfm2moe GGUF architecture")
        }
        let number =
            |key: &str| -> Result<usize> { Ok(get(&format!("lfm2moe.{key}"))?.to_u32()? as usize) };
        let layers_n = number("block_count")?;
        let hidden = number("embedding_length")?;
        let heads = number("attention.head_count")?;
        let dense_n = number("leading_dense_block_count")?;
        let dense_ff = number("feed_forward_length")?;
        let experts = number("expert_count")?;
        let topk = number("expert_used_count")?;
        let expert_ff = number("expert_feed_forward_length")?;
        let context = number("context_length")?;
        let vocab = number("vocab_size")?;
        let k = number("shortconv.l_cache")?;
        if number("expert_gating_func")? != 2 {
            bail!("LFM2 MoE requires sigmoid expert gating")
        }
        let eps = get("lfm2moe.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let theta = get("lfm2moe.rope.freq_base")?.to_f32()?;
        let kv: Vec<usize> = match get("lfm2moe.attention.head_count_kv")? {
            gguf_file::Value::Array(v) => v
                .iter()
                .map(|n| match n {
                    gguf_file::Value::U32(v) => Ok(*v as usize),
                    gguf_file::Value::I32(v) if *v >= 0 => Ok(*v as usize),
                    _ => bail!("KV head count must be a nonnegative integer"),
                })
                .collect::<Result<_>>()?,
            _ => bail!("lfm2moe requires per-layer KV head counts"),
        };
        if hidden == 0
            || heads == 0
            || hidden % heads != 0
            || (hidden / heads) % 2 != 0
            || layers_n == 0
            || kv.len() != layers_n
            || dense_n > layers_n
            || dense_ff == 0
            || expert_ff == 0
            || experts == 0
            || topk == 0
            || topk > experts
            || k == 0
            || context == 0
            || context > u32::MAX as usize
            || vocab == 0
            || kv.iter().any(|&v| v > heads || (v > 0 && heads % v != 0))
            || !eps.is_finite()
            || eps <= 0.
            || !theta.is_finite()
            || theta <= 0.
        {
            bail!("invalid LFM2 MoE configuration")
        }
        let hd = hidden / heads;
        let (cos, sin) = precomput_freqs_cis(hd, theta, context, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let mut tensor = |name: &str, dims: &[usize]| -> Result<QTensor> {
            let info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("missing tensor {name}")))?;
            if info.shape.dims() != dims {
                bail!("{name}: expected {dims:?}, got {:?}", info.shape)
            }
            ct.tensor(reader, name, device)
        };
        let embedding = Arc::new(tensor("token_embd.weight", &[vocab, hidden])?);
        let output = if ct.tensor_infos.contains_key("output.weight") {
            QMatMul::from_qtensor(tensor("output.weight", &[vocab, hidden])?)?
        } else {
            QMatMul::from_arc(embedding.clone())?
        };
        let norm = RmsNorm::from_qtensor(tensor("token_embd_norm.weight", &[hidden])?, eps)?;
        let mut layers = Vec::with_capacity(layers_n);
        for (i, &nkv) in kv.iter().enumerate() {
            let mut t = |name: &str, dims: &[usize]| tensor(&format!("blk.{i}.{name}"), dims);
            let norm = RmsNorm::from_qtensor(t("attn_norm.weight", &[hidden])?, eps)?;
            let ffn_norm = RmsNorm::from_qtensor(t("ffn_norm.weight", &[hidden])?, eps)?;
            let ffn = if i < dense_n {
                FeedForward::Dense(Mlp {
                    w1: QMatMul::from_qtensor(t("ffn_gate.weight", &[dense_ff, hidden])?)?,
                    w2: QMatMul::from_qtensor(t("ffn_down.weight", &[hidden, dense_ff])?)?,
                    w3: QMatMul::from_qtensor(t("ffn_up.weight", &[dense_ff, hidden])?)?,
                })
            } else {
                FeedForward::Moe(Moe {
                    gate: QMatMul::from_qtensor(t("ffn_gate_inp.weight", &[experts, hidden])?)?,
                    bias: t("exp_probs_b.bias", &[experts])?.dequantize(device)?,
                    gate_up: ExpertGateUp::new(
                        t("ffn_gate_exps.weight", &[experts, expert_ff, hidden])?,
                        t("ffn_up_exps.weight", &[experts, expert_ff, hidden])?,
                        device,
                    )?,
                    down: Experts::new(
                        t("ffn_down_exps.weight", &[experts, hidden, expert_ff])?,
                        device,
                    )?,
                    topk,
                })
            };
            let operator = if nkv > 0 {
                Operator::Attention(Box::new(AttentionLayer {
                    wq: QMatMul::from_qtensor(t("attn_q.weight", &[hidden, hidden])?)?,
                    wk: QMatMul::from_qtensor(t("attn_k.weight", &[nkv * hd, hidden])?)?,
                    wv: QMatMul::from_qtensor(t("attn_v.weight", &[nkv * hd, hidden])?)?,
                    wo: QMatMul::from_qtensor(t("attn_output.weight", &[hidden, hidden])?)?,
                    q_norm: RmsNorm::from_qtensor(t("attn_q_norm.weight", &[hd])?, eps)?,
                    k_norm: RmsNorm::from_qtensor(t("attn_k_norm.weight", &[hd])?, eps)?,
                    n_head: heads,
                    n_kv_head: nkv,
                    head_dim: hd,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    span_rot: tracing::span!(tracing::Level::TRACE, "rope"),
                }))
            } else {
                Operator::Conv(ShortConv {
                    input: QMatMul::from_qtensor(t(
                        "shortconv.in_proj.weight",
                        &[3 * hidden, hidden],
                    )?)?,
                    output: QMatMul::from_qtensor(t(
                        "shortconv.out_proj.weight",
                        &[hidden, hidden],
                    )?)?,
                    weight: t("shortconv.conv.weight", &[hidden, k])?.dequantize(device)?,
                })
            };
            layers.push(Layer {
                norm,
                ffn_norm,
                operator,
                ffn,
            });
        }
        Ok(Self {
            embedding: QMatMul::from_arc(embedding)?,
            output,
            norm,
            layers,
            owner: Arc::new(()),
            device: device.clone(),
            context,
            vocab,
            hidden,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, neg_inf: &Tensor) -> Result<Tensor> {
    mask.where_cond(&neg_inf.broadcast_as(mask.shape())?, on_false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn routing_bias_selects_but_does_not_weight() -> Result<()> {
        let d = Device::Cpu;
        let logits = Tensor::new(&[[0f32, 2., -2.]], &d)?;
        let bias = Tensor::new(&[0f32, 0., 2.], &d)?;
        let (ids, weights) = route(&logits, &bias, 2)?;
        assert_eq!(ids.to_vec2::<u32>()?, vec![vec![2, 1]]);
        let w = weights.to_vec2::<f32>()?;
        let a = 1. / (1. + 2f32.exp());
        let b = 1. / (1. + (-2f32).exp());
        assert!((w[0][0] - a / (a + b + 1e-6)).abs() < 1e-6);
        assert!((w[0][1] - b / (a + b + 1e-6)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn convolution_chunks_preserve_history_and_isolate_branches() -> Result<()> {
        let d = Device::Cpu;
        let weight = Tensor::new(&[[1f32, 2., 3.]], &d)?;
        let bx = Tensor::new(&[[[1f32, 2., 3., 4., 5., 6., 7.]]], &d)?;
        let (full, _) = causal_conv(&bx, &weight, None)?;
        assert_eq!(
            full.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 8., 14., 20., 26., 32., 38.]
        );
        for split in 1..7 {
            let (_, prefix) = causal_conv(&bx.narrow(2, 0, split)?, &weight, None)?;
            let frozen = prefix.flatten_all()?.to_vec1::<f32>()?;
            let (suffix, _) =
                causal_conv(&bx.narrow(2, split, 7 - split)?, &weight, Some(&prefix))?;
            assert_eq!(
                suffix.flatten_all()?.to_vec1::<f32>()?,
                full.narrow(2, split, 7 - split)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
            let _ = causal_conv(&Tensor::new(&[[[99f32, 88.]]], &d)?, &weight, Some(&prefix))?;
            assert_eq!(prefix.flatten_all()?.to_vec1::<f32>()?, frozen);
        }
        Ok(())
    }
    #[test]
    fn failure_after_all_layers_does_not_commit_state() -> Result<()> {
        let mut f = std::io::Cursor::new(include_bytes!("../../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = gguf_file::Content::read(&mut f)?;
        let mut model = Model::from_gguf(ct, &mut f, &Device::Cpu)?;
        let mut prefix = model.new_state();
        let _ = model.forward(&[1, 2, 3], &mut prefix)?;
        let saved = prefix.clone();
        let good_output = model.output.clone();
        model.output = QMatMul::Tensor(Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?);
        assert!(model.forward(&[4, 5], &mut prefix).is_err());
        assert_eq!(prefix.len(), 3);
        model.output = good_output;
        let a = model.forward(&[4, 5], &mut prefix)?;
        let b = model.forward(&[4, 5], &mut saved.clone())?;
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires actual ROCm hardware"]
    fn merged_gate_up_matches_separate_quantized_operations() -> Result<()> {
        use candle::quantized::GgmlDType;
        let dev = Device::new_rocm(0)?;
        let a = Tensor::from_vec(
            (0..4 * 96 * 256)
                .map(|i| (i as f32 / 113.).sin())
                .collect::<Vec<_>>(),
            (4, 96, 256),
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            (0..4 * 96 * 256)
                .map(|i| (i as f32 / 97.).cos())
                .collect::<Vec<_>>(),
            (4, 96, 256),
            &Device::Cpu,
        )?;
        for (gate_type, up_type) in [
            (GgmlDType::Q5K, GgmlDType::Q5K),
            (GgmlDType::Q6K, GgmlDType::Q6K),
            (GgmlDType::Q5K, GgmlDType::Q6K),
        ] {
            for batch in [1, 17] {
                let gate = QTensor::quantize_onto(&a, gate_type, &dev)?;
                let up = QTensor::quantize_onto(&b, up_type, &dev)?;
                let input = Tensor::from_vec(
                    (0..batch * 256)
                        .map(|i| (i as f32 / 127.).cos())
                        .collect::<Vec<_>>(),
                    (batch, 1, 256),
                    &dev,
                )?;
                let ids = Tensor::from_vec(
                    (0..batch * 2).map(|i| (i % 4) as u32).collect::<Vec<_>>(),
                    (batch, 2),
                    &dev,
                )?;
                let reference = (candle_nn::ops::silu(&gate.indexed_moe_forward(&input, &ids)?)?
                    * up.indexed_moe_forward(&input, &ids)?)?;
                let packed = ExpertGateUp::new(gate, up, &dev)?;
                assert_eq!(
                    matches!(packed, ExpertGateUp::Merged(_)),
                    gate_type == up_type
                );
                let actual = packed.forward(&input, &ids)?;
                assert_eq!(
                    actual.flatten_all()?.to_vec1::<f32>()?,
                    reference.flatten_all()?.to_vec1::<f32>()?,
                    "{gate_type:?}/{up_type:?} batch={batch}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires an actual ROCm GPU; device failure is an error"]
    fn rocm_quantized_experts_match_dequantized_reference() -> Result<()> {
        use candle::quantized::GgmlDType;
        let dev = Device::new_rocm(0)?;
        for dtype in [GgmlDType::Q5K, GgmlDType::Q6K] {
            let data: Vec<f32> = (0..4 * 96 * 256).map(|i| (i as f32 / 113.).sin()).collect();
            let dense = Tensor::from_vec(data, (4, 96, 256), &Device::Cpu)?;
            let q = QTensor::quantize(&dense, dtype)?;
            let cpu = Experts::Cpu(q.dequantize(&Device::Cpu)?);
            let gpu = Experts::new(QTensor::quantize_onto(&dense, dtype, &dev)?, &dev)?;
            let ids = Tensor::new(&[[3u32, 1], [2, 0]], &Device::Cpu)?;
            for slots in [1, 2] {
                let x = Tensor::from_vec(
                    (0..2 * slots * 256)
                        .map(|i| (i as f32 / 17.).cos())
                        .collect::<Vec<_>>(),
                    (2, slots, 256),
                    &Device::Cpu,
                )?;
                let reference = cpu.forward(&x, &ids)?.flatten_all()?.to_vec1::<f32>()?;
                let actual = gpu
                    .forward(&x.to_device(&dev)?, &ids.to_device(&dev)?)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let scale = reference.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let max_delta = reference
                    .iter()
                    .zip(actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_delta < 0.02 * scale,
                    "{dtype:?} slots={slots} delta={max_delta} scale={scale}"
                );
            }
        }
        Ok(())
    }
}
