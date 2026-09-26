//! Quantized LFM2 MoE causal decoder (including LFM2.5-8B-A1B).
//!
//! Weights and committed state prefixes are immutable. CPU/ROCm KV appends reserve fresh
//! tail space in growable buffers; conflicting branches copy their prefix first.
//! A failed forward leaves its input state's visible data and position unchanged.
//! Snapshots are process-local and tied to the loaded model instance.
mod attention;
mod batch;
mod kv_cache;

use self::kv_cache::KvCache;
use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{bail, DType, Device, IndexOp, Module, Result, Tensor, D};
use std::sync::Arc;

fn route(logits: &Tensor, bias: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    candle_nn::lfm2::moe_route(logits, bias, topk)
}

/// Read-only taps on one [`Model::forward_observed`] call, for examining the
/// model rather than serving it. Tensors stay on the model's device and are
/// the very values the forward goes on to use; copying to the host, and the
/// device synchronisation that costs, is the observer's decision. An error
/// fails the forward, which then commits no state.
pub trait Observer {
    /// Token embeddings entering layer 0: `(1, seq, hidden)`.
    fn embedding(&mut self, _x: &Tensor) -> Result<()> {
        Ok(())
    }
    /// The residual stream leaving `layer`, feed-forward included:
    /// `(1, seq, hidden)`. [`Model::project`] reads it as logits.
    fn residual(&mut self, _layer: usize, _x: &Tensor) -> Result<()> {
        Ok(())
    }
    /// One expert layer's routing. `logits` `(seq, experts)` are the raw router
    /// outputs before the sigmoid and the selection bias; `ids` `(seq, topk)`
    /// the chosen experts; `weights` `(seq, topk)` their normalised, unbiased
    /// combine weights. Dense layers never call this.
    fn routing(
        &mut self,
        _layer: usize,
        _logits: &Tensor,
        _ids: &Tensor,
        _weights: &Tensor,
    ) -> Result<()> {
        Ok(())
    }
}

/// The write side of [`Observer`]: replacement selection biases for chosen
/// expert layers, for one [`Model::forward_steered`] call.
///
/// A router chooses its experts by `sigmoid(logit) + bias` and weights them by
/// `sigmoid(logit)` alone, so replacing the bias is the whole vocabulary of
/// routing intervention: a large negative entry knocks an expert out, a large
/// positive one forces it in, a small delta nudges it, and in every case the
/// chosen experts are still weighted by their own scores. Layers not named keep
/// their trained bias. An empty `Steering` is exactly [`Model::forward`].
///
/// This is an instrument for examining a model. A state produced under steering
/// is a different computation from one that was not; nothing here records which
/// is which, so a caller that caches states must key them by their steering.
#[derive(Debug, Default, Clone)]
pub struct Steering {
    bias: std::collections::BTreeMap<usize, Tensor>,
}
impl Steering {
    /// Use `bias` `(experts,)` in place of `layer`'s selection bias.
    pub fn set_bias(&mut self, layer: usize, bias: Tensor) {
        self.bias.insert(layer, bias);
    }
    pub fn is_empty(&self) -> bool {
        self.bias.is_empty()
    }
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

#[cfg(all(test, feature = "rocm"))]
thread_local! {
    // Observe model wiring without exposing counters or synchronization in serving.
    static GROUPED_ROUTE_CALLS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

// One forward's routing decision. Grouped ROCm gate/up/down share its immutable
// packed assignments and scratch; decode and other backends keep indexed IDs.
struct ExpertRouting<'a> {
    ids: &'a Tensor,
    #[cfg(feature = "rocm")]
    grouped: Option<candle::quantized::rocm::GroupedMoeRouting>,
}
impl<'a> ExpertRouting<'a> {
    fn new(ids: &'a Tensor) -> Self {
        Self {
            ids,
            #[cfg(feature = "rocm")]
            grouped: None,
        }
    }
}

// CPU is an explicit reference implementation, not a GPU fallback.
#[derive(Debug)]
enum Experts {
    Quantized(Arc<QTensor>),
    Cpu(Tensor),
}
impl Experts {
    #[cfg(feature = "rocm")]
    fn supports_grouped(&self, batch: usize, topk: usize) -> bool {
        match self {
            Self::Quantized(w) => w.supports_grouped_moe(batch, topk),
            Self::Cpu(_) => false,
        }
    }
    fn new(w: QTensor, device: &Device) -> Result<Self> {
        if device.is_cpu() {
            Ok(Self::Cpu(w.dequantize(device)?))
        } else {
            Ok(Self::Quantized(Arc::new(w)))
        }
    }
    fn forward(&self, x: &Tensor, routing: &ExpertRouting<'_>) -> Result<Tensor> {
        match self {
            Self::Quantized(w) => {
                #[cfg(feature = "rocm")]
                if let Some(prepared) = &routing.grouped {
                    #[cfg(test)]
                    GROUPED_ROUTE_CALLS.with(|c| {
                        let (built, used) = c.get();
                        c.set((built, used + 1));
                    });
                    return w.grouped_moe_forward(x, prepared);
                }
                w.indexed_moe_forward(x, routing.ids)
            }
            Self::Cpu(w) => {
                let ids = routing.ids;
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
    #[cfg(feature = "rocm")]
    fn supports_grouped(&self, batch: usize, topk: usize) -> bool {
        match self {
            Self::Merged(w) => w.supports_grouped(batch, topk),
            Self::Separate { gate, up } => {
                gate.supports_grouped(batch, topk) && up.supports_grouped(batch, topk)
            }
        }
    }
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
    fn forward(&self, x: &Tensor, routing: &ExpertRouting<'_>) -> Result<Tensor> {
        match self {
            Self::Merged(proj) => candle_nn::lfm2::swiglu(&proj.forward(x, routing)?),
            Self::Separate { gate, up } => {
                candle_nn::ops::silu(&gate.forward(x, routing)?)? * up.forward(x, routing)?
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
    fn forward(
        &self,
        x: &Tensor,
        tap: Option<(usize, &mut dyn Observer)>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat = x.reshape((b * s, h))?;
        let logits = self.gate.forward(&flat)?;
        let (ids, weights) = route(&logits, bias.unwrap_or(&self.bias), self.topk)?;
        if let Some((layer, observer)) = tap {
            observer.routing(layer, &logits, &ids, &weights)?;
        }
        let routing = ExpertRouting::new(&ids);
        #[cfg(feature = "rocm")]
        let routing = if self.gate_up.supports_grouped(b * s, self.topk)
            && self.down.supports_grouped(b * s, self.topk)
        {
            #[cfg(test)]
            GROUPED_ROUTE_CALLS.with(|c| {
                let (built, used) = c.get();
                c.set((built + 1, used));
            });
            ExpertRouting {
                ids: &ids,
                grouped: Some(candle::quantized::rocm::GroupedMoeRouting::new(
                    &ids,
                    self.bias.dims1()?,
                )?),
            }
        } else {
            routing
        };
        let x = flat.unsqueeze(1)?;
        let activated = self.gate_up.forward(&x, &routing)?;
        candle_nn::lfm2::moe_combine(&self.down.forward(&activated, &routing)?, &weights)?
            .reshape((b, s, h))
    }
}
#[derive(Debug)]
enum FeedForward {
    Dense(Mlp),
    Moe(Moe),
}
impl FeedForward {
    fn forward(
        &self,
        x: &Tensor,
        tap: Option<(usize, &mut dyn Observer)>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x, tap, bias),
        }
    }
}

// Merge compatible packed projections at load time, following vLLM's QKV
// projection layout. Mixed-format V remains separate without requantization.
#[derive(Debug, Clone)]
enum QkvProjection {
    Fused {
        weight: QMatMul,
        q: usize,
        kv: usize,
    },
    Qk {
        weight: QMatMul,
        value: QMatMul,
        q: usize,
        kv: usize,
    },
    Separate {
        query: QMatMul,
        key: QMatMul,
        value: QMatMul,
    },
}
impl QkvProjection {
    fn new(query: QTensor, key: QTensor, value: QTensor) -> Result<Self> {
        let (q, k) = query.shape().dims2()?;
        let (kv, kk) = key.shape().dims2()?;
        if q == 0 || kv == 0 || k == 0 || kk != k || value.shape().dims() != [kv, k] {
            bail!("incompatible LFM2 QKV projection dimensions")
        }
        if query.dtype() == key.dtype() && key.dtype() == value.dtype() {
            Ok(Self::Fused {
                weight: QMatMul::from_qtensor(QTensor::cat(&[&query, &key, &value], 0)?)?,
                q,
                kv,
            })
        } else if query.dtype() == key.dtype() {
            Ok(Self::Qk {
                weight: QMatMul::from_qtensor(QTensor::cat(&[&query, &key], 0)?)?,
                value: QMatMul::from_qtensor(value)?,
                q,
                kv,
            })
        } else {
            Ok(Self::Separate {
                query: QMatMul::from_qtensor(query)?,
                key: QMatMul::from_qtensor(key)?,
                value: QMatMul::from_qtensor(value)?,
            })
        }
    }
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Fused { weight, q, kv } => {
                let y = weight.forward(x)?;
                Ok((
                    y.narrow(D::Minus1, 0, *q)?,
                    y.narrow(D::Minus1, *q, *kv)?,
                    y.narrow(D::Minus1, q + kv, *kv)?,
                ))
            }
            Self::Qk {
                weight,
                value,
                q,
                kv,
            } => {
                let y = weight.forward(x)?;
                Ok((
                    y.narrow(D::Minus1, 0, *q)?,
                    y.narrow(D::Minus1, *q, *kv)?,
                    value.forward(x)?,
                ))
            }
            Self::Separate { query, key, value } => {
                Ok((query.forward(x)?, key.forward(x)?, value.forward(x)?))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    qkv: QkvProjection,
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
        kv_cache: &mut Option<KvCache>,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = xs.dims3()?;

        let (q, k, v) = self.qkv.forward(xs)?;

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

        if kv_cache.as_ref().map_or(0, KvCache::len) != index_pos {
            bail!("LFM2 KV position does not match state")
        }
        let next = KvCache::append(kv_cache.as_ref(), &k, &v, self.cos.dim(0)?)?;
        let (k, v) = next.current()?;
        *kv_cache = Some(next);

        let y = attention::forward(&q, &k, &v, mask, &self.neg_inf)?;

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
    fn forward(&self, x: &Tensor, state: &mut Option<ConvState>) -> Result<Tensor> {
        let h = x.dim(2)?;
        let projected = self.input.forward(x)?;
        let previous = state.as_ref().map(ConvState::row).transpose()?;
        if x.dim(1)? == 1 {
            let previous = match previous {
                Some(s) => s,
                None => Tensor::zeros((x.dim(0)?, h, self.weight.dim(1)?), x.dtype(), x.device())?,
            };
            let (out, next) =
                candle_nn::lfm2::short_conv_step(&projected, &self.weight, &previous)?;
            *state = Some(ConvState::single(next));
            return self.output.forward(&out);
        }
        let bcx = projected.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, h)?;
        let c = bcx.narrow(1, h, h)?;
        let bx = (b * bcx.narrow(1, 2 * h, h)?)?;
        let (conv, next) = causal_conv(&bx, &self.weight, previous.as_ref())?;
        *state = Some(ConvState::single(next));
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

/// What one layer is made of, for labelling an [`Observer`]'s taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerInfo {
    /// Full attention; otherwise the gated short convolution.
    pub attention: bool,
    /// `(experts, experts used per token)`; `None` for a dense feed-forward.
    pub experts: Option<(usize, usize)>,
}

/// An opaque, branchable snapshot of the entire causal hybrid state.
/// Cloning shares immutable prefixes of append-only storage. It cannot be used
/// with another model, even when that model happens to have the same shape.
#[derive(Debug, Clone)]
pub struct State {
    owner: Arc<()>,
    len: usize,
    kv: Vec<Option<KvCache>>,
    conv: Vec<Option<ConvState>>,
}

/// One sequence's convolution state `(1, hidden, k)`: row `row` of the
/// immutable `(rows, hidden, k)` tensor the step that wrote it produced. A
/// batched step keeps the whole tensor so that the next batched step over the
/// same states in the same order reads it as is, instead of gathering its rows
/// again. A surviving state keeps its batch's tensor alive; it is small.
#[derive(Debug, Clone)]
struct ConvState {
    rows: Tensor,
    row: usize,
}
impl ConvState {
    fn single(state: Tensor) -> Self {
        Self {
            rows: state,
            row: 0,
        }
    }
    fn row(&self) -> Result<Tensor> {
        if self.rows.dim(0)? == 1 {
            Ok(self.rows.clone())
        } else {
            self.rows.narrow(0, self.row, 1)
        }
    }
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
    /// Whether a snapshot belongs to this loaded model instance. Useful when
    /// reusing saved logits without immediately executing another forward.
    pub fn owns_state(&self, state: &State) -> bool {
        Arc::ptr_eq(&state.owner, &self.owner)
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
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// An expert layer's selection bias `(experts,)`: added to the sigmoid
    /// scores to CHOOSE experts, never to weight them. `None` for a dense layer.
    pub fn router_bias(&self, layer: usize) -> Option<&Tensor> {
        match &self.layers.get(layer)?.ffn {
            FeedForward::Dense(_) => None,
            FeedForward::Moe(m) => Some(&m.bias),
        }
    }
    pub fn layers(&self) -> Vec<LayerInfo> {
        self.layers
            .iter()
            .map(|l| LayerInfo {
                attention: matches!(l.operator, Operator::Attention(_)),
                experts: match &l.ffn {
                    FeedForward::Dense(_) => None,
                    FeedForward::Moe(m) => Some((m.bias.dims()[0], m.topk)),
                },
            })
            .collect()
    }

    /// Batch-one forward, returning next-token logits. Caller bounds prefill
    /// chunks; all validation happens before committing the new state.
    pub fn forward(&self, tokens: &[u32], state: &mut State) -> Result<Tensor> {
        self.run(tokens, state, None, None)
    }

    /// [`Model::forward`] with the named routers' selection biases replaced for
    /// this call, and optionally observed; the observer sees the routing that
    /// actually ran. Steering a layer that has no router, or with a bias of the
    /// wrong shape, dtype or device, is refused before anything runs.
    pub fn forward_steered(
        &self,
        tokens: &[u32],
        state: &mut State,
        steering: &Steering,
        observer: Option<&mut dyn Observer>,
    ) -> Result<Tensor> {
        for (&layer, bias) in &steering.bias {
            let own = self.router_bias(layer).ok_or_else(|| {
                candle::Error::Msg(format!("layer {layer} has no router to steer"))
            })?;
            if bias.dims() != own.dims()
                || bias.dtype() != own.dtype()
                || !bias.device().same_device(own.device())
            {
                bail!("steering bias for layer {layer} must match the router's own: {own:?}")
            }
        }
        self.run(tokens, state, observer, Some(steering))
    }

    /// [`Model::forward`] with `observer` called at each tap. It executes the
    /// same operations in the same order, so logits and committed state are
    /// bit-identical to the unobserved call.
    pub fn forward_observed(
        &self,
        tokens: &[u32],
        state: &mut State,
        observer: &mut dyn Observer,
    ) -> Result<Tensor> {
        self.run(tokens, state, Some(observer), None)
    }

    /// The logit lens: read a residual stream `(1, seq, hidden)` from any depth
    /// through the model's own final norm and output projection, giving
    /// `(1, seq, vocab)`. On the last layer's residual this is exactly what
    /// [`Model::forward`] returns for the final position.
    pub fn project(&self, residual: &Tensor) -> Result<Tensor> {
        let (batch, _, hidden) = residual.dims3()?;
        if batch != 1 || hidden != self.hidden {
            bail!("residual must be (1, seq, {})", self.hidden)
        }
        self.output.forward(&self.norm.forward(residual)?)
    }

    fn run(
        &self,
        tokens: &[u32],
        state: &mut State,
        mut observer: Option<&mut dyn Observer>,
        steering: Option<&Steering>,
    ) -> Result<Tensor> {
        if !self.owns_state(state) {
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
        if let Some(o) = observer.as_deref_mut() {
            o.embedding(&x)?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm.forward(&x)?;
            let y = match &layer.operator {
                Operator::Attention(a) => {
                    a.forward(&normed, mask.as_ref(), state.len, &mut next.kv[i])?
                }
                Operator::Conv(c) => c.forward(&normed, &mut next.conv[i])?,
            };
            x = (x + y)?;
            // A match, not `map`: the reborrow has to pass through a coercion
            // site to shorten the trait object's lifetime for this iteration.
            let tap: Option<(usize, &mut dyn Observer)> = match &mut observer {
                Some(o) => Some((i, &mut **o)),
                None => None,
            };
            let bias = steering.and_then(|s| s.bias.get(&i));
            x = (&x + layer.ffn.forward(&layer.ffn_norm.forward(&x)?, tap, bias)?)?;
            if let Some(o) = observer.as_deref_mut() {
                o.residual(i, &x)?;
            }
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
                    qkv: QkvProjection::new(
                        t("attn_q.weight", &[hidden, hidden])?,
                        t("attn_k.weight", &[nkv * hd, hidden])?,
                        t("attn_v.weight", &[nkv * hd, hidden])?,
                    )?,
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
    fn check_failed_forward(inside_conv: bool) -> Result<()> {
        let device = &Device::Cpu;
        fn output_to_break(model: &mut Model, inside_conv: bool) -> &mut QMatMul {
            if inside_conv {
                match &mut model.layers[2].operator {
                    Operator::Conv(conv) => &mut conv.output,
                    _ => panic!("fixture must end with convolution"),
                }
            } else {
                &mut model.output
            }
        }
        let mut f = std::io::Cursor::new(include_bytes!("../../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = gguf_file::Content::read(&mut f)?;
        let mut model = Model::from_gguf(ct, &mut f, device)?;
        let mut prefix = model.new_state();
        let _ = model.forward(&[1, 2, 3], &mut prefix)?;
        let saved = prefix.clone();
        let bad_output = QMatMul::Tensor(Tensor::zeros((1, 1), DType::F32, device)?);
        let good_output = std::mem::replace(output_to_break(&mut model, inside_conv), bad_output);
        assert!(model.forward(&[4, 5], &mut prefix).is_err());
        assert_eq!(prefix.len(), 3);
        *output_to_break(&mut model, inside_conv) = good_output;
        // Retry different tokens and a different length. Comparing only two
        // clones could miss corruption shared by both; also compare cold truth.
        let a = model.forward(&[9, 8, 7], &mut prefix)?;
        let b = model.forward(&[9, 8, 7], &mut saved.clone())?;
        let cold = model.forward(&[1, 2, 3, 9, 8, 7], &mut model.new_state())?;
        for (actual, expected) in a
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(cold.flatten_all()?.to_vec1::<f32>()?)
        {
            assert!((actual - expected).abs() < 2e-5, "{actual} vs {expected}");
        }
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    fn tiny() -> Result<Model> {
        let mut f = std::io::Cursor::new(include_bytes!("../../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = gguf_file::Content::read(&mut f)?;
        Model::from_gguf(ct, &mut f, &Device::Cpu)
    }
    fn flat(t: &Tensor) -> Result<Vec<f32>> {
        t.flatten_all()?.to_vec1::<f32>()
    }

    #[derive(Default)]
    struct Recorder {
        embedding: Vec<Vec<f32>>,
        residual: Vec<(usize, Vec<usize>, Vec<f32>)>,
        routing: Vec<(usize, Vec<Vec<f32>>, Vec<Vec<u32>>, Vec<Vec<f32>>)>,
        fail_at_layer: Option<usize>,
    }
    impl Observer for Recorder {
        fn embedding(&mut self, x: &Tensor) -> Result<()> {
            self.embedding.push(flat(x)?);
            Ok(())
        }
        fn residual(&mut self, layer: usize, x: &Tensor) -> Result<()> {
            if self.fail_at_layer == Some(layer) {
                bail!("observer refused layer {layer}")
            }
            self.residual.push((layer, x.dims().to_vec(), flat(x)?));
            Ok(())
        }
        fn routing(
            &mut self,
            layer: usize,
            logits: &Tensor,
            ids: &Tensor,
            weights: &Tensor,
        ) -> Result<()> {
            self.routing.push((
                layer,
                logits.to_vec2::<f32>()?,
                ids.to_vec2::<u32>()?,
                weights.to_vec2::<f32>()?,
            ));
            Ok(())
        }
    }

    #[test]
    fn observed_forward_is_bit_identical_to_the_unobserved_one() -> Result<()> {
        let model = tiny()?;
        let (mut plain, mut seen) = (model.new_state(), model.new_state());
        let mut rec = Recorder::default();
        let a = model.forward(&[1, 2, 3, 4, 5], &mut plain)?;
        let b = model.forward_observed(&[1, 2, 3, 4, 5], &mut seen, &mut rec)?;
        assert_eq!(flat(&a)?, flat(&b)?);
        // The committed hybrid state must match too, not only this step's logits.
        assert_eq!(seen.len(), plain.len());
        assert_eq!(
            flat(&model.forward(&[6, 7], &mut plain)?)?,
            flat(&model.forward(&[6, 7], &mut seen)?)?
        );
        Ok(())
    }

    #[test]
    fn observer_sees_every_layer_once_and_routing_only_where_experts_are() -> Result<()> {
        let model = tiny()?;
        let mut rec = Recorder::default();
        model.forward_observed(&[1, 2, 3, 4, 5], &mut model.new_state(), &mut rec)?;
        assert_eq!(rec.embedding.len(), 1);
        assert_eq!(rec.embedding[0].len(), 5 * 8);
        assert_eq!(
            rec.residual.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        for (_, dims, _) in &rec.residual {
            assert_eq!(dims, &vec![1, 5, 8]);
        }
        // Fixture: conv, attention, conv; one leading dense block, then 3 experts
        // choosing 2. So layer 0 has no router.
        let moe = Some((3, 2));
        assert_eq!(
            model.layers(),
            vec![
                LayerInfo {
                    attention: false,
                    experts: None
                },
                LayerInfo {
                    attention: true,
                    experts: moe
                },
                LayerInfo {
                    attention: false,
                    experts: moe
                },
            ]
        );
        assert_eq!(
            rec.routing.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(model.router_bias(0).is_none() && model.router_bias(3).is_none());
        for (layer, logits, ids, weights) in &rec.routing {
            assert_eq!((logits.len(), ids.len(), weights.len()), (5, 5, 5));
            // The reported choice is reproducible from what the record holds:
            // the top sigmoid(logit) + bias, recomputed here on the host.
            let bias = model.router_bias(*layer).unwrap().to_vec1::<f32>()?;
            for (l, i) in logits.iter().zip(ids) {
                let mut order: Vec<u32> = (0..3).collect();
                let score = |e: &u32| 1. / (1. + (-l[*e as usize]).exp()) + bias[*e as usize];
                order.sort_by(|a, b| score(b).total_cmp(&score(a)));
                let (mut got, mut want) = (i.clone(), order[..2].to_vec());
                got.sort();
                want.sort();
                assert_eq!(got, want, "layer {layer}");
            }
            for ((l, i), w) in logits.iter().zip(ids).zip(weights) {
                assert_eq!((l.len(), i.len(), w.len()), (3, 2, 2), "layer {layer}");
                assert!(i[0] != i[1] && i.iter().all(|&e| e < 3), "{i:?}");
                // Reported weights are the ones the combine used: unbiased
                // sigmoid scores of the chosen experts, normalised with 1e-6.
                let s: Vec<f32> = i
                    .iter()
                    .map(|&e| 1. / (1. + (-l[e as usize]).exp()))
                    .collect();
                for (got, score) in w.iter().zip(&s) {
                    let want = score / (s[0] + s[1] + 1e-6);
                    assert!((got - want).abs() < 1e-6, "{got} vs {want}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn observation_is_causal_across_chunk_boundaries() -> Result<()> {
        // A position's routing and residual cannot depend on how the caller
        // chunked the prefill. Catches position or layer mix-ups in the tap.
        let model = tiny()?;
        let mut whole = Recorder::default();
        model.forward_observed(&[1, 2, 3, 4, 5], &mut model.new_state(), &mut whole)?;
        let (mut head, mut tail) = (Recorder::default(), Recorder::default());
        let mut state = model.new_state();
        model.forward_observed(&[1, 2, 3], &mut state, &mut head)?;
        model.forward_observed(&[4, 5], &mut state, &mut tail)?;
        for n in 0..2 {
            let mut ids = head.routing[n].2.clone();
            ids.extend(tail.routing[n].2.clone());
            assert_eq!(ids, whole.routing[n].2, "router {n}");
        }
        for n in 0..3 {
            let mut x = head.residual[n].2.clone();
            x.extend(tail.residual[n].2.clone());
            for (got, want) in x.iter().zip(&whole.residual[n].2) {
                assert!((got - want).abs() < 2e-5, "layer {n}: {got} vs {want}");
            }
        }
        Ok(())
    }

    #[test]
    fn projecting_the_last_residual_reproduces_the_returned_logits() -> Result<()> {
        // Pins the logit lens to the model's own head: final norm, then output.
        let model = tiny()?;
        let mut rec = Recorder::default();
        let logits = model.forward_observed(&[1, 2, 3, 4], &mut model.new_state(), &mut rec)?;
        let (_, dims, last) = rec.residual.last().unwrap();
        let x = Tensor::from_vec(last.clone(), dims.clone(), &Device::Cpu)?;
        let lens = model.project(&x)?;
        assert_eq!(lens.dims(), &[1, 4, 16]);
        assert_eq!(flat(&lens.i((.., 3, ..))?)?, flat(&logits)?);
        // An earlier layer is a different read, or the lens shows nothing.
        let (_, dims, first) = &rec.residual[0];
        let early = model.project(&Tensor::from_vec(
            first.clone(),
            dims.clone(),
            &Device::Cpu,
        )?)?;
        assert_ne!(flat(&early.i((.., 3, ..))?)?, flat(&logits)?);
        assert!(model
            .project(&Tensor::zeros((1, 4, 7), DType::F32, &Device::Cpu)?)
            .is_err());
        Ok(())
    }

    /// The experts layer `layer` chose at every position of one forward.
    fn chosen(
        model: &Model,
        tokens: &[u32],
        steering: &Steering,
        layer: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let mut rec = Recorder::default();
        model.forward_steered(tokens, &mut model.new_state(), steering, Some(&mut rec))?;
        Ok(rec.routing.into_iter().find(|r| r.0 == layer).unwrap().2)
    }

    #[test]
    fn steering_with_the_models_own_bias_changes_nothing() -> Result<()> {
        let model = tiny()?;
        let mut steering = Steering::default();
        for layer in [1, 2] {
            steering.set_bias(layer, model.router_bias(layer).unwrap().clone());
        }
        let (mut plain, mut steered) = (model.new_state(), model.new_state());
        let a = model.forward(&[1, 2, 3, 4, 5], &mut plain)?;
        let b = model.forward_steered(&[1, 2, 3, 4, 5], &mut steered, &steering, None)?;
        assert_eq!(flat(&a)?, flat(&b)?);
        let empty = Steering::default();
        let c = model.forward_steered(&[1, 2, 3, 4, 5], &mut model.new_state(), &empty, None)?;
        assert_eq!(flat(&a)?, flat(&c)?);
        Ok(())
    }

    #[test]
    fn a_large_negative_bias_knocks_an_expert_out_and_a_large_positive_one_forces_it_in(
    ) -> Result<()> {
        let model = tiny()?;
        let tokens = [1u32, 2, 3, 4, 5];
        let baseline = chosen(&model, &tokens, &Steering::default(), 1)?;
        // An expert the router really uses, so removing it has to change something.
        let used = baseline[0][0];
        let bias = model.router_bias(1).unwrap().to_vec1::<f32>()?;
        let with = |delta: f32| -> Result<Steering> {
            let mut b = bias.clone();
            b[used as usize] += delta;
            let mut s = Steering::default();
            s.set_bias(1, Tensor::from_vec(b, 3, &Device::Cpu)?);
            Ok(s)
        };
        let out = chosen(&model, &tokens, &with(-1e4)?, 1)?;
        assert!(out.iter().all(|row| !row.contains(&used)), "{out:?}");
        // Three experts choosing two: with one banned, the other two are chosen everywhere.
        assert!(out.iter().all(|row| row.len() == 2));
        let forced = chosen(&model, &tokens, &with(1e4)?, 1)?;
        assert!(forced.iter().all(|row| row.contains(&used)), "{forced:?}");
        // Layer 2 was not steered, but it reads a residual layer 1 changed.
        let plain = model.forward(&tokens, &mut model.new_state())?;
        let steered = model.forward_steered(&tokens, &mut model.new_state(), &with(-1e4)?, None)?;
        assert_ne!(flat(&plain)?, flat(&steered)?);
        Ok(())
    }

    #[test]
    fn a_forced_expert_is_weighted_by_its_own_score_not_by_the_bias() -> Result<()> {
        let model = tiny()?;
        let bias = model.router_bias(2).unwrap().to_vec1::<f32>()?;
        let mut b = bias.clone();
        b[0] += 1e4;
        let mut steering = Steering::default();
        steering.set_bias(2, Tensor::from_vec(b, 3, &Device::Cpu)?);
        let mut rec = Recorder::default();
        model.forward_steered(
            &[1, 2, 3],
            &mut model.new_state(),
            &steering,
            Some(&mut rec),
        )?;
        let (_, logits, ids, weights) = rec.routing.iter().find(|r| r.0 == 2).unwrap();
        for ((l, i), w) in logits.iter().zip(ids).zip(weights) {
            let s: Vec<f32> = i
                .iter()
                .map(|&e| 1. / (1. + (-l[e as usize]).exp()))
                .collect();
            for (got, score) in w.iter().zip(&s) {
                assert!((got - score / (s[0] + s[1] + 1e-6)).abs() < 1e-6, "{got}");
                assert!(*got < 1.0);
            }
        }
        Ok(())
    }

    #[test]
    fn steering_that_cannot_apply_is_refused_and_commits_nothing() -> Result<()> {
        let model = tiny()?;
        let mut state = model.new_state();
        model.forward(&[1, 2, 3], &mut state)?;
        let bad = |layer: usize, t: Tensor| {
            let mut s = Steering::default();
            s.set_bias(layer, t);
            s
        };
        let d = Device::Cpu;
        // Layer 0 is dense, layer 9 does not exist, and a router has 3 experts here.
        for steering in [
            bad(0, Tensor::zeros(3, DType::F32, &d)?),
            bad(9, Tensor::zeros(3, DType::F32, &d)?),
            bad(1, Tensor::zeros(4, DType::F32, &d)?),
            bad(1, Tensor::zeros(3, DType::F64, &d)?),
        ] {
            assert!(model
                .forward_steered(&[4, 5], &mut state, &steering, None)
                .is_err());
            assert_eq!(state.len(), 3);
        }
        Ok(())
    }

    #[test]
    fn failing_observer_fails_the_forward_and_does_not_commit_state() -> Result<()> {
        let model = tiny()?;
        let mut state = model.new_state();
        model.forward(&[1, 2, 3], &mut state)?;
        let mut rec = Recorder {
            fail_at_layer: Some(1),
            ..Default::default()
        };
        assert!(model
            .forward_observed(&[4, 5], &mut state, &mut rec)
            .is_err());
        assert_eq!(state.len(), 3);
        let a = model.forward(&[4, 5], &mut state)?;
        let cold = model.forward(&[1, 2, 3, 4, 5], &mut model.new_state())?;
        for (got, want) in flat(&a)?.iter().zip(flat(&cold)?) {
            assert!((got - want).abs() < 2e-5, "{got} vs {want}");
        }
        Ok(())
    }

    #[test]
    fn failure_after_all_layers_does_not_commit_state() -> Result<()> {
        check_failed_forward(false)
    }

    #[test]
    fn failure_inside_convolution_does_not_commit_state() -> Result<()> {
        check_failed_forward(true)
    }

    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires actual ROCm hardware; device failure is an error"]
    fn model_reuses_one_routing_map_across_expert_projections() -> Result<()> {
        use candle::quantized::GgmlDType;
        let device = Device::new_rocm(0)?;
        let data = |n: usize, scale: f32| {
            (0..n)
                .map(|i| (i as f32 / scale).sin() * 0.1)
                .collect::<Vec<_>>()
        };
        let dense = Tensor::from_vec(data(4 * 256 * 256, 61.), (4, 256, 256), &Device::Cpu)?;
        let x = Tensor::from_vec(data(17 * 256, 43.), (1, 17, 256), &device)?;
        for (up_dtype, projections) in [(GgmlDType::Q5K, 2), (GgmlDType::Q6K, 3)] {
            let moe = Moe {
                gate: QMatMul::Tensor(Tensor::from_vec(data(4 * 256, 17.), (4, 256), &device)?),
                bias: Tensor::new(&[0f32, 0.1, -0.1, 0.05], &device)?,
                gate_up: ExpertGateUp::new(
                    QTensor::quantize_onto(&dense, GgmlDType::Q5K, &device)?,
                    QTensor::quantize_onto(&dense, up_dtype, &device)?,
                    &device,
                )?,
                down: Experts::new(
                    QTensor::quantize_onto(&dense, GgmlDType::Q6K, &device)?,
                    &device,
                )?,
                topk: 2,
            };
            GROUPED_ROUTE_CALLS.with(|c| c.set((0, 0)));
            let actual = moe.forward(&x, None, None)?;
            GROUPED_ROUTE_CALLS.with(|c| assert_eq!(c.get(), (1, projections)));
            let flat = x.reshape((17, 256))?;
            let (ids, weights) = route(&moe.gate.forward(&flat)?, &moe.bias, moe.topk)?;
            let unprepared = ExpertRouting::new(&ids);
            let activated = moe.gate_up.forward(&flat.unsqueeze(1)?, &unprepared)?;
            let reference = moe
                .down
                .forward(&activated, &unprepared)?
                .broadcast_mul(&weights.unsqueeze(2)?)?
                .sum(1)?
                .reshape((1, 17, 256))?;
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                reference.flatten_all()?.to_vec1::<f32>()?
            );
        }
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
                let actual = packed.forward(&input, &ExpertRouting::new(&ids))?;
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
                let reference = cpu
                    .forward(&x, &ExpertRouting::new(&ids))?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let actual = gpu
                    .forward(
                        &x.to_device(&dev)?,
                        &ExpertRouting::new(&ids.to_device(&dev)?),
                    )?
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

#[cfg(test)]
mod qkv_tests {
    use super::*;
    use candle::quantized::GgmlDType;
    fn check(device: &Device) -> Result<()> {
        // Q/K share formats in our GGUF; V is Q6K in two attention layers.
        for types in [
            [GgmlDType::Q5K; 3],
            [GgmlDType::Q5K, GgmlDType::Q5K, GgmlDType::Q6K],
            [GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q5K],
        ] {
            let rows = [1024, 512, 512];
            let mut weights = Vec::new();
            let mut separate = Vec::new();
            for (i, &n) in rows.iter().enumerate() {
                let data = Tensor::from_vec(
                    (0..n * 512)
                        .map(|j| ((j + i * 331) as f32 / 113.).sin())
                        .collect::<Vec<_>>(),
                    (n, 512),
                    &Device::Cpu,
                )?;
                let w = QTensor::quantize_onto(&data, types[i], device)?;
                separate.push(QMatMul::from_qtensor(QTensor::quantize_onto(
                    &data, types[i], device,
                )?)?);
                weights.push(w);
            }
            let v = weights.pop().unwrap();
            let k = weights.pop().unwrap();
            let q = weights.pop().unwrap();
            let joined = QkvProjection::new(q, k, v)?;
            for seq in [1, 3, 17] {
                let x = Tensor::from_vec(
                    (0..seq * 512)
                        .map(|i| (i as f32 / 71.).cos())
                        .collect::<Vec<_>>(),
                    (1, seq, 512),
                    device,
                )?;
                let (q, k, v) = joined.forward(&x)?;
                for (i, out) in [q, k, v].into_iter().enumerate() {
                    assert_eq!(out.dims(), [1, seq, rows[i]]);
                    let expected = separate[i].forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
                    let got = out.flatten_all()?.to_vec1::<f32>()?;
                    let scale = expected.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                    for (a, b) in got.iter().zip(&expected) {
                        assert!(
                            (a - b).abs() < 2e-5 * scale,
                            "{types:?} seq={seq} projection={i}: {a} vs {b}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn qkv_bundle_matches_independent_projections_cpu() -> Result<()> {
        check(&Device::Cpu)
    }
    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires ROCm GPU; device failure is an error"]
    fn qkv_bundle_matches_independent_projections_rocm() -> Result<()> {
        check(&Device::new_rocm(0)?)
    }
}
