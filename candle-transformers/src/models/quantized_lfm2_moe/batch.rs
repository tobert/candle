//! Batched decode: one new token for each of several independent states.
//!
//! Everything whose cost is reading weights runs once for the whole batch:
//! embedding, norms, QKV and output projections, the gated convolution (each
//! row with its own convolution state), router, experts and output head.
//! Attention runs per row against that row's own KV, because the rows'
//! lengths differ and only a few layers have attention.
//!
//! Batch size selects the quantized kernel (on ROCm, B = 1 takes DMMV for
//! K-quants, 2..=8 MMVQ, larger MMQ), so a row's logits at B > 1 are a
//! different measurement from the batch-1 ones, not a reproduction of them.
use super::kv_cache::KvCache;
use super::{attention, AttentionLayer, ConvState, Model, Operator, ShortConv, State};
use candle::{bail, IndexOp, Module, Result, Tensor};

impl AttentionLayer {
    /// `xs` `(B, 1, hidden)`; row `i` sits at `positions` (a `(B,)` u32 tensor
    /// on the device) and appends to `kv[i]`, whose length must be `pos[i]`.
    fn decode_batch(
        &self,
        xs: &Tensor,
        pos: &[usize],
        positions: &Tensor,
        kv: &mut [&mut Option<KvCache>],
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, _, n_embd) = xs.dims3()?;
        let (q, k, v) = self.qkv.forward(xs)?;
        let q = q
            .reshape((b, 1, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, 1, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, 1, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        // Each row's own RoPE angle: (B, 1, head_dim / 2) rows of the table.
        let rope = |x: &Tensor| -> Result<Tensor> {
            let _enter = self.span_rot.enter();
            let half = self.head_dim / 2;
            let cos = self.cos.index_select(positions, 0)?.reshape((b, 1, half))?;
            let sin = self.sin.index_select(positions, 0)?.reshape((b, 1, half))?;
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        };
        let q = rope(&q)?;
        let k = rope(&k)?;
        let limit = self.cos.dim(0)?;
        let mut ys = Vec::with_capacity(b);
        for (i, cache) in kv.iter_mut().enumerate() {
            if cache.as_ref().map_or(0, KvCache::len) != pos[i] {
                bail!("LFM2 KV position does not match state")
            }
            let next = KvCache::append(
                cache.as_ref(),
                &k.narrow(0, i, 1)?,
                &v.narrow(0, i, 1)?,
                limit,
            )?;
            let (ki, vi) = next.current()?;
            **cache = Some(next);
            ys.push(attention::forward(
                &q.narrow(0, i, 1)?,
                &ki,
                &vi,
                None,
                &self.neg_inf,
            )?);
        }
        let y = Tensor::cat(&ys, 0)?;
        let y = y.transpose(1, 2)?.reshape((b, 1, n_embd))?;
        self.wo.forward(&y)
    }
}

/// The `(B, hidden, k)` tensor a previous batched step wrote, when `state` is
/// exactly its rows in order: row `i` of the batch is row `i` of the tensor.
fn previous_batch(state: &[&mut Option<ConvState>]) -> Option<Tensor> {
    let first = state.first()?.as_ref()?;
    let rows = &first.rows;
    let whole = rows.dim(0).ok()? == state.len();
    let in_order = state.iter().enumerate().all(|(i, s)| {
        s.as_ref()
            .is_some_and(|s| s.row == i && s.rows.id() == rows.id())
    });
    (whole && in_order).then(|| rows.clone())
}

impl ShortConv {
    /// `x` `(B, 1, hidden)`; row `i` reads and replaces `state[i]`.
    fn decode_batch(&self, x: &Tensor, state: &mut [&mut Option<ConvState>]) -> Result<Tensor> {
        let h = x.dim(2)?;
        let k = self.weight.dim(1)?;
        let projected = self.input.forward(x)?;
        let previous = match previous_batch(state) {
            Some(rows) => rows,
            None => {
                let rows = state
                    .iter()
                    .map(|s| match s.as_ref() {
                        Some(s) => s.row(),
                        None => Tensor::zeros((1, h, k), x.dtype(), x.device()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&rows, 0)?
            }
        };
        let (out, next) = candle_nn::lfm2::short_conv_step(&projected, &self.weight, &previous)?;
        for (i, s) in state.iter_mut().enumerate() {
            **s = Some(ConvState {
                rows: next.clone(),
                row: i,
            });
        }
        self.output.forward(&out)
    }
}

impl Model {
    /// One decode step for `B = tokens.len()` independent sequences: `tokens[i]`
    /// continues `states[i]`. Returns next-token logits `(B, vocab)`, row `i`
    /// for `states[i]`. States may have different lengths and may share
    /// prefixes (branches of one parent); every state keeps the branch
    /// semantics of [`Model::forward`]. Validation happens before anything
    /// runs, and either every state commits or none does.
    ///
    /// B = 1 runs the same operations as `forward(&[token], state)` and gives
    /// bit-identical logits and state. Observation and steering are not
    /// offered here; use [`Model::forward_observed`] per sequence.
    pub fn decode_batch(&self, tokens: &[u32], states: &mut [&mut State]) -> Result<Tensor> {
        let b = tokens.len();
        if b == 0 {
            bail!("decode_batch needs at least one sequence")
        }
        if states.len() != b {
            bail!("decode_batch has {b} tokens for {} states", states.len())
        }
        for (i, (s, &t)) in states.iter().zip(tokens).enumerate() {
            if !self.owns_state(s) {
                bail!("row {i}: snapshot belongs to another model")
            }
            if s.len >= self.context {
                bail!("row {i}: context limit exceeded")
            }
            if t as usize >= self.vocab {
                bail!("row {i}: token outside model vocabulary")
            }
        }
        let pos: Vec<usize> = states.iter().map(|s| s.len).collect();
        let positions = Tensor::from_iter(pos.iter().map(|&p| p as u32), &self.device)?;
        let mut next: Vec<State> = states.iter().map(|s| (**s).clone()).collect();
        let ids = Tensor::from_slice(tokens, (b, 1), &self.device)?;
        let mut x = self.embedding.embedding(&ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm.forward(&x)?;
            let y = match &layer.operator {
                Operator::Attention(a) => {
                    let mut kv: Vec<_> = next.iter_mut().map(|s| &mut s.kv[i]).collect();
                    a.decode_batch(&normed, &pos, &positions, &mut kv)?
                }
                Operator::Conv(c) => {
                    let mut conv: Vec<_> = next.iter_mut().map(|s| &mut s.conv[i]).collect();
                    c.decode_batch(&normed, &mut conv)?
                }
            };
            x = (x + y)?;
            x = (&x
                + layer
                    .ffn
                    .forward(&layer.ffn_norm.forward(&x)?, None, None)?)?;
        }
        let x = self.norm.forward(&x)?.i((.., 0, ..))?.contiguous()?;
        let logits = self.output.forward(&x)?;
        for (s, mut n) in states.iter_mut().zip(next) {
            n.len += 1;
            **s = n;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::{gguf_file, QMatMul};
    use candle::{DType, Device};

    fn tiny() -> Result<Model> {
        let mut f =
            std::io::Cursor::new(include_bytes!("../../../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = gguf_file::Content::read(&mut f)?;
        Model::from_gguf(ct, &mut f, &Device::Cpu)
    }
    fn rows(t: &Tensor) -> Result<Vec<Vec<f32>>> {
        t.to_vec2::<f32>()
    }
    fn prefilled(model: &Model, tokens: &[u32]) -> Result<State> {
        let mut s = model.new_state();
        if !tokens.is_empty() {
            model.forward(tokens, &mut s)?;
        }
        Ok(s)
    }
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max)
    }
    /// What a batch-1 step gives on a copy of `state`, plus the next step after
    /// it, so a test can check both the logits and the committed state.
    fn single(model: &Model, state: &State, t: u32, then: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut s = state.clone();
        let a = model.forward(&[t], &mut s)?.flatten_all()?.to_vec1()?;
        let b = model.forward(&[then], &mut s)?.flatten_all()?.to_vec1()?;
        Ok((a, b))
    }

    // Tiny fixture on CPU: plain f32 weights, so every op is the same maths at
    // B rows as at one. The tolerance is tight; the measured value is printed.
    const CPU_TOL: f32 = 1e-5;

    #[test]
    fn mixed_lengths_match_per_row_single_steps_and_commit_the_same_state() -> Result<()> {
        let model = tiny()?;
        // Lengths 0 (a first token), 1, 3 and 7: conv state absent and
        // present, KV at several positions, different RoPE angles per row.
        let prefixes: [&[u32]; 4] = [&[], &[5], &[1, 2, 3], &[9, 8, 7, 6, 5, 4, 3]];
        let tokens = [11u32, 2, 4, 15];
        let mut states: Vec<State> = prefixes
            .iter()
            .map(|p| prefilled(&model, p))
            .collect::<Result<_>>()?;
        let expected: Vec<_> = states
            .iter()
            .zip(tokens)
            .map(|(s, t)| single(&model, s, t, 7))
            .collect::<Result<_>>()?;
        let mut refs: Vec<&mut State> = states.iter_mut().collect();
        let got = model.decode_batch(&tokens, &mut refs)?;
        assert_eq!(got.dims(), &[4, model.vocab_size()]);
        let got = rows(&got)?;
        let mut worst = 0f32;
        for (i, (row, (want, _))) in got.iter().zip(&expected).enumerate() {
            let d = max_abs(row, want);
            assert!(d < CPU_TOL, "row {i}: max |delta| {d}");
            worst = worst.max(d);
        }
        eprintln!("cpu batched vs batch-1 max |delta logit| = {worst:e}");
        for (i, (s, (_, then))) in states.iter_mut().zip(&expected).enumerate() {
            assert_eq!(s.len(), prefixes[i].len() + 1);
            let next = model.forward(&[7], s)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(max_abs(&next, then) < CPU_TOL, "row {i} committed state");
        }
        Ok(())
    }

    #[test]
    fn repeated_batched_steps_track_single_step_decoding() -> Result<()> {
        // Several consecutive batched steps, so a row reads conv/KV state that
        // an earlier batched step wrote, not only state from a batch-1 prefill.
        let model = tiny()?;
        let mut batched: Vec<State> = [&[1u32, 2][..], &[3, 4, 5, 6, 7]]
            .iter()
            .map(|p| prefilled(&model, p))
            .collect::<Result<_>>()?;
        let mut singles = batched.clone();
        for step in 0..6u32 {
            let tokens = [step % 16, (step * 5 + 3) % 16];
            let mut refs: Vec<&mut State> = batched.iter_mut().collect();
            let got = rows(&model.decode_batch(&tokens, &mut refs)?)?;
            for (i, s) in singles.iter_mut().enumerate() {
                let want = model
                    .forward(&[tokens[i]], s)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let d = max_abs(&got[i], &want);
                assert!(d < CPU_TOL, "step {step} row {i}: {d}");
            }
        }
        Ok(())
    }

    #[test]
    fn a_batched_step_reads_the_previous_batch_only_for_the_same_rows_in_order() -> Result<()> {
        let model = tiny()?;
        let mut s: Vec<State> = [&[1u32][..], &[2, 3], &[4, 5, 6]]
            .iter()
            .map(|p| prefilled(&model, p))
            .collect::<Result<_>>()?;
        let layer = model
            .layers
            .iter()
            .position(|l| matches!(l.operator, Operator::Conv(_)))
            .unwrap();
        let conv = |s: &mut [State]| -> Option<Tensor> {
            let refs: Vec<&mut Option<ConvState>> =
                s.iter_mut().map(|s| &mut s.conv[layer]).collect();
            previous_batch(&refs)
        };
        // After batch-1 prefills: separate tensors, gather.
        assert!(conv(&mut s).is_none());
        let mut refs: Vec<&mut State> = s.iter_mut().collect();
        model.decode_batch(&[7, 8, 9], &mut refs)?;
        // The same rows in the same order: the step's own tensor, as is.
        let whole = conv(&mut s).expect("same rows, same order");
        assert_eq!(whole.dims(), &[3, 8, 3]);
        // Any other arrangement gathers: permuted, a subset, a clone repeated.
        s.swap(0, 1);
        assert!(conv(&mut s).is_none());
        s.swap(0, 1);
        assert!(conv(&mut s[..2]).is_none());
        let mut twice = vec![s[0].clone(), s[0].clone(), s[2].clone()];
        assert!(conv(&mut twice).is_none());
        // A batch-1 step on one row leaves the others' shared tensor usable,
        // but the batch no longer matches it.
        model.forward(&[1], &mut s[1])?;
        assert!(conv(&mut s).is_none());
        // Row 0 of one batch beside row 1 of another batch of the same size:
        // right indices, different tensors. Gather, and decode each correctly.
        let mut x: Vec<State> = [&[1u32][..], &[2, 3]]
            .iter()
            .map(|p| prefilled(&model, p))
            .collect::<Result<_>>()?;
        let mut y: Vec<State> = [&[9u32, 9, 9][..], &[4]]
            .iter()
            .map(|p| prefilled(&model, p))
            .collect::<Result<_>>()?;
        for pair in [&mut x, &mut y] {
            let mut refs: Vec<&mut State> = pair.iter_mut().collect();
            model.decode_batch(&[5, 6], &mut refs)?;
        }
        let mut mixed = vec![x[0].clone(), y[1].clone()];
        assert!(conv(&mut mixed).is_none());
        let want: Vec<Vec<f32>> = mixed
            .iter()
            .map(|st| {
                model
                    .forward(&[11], &mut st.clone())
                    .and_then(|l| l.flatten_all()?.to_vec1())
            })
            .collect::<Result<_>>()?;
        let mut refs: Vec<&mut State> = mixed.iter_mut().collect();
        let got = rows(&model.decode_batch(&[11, 11], &mut refs)?)?;
        for i in 0..2 {
            assert!(max_abs(&got[i], &want[i]) < CPU_TOL, "mixed row {i}");
        }
        Ok(())
    }

    #[test]
    fn rows_rearranged_between_batched_steps_still_track_single_steps() -> Result<()> {
        // The reuse must only ever fire for the exact arrangement: permuting,
        // shrinking, growing and duplicating the batch between steps must give
        // each row its own state, as batch-1 decoding would.
        let model = tiny()?;
        let mut batched: Vec<State> = [&[1u32, 2][..], &[3], &[4, 5, 6], &[7, 8]]
            .iter()
            .map(|p| prefilled(&model, p))
            .collect::<Result<_>>()?;
        let mut singles = batched.clone();
        let orders: [&[usize]; 6] = [
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &[3, 1, 0, 2],
            &[1, 2],
            &[0, 1, 2, 3],
            &[2, 0],
        ];
        for (step, order) in orders.iter().enumerate() {
            let tokens: Vec<u32> = order
                .iter()
                .map(|&i| ((step * 3 + i) % 16) as u32)
                .collect();
            let mut picked: Vec<State> = order.iter().map(|&i| batched[i].clone()).collect();
            let mut refs: Vec<&mut State> = picked.iter_mut().collect();
            let got = rows(&model.decode_batch(&tokens, &mut refs)?)?;
            for (j, &i) in order.iter().enumerate() {
                let want = model
                    .forward(&[tokens[j]], &mut singles[i])?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(max_abs(&got[j], &want) < CPU_TOL, "step {step} row {i}");
                batched[i] = picked[j].clone();
            }
        }
        Ok(())
    }

    #[test]
    fn a_batch_of_one_is_bit_identical_to_forward() -> Result<()> {
        let model = tiny()?;
        for prefix in [&[][..], &[3], &[1, 2, 3, 4, 5]] {
            let mut a = prefilled(&model, prefix)?;
            let mut b = a.clone();
            let want = model.forward(&[9], &mut a)?;
            let got = model.decode_batch(&[9], &mut [&mut b])?;
            assert_eq!(got.dims(), want.dims());
            assert_eq!(rows(&got)?, rows(&want)?);
            // And the committed state is the same state, bit for bit.
            assert_eq!(
                rows(&model.forward(&[4, 2], &mut a)?)?,
                rows(&model.forward(&[4, 2], &mut b)?)?
            );
        }
        Ok(())
    }

    #[test]
    fn siblings_of_one_parent_decode_together_without_touching_each_other() -> Result<()> {
        let model = tiny()?;
        let parent = prefilled(&model, &[1, 2, 3, 4])?;
        // Before anything else touches the parent: its own continuations.
        let parent_next = single(&model, &parent, 6, 1)?;
        let mut a = parent.clone();
        let mut b = parent.clone();
        // c is a sibling that already ran ahead: it holds a reservation past
        // the parent's length in the shared KV storage.
        let mut c = parent.clone();
        model.forward(&[10, 11], &mut c)?;
        let a_want = single(&model, &a, 6, 12)?;
        let b_want = single(&model, &b, 13, 14)?;
        let c_want = single(&model, &c, 2, 3)?;
        let mut p = parent.clone();
        let p_want = single(&model, &p, 8, 9)?;
        let got =
            rows(&model.decode_batch(&[6, 13, 2, 8], &mut [&mut a, &mut b, &mut c, &mut p])?)?;
        for (i, want) in [&a_want, &b_want, &c_want, &p_want].iter().enumerate() {
            assert!(max_abs(&got[i], &want.0) < CPU_TOL, "row {i}");
        }
        // Each committed branch continues as a batch-1 branch would.
        for (i, (s, t, want)) in [
            (&mut a, 12, &a_want),
            (&mut b, 14, &b_want),
            (&mut c, 3, &c_want),
            (&mut p, 9, &p_want),
        ]
        .into_iter()
        .enumerate()
        {
            let next = model.forward(&[t], s)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(max_abs(&next, &want.1) < CPU_TOL, "row {i} after the batch");
        }
        // The parent is untouched by four branches writing past it.
        assert_eq!(parent.len(), 4);
        let again = single(&model, &parent, 6, 1)?;
        assert_eq!(again, parent_next);
        Ok(())
    }

    #[test]
    fn bad_input_fails_loudly_and_commits_nothing() -> Result<()> {
        let model = tiny()?;
        let other = tiny()?;
        let base = prefilled(&model, &[1, 2, 3])?;
        let want = single(&model, &base, 4, 5)?;
        let context = model.context_length();
        let full = prefilled(&model, &vec![1u32; context])?;
        let foreign = prefilled(&other, &[1])?;
        let cases: Vec<(&str, Vec<u32>, Vec<State>)> = vec![
            ("empty batch", vec![], vec![]),
            (
                "fewer tokens than states",
                vec![4],
                vec![base.clone(), base.clone()],
            ),
            ("more tokens than states", vec![4, 4], vec![base.clone()]),
            (
                "token outside vocabulary",
                vec![4, 16],
                vec![base.clone(), base.clone()],
            ),
            (
                "state at the context limit",
                vec![4, 4],
                vec![base.clone(), full.clone()],
            ),
            (
                "state from another model",
                vec![4, 4],
                vec![base.clone(), foreign.clone()],
            ),
        ];
        for (what, tokens, mut states) in cases {
            let lens: Vec<usize> = states.iter().map(State::len).collect();
            let mut refs: Vec<&mut State> = states.iter_mut().collect();
            assert!(
                model.decode_batch(&tokens, &mut refs).is_err(),
                "{what} accepted"
            );
            assert_eq!(
                states.iter().map(State::len).collect::<Vec<_>>(),
                lens,
                "{what}"
            );
            // The valid rows beside the bad one are still exactly usable.
            if let Some(s) = states.first() {
                if s.len() == 3 {
                    assert_eq!(single(&model, s, 4, 5)?, want, "{what}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn a_failure_inside_the_forward_commits_no_row() -> Result<()> {
        let mut model = tiny()?;
        let mut a = prefilled(&model, &[1, 2, 3])?;
        let mut b = prefilled(&model, &[4])?;
        let (a_want, b_want) = (single(&model, &a, 5, 6)?, single(&model, &b, 7, 8)?);
        let broken = QMatMul::Tensor(Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?);
        let good = std::mem::replace(&mut model.output, broken);
        assert!(model.decode_batch(&[5, 7], &mut [&mut a, &mut b]).is_err());
        model.output = good;
        assert_eq!((a.len(), b.len()), (3, 1));
        assert_eq!(single(&model, &a, 5, 6)?, a_want);
        assert_eq!(single(&model, &b, 7, 8)?, b_want);
        // And a retry of the same batch on the same states succeeds exactly.
        let got = rows(&model.decode_batch(&[5, 7], &mut [&mut a, &mut b])?)?;
        assert!(max_abs(&got[0], &a_want.0) < CPU_TOL);
        assert!(max_abs(&got[1], &b_want.0) < CPU_TOL);
        Ok(())
    }
}

/// Real-checkpoint tests and the batched-decode bench, on ROCm. Both need
/// `LFM25_GGUF` (the LFM2.5-8B-A1B GGUF); the bench also needs `LFM25_PROMPTS`
/// (JSON `{"prompts": [[token ids]...], "sha256": ...}`) and writes its record
/// to `LFM25_BENCH_OUT`. Missing inputs are an error, never a skip.
#[cfg(all(test, feature = "rocm"))]
mod rocm_real {
    use super::*;
    use candle::quantized::gguf_file;
    use candle::{Device, D};
    use std::time::Instant;

    fn env(key: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| panic!("set {key}"))
    }
    fn load(device: &Device) -> Result<Model> {
        let path = env("LFM25_GGUF");
        let mut f = std::fs::File::open(&path)?;
        let ct = gguf_file::Content::read(&mut f)?;
        Model::from_gguf(ct, &mut f, device)
    }
    /// Abort anything large before the host runs short: the GPU here shares
    /// system memory.
    fn memory_guard() -> Result<u64> {
        let info = std::fs::read_to_string("/proc/meminfo")?;
        let kib: u64 = info
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .expect("MemAvailable in /proc/meminfo");
        if kib < 6 * 1024 * 1024 {
            bail!("memory guard: MemAvailable {kib} KiB is under 6 GiB")
        }
        Ok(kib)
    }
    fn prefill(model: &Model, tokens: &[u32], stops: &[usize]) -> Result<Vec<State>> {
        // Chunked like a server; returns a snapshot at every requested length.
        let mut state = model.new_state();
        let mut snaps = Vec::new();
        let mut done = 0;
        for &stop in stops {
            while done < stop {
                memory_guard()?;
                let n = (stop - done).min(512);
                model.forward(&tokens[done..done + n], &mut state)?;
                done += n;
            }
            snaps.push(state.clone());
        }
        Ok(snaps)
    }
    fn host_rows(t: &Tensor) -> Result<Vec<Vec<f32>>> {
        t.to_device(&Device::Cpu)?.to_vec2::<f32>()
    }
    fn log_softmax(row: &[f32]) -> Vec<f64> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
        let lse = max + sum.ln();
        row.iter().map(|&v| v as f64 - lse).collect()
    }
    fn argmax(row: &[f32]) -> u32 {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    /// Every kernel on the batched path computes each row on its own, so a
    /// row's bits cannot depend on where it sits in the batch. Reversing the
    /// rows over several steps catches any row mixing (a convolution state or
    /// KV written to or read from the wrong row) that would otherwise pass as
    /// batch-size drift. B = 3 takes MMVQ, B = 12 MMQ and task-major experts.
    #[test]
    #[ignore = "requires ROCm and LFM25_GGUF; device failure is an error"]
    fn rocm_rows_are_position_invariant_over_several_steps() -> Result<()> {
        let device = Device::new_rocm(0)?;
        let model = load(&device)?;
        for b in [3usize, 12] {
            // Different lengths and contents per row.
            let base: Vec<State> = (0..b)
                .map(|i| {
                    let text: Vec<u32> = (0..40 + 7 * i as u32)
                        .map(|j| 500 + (j * 131 + i as u32 * 977) % 30000)
                        .collect();
                    prefill(&model, &text, &[text.len()]).map(|mut v| v.remove(0))
                })
                .collect::<Result<_>>()?;
            let mut fwd = base.clone();
            let mut rev: Vec<State> = base.iter().rev().cloned().collect();
            let mut tokens: Vec<u32> = (0..b as u32).map(|i| 1000 + 17 * i).collect();
            for step in 0..6 {
                let mut a: Vec<&mut State> = fwd.iter_mut().collect();
                let x = host_rows(&model.decode_batch(&tokens, &mut a)?)?;
                let rtokens: Vec<u32> = tokens.iter().rev().copied().collect();
                let mut r: Vec<&mut State> = rev.iter_mut().collect();
                let y = host_rows(&model.decode_batch(&rtokens, &mut r)?)?;
                for i in 0..b {
                    assert!(
                        x[i] == y[b - 1 - i],
                        "B={b} step {step}: row {i} depends on its position"
                    );
                }
                tokens = x.iter().map(|row| argmax(row)).collect();
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires ROCm and LFM25_GGUF; device failure is an error"]
    fn rocm_batch_of_one_is_bit_identical_and_siblings_match_independent_rows() -> Result<()> {
        let device = Device::new_rocm(0)?;
        let model = load(&device)?;
        let text: Vec<u32> = (0..200u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        // B = 1: same kernels as forward, so the same bits, state included.
        let mut a = prefill(&model, &text, &[150])?.remove(0);
        let mut b = a.clone();
        let want = host_rows(&model.forward(&[text[150]], &mut a)?)?;
        let got = host_rows(&model.decode_batch(&[text[150]], &mut [&mut b])?)?;
        assert_eq!(got, want, "B=1 logits");
        assert_eq!(
            host_rows(&model.forward(&[text[151]], &mut a)?)?,
            host_rows(&model.forward(&[text[151]], &mut b)?)?,
            "B=1 committed state"
        );
        // Four branches of one parent (shared KV storage, one of them already
        // ahead of the parent) against four rows prefilled independently: the
        // batch composition is identical, so the bits must be too.
        let parent = prefill(&model, &text, &[120])?.remove(0);
        let parent_next = host_rows(&model.forward(&[5], &mut parent.clone())?)?;
        let mut ahead = parent.clone();
        model.forward(&text[120..123], &mut ahead)?;
        let mut shared = [parent.clone(), parent.clone(), ahead, parent.clone()];
        // Built by the same chunking as its twin: chunk shape selects kernels.
        let mut fresh_ahead = prefill(&model, &text, &[120])?.remove(0);
        model.forward(&text[120..123], &mut fresh_ahead)?;
        let mut fresh = [
            prefill(&model, &text, &[120])?.remove(0),
            prefill(&model, &text, &[120])?.remove(0),
            fresh_ahead,
            prefill(&model, &text, &[120])?.remove(0),
        ];
        let tokens = [7u32, 8, 9, 10];
        for step in 0..3 {
            let [s0, s1, s2, s3] = &mut shared;
            let x = host_rows(&model.decode_batch(&tokens, &mut [s0, s1, s2, s3])?)?;
            let [f0, f1, f2, f3] = &mut fresh;
            let y = host_rows(&model.decode_batch(&tokens, &mut [f0, f1, f2, f3])?)?;
            for i in 0..4 {
                assert!(
                    x[i] == y[i],
                    "step {step} row {i}: branch differs from its independent twin"
                );
            }
        }
        assert_eq!(parent.len(), 120);
        assert_eq!(
            host_rows(&model.forward(&[5], &mut parent.clone())?)?,
            parent_next
        );
        Ok(())
    }

    #[derive(serde::Serialize, Default)]
    struct Drift {
        b: usize,
        rows_x_steps: usize,
        top1_agree: usize,
        dtop1_nats_median: f64,
        dtop1_nats_max: f64,
        dtop20_nats_median: f64,
        dtop20_nats_max: f64,
        /// Largest |delta logprob| over tokens the batch-1 read gives >= 1%.
        dp1pct_nats_max: f64,
        /// KL(batch-1 || batched) in nats, per row and step.
        kl_median: f64,
        kl_max: f64,
        /// The first step only: one forward's kernel difference, before the
        /// rows' own KV (written by the batched kernels) compounds it.
        step0_dtop1_nats_max: f64,
        step0_kl_max: f64,
    }
    #[derive(serde::Serialize)]
    struct Timing {
        regime: String,
        b: usize,
        mean_ctx: f64,
        steps: usize,
        step_ms_p50: f64,
        step_ms_p90: f64,
        tok_per_s: f64,
        serial_step_ms_p50: f64,
        serial_tok_per_s: f64,
        kv_bytes: usize,
        device_used_delta_mib: f64,
    }
    fn pct(v: &mut [f64], p: f64) -> f64 {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p).round() as usize]
    }
    fn device_used() -> f64 {
        let m = candle::rocm_backend::rocm_rs::hip::memory::memory_info().unwrap();
        (m.total - m.free) as f64 / (1u64 << 20) as f64
    }
    fn kv_bytes(model: &Model, states: &[State]) -> Result<usize> {
        // Committed KV plus convolution state, at f32; capacity slack excluded.
        let mut total = 0;
        for s in states {
            for (i, l) in model.layers.iter().enumerate() {
                if let (Operator::Attention(a), Some(_)) = (&l.operator, &s.kv[i]) {
                    total += 2 * a.n_kv_head * a.head_dim * s.len * 4;
                }
                if let Some(c) = &s.conv[i] {
                    total += c.row()?.elem_count() * 4;
                }
            }
        }
        Ok(total)
    }

    /// decode_batch with a device synchronisation after each part, charging
    /// the wall time to a category. It must give decode_batch's own logits
    /// (checked by the caller), so the mirror cannot drift from the model.
    fn profiled_step(
        model: &Model,
        tokens: &[u32],
        states: &mut [State],
        acc: &mut std::collections::BTreeMap<&'static str, f64>,
    ) -> Result<Tensor> {
        let dev = model.device.clone();
        let mut t = Instant::now();
        let mut lap = |acc: &mut std::collections::BTreeMap<&'static str, f64>,
                       k: &'static str|
         -> Result<()> {
            dev.synchronize()?;
            *acc.entry(k).or_default() += t.elapsed().as_secs_f64() * 1e3;
            t = Instant::now();
            Ok(())
        };
        let b = tokens.len();
        let pos: Vec<usize> = states.iter().map(|s| s.len).collect();
        let positions = Tensor::from_iter(pos.iter().map(|&p| p as u32), &model.device)?;
        let ids = Tensor::from_slice(tokens, (b, 1), &model.device)?;
        let mut x = model.embedding.embedding(&ids)?;
        lap(acc, "embed")?;
        for (i, layer) in model.layers.iter().enumerate() {
            let normed = layer.norm.forward(&x)?;
            lap(acc, "norms+residual")?;
            let y = match &layer.operator {
                Operator::Attention(a) => {
                    let (q, k, v) = a.qkv.forward(&normed)?;
                    let q = q.reshape((b, 1, a.n_head, a.head_dim))?.transpose(1, 2)?;
                    let k = k
                        .reshape((b, 1, a.n_kv_head, a.head_dim))?
                        .transpose(1, 2)?;
                    let v = v
                        .reshape((b, 1, a.n_kv_head, a.head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?;
                    let q = a.q_norm.forward(&q.contiguous()?)?;
                    let k = a.k_norm.forward(&k.contiguous()?)?;
                    let half = a.head_dim / 2;
                    let cos = a.cos.index_select(&positions, 0)?.reshape((b, 1, half))?;
                    let sin = a.sin.index_select(&positions, 0)?.reshape((b, 1, half))?;
                    let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
                    let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
                    lap(acc, "attn qkv+norm+rope (batched)")?;
                    let mut ys = Vec::with_capacity(b);
                    for (r, s) in states.iter_mut().enumerate() {
                        let next = KvCache::append(
                            s.kv[i].as_ref(),
                            &k.narrow(0, r, 1)?,
                            &v.narrow(0, r, 1)?,
                            a.cos.dim(0)?,
                        )?;
                        let (ki, vi) = next.current()?;
                        s.kv[i] = Some(next);
                        ys.push(attention::forward(
                            &q.narrow(0, r, 1)?,
                            &ki,
                            &vi,
                            None,
                            &a.neg_inf,
                        )?);
                    }
                    let y = Tensor::cat(&ys, 0)?
                        .transpose(1, 2)?
                        .reshape((b, 1, model.hidden))?;
                    lap(acc, "attn kv append+attention (per row)")?;
                    let y = a.wo.forward(&y)?;
                    lap(acc, "attn output proj (batched)")?;
                    y
                }
                Operator::Conv(c) => {
                    let mut conv: Vec<_> = states.iter_mut().map(|s| &mut s.conv[i]).collect();
                    let y = c.decode_batch(&normed, &mut conv)?;
                    lap(acc, "conv (batched)")?;
                    y
                }
            };
            x = (x + y)?;
            let f = layer.ffn_norm.forward(&x)?;
            lap(acc, "norms+residual")?;
            let f = layer.ffn.forward(&f, None, None)?;
            lap(
                acc,
                match layer.ffn {
                    super::super::FeedForward::Dense(_) => "dense ffn",
                    _ => "moe (router+experts)",
                },
            )?;
            x = (&x + f)?;
        }
        let x = model.norm.forward(&x)?.i((.., 0, ..))?.contiguous()?;
        let logits = model.output.forward(&x)?;
        lap(acc, "head")?;
        for s in states.iter_mut() {
            s.len += 1;
        }
        Ok(logits)
    }

    /// Prefill throughput of ONE sequence by chunk size, from a 512-token
    /// context: what a ragged multi-sequence prefill could gain is what larger
    /// chunks gain here, since the weight-bound work sees only the token count.
    #[test]
    #[ignore = "bench: requires ROCm, LFM25_GGUF, LFM25_PROMPTS"]
    fn rocm_prefill_chunk_sweep() -> Result<()> {
        let device = Device::new_rocm(0)?;
        let model = load(&device)?;
        let spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(env("LFM25_PROMPTS"))?).unwrap();
        let prompts: Vec<Vec<u32>> = serde_json::from_value(spec["prompts"].clone()).unwrap();
        let p = prompts
            .iter()
            .find(|p| p.len() >= 3072)
            .expect("a 3072-token prompt");
        let base = prefill(&model, p, &[512])?.remove(0);
        for chunk in [1usize, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            let mut ms = Vec::new();
            for round in 0..6 {
                memory_guard()?;
                let mut s = base.clone();
                let t = Instant::now();
                let l = model.forward(&p[512..512 + chunk], &mut s)?;
                let _ = l.argmax(D::Minus1)?.to_vec1::<u32>()?;
                if round > 0 {
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let m = pct(&mut ms, 0.5);
            eprintln!(
                "PREFILL_SWEEP chunk={chunk} ms={m:.2} tok_per_s={:.0}",
                chunk as f64 * 1e3 / m
            );
        }
        Ok(())
    }

    /// A short, repeatable decode timing for process-level A/B runs (kernel
    /// geometry, env switches): 8 prompts at 512 tokens, 40 timed steps at each
    /// B in LFM25_AB_BS (default 1,8,32). Prints one line per B.
    #[test]
    #[ignore = "bench: requires ROCm, LFM25_GGUF, LFM25_PROMPTS"]
    fn rocm_decode_ab_quick() -> Result<()> {
        let device = Device::new_rocm(0)?;
        let model = load(&device)?;
        let spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(env("LFM25_PROMPTS"))?).unwrap();
        let prompts: Vec<Vec<u32>> = serde_json::from_value(spec["prompts"].clone()).unwrap();
        let bs: Vec<usize> = std::env::var("LFM25_AB_BS")
            .unwrap_or("1,8,32".into())
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect();
        let pool: Vec<State> = prompts
            .iter()
            .filter(|p| p.len() > 512)
            .take(8)
            .map(|p| prefill(&model, p, &[512]).map(|mut v| v.remove(0)))
            .collect::<Result<_>>()?;
        for b in bs {
            memory_guard()?;
            let mut rows: Vec<State> = (0..b).map(|i| pool[i % pool.len()].clone()).collect();
            let mut tokens = vec![1u32; b];
            let mut ms = Vec::new();
            for step in 0..45 {
                let t = Instant::now();
                let mut refs: Vec<&mut State> = rows.iter_mut().collect();
                tokens = model
                    .decode_batch(&tokens, &mut refs)?
                    .argmax(D::Minus1)?
                    .to_vec1::<u32>()?;
                if step >= 5 {
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let p50 = pct(&mut ms, 0.5);
            eprintln!(
                "AB B={b} step_ms_p50={p50:.3} min={:.3} tok_per_s={:.1}",
                pct(&mut ms, 0.0),
                b as f64 * 1e3 / p50
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "profile: requires ROCm, LFM25_GGUF, LFM25_PROMPTS"]
    fn rocm_decode_batch_profile() -> Result<()> {
        let device = Device::new_rocm(0)?;
        let model = load(&device)?;
        let spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(env("LFM25_PROMPTS"))?).unwrap();
        let prompts: Vec<Vec<u32>> = serde_json::from_value(spec["prompts"].clone()).unwrap();
        let ctx: usize = std::env::var("LFM25_PROFILE_CTX").map_or(512, |v| v.parse().unwrap());
        let pool: Vec<State> = prompts
            .iter()
            .filter(|p| p.len() > ctx)
            .map(|p| prefill(&model, p, &[ctx]).map(|mut v| v.remove(0)))
            .collect::<Result<_>>()?;
        for b in [1usize, 8, 32] {
            let mut rows: Vec<State> = (0..b).map(|i| pool[i % pool.len()].clone()).collect();
            let tokens: Vec<u32> = (0..b as u32).map(|i| 1000 + i).collect();
            // Warm and fork every row off the shared pool storage first.
            let mut warm: Vec<&mut State> = rows.iter_mut().collect();
            model.decode_batch(&tokens, &mut warm)?;
            let mut acc = std::collections::BTreeMap::new();
            let steps = 10;
            for _ in 0..steps {
                // The profiled rows append in place; the untimed twin forks.
                let mut twin = rows.clone();
                let got = host_rows(&profiled_step(&model, &tokens, &mut rows, &mut acc)?)?;
                let mut refs: Vec<&mut State> = twin.iter_mut().collect();
                let want = host_rows(&model.decode_batch(&tokens, &mut refs)?)?;
                assert!(
                    got == want,
                    "profiled mirror diverged from decode_batch at B={b}"
                );
            }
            let total: f64 = acc.values().sum();
            eprintln!(
                "PROFILE B={b} ctx={ctx}: {:.2} ms/step (synchronised)",
                total / steps as f64
            );
            for (k, v) in &acc {
                eprintln!(
                    "PROFILE   {k:<38} {:>8.3} ms  {:>5.1}%",
                    v / steps as f64,
                    100. * v / total
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "bench: requires ROCm, LFM25_GGUF, LFM25_PROMPTS, LFM25_BENCH_OUT"]
    fn rocm_decode_batch_bench() -> Result<()> {
        let device = Device::new_rocm(0)?;
        let used0 = device_used();
        let model = load(&device)?;
        let loaded = device_used();
        let spec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(env("LFM25_PROMPTS"))?).unwrap();
        let prompts: Vec<Vec<u32>> = serde_json::from_value(spec["prompts"].clone()).unwrap();
        let bs = [1usize, 2, 4, 8, 16, 32];
        let t0 = Instant::now();
        // Snapshots at 16, 512 and 2048 tokens (when the prompt reaches them)
        // and one token short of the prompt's full length.
        let mut snaps: Vec<Vec<State>> = Vec::new();
        for p in &prompts {
            let mut stops: Vec<usize> = [16, 512, 2048]
                .into_iter()
                .filter(|&s| s < p.len())
                .collect();
            stops.push(p.len() - 1);
            snaps.push(prefill(&model, p, &stops)?);
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        let at = |len: usize| -> Vec<State> {
            snaps
                .iter()
                .flatten()
                .filter(|s| s.len() == len)
                .cloned()
                .collect()
        };
        let regimes: Vec<(String, Vec<State>)> = vec![
            ("ctx16".into(), at(16)),
            ("ctx512".into(), at(512)),
            ("ctx2048".into(), at(2048)),
            (
                "mixed".into(),
                snaps.iter().map(|s| s.last().unwrap().clone()).collect(),
            ),
        ];
        eprintln!(
            "prefill {prefill_s:.1}s for {} tokens",
            prompts.iter().map(Vec::len).sum::<usize>()
        );

        // Drift: batched vs batch-1, teacher-forced on the batch-1 greedy token.
        // LFM25_BENCH_PHASES picks among drift and timing (default both);
        // LFM25_BENCH_SERIAL=0 skips the serial baseline.
        let phases = std::env::var("LFM25_BENCH_PHASES").unwrap_or("drift,timing".into());
        let serial_on = std::env::var("LFM25_BENCH_SERIAL").map_or(true, |v| v != "0");
        let mixed = &regimes[3].1;
        let mut drift = Vec::new();
        let steps = 8;
        for &b in bs.iter().filter(|_| phases.contains("drift")) {
            let mut batched: Vec<State> = mixed[..b].to_vec();
            let mut single: Vec<State> = mixed[..b].to_vec();
            // The snapshots stop one token short: the first step feeds each
            // prompt's real last token, later steps the batch-1 greedy token.
            let mut tokens: Vec<u32> = (0..b).map(|i| *prompts[i].last().unwrap()).collect();
            let (mut d1, mut d20, mut agree) = (Vec::new(), Vec::new(), 0);
            let (mut dp, mut kl, mut s0d1, mut s0kl) = (Vec::new(), Vec::new(), 0f64, 0f64);
            for step in 0..steps {
                memory_guard()?;
                let mut refs: Vec<&mut State> = batched.iter_mut().collect();
                let got = host_rows(&model.decode_batch(&tokens, &mut refs)?)?;
                for i in 0..b {
                    let want = host_rows(&model.forward(&[tokens[i]], &mut single[i])?)?.remove(0);
                    let (lw, lg) = (log_softmax(&want), log_softmax(&got[i]));
                    let top = argmax(&want);
                    agree += usize::from(argmax(&got[i]) == top);
                    d1.push((lg[top as usize] - lw[top as usize]).abs());
                    let mut order: Vec<usize> = (0..lw.len()).collect();
                    order.select_nth_unstable_by(20, |a, b| lw[*b].total_cmp(&lw[*a]));
                    d20.push(
                        order[..20]
                            .iter()
                            .map(|&j| (lg[j] - lw[j]).abs())
                            .fold(0., f64::max),
                    );
                    let mut k = 0.;
                    let mut pmax = 0f64;
                    for (w, g) in lw.iter().zip(&lg) {
                        let p = w.exp();
                        k += p * (w - g);
                        if p >= 0.01 {
                            pmax = pmax.max((w - g).abs());
                        }
                    }
                    dp.push(pmax);
                    kl.push(k);
                    if step == 0 {
                        s0d1 = s0d1.max(*d1.last().unwrap());
                        s0kl = s0kl.max(k);
                    }
                    tokens[i] = top;
                }
            }
            let n = d1.len();
            let row = Drift {
                b,
                rows_x_steps: n,
                top1_agree: agree,
                dtop1_nats_median: pct(&mut d1, 0.5),
                dtop1_nats_max: pct(&mut d1, 1.0),
                dtop20_nats_median: pct(&mut d20, 0.5),
                dtop20_nats_max: pct(&mut d20, 1.0),
                dp1pct_nats_max: pct(&mut dp, 1.0),
                kl_median: pct(&mut kl, 0.5),
                kl_max: pct(&mut kl, 1.0),
                step0_dtop1_nats_max: s0d1,
                step0_kl_max: s0kl,
            };
            eprintln!(
                "drift B={b}: top1 {agree}/{n}, top1 |d| med {:.4} max {:.4}, top20 max|d| med {:.4} max {:.4}, p>=1% max|d| {:.4}, KL med {:.2e} max {:.2e}, step0 top1 max {:.4} KL max {:.2e}",
                row.dtop1_nats_median, row.dtop1_nats_max, row.dtop20_nats_median, row.dtop20_nats_max,
                row.dp1pct_nats_max, row.kl_median, row.kl_max, row.step0_dtop1_nats_max, row.step0_kl_max
            );
            drift.push(row);
        }

        // Throughput: per-step wall time including the device-to-host read of
        // each row's greedy token, which a real decode loop needs anyway.
        let (warm, timed) = (3, 24);
        let mut timing = Vec::new();
        for (name, pool) in regimes.iter().filter(|_| phases.contains("timing")) {
            for &b in &bs {
                memory_guard()?;
                let rows: Vec<State> = (0..b).map(|i| pool[i % pool.len()].clone()).collect();
                let mean_ctx = rows.iter().map(|s| s.len() as f64).sum::<f64>() / b as f64;
                let mut batched = rows.clone();
                let mut tokens = vec![1u32; b];
                let mut ms = Vec::new();
                for step in 0..warm + timed {
                    let t = Instant::now();
                    let mut refs: Vec<&mut State> = batched.iter_mut().collect();
                    let logits = model.decode_batch(&tokens, &mut refs)?;
                    tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
                    if step >= warm {
                        ms.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                let used = device_used();
                let kv = kv_bytes(&model, &batched)?;
                drop(batched);
                // Serial baseline: the same rows, one batch-1 forward each.
                let mut serial = rows.clone();
                let mut stoks = vec![1u32; b];
                let mut sms = Vec::new();
                for step in 0..(warm + timed / 2) * usize::from(serial_on) {
                    let t = Instant::now();
                    for (i, s) in serial.iter_mut().enumerate() {
                        let l = model.forward(&[stoks[i]], s)?;
                        stoks[i] = l.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
                    }
                    if step >= warm {
                        sms.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                let p50 = pct(&mut ms, 0.5);
                let sp50 = if sms.is_empty() {
                    f64::NAN
                } else {
                    pct(&mut sms, 0.5)
                };
                let row = Timing {
                    regime: name.clone(),
                    b,
                    mean_ctx,
                    steps: timed,
                    step_ms_p50: p50,
                    step_ms_p90: pct(&mut ms, 0.9),
                    tok_per_s: b as f64 * 1e3 / p50,
                    serial_step_ms_p50: sp50,
                    serial_tok_per_s: b as f64 * 1e3 / sp50,
                    kv_bytes: kv,
                    device_used_delta_mib: used - loaded,
                };
                eprintln!("{name} B={b} ctx {mean_ctx:.0}: step p50 {p50:.2} ms p90 {:.2} -> {:.1} tok/s; serial {sp50:.2} ms -> {:.1} tok/s; kv {:.1} MiB",
                    row.step_ms_p90, row.tok_per_s, row.serial_tok_per_s, kv as f64 / 1048576.);
                timing.push(row);
            }
        }
        let record = serde_json::json!({
            "prompts_sha256": spec["sha256"],
            "phases": phases,
            "moe_row_major_forced": std::env::var("CANDLE_ROCM_MOE_ROW_MAJOR").ok(),
            "force_dmmv": std::env::var("CANDLE_ROCM_FORCE_DMMV").ok(),
            "prompt_lengths": prompts.iter().map(Vec::len).collect::<Vec<_>>(),
            "prefill_seconds": prefill_s,
            "device_used_mib_before_load": used0,
            "device_used_mib_after_load": loaded,
            "drift_steps_teacher_forced": steps,
            "drift": drift,
            "timing": timing,
        });
        std::fs::write(
            env("LFM25_BENCH_OUT"),
            serde_json::to_string_pretty(&record).unwrap(),
        )?;
        Ok(())
    }
}
