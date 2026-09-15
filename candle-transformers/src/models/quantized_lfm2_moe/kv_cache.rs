//! Append-only KV storage, private to the LFM2 MoE state machine.
//!
//! Each view exposes only its committed prefix. A successful reservation owns
//! fresh tail positions forever, including when a later operation fails. An
//! older view therefore forks before writing a conflicting continuation. This
//! lets Model::forward keep its transactional state clone without copying KV
//! on every token. Tensor handles never escape the model's attention code.
//!
//! In-place append is enabled for CPU and ROCm only. Concurrent ROCm callers
//! must share the same Device instance (checked below), whose backend queues
//! every operation on one owning stream. The atomic is not a GPU fence. Other
//! backends retain the original concatenation path until their stream/lifetime
//! contracts are validated; in particular CUDA's default stream is per-thread.
use candle::{bail, Result, Tensor};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
struct Storage {
    k: Tensor,
    v: Tensor,
    // Monotonic reservation boundary, not the length of any committed view.
    // Never roll it back: a failed/asynchronous write may have touched the tail.
    reserved: AtomicUsize,
}

#[derive(Debug, Clone)]
pub(super) struct KvCache {
    storage: Arc<Storage>,
    len: usize,
}

impl KvCache {
    pub(super) fn append(
        previous: Option<&Self>,
        k: &Tensor,
        v: &Tensor,
        limit: usize,
    ) -> Result<Self> {
        let (batch, heads, seq, width) = k.dims4()?;
        if batch == 0
            || heads == 0
            || seq == 0
            || width == 0
            || k.dims() != v.dims()
            || k.dtype() != v.dtype()
            || !k.device().same_device(v.device())
        {
            bail!("invalid LFM2 KV pair")
        }
        let len = previous.map_or(0, |p| p.len);
        let end = len
            .checked_add(seq)
            .filter(|&end| end <= limit)
            .ok_or_else(|| candle::Error::Msg("LFM2 KV context limit exceeded".into()))?;
        if let Some(p) = previous {
            let old = &p.storage.k;
            let (b, h, _, d) = old.dims4()?;
            if (b, h, d) != (batch, heads, width)
                || old.dtype() != k.dtype()
                || !old.device().same_device(k.device())
            {
                bail!("LFM2 KV append shape, dtype, or device mismatch")
            }
        }
        // Explicit backend policy, never an error-triggered retry/fallback.
        if !k.device().is_cpu() && !k.device().is_rocm() {
            return Self::append_concat(previous, k, v, end);
        }
        // slice_set requires contiguous sources. Do fallible preparation before
        // reserving any shared storage. Sources may be offset/strided views.
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let reusable = previous.filter(|p| {
            end <= p.storage.k.dims()[2]
                && p.storage
                    .reserved
                    .compare_exchange(len, end, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
        });
        let storage = match reusable {
            Some(p) => p.storage.clone(),
            None => {
                // Geometric growth bounds total copying along a linear decode.
                // A conflicting branch sizes from its own prefix, not a sibling's
                // possibly much larger allocation. Every allocation is context-bounded.
                let capacity = end
                    .max(128)
                    .checked_next_power_of_two()
                    .unwrap_or(limit)
                    .min(limit);
                let shape = (batch, heads, capacity, width);
                let storage = Arc::new(Storage {
                    k: Tensor::zeros(shape, k.dtype(), k.device())?,
                    v: Tensor::zeros(shape, v.dtype(), v.device())?,
                    reserved: AtomicUsize::new(end),
                });
                if let Some(p) = previous {
                    let (pk, pv) = p.current()?;
                    storage.k.slice_set(&pk.contiguous()?, 2, 0)?;
                    storage.v.slice_set(&pv.contiguous()?, 2, 0)?;
                }
                storage
            }
        };
        // Only this reservation can write [len, end). All existing views end at
        // or before len. Failure may waste capacity but cannot change their data.
        storage.k.slice_set(&k, 2, len)?;
        storage.v.slice_set(&v, 2, len)?;
        Ok(Self { storage, len: end })
    }

    fn append_concat(previous: Option<&Self>, k: &Tensor, v: &Tensor, end: usize) -> Result<Self> {
        let (k, v) = match previous {
            None => (k.clone(), v.clone()),
            Some(p) => {
                let (pk, pv) = p.current()?;
                (Tensor::cat(&[&pk, k], 2)?, Tensor::cat(&[&pv, v], 2)?)
            }
        };
        Ok(Self {
            storage: Arc::new(Storage {
                k,
                v,
                reserved: AtomicUsize::new(end),
            }),
            len: end,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn current(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.storage.k.narrow(2, 0, self.len)?,
            self.storage.v.narrow(2, 0, self.len)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn values(t: &Tensor) -> Result<Vec<f32>> {
        t.flatten_all()?.to_vec1::<f32>()
    }

    fn chunk(start: usize, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        // Multiple batches and heads catch incorrectly copying a padded sequence stride.
        let data = (0..2)
            .flat_map(|b| {
                (0..3).flat_map(move |h| {
                    (start..start + len).flat_map(move |s| {
                        (0..4).map(move |d| (b * 10000 + h * 1000 + s * 4 + d) as f32)
                    })
                })
            })
            .collect::<Vec<_>>();
        let k = Tensor::from_vec(data, (2, 3, len, 4), device)?;
        let v = (&k + 0.5)?;
        Ok((k, v))
    }

    fn check(cache: &KvCache, k: &Tensor, v: &Tensor) -> Result<()> {
        let (actual_k, actual_v) = cache.current()?;
        assert_eq!(values(&actual_k)?, values(k)?);
        assert_eq!(values(&actual_v)?, values(v)?);
        Ok(())
    }

    fn append_and_branch(device: &Device) -> Result<()> {
        let (k, v) = chunk(0, 3, device)?;
        let prefix = KvCache::append(None, &k, &v, 300)?;
        let frozen = prefix.current()?;
        let mut a = prefix.clone();
        let mut allocations = 1;
        for s in 3..300 {
            let (k, v) = chunk(s, 1, device)?;
            let next = KvCache::append(Some(&a), &k, &v, 300)?;
            allocations += usize::from(!Arc::ptr_eq(&a.storage, &next.storage));
            a = next;
        }
        // A growing continuation must not allocate/copy its prefix for every token.
        assert!(
            allocations <= 4,
            "{allocations} cache allocations for 297 appends"
        );
        let (ak, av) = chunk(0, 300, device)?;
        check(&a, &ak, &av)?;
        let (pk, pv) = chunk(0, 3, device)?;
        check(&prefix, &pk, &pv)?;
        assert_eq!(values(&frozen.0)?, values(&pk)?);
        assert_eq!(values(&frozen.1)?, values(&pv)?);
        // Branch at an older prefix after the original continuation grew twice.
        let (bk, bv) = chunk(400, 7, device)?;
        let b = KvCache::append(Some(&prefix), &bk, &bv, 300)?;
        assert!(!Arc::ptr_eq(&prefix.storage, &b.storage));
        check(
            &b,
            &Tensor::cat(&[&pk, &bk], 2)?,
            &Tensor::cat(&[&pv, &bv], 2)?,
        )?;
        check(&a, &ak, &av)?;
        check(&prefix, &pk, &pv)?;
        // Limit rejection leaves the most recent branch intact.
        assert!(KvCache::append(Some(&a), &bk, &bv, 300).is_err());
        check(&a, &ak, &av)?;
        Ok(())
    }

    #[test]
    fn sequential_appends_are_amortized_and_snapshots_stay_frozen() -> Result<()> {
        append_and_branch(&Device::Cpu)
    }

    #[test]
    fn abandoned_append_cannot_be_overwritten_by_a_retry() -> Result<()> {
        let d = Device::Cpu;
        let (pk, pv) = chunk(0, 5, &d)?;
        let prefix = KvCache::append(None, &pk, &pv, 64)?;
        let (ak, av) = chunk(100, 2, &d)?;
        let abandoned = KvCache::append(Some(&prefix), &ak, &av, 64)?;
        let frozen = abandoned.current()?;
        drop(abandoned); // Like a model forward failing after attention.
        let (bk, bv) = chunk(200, 3, &d)?;
        let retry = KvCache::append(Some(&prefix), &bk, &bv, 64)?;
        check(
            &retry,
            &Tensor::cat(&[&pk, &bk], 2)?,
            &Tensor::cat(&[&pv, &bv], 2)?,
        )?;
        assert_eq!(values(&frozen.0)?, values(&Tensor::cat(&[&pk, &ak], 2)?)?);
        assert_eq!(values(&frozen.1)?, values(&Tensor::cat(&[&pv, &av], 2)?)?);
        check(&prefix, &pk, &pv)?;
        Ok(())
    }

    #[test]
    fn invalid_pairs_do_not_change_a_saved_prefix() -> Result<()> {
        let (k, v) = chunk(0, 3, &Device::Cpu)?;
        let prefix = KvCache::append(None, &k, &v, 64)?;
        assert!(KvCache::append(Some(&prefix), &k, &v.narrow(2, 0, 2)?, 64).is_err());
        assert!(KvCache::append(Some(&prefix), &k, &v.to_dtype(DType::F64)?, 64).is_err());
        assert!(KvCache::append(Some(&prefix), &k, &v, 5).is_err());
        assert!(KvCache::append(None, &k.narrow(2, 0, 0)?, &v.narrow(2, 0, 0)?, 64).is_err());
        check(&prefix, &k, &v)?;
        Ok(())
    }

    #[test]
    fn strided_offset_sources_and_short_limits() -> Result<()> {
        let (k, v) = chunk(0, 9, &Device::Cpu)?;
        let k = k.narrow(2, 2, 5)?;
        let v = v.narrow(2, 2, 5)?;
        assert!(!k.is_contiguous());
        let cache = KvCache::append(None, &k, &v, 7)?;
        let (tail_k, tail_v) = chunk(20, 2, &Device::Cpu)?;
        let next = KvCache::append(Some(&cache), &tail_k, &tail_v, 7)?;
        check(
            &next,
            &Tensor::cat(&[&k, &tail_k], 2)?,
            &Tensor::cat(&[&v, &tail_v], 2)?,
        )?;
        assert!(next.storage.k.dim(2)? <= 7);
        Ok(())
    }

    fn partial_write_failure(device: &Device) -> Result<()> {
        let k = Tensor::zeros((1, 1, 3, 4), DType::F32, device)?;
        let v = (&k + 1.)?;
        let prefix = KvCache::append(None, &k, &v, 64)?;
        let sibling = prefix.clone();
        let tail_k = Tensor::full(99f32, (1, 1, 2, 4), device)?;
        // Deliberately alias the value source. slice_set rejects this after K
        // has been written, giving a real partial-write error without a test hook.
        let aliased_v = prefix.current()?.1.narrow(2, 0, 2)?;
        let err = KvCache::append(Some(&prefix), &tail_k, &aliased_v, 64).unwrap_err();
        assert!(err.to_string().contains("share their storage"), "{err}");
        let retry_k = Tensor::full(7f32, (1, 1, 2, 4), device)?;
        let retry_v = (&retry_k + 1.)?;
        let retry = KvCache::append(Some(&sibling), &retry_k, &retry_v, 64)?;
        // Read back only after enqueueing the retry: no artificial fence between
        // the failed asynchronous append and a sibling continuation. Prove K
        // was written, so swapping the K/V statements cannot weaken this witness.
        assert_eq!(
            values(&prefix.storage.k.narrow(2, 3, 2)?)?,
            values(&tail_k)?
        );
        assert!(!Arc::ptr_eq(&prefix.storage, &retry.storage));
        check(
            &retry,
            &Tensor::cat(&[&k, &retry_k], 2)?,
            &Tensor::cat(&[&v, &retry_v], 2)?,
        )?;
        check(&prefix, &k, &v)?;
        Ok(())
    }

    #[test]
    fn failure_between_key_and_value_writes_preserves_prefix_and_burns_tail() -> Result<()> {
        partial_write_failure(&Device::Cpu)
    }

    fn simultaneous_branches(device: &Device) -> Result<()> {
        let (pk, pv) = chunk(0, 3, device)?;
        let prefix = KvCache::append(None, &pk, &pv, 64)?;
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let handles = (0..4)
            .map(|i| {
                let prefix = prefix.clone();
                let barrier = barrier.clone();
                let device = device.clone();
                std::thread::spawn(move || -> Result<_> {
                    let (k, v) = chunk(50 + i * 10, 5, &device)?;
                    barrier.wait();
                    let branch = KvCache::append(Some(&prefix), &k, &v, 64)?;
                    Ok((branch, k, v))
                })
            })
            .collect::<Vec<_>>();
        let branches = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            branches
                .iter()
                .filter(|(b, _, _)| Arc::ptr_eq(&b.storage, &prefix.storage))
                .count(),
            1
        );
        for (i, (branch, k, v)) in branches.iter().enumerate() {
            for (other, _, _) in &branches[..i] {
                assert!(!Arc::ptr_eq(&branch.storage, &other.storage));
            }
            check(
                branch,
                &Tensor::cat(&[&pk, k], 2)?,
                &Tensor::cat(&[&pv, v], 2)?,
            )?;
        }
        check(&prefix, &pk, &pv)?;
        Ok(())
    }

    #[test]
    fn simultaneous_branches_never_overwrite_each_other() -> Result<()> {
        simultaneous_branches(&Device::Cpu)
    }

    #[test]
    fn conservative_concat_path_preserves_snapshots() -> Result<()> {
        let (k, v) = chunk(0, 3, &Device::Cpu)?;
        let prefix = KvCache::append_concat(None, &k, &v, 3)?;
        let (tail_k, tail_v) = chunk(10, 2, &Device::Cpu)?;
        let next = KvCache::append_concat(Some(&prefix), &tail_k, &tail_v, 5)?;
        assert!(!Arc::ptr_eq(&prefix.storage, &next.storage));
        check(&prefix, &k, &v)?;
        check(
            &next,
            &Tensor::cat(&[&k, &tail_k], 2)?,
            &Tensor::cat(&[&v, &tail_v], 2)?,
        )
    }

    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires actual ROCm hardware; device failure is an error"]
    fn rocm_append_growth_and_snapshot_isolation() -> Result<()> {
        let device = Device::new_rocm(0)?;
        append_and_branch(&device)?;
        partial_write_failure(&device)?;
        simultaneous_branches(&device)
    }
}
