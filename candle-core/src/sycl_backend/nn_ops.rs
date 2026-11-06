//! Native SYCL kernels for the candle-nn fused ops (softmax / rms_norm / rope).
//! Entry points called from `candle-nn`'s `sycl_fwd` impls, replacing the
//! host round-trip shim.
use super::{ffi_layout, k, storage_from_buffer, to_sycl_dtype, wrap, SyclStorage};
use crate::backend::BackendStorage;
use crate::{Layout, Result};

fn same_dense(s: &SyclStorage, numel: usize) -> Result<SyclStorage> {
    let buf = wrap(k::DeviceBuffer::alloc(
        s.device().queue(),
        numel * s.dtype().size_in_bytes(),
    ))?;
    Ok(storage_from_buffer(s.device(), buf, s.dtype(), numel))
}

/// A contiguous, zero-offset view of a storage: either the original, or a fresh
/// dense copy (for offset views such as a KV-cache slice). The kernels below all
/// assume dense buffers. `SyclStorage` is not `Clone`, hence the hand-rolled Cow.
enum Dense<'a> {
    Borrowed(&'a SyclStorage),
    Owned(SyclStorage),
}
impl Dense<'_> {
    fn get(&self) -> &SyclStorage {
        match self {
            Dense::Borrowed(s) => s,
            Dense::Owned(s) => s,
        }
    }
}

fn densify<'a>(s: &'a SyclStorage, l: &Layout, what: &str) -> Result<Dense<'a>> {
    if l.start_offset() == 0 && l.is_contiguous() {
        return Ok(Dense::Borrowed(s));
    }
    if !l.is_contiguous() {
        crate::bail!("sycl {what}: input must be contiguous");
    }
    let numel = l.shape().elem_count();
    let dense = same_dense(s, numel)?;
    wrap(k::copy_strided(
        s.device().queue(),
        to_sycl_dtype(s.dtype())?,
        &ffi_layout(l)?,
        s.buf(),
        dense.buf(),
        0,
        numel,
    ))?;
    Ok(Dense::Owned(dense))
}

pub fn softmax_last_dim(s: &SyclStorage, l: &Layout) -> Result<SyclStorage> {
    let s = densify(s, l, "softmax")?;
    let s = s.get();
    let dims = l.dims();
    let d = dims[dims.len() - 1];
    let rows = l.shape().elem_count() / d.max(1);
    let out = same_dense(s, l.shape().elem_count())?;
    wrap(k::softmax_lastdim(
        s.device().queue(),
        to_sycl_dtype(s.dtype())?,
        s.buf(),
        out.buf(),
        rows,
        d,
    ))?;
    Ok(out)
}

/// Elementwise sigmoid `1 / (1 + e^-x)`. candle-nn's `Sigmoid` is a `CustomOp1`
/// with no candle-core builtin, so it routes here rather than through
/// `unary_impl`.
pub fn sigmoid(s: &SyclStorage, l: &Layout) -> Result<SyclStorage> {
    let s = densify(s, l, "sigmoid")?;
    let s = s.get();
    let n = l.shape().elem_count();
    let out = same_dense(s, n)?;
    wrap(k::unary(
        s.device().queue(),
        k::UnaryOp::Sigmoid,
        to_sycl_dtype(s.dtype())?,
        &k::Layout::dense(),
        s.buf(),
        out.buf(),
        n,
    ))?;
    Ok(out)
}

pub fn rms_norm(
    x: &SyclStorage,
    xl: &Layout,
    alpha: &SyclStorage,
    al: &Layout,
    eps: f32,
) -> Result<SyclStorage> {
    let dims = xl.dims();
    let Some(&d) = dims.last() else {
        crate::bail!("sycl rms_norm: x must have at least one dimension");
    };
    // The CPU path takes both operands in one dtype and indexes `alpha` by the
    // last dimension; the kernel below assumes the same.
    if alpha.dtype() != x.dtype() {
        crate::bail!(
            "sycl rms_norm: alpha dtype {:?} != x dtype {:?}",
            alpha.dtype(),
            x.dtype()
        );
    }
    if al.shape().elem_count() != d {
        crate::bail!(
            "sycl rms_norm: alpha has {} elements, expected {d} (x's last dimension)",
            al.shape().elem_count()
        );
    }
    let x = densify(x, xl, "rms_norm x")?;
    let x = x.get();
    let alpha = densify(alpha, al, "rms_norm alpha")?;
    let alpha = alpha.get();
    let rows = xl.shape().elem_count() / d.max(1);
    let out = same_dense(x, xl.shape().elem_count())?;
    wrap(k::rms_norm(
        x.device().queue(),
        to_sycl_dtype(x.dtype())?,
        x.buf(),
        alpha.buf(),
        out.buf(),
        rows,
        d,
        eps,
    ))?;
    Ok(out)
}

/// `mode`: 0 interleaved (b,h,t,d), 1 half-split (b,h,t,d), 2 half-split (b,t,h,d).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    mode: u32,
    x: &SyclStorage,
    xl: &Layout,
    cos: &SyclStorage,
    cl: &Layout,
    sin: &SyclStorage,
    sl: &Layout,
) -> Result<SyclStorage> {
    if xl.shape().dims().len() != 4 {
        crate::bail!("sycl rope: x must be rank 4, got {:?}", xl.shape());
    }
    let (d0, d1, d2, d3) = xl.shape().dims4()?;
    if mode > 2 {
        crate::bail!("sycl rope: mode {mode} is not one of 0, 1, 2");
    }
    if d3 % 2 != 0 {
        crate::bail!("sycl rope: head dim {d3} must be even (pairs of elements rotate)");
    }
    // The CPU path takes all three operands in one dtype, and the kernel reads
    // cos/sin with one element type.
    if cos.dtype() != x.dtype() || sin.dtype() != x.dtype() {
        crate::bail!(
            "sycl rope: cos {:?} / sin {:?} dtype != x {:?}",
            cos.dtype(),
            sin.dtype(),
            x.dtype()
        );
    }
    // mode 0/1: (b, h, t, d); mode 2: (b, t, h, d).
    let (b, h, t) = if mode == 2 {
        (d0, d2, d1)
    } else {
        (d0, d1, d2)
    };
    let half = d3 / 2;
    let batched = cl.dims().len() == 3;
    if batched != (sl.dims().len() == 3) {
        crate::bail!(
            "sycl rope: cos has {} dims, sin has {}; both are batched (3) or unbatched",
            cl.dims().len(),
            sl.dims().len()
        );
    }
    // Batched cos/sin are (b, t, d/2): the kernel offsets batch b_i by b_i * t
    // rows, so each batch holds exactly t rows (candle_nn's wrappers cut a
    // longer batched table down before it gets here). An unbatched (rows, d/2)
    // table may run past the sequence, as on every other backend: the kernel
    // reads its first t rows.
    for (what, l) in [("cos", cl), ("sin", sl)] {
        let ok = match *l.dims() {
            [rb, rows, hd] => rb == b && rows == t && hd == half,
            [rows, hd] => rows >= t && hd == half,
            _ => false,
        };
        if !ok {
            let want = if batched {
                format!("({b}, {t}, {half})")
            } else {
                format!("(>= {t}, {half})")
            };
            crate::bail!(
                "sycl rope: {what} is {:?}, expected {want} for mode {mode} \
                 (b {b}, t {t}, half {half})",
                l.shape()
            );
        }
    }
    let x = densify(x, xl, "rope x")?;
    let x = x.get();
    let cos = densify(cos, cl, "rope cos")?;
    let cos = cos.get();
    let sin = densify(sin, sl, "rope sin")?;
    let sin = sin.get();
    let cos_batched = batched;
    let out = same_dense(x, xl.shape().elem_count())?;
    wrap(k::rope(
        x.device().queue(),
        mode,
        to_sycl_dtype(x.dtype())?,
        x.buf(),
        cos.buf(),
        sin.buf(),
        out.buf(),
        b,
        h,
        t,
        d3,
        cos_batched,
    ))?;
    Ok(out)
}
