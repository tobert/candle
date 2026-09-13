use candle::rocm_backend::{launch_config, RocmStorage, RocmStorageSlice};
use candle::{bail, CpuStorage, CustomOp1, CustomOp3, Layout, Result, Shape};
use std::ffi::c_void;
const SOURCE: &str = include_str!("kernels.cu");
fn arg<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}
fn ptr(s: &RocmStorage, l: &Layout) -> Result<*mut c_void> {
    let RocmStorageSlice::F32(s) = &s.slice else {
        bail!("expected f32 ROCm input")
    };
    if !l.is_contiguous() {
        bail!("expected contiguous ROCm input")
    }
    Ok(unsafe { s.ptr_at(l.start_offset()) })
}
pub(super) struct Step {
    pub b: usize,
    pub h: usize,
    pub k: usize,
}
impl CustomOp3 for Step {
    fn name(&self) -> &'static str {
        "lfm2-conv-step"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        bail!("ROCm-only fused operation")
    }
    fn rocm_fwd(
        &self,
        x: &RocmStorage,
        xl: &Layout,
        w: &RocmStorage,
        wl: &Layout,
        old: &RocmStorage,
        ol: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let dev = &x.device;
        let n = self.b * self.h * (self.k + 1);
        let output = dev.alloc::<f32>(n)?;
        let (x, w, old, out) = (ptr(x, xl)?, ptr(w, wl)?, ptr(old, ol)?, output.as_ptr());
        let f = dev.get_or_load_custom_func("lfm2_conv_step", "candle_lfm2_ops_v1", SOURCE)?;
        let (grid, block) = launch_config(dev, self.b * self.h);
        f.launch(
            grid,
            block,
            0,
            Some(dev.stream()),
            &mut [
                arg(&x),
                arg(&w),
                arg(&old),
                arg(&out),
                arg(&self.b),
                arg(&self.h),
                arg(&self.k),
            ],
        )
        .map_err(candle::Error::wrap)?;
        Ok((
            RocmStorage {
                slice: RocmStorageSlice::F32(output),
                device: dev.clone(),
            },
            n.into(),
        ))
    }
}
pub(super) struct SwiGlu;
impl CustomOp1 for SwiGlu {
    fn name(&self) -> &'static str {
        "lfm2-swiglu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("ROCm-only fused operation")
    }
    fn rocm_fwd(&self, x: &RocmStorage, l: &Layout) -> Result<(RocmStorage, Shape)> {
        let dev = &x.device;
        let mut shape = l.dims().to_vec();
        let last = shape
            .last_mut()
            .ok_or_else(|| candle::Error::Msg("missing SwiGLU width".into()))?;
        *last /= 2;
        let h = *last;
        let n = l.shape().elem_count() / 2;
        let output = dev.alloc::<f32>(n)?;
        let (x, out) = (ptr(x, l)?, output.as_ptr());
        let f = dev.get_or_load_custom_func("lfm2_swiglu", "candle_lfm2_ops_v1", SOURCE)?;
        let (grid, block) = launch_config(dev, n);
        f.launch(
            grid,
            block,
            0,
            Some(dev.stream()),
            &mut [arg(&x), arg(&out), arg(&n), arg(&h)],
        )
        .map_err(candle::Error::wrap)?;
        Ok((
            RocmStorage {
                slice: RocmStorageSlice::F32(output),
                device: dev.clone(),
            },
            shape.into(),
        ))
    }
}
