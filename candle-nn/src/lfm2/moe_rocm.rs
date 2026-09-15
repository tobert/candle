use candle::op::BackpropOp;
use candle::rocm_backend::rocm_rs::hip::Dim3;
use candle::rocm_backend::{launch_config, RocmStorage, RocmStorageSlice};
use candle::{bail, CpuStorage, CustomOp2, Layout, Result, Shape, Storage, Tensor};
use std::ffi::c_void;

const SOURCE: &str = include_str!("moe.cu");
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

// Called only after the public function validates dimensions, dtype and device.
pub(super) fn route(logits: &Tensor, bias: &Tensor) -> Result<(Tensor, Tensor)> {
    let rows = logits.dim(0)?;
    let grid = u32::try_from(rows).map_err(candle::Error::wrap)?;
    let (xs, xl) = logits.storage_and_layout();
    let (bs, bl) = bias.storage_and_layout();
    let (Storage::Rocm(xs), Storage::Rocm(bs)) = (&*xs, &*bs) else {
        bail!("expected ROCm routing inputs")
    };
    let dev = &xs.device;
    let ids = dev.alloc::<u32>(rows * 4)?;
    let weights = dev.alloc::<f32>(rows * 4)?;
    let (x, bias, out_ids, out_weights) =
        (ptr(xs, xl)?, ptr(bs, bl)?, ids.as_ptr(), weights.as_ptr());
    let f = dev.get_or_load_custom_func("lfm2_route_32_4", "candle_lfm2_moe_v1", SOURCE)?;
    f.launch(
        Dim3::new_1d(grid),
        Dim3::new_1d(32),
        0,
        Some(dev.stream()),
        &mut [arg(&x), arg(&bias), arg(&out_ids), arg(&out_weights)],
    )
    .map_err(candle::Error::wrap)?;
    let tensor = |slice| {
        Tensor::from_storage(
            Storage::Rocm(RocmStorage {
                slice,
                device: dev.clone(),
            }),
            (rows, 4),
            BackpropOp::none(),
            false,
        )
    };
    Ok((
        tensor(RocmStorageSlice::U32(ids)),
        tensor(RocmStorageSlice::F32(weights)),
    ))
}

pub(super) struct Combine;
impl CustomOp2 for Combine {
    fn name(&self) -> &'static str {
        "lfm2-moe-combine"
    }
    fn cpu_fwd(
        &self,
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
    ) -> Result<(RocmStorage, Shape)> {
        let (rows, _, hidden) = xl.shape().dims3()?;
        let n = rows * hidden;
        let dev = &x.device;
        let output = dev.alloc::<f32>(n)?;
        let (x, w, out) = (ptr(x, xl)?, ptr(w, wl)?, output.as_ptr());
        let f = dev.get_or_load_custom_func("lfm2_combine_4", "candle_lfm2_moe_v1", SOURCE)?;
        let (grid, block) = launch_config(dev, n);
        f.launch(
            grid,
            block,
            0,
            Some(dev.stream()),
            &mut [arg(&x), arg(&w), arg(&out), arg(&n), arg(&hidden)],
        )
        .map_err(candle::Error::wrap)?;
        Ok((
            RocmStorage {
                slice: RocmStorageSlice::F32(output),
                device: dev.clone(),
            },
            (rows, hidden).into(),
        ))
    }
}
