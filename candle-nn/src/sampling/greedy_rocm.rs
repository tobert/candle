use candle::rocm_backend::rocm_rs::hip::{Dim3, Function};
use candle::rocm_backend::{RocmDevice, RocmStorageSlice, SendSyncDeviceMemory};
use candle::{bail, Result, Storage, Tensor};
use std::ffi::c_void;
const SOURCE: &str = include_str!("greedy.cu");
fn arg<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}
pub(super) struct Sampler {
    dev: RocmDevice,
    seen: SendSyncDeviceMemory<u32>,
    values: SendSyncDeviceMemory<f32>,
    ids: SendSyncDeviceMemory<u32>,
    errors: SendSyncDeviceMemory<u32>,
    output: SendSyncDeviceMemory<u32>,
    tiles: u32,
    vocab: u32,
    scan: Function,
    finish: Function,
}
impl Sampler {
    pub(super) fn new(dev: &RocmDevice, seen: &[bool]) -> Result<Self> {
        let tiles = seen.len().div_ceil(1024);
        let scan = dev.get_or_load_custom_func("greedy_tiles", "candle_greedy_v1", SOURCE)?;
        let finish = dev.get_or_load_custom_func("greedy_finish", "candle_greedy_v1", SOURCE)?;
        Ok(Self {
            dev: dev.clone(),
            seen: dev.clone_htod(&seen.iter().map(|v| u32::from(*v)).collect::<Vec<_>>())?,
            values: dev.alloc::<f32>(tiles)?,
            ids: dev.alloc::<u32>(tiles)?,
            errors: dev.alloc::<u32>(tiles)?,
            output: dev.alloc::<u32>(1)?,
            tiles: tiles as u32,
            vocab: seen.len() as u32,
            scan,
            finish,
        })
    }
    pub(super) fn sample(&mut self, logits: &Tensor, penalty: f32) -> Result<u32> {
        let (storage, layout) = logits.storage_and_layout();
        let Storage::Rocm(storage) = &*storage else {
            bail!("expected ROCm logits")
        };
        let RocmStorageSlice::F32(src) = &storage.slice else {
            bail!("expected f32 logits")
        };
        // The public sampler verifies device, dtype, length and contiguity.
        let src = unsafe { src.ptr_at(layout.start_offset()) };
        let seen = self.seen.as_ptr();
        let values = self.values.as_ptr();
        let ids = self.ids.as_ptr();
        let errors = self.errors.as_ptr();
        let output = self.output.as_ptr();
        self.scan
            .launch(
                Dim3::new_1d(self.tiles),
                Dim3::new_1d(256),
                0,
                Some(self.dev.stream()),
                &mut [
                    arg(&src),
                    arg(&seen),
                    arg(&values),
                    arg(&ids),
                    arg(&errors),
                    arg(&self.vocab),
                    arg(&penalty),
                ],
            )
            .map_err(candle::Error::wrap)?;
        self.finish
            .launch(
                Dim3::new_1d(1),
                Dim3::new_1d(256),
                0,
                Some(self.dev.stream()),
                &mut [
                    arg(&values),
                    arg(&ids),
                    arg(&errors),
                    arg(&seen),
                    arg(&output),
                    arg(&self.tiles),
                ],
            )
            .map_err(candle::Error::wrap)?;
        // clone_dtoh -> copy_to_host synchronizes the owning stream before
        // its blocking HIP copy; the scalar is host-visible on return.
        let id = self.dev.clone_dtoh(&self.output)?[0];
        if id == u32::MAX {
            bail!("nonfinite model logits")
        }
        Ok(id)
    }
}
