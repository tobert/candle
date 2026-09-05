//! Process-exit guard shared by every HIP handle wrapper.
//!
//! A host that calls `std::process::exit` while another thread still owns
//! device state runs libamdhip64's own atexit teardown first; any
//! `hipFree`, `hipModuleUnload`, `rocrand_destroy_generator` or
//! `rocblas_destroy_handle` issued after that faults or corrupts the heap
//! instead of returning an error (CUDA returns cudaErrorCudartUnloading
//! there; HIP does not). The hook below is registered when the first device
//! is created — after `hipInit`, so it runs *before* HIP's handler, atexit
//! being LIFO — and raises a flag that turns every later release into a
//! leak. The OS reclaims device memory and handles at exit regardless.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

static PROCESS_EXITING: AtomicBool = AtomicBool::new(false);
static EXIT_HOOK: Once = Once::new();

extern "C" fn mark_process_exiting() {
    PROCESS_EXITING.store(true, Ordering::SeqCst);
}

/// Register the exit hook. Idempotent; call once a device exists.
pub fn register_exit_hook() {
    EXIT_HOOK.call_once(|| {
        // SAFETY: `mark_process_exiting` is a plain `extern "C" fn()`, which
        // is exactly the signature `atexit` takes.
        let rc = unsafe { libc::atexit(mark_process_exiting) };
        if rc != 0 {
            // Not fatal: the process merely keeps the crash-at-exit behaviour.
            eprintln!(
                "candle rocm: atexit registration failed ({rc}); device frees at exit are not guarded"
            );
        }
    });
}

/// True once `exit` has begun.
pub fn process_exiting() -> bool {
    PROCESS_EXITING.load(Ordering::SeqCst)
}

/// Drop `handle` unless the process is exiting, in which case leak it.
pub fn drop_unless_exiting<T>(handle: &mut ManuallyDrop<T>) {
    if !process_exiting() {
        // SAFETY: called exactly once, from the owning wrapper's `Drop`.
        unsafe { ManuallyDrop::drop(handle) }
    }
}
