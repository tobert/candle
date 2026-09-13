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
//!
//! A flag read once is not enough. A release that read it just before exit
//! began kept issuing driver calls straight through the teardown:
//! `RocmAllocator::release_all` looping `hipFree` over its free list on one
//! thread while another called `exit` (lfm2d's 2026-09-13 core dump, and
//! `candle-core/tests/rocm_exit_race.rs`, which crashed 24 of 24 times). So
//! every release is bracketed: [`begin_release`] counts it in and refuses
//! once exit has begun, and the hook waits, bounded, for every release
//! already counted in before letting HIP's teardown run.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Once;
use std::time::{Duration, Instant};

/// How long the exit hook waits for releases already in progress. Freeing a
/// few thousand blocks takes well under a second; the bound exists so a
/// stuck driver call cannot hang `exit` forever. Past it, HIP's teardown runs
/// under the release and the original crash is possible again.
const EXIT_WAIT: Duration = Duration::from_secs(10);

static GATE: ExitGate = ExitGate::new();
static EXIT_HOOK: Once = Once::new();

/// The exit flag plus a count of driver releases in progress.
///
/// Dekker-style: a release increments `in_flight` *before* reading `exiting`,
/// and the hook sets `exiting` *before* reading `in_flight`, all `SeqCst`.
/// Either the release sees the flag and backs out, or the hook sees the
/// release and waits for it; no release can start once the hook has finished
/// waiting. A counter rather than a lock, so nested releases on one thread (a
/// wrapper dropped inside another release) cannot deadlock.
struct ExitGate {
    exiting: AtomicBool,
    in_flight: AtomicUsize,
}

impl ExitGate {
    const fn new() -> Self {
        Self {
            exiting: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
        }
    }

    fn begin_release(&self) -> Option<ReleaseGuard<'_>> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        if self.exiting.load(Ordering::SeqCst) {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(ReleaseGuard(self))
    }

    /// Raise the flag, then wait up to `wait` for releases already admitted.
    /// `true` iff they all finished.
    fn mark_exiting(&self, wait: Duration) -> bool {
        self.exiting.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + wait;
        while self.in_flight.load(Ordering::SeqCst) != 0 {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }
}

/// An admitted driver release. Hold it for the whole release — every
/// `hipFree` of a loop, not just the first — and drop it when done.
pub struct ReleaseGuard<'a>(&'a ExitGate);

impl Drop for ReleaseGuard<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

extern "C" fn mark_process_exiting() {
    if !GATE.mark_exiting(EXIT_WAIT) {
        eprintln!(
            "candle rocm: a device release was still running {EXIT_WAIT:?} into exit; \
             the HIP runtime tears down under it"
        );
    }
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

/// Admit a driver release, or `None` once exit has begun (leak instead).
pub fn begin_release() -> Option<ReleaseGuard<'static>> {
    GATE.begin_release()
}

/// True once `exit` has begun.
pub fn process_exiting() -> bool {
    GATE.exiting.load(Ordering::SeqCst)
}

/// Drop `handle` unless the process is exiting, in which case leak it.
pub fn drop_unless_exiting<T>(handle: &mut ManuallyDrop<T>) {
    if let Some(_release) = begin_release() {
        // SAFETY: called exactly once, from the owning wrapper's `Drop`.
        unsafe { ManuallyDrop::drop(handle) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_waits_for_a_release_admitted_before_it() {
        let gate = ExitGate::new();
        let release = gate.begin_release().expect("admitted before exit");
        std::thread::scope(|s| {
            s.spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                drop(release);
            });
            let start = Instant::now();
            assert!(gate.mark_exiting(Duration::from_secs(5)));
            assert!(
                start.elapsed() >= Duration::from_millis(150),
                "exit did not wait for the in-flight release"
            );
        });
    }

    #[test]
    fn a_release_after_exit_is_refused_and_counts_itself_back_out() {
        let gate = ExitGate::new();
        assert!(gate.mark_exiting(Duration::ZERO));
        assert!(gate.begin_release().is_none());
        assert_eq!(gate.in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn exit_gives_up_on_a_stuck_release_at_the_deadline() {
        let gate = ExitGate::new();
        let release = gate.begin_release().expect("admitted before exit");
        let start = Instant::now();
        assert!(!gate.mark_exiting(Duration::from_millis(100)));
        assert!(start.elapsed() < Duration::from_secs(2));
        drop(release);
    }

    #[test]
    fn nested_releases_on_one_thread_are_all_waited_for() {
        let gate = ExitGate::new();
        let outer = gate.begin_release().expect("outer admitted");
        let inner = gate.begin_release().expect("inner admitted");
        std::thread::scope(|s| {
            s.spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                drop(inner);
                std::thread::sleep(Duration::from_millis(100));
                drop(outer);
            });
            let start = Instant::now();
            assert!(gate.mark_exiting(Duration::from_secs(5)));
            assert!(
                start.elapsed() >= Duration::from_millis(180),
                "exit returned before the outer release finished"
            );
        });
    }
}
