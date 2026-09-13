//! A process that exits while another thread is releasing ROCm device memory
//! must not crash.
//!
//! libamdhip64 tears its runtime down in its own atexit handler, and any
//! `hipFree` issued after that faults instead of returning an error. The
//! `candle_rocm_kernels::exit` guard leaks device handles once exit has begun,
//! but it only looked at its flag *before* starting a release: a thread
//! already inside `RocmAllocator::release_all`'s free loop kept calling
//! `hipFree` straight through the teardown. lfm2d hit exactly that on
//! 2026-09-13 — a daemon logged a clean shutdown and then died of SIGSEGV
//! (core dump: `release_all` -> `hipFree` on its worker thread, racing
//! `main`'s `exit(0)`).
//!
//! The race needs a real process exit, so the parent test re-executes this
//! test binary. Each child parks a couple of thousand blocks on the
//! allocator's free list, hands the device to a thread that drops it (the
//! last owner, so the drop runs `release_all` over every block), and calls
//! `std::process::exit(0)` a few milliseconds later, spread across the free
//! loop. Every child must exit 0.
#![cfg(feature = "rocm")]

use std::process::{Command, Stdio};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use candle_core::{DType, Device, Tensor};

const CHILD_ENV: &str = "CANDLE_ROCM_EXIT_RACE_CHILD";
const ROUNDS: usize = 24;
const EXIT_DELAYS_MS: [u64; 6] = [0, 1, 2, 5, 10, 20];

/// The child half. It does nothing unless the parent re-executed this binary
/// with [`CHILD_ENV`] set; run by hand it passes without touching the GPU.
#[test]
fn exit_race_child() {
    let Ok(delay_ms) = std::env::var(CHILD_ENV) else {
        return;
    };
    let delay = Duration::from_millis(delay_ms.parse().expect("delay in ms"));
    let device = Device::new_rocm(0).expect("rocm device 0");
    {
        // Distinct 512-byte buckets up to ~1 GiB in total: each tensor gets its
        // own block, and dropping them parks every block on the free list.
        let tensors: Vec<Tensor> = (1..=2000)
            .map(|i| Tensor::zeros(i * 512, DType::U8, &device).expect("allocate"))
            .collect();
        drop(tensors);
    }
    let barrier = Arc::new(Barrier::new(2));
    let releaser = barrier.clone();
    std::thread::spawn(move || {
        releaser.wait();
        // Last owner: RocmAllocator::drop -> release_all -> hipFree per block.
        drop(device);
        std::thread::sleep(Duration::from_secs(60));
    });
    barrier.wait();
    std::thread::sleep(delay);
    std::process::exit(0);
}

#[test]
fn exiting_while_a_thread_releases_device_memory_does_not_crash() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let mut failures = Vec::new();
    for round in 0..ROUNDS {
        let delay_ms = EXIT_DELAYS_MS[round % EXIT_DELAYS_MS.len()];
        let status = Command::new(&exe)
            .args(["exit_race_child", "--exact", "--test-threads=1"])
            .env(CHILD_ENV, delay_ms.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("spawn the child test process");
        if !status.success() {
            failures.push((delay_ms, status.to_string()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {ROUNDS} processes crashed exiting during a device release \
         (exit delay ms, status): {failures:?}",
        failures.len()
    );
}
