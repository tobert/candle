// Requires a real Intel GPU + oneAPI. Run inside the `candle-sycl-dev` container.
use candle_sycl_kernels::*;

fn bytes_of<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn bytes_of_mut<T: Copy>(s: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

#[test]
fn device_opens() {
    let q = Queue::new(0).unwrap();
    let info = q.device_info().unwrap();
    println!(
        "{} | {} CU | {:.1} GB | integrated={} fp16={}",
        info.name,
        info.max_compute_units,
        info.global_mem_bytes as f64 / 1e9,
        info.is_integrated,
        info.supports_fp16
    );
    assert!(info.max_compute_units > 0);
}

#[test]
fn roundtrip_and_affine() {
    let q = Queue::new(0).unwrap();
    let host: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let buf = DeviceBuffer::alloc(&q, host.len() * 4).unwrap();
    buf.copy_from_host(bytes_of(&host)).unwrap();

    let out = DeviceBuffer::alloc(&q, host.len() * 4).unwrap();
    affine(
        &q,
        SyclDType::F32,
        &Layout::dense(),
        &buf,
        &out,
        host.len(),
        3.0,
        1.0,
    )
    .unwrap();

    let mut got = vec![0f32; 16];
    out.copy_to_host(bytes_of_mut(&mut got)).unwrap();
    for (i, v) in got.iter().enumerate() {
        assert_eq!(*v, i as f32 * 3.0 + 1.0, "idx {i}");
    }
}

#[test]
fn unary_exp_and_binary_add() {
    let q = Queue::new(0).unwrap();
    let a: Vec<f32> = vec![0.0, 1.0, 2.0];
    let b: Vec<f32> = vec![10.0, 20.0, 30.0];
    let da = DeviceBuffer::alloc(&q, 12).unwrap();
    let db = DeviceBuffer::alloc(&q, 12).unwrap();
    let dc = DeviceBuffer::alloc(&q, 12).unwrap();
    da.copy_from_host(bytes_of(&a)).unwrap();
    db.copy_from_host(bytes_of(&b)).unwrap();

    binary(
        &q,
        BinaryOp::Add,
        SyclDType::F32,
        &Layout::dense(),
        &da,
        &Layout::dense(),
        &db,
        &dc,
        3,
    )
    .unwrap();
    let mut c = vec![0f32; 3];
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    assert_eq!(c, vec![10.0, 21.0, 32.0]);

    unary(
        &q,
        UnaryOp::Exp,
        SyclDType::F32,
        &Layout::dense(),
        &da,
        &dc,
        3,
    )
    .unwrap();
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    assert!((c[0] - 1.0).abs() < 1e-5 && (c[1] - std::f32::consts::E).abs() < 1e-4);
}

#[test]
fn gemm_f32_identity() {
    let q = Queue::new(0).unwrap();
    // 2x3 @ 3x2 = 2x2, row-major, all ones -> each entry = 3
    let a = vec![1f32; 6];
    let b = vec![1f32; 6];
    let da = DeviceBuffer::alloc(&q, 24).unwrap();
    let db = DeviceBuffer::alloc(&q, 24).unwrap();
    let dc = DeviceBuffer::alloc(&q, 16).unwrap();
    da.copy_from_host(bytes_of(&a)).unwrap();
    db.copy_from_host(bytes_of(&b)).unwrap();
    gemm(
        &q,
        SyclDType::F32,
        false,
        false,
        2,
        2,
        3,
        1.0,
        0.0,
        &da,
        &db,
        &dc,
        1,
        0,
        0,
        0,
        0,
        0,
    )
    .unwrap();
    let mut c = vec![0f32; 4];
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    assert_eq!(c, vec![3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn batched_gemm_matches_loop() {
    let q = Queue::new(0).unwrap();
    let (batch, m, n, kk) = (4usize, 3usize, 2usize, 5usize);
    let mut a = vec![0f32; batch * m * kk];
    let mut b = vec![0f32; batch * kk * n];
    for (i, x) in a.iter_mut().enumerate() {
        *x = (i % 7) as f32 - 3.0;
    }
    for (i, x) in b.iter_mut().enumerate() {
        *x = (i % 5) as f32 - 2.0;
    }
    let da = DeviceBuffer::alloc(&q, a.len() * 4).unwrap();
    let db = DeviceBuffer::alloc(&q, b.len() * 4).unwrap();
    let dc = DeviceBuffer::alloc(&q, batch * m * n * 4).unwrap();
    da.copy_from_host(bytes_of(&a)).unwrap();
    db.copy_from_host(bytes_of(&b)).unwrap();
    gemm(
        &q,
        SyclDType::F32,
        false,
        false,
        m as i64,
        n as i64,
        kk as i64,
        1.0,
        0.0,
        &da,
        &db,
        &dc,
        batch as i64,
        (m * kk) as i64,
        (kk * n) as i64,
        (m * n) as i64,
        0,
        0,
    )
    .unwrap();
    let mut c = vec![0f32; batch * m * n];
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    // reference
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..kk {
                    acc += a[bi * m * kk + i * kk + p] * b[bi * kk * n + p * n + j];
                }
                let got = c[bi * m * n + i * n + j];
                assert!(
                    (got - acc).abs() < 1e-3,
                    "b{bi} [{i},{j}]: got {got} want {acc}"
                );
            }
        }
    }
}

/// A wait failure seen by an internal flush (here `drain_pool`; also
/// `defer_free`'s batched free) cannot be returned from there, but must not be
/// lost: the next synchronize reports it, once.
#[test]
fn a_failed_internal_flush_reaches_synchronize() {
    let q = Queue::new(0).unwrap();
    q.fail_next_wait_for_test(7);
    q.drain_pool();
    let err = q
        .synchronize()
        .expect_err("the flush's failure must reach synchronize");
    assert!(err.to_string().contains("status 7"), "unexpected error: {err}");
    q.synchronize().unwrap();
}

/// As above, for the first device-to-host copy: async device-to-device copies
/// and memsets report execution failures only at a later wait, and a host
/// copy is where most callers look.
#[test]
fn a_failed_internal_flush_reaches_copy_to_host() {
    let q = Queue::new(0).unwrap();
    let buf = DeviceBuffer::alloc(&q, 16).unwrap();
    buf.memset_zero().unwrap();
    q.fail_next_wait_for_test(7);
    q.drain_pool();
    let mut host = [0u8; 16];
    let err = buf
        .copy_to_host(&mut host)
        .expect_err("the flush's failure must reach the host copy");
    assert!(err.to_string().contains("status 7"), "unexpected error: {err}");
    buf.copy_to_host(&mut host).unwrap();
    assert_eq!(host, [0u8; 16]);
}

/// A failure in the reporting wait itself is returned there and not again.
#[test]
fn a_failed_synchronize_is_reported_once() {
    let q = Queue::new(0).unwrap();
    q.fail_next_wait_for_test(7);
    q.synchronize().expect_err("the injected failure");
    q.synchronize().unwrap();
}

#[test]
#[should_panic(expected = "memcpy_dtod")]
fn copy_from_device_refuses_more_than_the_source_holds() {
    let q = Queue::new(0).unwrap();
    let src = DeviceBuffer::alloc(&q, 16).unwrap();
    let dst = DeviceBuffer::alloc(&q, 64).unwrap();
    let _ = dst.copy_from_device(&src, 32);
}

#[test]
#[should_panic(expected = "memcpy_dtod")]
fn copy_from_device_refuses_more_than_the_destination_holds() {
    let q = Queue::new(0).unwrap();
    let src = DeviceBuffer::alloc(&q, 64).unwrap();
    let dst = DeviceBuffer::alloc(&q, 16).unwrap();
    let _ = dst.copy_from_device(&src, 32);
}

/// Release builds too: a clamped view would silently alias the wrong bytes.
#[test]
#[should_panic(expected = "view_at")]
fn view_at_refuses_an_offset_past_the_end() {
    let q = Queue::new(0).unwrap();
    let buf = DeviceBuffer::alloc(&q, 16).unwrap();
    let _ = unsafe { buf.view_at(17) };
}

#[test]
fn view_at_the_end_is_empty() {
    let q = Queue::new(0).unwrap();
    let buf = DeviceBuffer::alloc(&q, 16).unwrap();
    assert_eq!(unsafe { buf.view_at(16) }.len_bytes(), 0);
}

#[test]
#[ignore = "benchmark: prints numbers, asserts nothing"]
fn alloc_cost() {
    let q = Queue::new(0).unwrap();
    let sz = 4096 * 4096 * 4;
    // warmup
    for _ in 0..3 {
        let _b = DeviceBuffer::alloc(&q, sz).unwrap();
    }
    let t = std::time::Instant::now();
    for _ in 0..50 {
        let _b = DeviceBuffer::alloc(&q, sz).unwrap();
    }
    println!(
        "malloc_device+free (64MB): {:.3} ms/iter",
        t.elapsed().as_secs_f64() * 1e3 / 50.0
    );

    let t = std::time::Instant::now();
    for _ in 0..50 {
        let _b = DeviceBuffer::alloc(&q, 16 * 1024).unwrap();
    }
    println!(
        "malloc_device+free (16KB): {:.3} ms/iter",
        t.elapsed().as_secs_f64() * 1e3 / 50.0
    );
}

#[test]
#[ignore = "benchmark: prints numbers, asserts nothing"]
fn mmvq_speed() {
    let q = Queue::new(0).unwrap();
    let (n, k, m) = (4096usize, 4096usize, 1usize);
    let nblk = k / 32;
    let wbytes = n * nblk * 18; // q4_0 block = 18 bytes
    let w = DeviceBuffer::alloc(&q, wbytes).unwrap();
    let act = DeviceBuffer::alloc(&q, m * k * 4).unwrap();
    let out = DeviceBuffer::alloc(&q, m * n * 4).unwrap();
    w.copy_from_host(&vec![1u8; wbytes]).unwrap();
    act.copy_from_host(bytes_of(&vec![1f32; m * k])).unwrap();
    for _ in 0..3 {
        mmvq(&q, GgmlDType::Q4_0, &w, &act, &out, n, k, m).unwrap();
    }
    let t = std::time::Instant::now();
    for _ in 0..100 {
        mmvq(&q, GgmlDType::Q4_0, &w, &act, &out, n, k, m).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 100.0;
    println!(
        "mmvq q4_0 4096x4096 m=1: {ms:.3} ms  ({:.1} GB/s)",
        wbytes as f64 / (ms * 1e-3) / 1e9
    );
}

#[test]
#[ignore = "benchmark: prints numbers, asserts nothing"]
fn kernel_launch_overhead() {
    let q = Queue::new(0).unwrap();
    let n = 1024usize;
    let a = DeviceBuffer::alloc(&q, n * 4).unwrap();
    let b = DeviceBuffer::alloc(&q, n * 4).unwrap();
    a.copy_from_host(bytes_of(&vec![1f32; n])).unwrap();
    // warmup
    for _ in 0..20 {
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &b, n, 2.0, 1.0).unwrap();
    }
    q.synchronize().unwrap();
    let t = std::time::Instant::now();
    let iters = 2000;
    for _ in 0..iters {
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &b, n, 2.0, 1.0).unwrap();
    }
    q.synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("affine(1024) enqueue: {us:.1} us/iter (no per-op sync)");

    // with an alloc each iter (simulating the op dispatch)
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let out = DeviceBuffer::alloc(&q, n * 4).unwrap();
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &out, n, 2.0, 1.0).unwrap();
    }
    q.synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("affine(1024) + pooled alloc: {us:.1} us/iter");

    // with a sync each iter (simulating the OLD behaviour)
    let t = std::time::Instant::now();
    for _ in 0..500 {
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &b, n, 2.0, 1.0).unwrap();
        q.synchronize().unwrap();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / 500.0;
    println!("affine(1024) + sync each: {us:.1} us/iter");
}

#[test]
#[ignore = "benchmark: prints numbers, asserts nothing"]
fn tiny_gemm_overhead() {
    let q = Queue::new(0).unwrap();
    // decode attention shapes
    let da = DeviceBuffer::alloc(&q, 128 * 4).unwrap();
    let db = DeviceBuffer::alloc(&q, 128 * 64 * 4).unwrap();
    let dc = DeviceBuffer::alloc(&q, 64 * 4).unwrap();
    da.copy_from_host(bytes_of(&vec![1f32; 128])).unwrap();
    db.copy_from_host(bytes_of(&vec![1f32; 128 * 64])).unwrap();
    for _ in 0..5 {
        gemm(
            &q,
            SyclDType::F32,
            false,
            false,
            1,
            64,
            128,
            1.0,
            0.0,
            &da,
            &db,
            &dc,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap();
    }
    q.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        gemm(
            &q,
            SyclDType::F32,
            false,
            false,
            1,
            64,
            128,
            1.0,
            0.0,
            &da,
            &db,
            &dc,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap();
    }
    let enq = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    q.synchronize().unwrap();
    let tot = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    println!("tiny gemm(1x64x128): enqueue {enq:.1}us  total {tot:.1}us");

    // mmvq decode shape (qwen 0.6b q_proj ~ 2048x1024 q4k)
    let nblk = 1024 / 256;
    let w = DeviceBuffer::alloc(&q, 2048 * nblk * 144).unwrap();
    let act = DeviceBuffer::alloc(&q, 1024 * 4).unwrap();
    let o = DeviceBuffer::alloc(&q, 2048 * 4).unwrap();
    w.copy_from_host(&vec![0u8; 2048 * nblk * 144]).unwrap();
    act.copy_from_host(bytes_of(&vec![1f32; 1024])).unwrap();
    for _ in 0..5 {
        mmvq(&q, GgmlDType::Q4K, &w, &act, &o, 2048, 1024, 1).unwrap();
    }
    q.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        mmvq(&q, GgmlDType::Q4K, &w, &act, &o, 2048, 1024, 1).unwrap();
    }
    let enq = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    q.synchronize().unwrap();
    let tot = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    println!("mmvq q4k 2048x1024 m=1: enqueue {enq:.1}us  total {tot:.1}us");
}
