use super::*;

#[test]
#[ignore = "requires actual ROCm hardware"]
fn quantizer_overwrites_dirty_payload_and_padding_rocm() -> Result<()> {
    let dev = RocmDevice::new(0)?;
    let block = GgmlDType::Q8_1.block_size();
    let bytes = GgmlDType::Q8_1.type_size();
    assert_eq!((block, bytes), (32, 36));
    let mut cases: Vec<_> = [1, 31, 32, 33, 255, 256, 511, 512, 513, 768, 2048]
        .into_iter()
        .map(|k| (k, 3))
        .collect();
    // Exercise the second launch and both source/destination row offsets.
    cases.push((1, MAX_GRID_Y + 1));
    for (k, rows) in cases {
        let source: Vec<f32> = (0..3 + k * rows)
            .map(|i| {
                if i % 7 == 0 {
                    0.
                } else {
                    (i as f32 / 17.).sin()
                }
            })
            .collect();
        let source = dev.clone_htod(&source)?;
        let size = buffer_bytes(k, rows);
        let stride = buffer_bytes(k, 1);
        let mut expected = None;
        for pattern in [0, 0xff, 0x5a] {
            let mut dst = dev.alloc::<u8>(size + 64)?;
            dst.memset(pattern).map_err(crate::Error::wrap)?;
            quantize_q8_1(&source, 3, &dst, k, rows, &dev)?;
            let actual = dev.clone_dtoh(&dst)?;
            assert!(
                actual[size..].iter().all(|&v| v == pattern as u8),
                "wrote beyond q8 payload"
            );
            let actual = &actual[..size];
            if let Some(expected) = &expected {
                assert!(
                    actual == expected,
                    "unwritten bytes: k={k}, rows={rows}, pattern={pattern}, first mismatch={:?}",
                    actual.iter().zip(expected).position(|(a, b)| a != b)
                );
            } else {
                expected = Some(actual.to_vec());
            }
            for row in actual.chunks_exact(stride) {
                // Padding in a partial block still requires zero quant values.
                for col in k..pad(k, MATRIX_ROW_PADDING) {
                    assert_eq!(row[col / block * bytes + 4 + col % block], 0);
                }
                // Fully padded blocks must have zero scale and sum as well.
                assert!(row[k.div_ceil(block) * bytes..].iter().all(|&v| v == 0));
            }
        }
    }
    Ok(())
}
