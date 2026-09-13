use candle::quantized::gguf_file;
use candle::{Device, Result};
use candle_transformers::models::quantized_lfm2_moe::{Model, State};
fn model() -> Result<Model> {
    let mut f = std::io::Cursor::new(include_bytes!("fixtures/lfm2-moe/tiny.gguf"));
    let ct = gguf_file::Content::read(&mut f)?;
    Model::from_gguf(ct, &mut f, &Device::Cpu)
}
fn output(model: &Model, tokens: &[u32], state: &mut State) -> Result<Vec<f32>> {
    model
        .forward(tokens, state)?
        .flatten_all()?
        .to_vec1::<f32>()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 2e-5, "logit {i}: {a} vs {b}");
    }
}
#[test]
fn whole_and_incremental_logits_match_numpy() -> Result<()> {
    let m = model()?;
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lfm2-moe/reference.json")).unwrap();
    for c in cases.as_array().unwrap() {
        let tokens: Vec<u32> = serde_json::from_value(c["tokens"].clone()).unwrap();
        let logits: Vec<Vec<f32>> = serde_json::from_value(c["logits"].clone()).unwrap();
        close(
            &output(&m, &tokens, &mut m.new_state())?,
            logits.last().unwrap(),
        );
        let mut state = m.new_state();
        for (i, &t) in tokens.iter().enumerate() {
            close(&output(&m, &[t], &mut state)?, &logits[i]);
        }
        for split in 1..tokens.len() {
            let mut state = m.new_state();
            output(&m, &tokens[..split], &mut state)?;
            close(
                &output(&m, &tokens[split..], &mut state)?,
                logits.last().unwrap(),
            );
        }
    }
    Ok(())
}
#[test]
fn snapshot_a_b_a_and_rejections_leave_prefix_usable() -> Result<()> {
    let m = model()?;
    let mut prefix = m.new_state();
    output(&m, &[1, 2, 3], &mut prefix)?;
    let mut branch = prefix.clone();
    let a = output(&m, &[4, 5, 6], &mut branch)?;
    assert_eq!(prefix.len(), 3);
    output(&m, &[9, 8, 7], &mut prefix.clone())?;
    close(&a, &output(&m, &[4, 5, 6], &mut prefix.clone())?);
    close(&a, &output(&m, &[1, 2, 3, 4, 5, 6], &mut m.new_state())?);
    for invalid in [vec![], vec![16], vec![1; 32]] {
        assert!(m.forward(&invalid, &mut prefix).is_err());
        assert_eq!(prefix.len(), 3);
    }
    assert!(model()?.forward(&[4], &mut prefix).is_err());
    close(&a, &output(&m, &[4, 5, 6], &mut prefix)?);
    Ok(())
}
