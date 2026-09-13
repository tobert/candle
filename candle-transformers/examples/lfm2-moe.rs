//! Minimal GGUF/state parity probe. Arguments: GGUF, JSON token IDs, cpu|rocm.
use candle::quantized::gguf_file;
use candle::{Device, Result};
use candle_transformers::models::quantized_lfm2_moe::Model;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        candle::bail!("usage: lfm2-moe MODEL.gguf TOKENS.json cpu|rocm [PREFIX_LENGTH]")
    }
    let device = match args[3].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "rocm")]
        "rocm" => Device::new_rocm(0)?,
        #[cfg(not(feature = "rocm"))]
        "rocm" => candle::bail!("rebuild with --features rocm"),
        _ => candle::bail!("unknown device"),
    };
    let mut file = std::fs::File::open(&args[1])?;
    let ct = gguf_file::Content::read(&mut file)?;
    let start = std::time::Instant::now();
    let model = Model::from_gguf(ct, &mut file, &device)?;
    eprintln!("loaded {:?} in {:?}", device, start.elapsed());
    let tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read(&args[2])?).map_err(candle::Error::wrap)?;
    if let Some(split) = args.get(4) {
        let split: usize = split.parse().map_err(candle::Error::wrap)?;
        if split == 0 || split >= tokens.len() {
            candle::bail!("prefix must split token input")
        }
        let prefill = |parts: &[&[u32]]| -> Result<Vec<f32>> {
            let mut state = model.new_state();
            let mut last = None;
            for part in parts {
                for chunk in part.chunks(128) {
                    last = Some(model.forward(chunk, &mut state)?);
                }
            }
            last.unwrap().flatten_all()?.to_vec1::<f32>()
        };
        let cold = prefill(&[&tokens])?;
        let cached = prefill(&[&tokens[..split], &tokens[split..]])?;
        let max = cold
            .iter()
            .zip(&cached)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms = (cold
            .iter()
            .zip(&cached)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / cold.len() as f64)
            .sqrt();
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        eprintln!(
            "cold/cached fixed-input logits: max_abs={max}, rms={rms}, top1={}/{}",
            argmax(&cold),
            argmax(&cached)
        );
        return Ok(());
    }
    let mut state = model.new_state();
    let start = std::time::Instant::now();
    let mut logits = model.forward(&tokens, &mut state)?;
    let saved = state.clone();
    eprintln!("prefill {} tokens {:?}", tokens.len(), start.elapsed());
    let mut generated = Vec::new();
    for _ in 0..40 {
        let t = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
        generated.push(t);
        if t == 124900 {
            break;
        }
        logits = model.forward(&[t], &mut state)?;
    }
    println!(
        "{}",
        serde_json::to_string(&generated).map_err(candle::Error::wrap)?
    );
    let suffix = [42u32, 123, 456];
    let mut a = saved.clone();
    let a = model.forward(&suffix, &mut a)?;
    let mut b = saved.clone();
    let _ = model.forward(&[43, 44], &mut b)?;
    let mut again = saved.clone();
    let again = model.forward(&suffix, &mut again)?;
    let delta = (a - again)?.abs()?.max_all()?.to_scalar::<f32>()?;
    if delta != 0. {
        candle::bail!("branch reuse drift {delta}")
    }
    eprintln!("A/B/A isolation exact; total {:?}", start.elapsed());
    Ok(())
}
