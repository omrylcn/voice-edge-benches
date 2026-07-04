//! Kokoro Rust benchmark via Kokoros lib — mirror of kokoro-python/bench.py.
//! Same fixtures, same N_RUNS protocol, same output schema for direct comparison.

use kokoros::tts::koko::TTSKoko;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const N_RUNS: usize = 10;
const VOICE: &str = "af_heart";
const SR: u32 = 24000;

#[derive(Deserialize)]
struct Fixtures { sentences: Vec<Sentence> }

#[derive(Deserialize)]
struct Sentence { id: String, words: usize, text: String }

#[derive(Serialize)]
struct SentenceResult {
    id: String,
    words: usize,
    audio_duration_s: f64,
    sample_rate: u32,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    rtf: f64,
    wav_bytes: usize,
    runs: usize,
}

#[derive(Serialize)]
struct Result {
    engine: &'static str,
    lang: &'static str,
    voice: &'static str,
    cold_start_ms: f64,
    rss_after_load_mb: f64,
    rss_load_delta_mb: f64,
    rss_final_mb: f64,
    sentences: Vec<SentenceResult>,
}

fn rss_mb() -> f64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(kb_str) = rest.split_whitespace().next() {
                if let Ok(kb) = kb_str.parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn write_wav(path: &Path, samples: &[f32], sr: u32) -> std::io::Result<usize> {
    let samples_i16: Vec<i16> = samples.iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    let data_len = (samples_i16.len() * 2) as u32;
    let mut f = fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sr.to_le_bytes())?;
    f.write_all(&(sr * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in &samples_i16 { f.write_all(&s.to_le_bytes())?; }
    Ok(44 + data_len as usize)
}

#[tokio::main]
async fn main() {
    // repo root = benches/kokoro-rust/../.. ; models via TTS_MODELS_DIR (default <repo>/models)
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()   // benches/
        .parent().unwrap()   // <repo>/
        .to_path_buf();
    let models_dir = std::env::var("TTS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("models"));
    let model_path = models_dir.join("kokoro-v1.0.onnx");
    let voices_path = models_dir.join("voices-v1.0.bin");
    let fixtures_path = repo_root.join("fixtures/sentences.json");
    let results_dir = repo_root.join("results/kokoro-rust");
    fs::create_dir_all(&results_dir).unwrap();

    println!("Loading Kokoro voice: {}", VOICE);
    let rss_before = rss_mb();
    let t0 = Instant::now();
    let tts = TTSKoko::new(
        model_path.to_str().unwrap(),
        voices_path.to_str().unwrap(),
    ).await;
    let cold_start = t0.elapsed();
    let rss_after = rss_mb();

    println!("  cold_start: {:.0} ms", cold_start.as_secs_f64() * 1000.0);
    println!("  rss delta:  {:.1} MB", rss_after - rss_before);
    println!();

    let fixtures: Fixtures = serde_json::from_str(
        &fs::read_to_string(&fixtures_path).unwrap()
    ).unwrap();

    let mut all_results = Vec::new();

    for s in &fixtures.sentences {
        println!("--- {} ({} words) ---", s.id, s.words);
        let mut timings_ms: Vec<f64> = Vec::new();
        let mut last_samples: Vec<f32> = Vec::new();

        for i in 0..N_RUNS {
            let t0 = Instant::now();
            let samples = tts.tts_raw_audio(
                &s.text,
                "en-us",
                VOICE,
                1.0,
                None, None, None, None,
            ).expect("synth failed");
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if i > 0 { timings_ms.push(elapsed_ms); }
            last_samples = samples;
        }

        let wav_path = results_dir.join(format!("{}.wav", s.id));
        let wav_bytes = write_wav(&wav_path, &last_samples, SR).unwrap();

        let duration_s = last_samples.len() as f64 / SR as f64;
        let mut sorted = timings_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let mean = timings_ms.iter().sum::<f64>() / timings_ms.len() as f64;
        let rtf = (mean / 1000.0) / duration_s;

        println!("  p50: {:.0} ms | p95: {:.0} ms | mean: {:.0} ms", p50, p95, mean);
        println!("  audio: {:.2}s @ {}Hz | RTF: {:.3}", duration_s, SR, rtf);
        println!("  wav:   {} ({} bytes)", wav_path.file_name().unwrap().to_string_lossy(), wav_bytes);

        all_results.push(SentenceResult {
            id: s.id.clone(),
            words: s.words,
            audio_duration_s: (duration_s * 1000.0).round() / 1000.0,
            sample_rate: SR,
            p50_ms: (p50 * 10.0).round() / 10.0,
            p95_ms: (p95 * 10.0).round() / 10.0,
            mean_ms: (mean * 10.0).round() / 10.0,
            rtf: (rtf * 10000.0).round() / 10000.0,
            wav_bytes,
            runs: timings_ms.len(),
        });
    }

    let rss_final = rss_mb();
    let result = Result {
        engine: "kokoro",
        lang: "rust",
        voice: VOICE,
        cold_start_ms: (cold_start.as_secs_f64() * 1000.0 * 10.0).round() / 10.0,
        rss_after_load_mb: (rss_after * 10.0).round() / 10.0,
        rss_load_delta_mb: ((rss_after - rss_before) * 10.0).round() / 10.0,
        rss_final_mb: (rss_final * 10.0).round() / 10.0,
        sentences: all_results,
    };

    let out_path = results_dir.join("result.json");
    fs::write(&out_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("\nResults → {}", out_path.display());
    println!("RSS final: {:.1} MB", rss_final);
}
