//! Supertonic Rust benchmark — mirror of Piper/Kokoro bench protocol.
//! Same fixtures, same N_RUNS, same output schema for direct comparison.
//! Adds a Turkish fixture set since this is Supertonic's killer feature.

mod helper;

use anyhow::Result;
use helper::{load_text_to_speech, load_voice_style, write_wav_file};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const N_RUNS: usize = 10;
// Model dir resolved at runtime from TTS_MODELS_DIR/supertonic (default <repo>/models/supertonic).
// Layout: <supertonic>/onnx/*.onnx + <supertonic>/voice_styles/F1.json
const VOICE_STYLE_REL: &str = "voice_styles/F1.json";
const ONNX_DIR_REL: &str = "onnx";
const DEFAULT_STEP: usize = 8; // SUPERTONIC_STEP overrides (e.g. 4)
const SPEED: f32 = 1.05;
const P: f32 = 0.3;

#[derive(Deserialize)]
struct Fixtures { sentences: Vec<Sentence> }

#[derive(Deserialize, Clone)]
struct Sentence { id: String, words: usize, text: String, lang: Option<String> }

#[derive(Serialize)]
struct SentenceResult {
    id: String,
    lang: String,
    words: usize,
    audio_duration_s: f64,
    sample_rate: i32,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    ttfa_ms: f64,
    rtf: f64,
    wav_bytes: usize,
    runs: usize,
}

#[derive(Serialize)]
struct BenchResult {
    engine: &'static str,
    lang: &'static str,
    voice: String,
    total_step: usize,
    speed: f32,
    cold_start_ms: f64,
    warmup_ms: f64,
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

fn main() -> Result<()> {
    // repo root = benches/supertonic-rust/../.. ; models via TTS_MODELS_DIR (default <repo>/models)
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()   // benches/
        .parent().unwrap()   // <repo>/
        .to_path_buf();
    let models_dir = std::env::var("TTS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("models"));
    let total_step: usize = std::env::var("SUPERTONIC_STEP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_STEP);
    let supertonic_dir = models_dir.join("supertonic");
    let onnx_dir = supertonic_dir.join(ONNX_DIR_REL);
    let voice_style = supertonic_dir.join(VOICE_STYLE_REL);
    let fixtures_path = repo_root.join("fixtures/sentences_multi.json");
    let results_dir = if total_step == DEFAULT_STEP {
        repo_root.join("results/supertonic-rust")
    } else {
        repo_root.join(format!("results/supertonic-rust-step{}", total_step))
    };
    fs::create_dir_all(&results_dir)?;
    let fixtures_text = fs::read_to_string(&fixtures_path)?;

    println!("Loading Supertonic (Rust)…");
    let rss_before = rss_mb();
    let t0 = Instant::now();
    let mut tts = load_text_to_speech(onnx_dir.to_str().unwrap(), false)?;
    let style = load_voice_style(&[voice_style.to_str().unwrap().to_string()], false)?;
    let cold_start = t0.elapsed();
    let rss_after = rss_mb();
    let sr = tts.sample_rate;

    println!("  cold_start: {:.0} ms", cold_start.as_secs_f64() * 1000.0);
    println!("  rss delta:  {:.1} MB", rss_after - rss_before);
    println!("  sample_rate: {} Hz", sr);
    println!();

    let fixtures: Fixtures = serde_json::from_str(&fixtures_text)?;

    let mut all_results = Vec::new();
    let mut warmup_ms_total = 0.0f64;

    for (si, s) in fixtures.sentences.iter().enumerate() {
        let lang = s.lang.clone().unwrap_or_else(|| "en".to_string());
        println!("--- {} [{}] ({} words) ---", s.id, lang, s.words);
        let mut timings_ms: Vec<f64> = Vec::new();
        let mut ttfa_ms_runs: Vec<f64> = Vec::new();
        let mut last_audio: Vec<f32> = Vec::new();
        let mut last_duration: f32 = 0.0;

        for i in 0..N_RUNS {
            let t0 = Instant::now();
            let mut ttfa_ms = 0.0f64;
            let (audio, duration) = tts.call_with_chunk_hook(
                &s.text, &lang, &style, total_step, SPEED, P,
                &mut |ci| {
                    if ci == 0 {
                        ttfa_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    }
                },
            )?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if i > 0 {
                timings_ms.push(elapsed_ms);
                ttfa_ms_runs.push(ttfa_ms);
            } else if si == 0 {
                warmup_ms_total = elapsed_ms; // first-ever inference after load
            }
            last_audio = audio;
            last_duration = duration;
        }

        // Trim to actual audio length using duration metadata
        let actual_len = (sr as f32 * last_duration) as usize;
        let wav_slice = &last_audio[..actual_len.min(last_audio.len())];

        let wav_path = results_dir.join(format!("{}.wav", s.id));
        write_wav_file(&wav_path, wav_slice, sr)?;
        let wav_bytes = fs::metadata(&wav_path)?.len() as usize;

        let duration_s = wav_slice.len() as f64 / sr as f64;
        let mut sorted = timings_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let mean = timings_ms.iter().sum::<f64>() / timings_ms.len() as f64;
        let ttfa = ttfa_ms_runs.iter().sum::<f64>() / ttfa_ms_runs.len() as f64;
        let rtf = (mean / 1000.0) / duration_s;

        println!("  p50: {:.0} ms | p95: {:.0} ms | mean: {:.0} ms | ttfa: {:.0} ms", p50, p95, mean, ttfa);
        println!("  audio: {:.2}s @ {}Hz | RTF: {:.3}", duration_s, sr, rtf);

        all_results.push(SentenceResult {
            id: s.id.clone(),
            lang,
            words: s.words,
            audio_duration_s: (duration_s * 1000.0).round() / 1000.0,
            sample_rate: sr,
            p50_ms: (p50 * 10.0).round() / 10.0,
            p95_ms: (p95 * 10.0).round() / 10.0,
            mean_ms: (mean * 10.0).round() / 10.0,
            ttfa_ms: (ttfa * 10.0).round() / 10.0,
            rtf: (rtf * 10000.0).round() / 10000.0,
            wav_bytes,
            runs: timings_ms.len(),
        });
    }

    let rss_final = rss_mb();
    let result = BenchResult {
        engine: "supertonic",
        lang: "rust",
        voice: PathBuf::from(VOICE_STYLE_REL).file_stem().unwrap().to_string_lossy().to_string(),
        total_step,
        speed: SPEED,
        cold_start_ms: (cold_start.as_secs_f64() * 1000.0 * 10.0).round() / 10.0,
        warmup_ms: (warmup_ms_total * 10.0).round() / 10.0,
        rss_after_load_mb: (rss_after * 10.0).round() / 10.0,
        rss_load_delta_mb: ((rss_after - rss_before) * 10.0).round() / 10.0,
        rss_final_mb: (rss_final * 10.0).round() / 10.0,
        sentences: all_results,
    };

    let out_path = results_dir.join("result.json");
    fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
    println!("\nResults → {}", out_path.display());
    println!("RSS final: {:.1} MB", rss_final);

    // Prevent ONNX session mutex-cleanup hang on drop
    std::mem::forget(tts);
    Ok(())
}
