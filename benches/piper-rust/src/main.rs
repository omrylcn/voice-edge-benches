//! Piper Rust benchmark — mirror of piper-python/bench.py.
//! Loads en_US-hfc_female-medium, runs N_RUNS per fixture sentence
//! (1 warmup discarded), writes WAV outputs and result.json.

use piper_rs::synth::PiperSpeechSynthesizer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const N_RUNS: usize = 10;
const DEFAULT_VOICE: &str = "en_US-ryan-high";

#[derive(Deserialize)]
struct Fixtures {
    sentences: Vec<Sentence>,
}

#[derive(Deserialize)]
struct Sentence {
    id: String,
    words: usize,
    text: String,
}

#[derive(Serialize)]
struct SentenceResult {
    id: String,
    words: usize,
    audio_duration_s: f64,
    sample_rate: u32,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    ttfa_ms: f64,
    rtf: f64,
    wav_bytes: usize,
    runs: usize,
}

#[derive(Serialize)]
struct Result {
    engine: &'static str,
    lang: &'static str,
    voice: String,
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
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 1 {
                if let Ok(kb) = parts[0].parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn write_wav(path: &Path, samples: &[f32], sr: u32) -> std::io::Result<usize> {
    let samples_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    let data_len = (samples_i16.len() * 2) as u32;
    let mut f = fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // channels
    f.write_all(&sr.to_le_bytes())?;
    f.write_all(&(sr * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in &samples_i16 {
        f.write_all(&s.to_le_bytes())?;
    }
    Ok(44 + data_len as usize)
}

/// Split text into sentences (keeps the terminator). Mirrors what piper's
/// Python lib does internally before synthesizing chunk-by-chunk.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            out.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out.retain(|s| !s.is_empty());
    if out.is_empty() { out.push(text.to_string()); }
    out
}

fn main() {
    // repo root = benches/piper-rust/../.. ; models via TTS_MODELS_DIR (default <repo>/models)
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()   // benches/
        .parent().unwrap()   // <repo>/
        .to_path_buf();
    let models_dir = std::env::var("TTS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("models"));
    // PIPER_VOICE switches the voice (default en_US-ryan-high; the default
    // keeps its historical results/piper-rust dir, others get a suffixed dir).
    let voice = std::env::var("PIPER_VOICE").unwrap_or_else(|_| DEFAULT_VOICE.to_string());
    let piper_dir = models_dir.join("piper");
    let onnx_path = piper_dir.join(format!("{}.onnx", voice));
    let config_path = piper_dir.join(format!("{}.onnx.json", voice));
    let fixtures_path = repo_root.join("fixtures/sentences.json");
    let results_dir = if voice == DEFAULT_VOICE {
        repo_root.join("results/piper-rust")
    } else {
        repo_root.join(format!("results/piper-rust-{}", voice))
    };
    fs::create_dir_all(&results_dir).unwrap();

    println!("Loading Piper voice: {}", voice);
    let rss_before = rss_mb();
    let t0 = Instant::now();
    let _ = &onnx_path; // 0.1.9 config'ten onnx yolunu kendi çözer
    let model = piper_rs::from_config_path(&config_path).expect("failed to load Piper config");
    let piper = PiperSpeechSynthesizer::new(model).expect("failed to build synthesizer");
    let cold_start = t0.elapsed();
    let rss_after = rss_mb();

    println!("  cold_start: {:.0} ms", cold_start.as_secs_f64() * 1000.0);
    println!("  rss delta:  {:.1} MB", rss_after - rss_before);
    println!();

    let fixtures: Fixtures =
        serde_json::from_str(&fs::read_to_string(&fixtures_path).unwrap()).unwrap();

    let mut all_results = Vec::new();
    let mut warmup_ms_total = 0.0f64;

    for (si, s) in fixtures.sentences.iter().enumerate() {
        println!("--- {} ({} words) ---", s.id, s.words);
        let mut timings_ms: Vec<f64> = Vec::new();
        let mut ttfa_ms_runs: Vec<f64> = Vec::new();
        let mut last_samples: Vec<f32> = Vec::new();
        let mut last_sr: u32 = 22050;

        let sr = piper.clone_model().audio_output_info().unwrap().sample_rate as u32;
        // PIPER_MODE: sentences (default — split text and synthesize
        // sentence-by-sentence, exactly what piper's Python lib does
        // internally; gives a real time-to-first-audio) | parallel | lazy |
        // streamed (needs a streaming-VITS model — standard voices panic).
        // piper-rs itself synthesizes the whole text as ONE task, so in the
        // non-sentence modes ttfa == total by construction.
        let mode = std::env::var("PIPER_MODE").unwrap_or_else(|_| "sentences".to_string());
        for i in 0..N_RUNS {
            let t0 = Instant::now();
            let mut samples: Vec<f32> = Vec::new();
            let mut ttfa_ms = 0.0f64;
            let mut stamp = |samples: &Vec<f32>, ttfa: &mut f64, t0: &Instant| {
                if samples.is_empty() {
                    *ttfa = t0.elapsed().as_secs_f64() * 1000.0;
                }
            };
            match mode.as_str() {
                "sentences" => {
                    for sentence in split_sentences(&s.text) {
                        let stream = piper.synthesize_parallel(sentence, None).expect("synth failed");
                        for chunk in stream {
                            stamp(&samples, &mut ttfa_ms, &t0);
                            samples.extend_from_slice(chunk.expect("chunk failed").samples.as_slice());
                        }
                    }
                }
                "parallel" => {
                    let stream = piper.synthesize_parallel(s.text.clone(), None).expect("synth failed");
                    for chunk in stream {
                        stamp(&samples, &mut ttfa_ms, &t0);
                        samples.extend_from_slice(chunk.expect("chunk failed").samples.as_slice());
                    }
                }
                "lazy" => {
                    let stream = piper.synthesize_lazy(s.text.clone(), None).expect("synth failed");
                    for chunk in stream {
                        stamp(&samples, &mut ttfa_ms, &t0);
                        samples.extend_from_slice(chunk.expect("chunk failed").samples.as_slice());
                    }
                }
                _ => {
                    // 100ms chunks (sr samples / 10), small padding
                    let sr_usize = sr as usize;
                    let stream = piper
                        .synthesize_streamed(s.text.clone(), None, sr_usize / 10, 3)
                        .expect("synth failed");
                    for chunk in stream {
                        stamp(&samples, &mut ttfa_ms, &t0);
                        samples.extend_from_slice(chunk.expect("chunk failed").as_slice());
                    }
                }
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if i > 0 {
                timings_ms.push(elapsed_ms);
                ttfa_ms_runs.push(ttfa_ms);
            } else if si == 0 {
                warmup_ms_total = elapsed_ms; // first-ever inference after load
            }
            last_samples = samples;
            last_sr = sr;
        }

        let wav_path = results_dir.join(format!("{}.wav", s.id));
        let wav_bytes = write_wav(&wav_path, &last_samples, last_sr).unwrap();

        let duration_s = last_samples.len() as f64 / last_sr as f64;
        let mut sorted = timings_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let mean = timings_ms.iter().sum::<f64>() / timings_ms.len() as f64;
        let ttfa = ttfa_ms_runs.iter().sum::<f64>() / ttfa_ms_runs.len() as f64;
        let rtf = (mean / 1000.0) / duration_s;

        println!(
            "  p50: {:.0} ms | p95: {:.0} ms | mean: {:.0} ms | ttfa: {:.0} ms",
            p50, p95, mean, ttfa
        );
        println!(
            "  audio: {:.2}s @ {}Hz | RTF: {:.3}",
            duration_s, last_sr, rtf
        );
        println!("  wav:   {} ({} bytes)", wav_path.file_name().unwrap().to_string_lossy(), wav_bytes);

        all_results.push(SentenceResult {
            id: s.id.clone(),
            words: s.words,
            audio_duration_s: (duration_s * 1000.0).round() / 1000.0,
            sample_rate: last_sr,
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
    let result = Result {
        engine: "piper",
        lang: "rust",
        voice,
        cold_start_ms: (cold_start.as_secs_f64() * 1000.0 * 10.0).round() / 10.0,
        warmup_ms: (warmup_ms_total * 10.0).round() / 10.0,
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
