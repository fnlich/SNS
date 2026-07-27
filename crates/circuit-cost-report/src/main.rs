//! Offline report of `estimate_circuit`'s constraint cost for every slice
//! sitting in a local circuit_cache directory -- the same figure
//! sn2-miner's `--max-heaviest-n` ranks circuits by, computed here without
//! needing a running miner or live validator traffic. Meant for sizing
//! `--max-heaviest-n`/`--cpu-pressure-threshold` against real data instead
//! of guessing.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Reports estimate_circuit's constraint cost for every slice in a circuit_cache directory")]
struct Args {
    /// Circuit cache directory to scan. Defaults to the same location
    /// sn2-miner uses.
    #[arg(long, default_value = "~/.bittensor/subnet-2/circuit_cache")]
    circuit_cache_dir: String,

    /// Write the full per-slice list to this CSV path (model_id, slice_num,
    /// constraints, declared_io_elements). Always prints a summary to
    /// stdout regardless of whether this is set.
    #[arg(long)]
    csv_out: Option<PathBuf>,

    /// How many of the heaviest (and lightest) slices to print inline.
    #[arg(long, default_value_t = 20)]
    top_n: usize,
}

struct SliceCost {
    model_id: String,
    slice_num: String,
    constraints: u64,
    declared_io_elements: u64,
}

fn find_slice_onnx(slice_dir: &Path) -> Result<PathBuf> {
    let payload_dir = slice_dir.join("payload");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&payload_dir)
        .with_context(|| format!("reading {}", payload_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
        .collect();
    candidates.sort();
    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        n if n > 1 => anyhow::bail!("multiple .onnx files in {}: {candidates:?}", payload_dir.display()),
        _ => anyhow::bail!("no .onnx file found in {}", payload_dir.display()),
    }
}

fn estimate_slice(
    backend: &dsperse::backend::jstprove::JstproveBackend,
    slice_dir: &Path,
) -> Result<(u64, u64)> {
    let circuit_path = slice_dir.join("jstprove").join("circuit.bundle");
    anyhow::ensure!(circuit_path.is_dir(), "no compiled bundle at {}", circuit_path.display());

    let onnx_path = find_slice_onnx(slice_dir)?;

    let params = backend
        .load_params(&circuit_path)
        .map_err(|e| anyhow::anyhow!("loading circuit params: {e}"))?
        .context("compiled bundle has no metadata")?;

    let declared_io_elements: u64 = params
        .inputs
        .iter()
        .chain(params.outputs.iter())
        .map(|io| io.shape.iter().product::<usize>() as u64)
        .sum();

    let (est_params, architecture, _wandb) =
        dsperse::converter::prepare_jstprove_artifacts(&onnx_path, params.weights_as_inputs)
            .map_err(|e| anyhow::anyhow!("preparing artifacts for estimate: {e}"))?;

    // 31-bit field, matching the goldilocks_ext4_whir proof_config observed
    // in this workspace's manifests; from_circuit_params otherwise derives
    // everything else (chunk_bits, scale_exponent, freivalds_reps,
    // rescale_config) straight from the circuit's own params, not this
    // default -- see dsperse.rs's estimate_circuit_constraints for the
    // matching logic this mirrors.
    let total = jstprove_circuits::api::estimate_circuit(&est_params, &architecture, 31).total();

    Ok((total, declared_io_elements))
}

fn percentile(sorted_desc: &[u64], p: f64) -> u64 {
    if sorted_desc.is_empty() {
        return 0;
    }
    // sorted_desc is heaviest-first; convert p (e.g. 0.95 = p95) to an index
    // counting from the heavy end.
    let idx = (((1.0 - p) * sorted_desc.len() as f64).round() as usize).min(sorted_desc.len() - 1);
    sorted_desc[idx]
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cache_dir = PathBuf::from(shellexpand::tilde(&args.circuit_cache_dir).to_string());
    anyhow::ensure!(
        cache_dir.is_dir(),
        "circuit cache directory not found: {}",
        cache_dir.display()
    );

    let backend = dsperse::backend::jstprove::JstproveBackend::new();

    let mut results: Vec<SliceCost> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut models_scanned: BTreeSet<String> = BTreeSet::new();

    for model_entry in std::fs::read_dir(&cache_dir)
        .with_context(|| format!("reading {}", cache_dir.display()))?
        .filter_map(|e| e.ok())
    {
        let model_name = model_entry.file_name().to_string_lossy().to_string();
        let Some(model_id) = model_name.strip_prefix("model_") else {
            continue;
        };

        let slices_dir = model_entry.path().join("slices");
        if !slices_dir.is_dir() {
            continue;
        }
        models_scanned.insert(model_id.to_string());

        for slice_entry in std::fs::read_dir(&slices_dir)
            .with_context(|| format!("reading {}", slices_dir.display()))?
            .filter_map(|e| e.ok())
        {
            let slice_num = slice_entry.file_name().to_string_lossy().to_string();
            if !slice_num.starts_with("slice_") {
                continue;
            }
            let slice_dir = slice_entry.path();

            match estimate_slice(&backend, &slice_dir) {
                Ok((constraints, declared_io_elements)) => {
                    results.push(SliceCost {
                        model_id: model_id.to_string(),
                        slice_num: slice_num.clone(),
                        constraints,
                        declared_io_elements,
                    });
                }
                Err(e) => {
                    errors.push(format!("model_{model_id}/{slice_num}: {e:#}"));
                }
            }
            print!(
                "\rscanned {} slice(s) across {} model(s), {} error(s)...",
                results.len(),
                models_scanned.len(),
                errors.len()
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();

    anyhow::ensure!(
        !results.is_empty(),
        "no estimable slices found under {}",
        cache_dir.display()
    );

    results.sort_by(|a, b| b.constraints.cmp(&a.constraints));

    if let Some(csv_path) = &args.csv_out {
        let mut out = String::from("model_id,slice_num,constraints,declared_io_elements\n");
        for r in &results {
            out.push_str(&format!(
                "{},{},{},{}\n",
                r.model_id, r.slice_num, r.constraints, r.declared_io_elements
            ));
        }
        std::fs::write(csv_path, out)
            .with_context(|| format!("writing {}", csv_path.display()))?;
        println!("wrote full list ({} rows) to {}", results.len(), csv_path.display());
    }

    let constraints: Vec<u64> = results.iter().map(|r| r.constraints).collect();
    let sum: u64 = constraints.iter().sum();
    let n = constraints.len();

    println!("\n=== summary ({n} slices across {} models, {} error(s)) ===", models_scanned.len(), errors.len());
    println!("min:    {}", constraints.last().copied().unwrap_or(0));
    println!("p10:    {}", percentile(&constraints, 0.10));
    println!("p50:    {}", percentile(&constraints, 0.50));
    println!("p90:    {}", percentile(&constraints, 0.90));
    println!("p95:    {}", percentile(&constraints, 0.95));
    println!("p99:    {}", percentile(&constraints, 0.99));
    println!("max:    {}", constraints.first().copied().unwrap_or(0));
    println!("mean:   {}", if n > 0 { sum / n as u64 } else { 0 });

    println!("\n=== {} heaviest ===", args.top_n.min(n));
    for r in results.iter().take(args.top_n) {
        println!(
            "  {:>14}  model_{}/{}  (declared I/O: {})",
            r.constraints, r.model_id, r.slice_num, r.declared_io_elements
        );
    }

    println!("\n=== {} lightest ===", args.top_n.min(n));
    for r in results.iter().rev().take(args.top_n) {
        println!(
            "  {:>14}  model_{}/{}  (declared I/O: {})",
            r.constraints, r.model_id, r.slice_num, r.declared_io_elements
        );
    }

    if !errors.is_empty() {
        println!("\n=== {} slice(s) failed to estimate (skipped, not counted above) ===", errors.len());
        for e in errors.iter().take(20) {
            println!("  {e}");
        }
        if errors.len() > 20 {
            println!("  ... and {} more", errors.len() - 20);
        }
    }

    Ok(())
}
