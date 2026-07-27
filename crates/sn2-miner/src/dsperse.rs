use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::{Context, Result};
use sn2_types::json_tensor::flatten_json_to_f64;
use tracing::info;

pub struct DSperseClient {
    cache_dir: PathBuf,
    /// Shared across every request so its internal bundle cache
    /// (`load_bundle_cached`) actually gets reused instead of being
    /// discarded and rebuilt on every single call. A circuit bundle is
    /// tens of megabytes read and parsed from disk; a fresh backend per
    /// request means a fresh cache-miss every time, even for a circuit
    /// that was just proved a moment ago. Mirrors the pattern already
    /// used by `sn2-verify`'s validator-side `BACKEND` static.
    backend: Arc<dsperse::backend::jstprove::JstproveBackend>,
    /// Caches `resolve_component`'s (component_sha, slice_id) -> slice_dir
    /// lookups, avoiding a full scan of every locally cached model
    /// directory on repeat DSlice requests for the same component.
    component_cache: RwLock<HashMap<(String, String), PathBuf>>,
    /// Caches `find_slice_onnx`'s slice_dir -> onnx_path resolution,
    /// avoiding a directory listing on every repeat prove_slice call.
    onnx_cache: RwLock<HashMap<PathBuf, PathBuf>>,
    /// Caches the fallback whole-file ONNX initializer extraction
    /// (`extract_onnx_initializers`), keyed by onnx_path. Only hit when
    /// the known-constants table is missing an entry, but when it is,
    /// the result is fully deterministic for a given circuit -- re-reading
    /// and re-parsing the same donor ONNX file on every repeat request
    /// for that slice is pure waste. `Arc`-wrapped since it's read and
    /// populated from inside `spawn_blocking`, not just `&self` methods.
    onnx_initializers_cache: Arc<RwLock<HashMap<PathBuf, Vec<(Vec<f64>, Vec<usize>)>>>>,
    /// If set, `prove_slice` skips a request outright -- before calling into
    /// witness/proof generation -- when its circuit ranks among the N
    /// heaviest distinct circuits observed so far (by `estimate_circuit`'s
    /// total constraint count, ranked against everything currently in
    /// `circuit_estimate_cache`). Early on, with few distinct circuits
    /// cached, this can trivially skip most or all of them -- the ranking is
    /// relative to what's been *seen*, not an absolute budget, so it widens
    /// out on its own as more circuits get observed.
    ///
    /// The N actually used per request isn't this value directly -- it's
    /// computed fresh each time from current CPU utilization (`cpu_load`),
    /// scaling linearly from 0 at `cpu_pressure_threshold` up to this
    /// ceiling at 100% utilization:
    /// `n = round(max_heaviest_n * (utilization - threshold) / (1 - threshold))`,
    /// clamped to `[0, max_heaviest_n]`. Below the threshold this always
    /// computes to 0, which is a no-op through the same `heavier_count >= n`
    /// comparison used above the threshold (trivially true for n=0) -- so
    /// "disabled" isn't a separate state, it's just the low end of the same
    /// ramp. Setting this to 0 disables the mechanism entirely, at any
    /// utilization. No hysteresis/deadband: since n itself is continuously
    /// recomputed, not just gated on/off, this ramps smoothly through the
    /// threshold rather than flipping a single check discontinuously.
    max_heaviest_n: usize,
    cpu_load: Arc<crate::cpu_load::CpuLoad>,
    cpu_pressure_threshold: f64,
    /// Caches `estimate_circuit`'s total constraint count per onnx_path.
    /// Computing it requires a full ONNX parse (`generate_metadata`: shape
    /// inference + quantization across the whole graph) -- real work for
    /// donor files we've directly seen exceed 100MB. The result is
    /// deterministic for a given (immutable, already-cached) donor ONNX
    /// file, so repeat requests for the same slice -- including retry
    /// bursts -- reuse it instead of re-parsing from scratch every time.
    /// Also doubles as the ranking pool for `max_heaviest_n`.
    circuit_estimate_cache: Arc<RwLock<HashMap<PathBuf, u64>>>,
    /// Caches the count of distinct activation placeholders a compiled
    /// circuit declares, keyed by circuit_path. Always enforced (not
    /// gated by a flag): the validator only ever sends one activation
    /// tensor per request, so any circuit declaring more than one is
    /// structurally unprovable here -- skipping it costs nothing, since
    /// it was never going to succeed. Fixed per circuit_path, so caching
    /// it is a pure win on repeat requests, including retry bursts.
    activation_mismatch_cache: Arc<RwLock<HashMap<PathBuf, usize>>>,
    /// Caches `backend.load_params`'s result per circuit_path, sitting in
    /// front of `JstproveBackend`'s own internal bundle cache. That cache is
    /// correct but not free even on a hit: it canonicalizes circuit_path
    /// (a filesystem syscall) and serializes through a `Mutex` (not a
    /// `RwLock`) on every call, so concurrent requests for an
    /// already-cached circuit still briefly contend on one lock. A hit here
    /// skips calling into the backend entirely.
    params_cache: Arc<RwLock<HashMap<PathBuf, Option<jstprove_circuits::api::CircuitParamsType>>>>,
}

fn validate_circuit_id(id: &str) -> Result<()> {
    anyhow::ensure!(
        id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid circuit id: expected 64-char hex string"
    );
    Ok(())
}

pub fn normalize_slice_id(slice_num: &str) -> Result<String> {
    let idx: usize = slice_num
        .strip_prefix("slice_")
        .unwrap_or(slice_num)
        .parse()
        .context("parsing slice_num")?;
    Ok(format!("slice_{idx}"))
}

fn find_slice_onnx(slice_dir: &Path) -> Result<PathBuf> {
    let payload_dir = slice_dir.join("payload");
    if payload_dir.is_dir() {
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(&payload_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
            .collect();
        candidates.sort();
        match candidates.len() {
            1 => return Ok(candidates.remove(0)),
            n if n > 1 => anyhow::bail!(
                "multiple .onnx files in {}: {:?}",
                payload_dir.display(),
                candidates,
            ),
            _ => {}
        }
    }
    anyhow::bail!("no .onnx file found in {}", payload_dir.display())
}

/// Returns (this circuit's `estimate_circuit` total constraint count, how
/// many *other* cached circuits have a strictly higher one) in a single lock
/// acquisition. Combines the estimate lookup/insert with the heaviest-N
/// ranking scan that callers need right after it -- doing them as two
/// separate lock acquisitions (compute-or-cache, then re-lock to scan)
/// costs an extra acquire/release under exactly the CPU-pressure conditions
/// where this runs and lock contention is least welcome.
///
/// Computing the estimate on a cache miss requires a full ONNX parse
/// (`generate_metadata` inside `prepare_jstprove_artifacts`: shape inference
/// and quantization across the whole graph) -- real work for donor files
/// we've directly observed exceeding 100MB -- so this is only cheap on
/// repeat calls for the same onnx_path. The parse itself runs outside any
/// lock; only the cache lookup/insert and ranking scan are guarded.
fn estimate_circuit_constraints_and_rank(
    cache: &RwLock<HashMap<PathBuf, u64>>,
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<(u64, usize)> {
    {
        let guard = cache.read().unwrap_or_else(|e| e.into_inner());
        if let Some(&total) = guard.get(onnx_path) {
            let heavier_count = guard.values().filter(|&&v| v > total).count();
            return Ok((total, heavier_count));
        }
    }

    let total = estimate_circuit_constraints(cache, onnx_path, weights_as_inputs)?;
    let heavier_count = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter(|&&v| v > total)
        .count();
    Ok((total, heavier_count))
}

/// Same lookup/compute/cache as `estimate_circuit_constraints_and_rank`, but
/// without the ranking scan -- for callers that only need the estimate
/// cached (e.g. warming the cache while CPU pressure is below the point
/// where anything is actually being ranked/enforced), where computing
/// `heavier_count` would be a wasted O(cache size) scan on every call.
fn estimate_circuit_constraints(
    cache: &RwLock<HashMap<PathBuf, u64>>,
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<u64> {
    if let Some(&total) = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(onnx_path)
    {
        return Ok(total);
    }

    let (params, architecture, _wandb) =
        dsperse::converter::prepare_jstprove_artifacts(onnx_path, weights_as_inputs)
            .map_err(|e| anyhow::anyhow!("estimating circuit cost: {e}"))?;
    // 31-bit field, matching this deployment's observed goldilocks_ext4_whir
    // proof_config; from_circuit_params otherwise derives everything else
    // (chunk_bits, scale_exponent, freivalds_reps, rescale_config) straight
    // from the circuit's own params, not this default.
    let total = jstprove_circuits::api::estimate_circuit(&params, &architecture, 31).total();

    cache
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(onnx_path.to_path_buf(), total);
    Ok(total)
}

/// Returns how many distinct activation placeholders `p` declares, caching
/// the count per circuit_path -- a fixed, deterministic property of a
/// compiled bundle. The validator only ever sends one activation tensor per
/// request; more than one placeholder means there's no safe way to know
/// which data belongs to which input (see the WAI fallback below), so this
/// is checked up front to skip the request before spending any time on it.
fn activation_placeholder_count(
    cache: &RwLock<HashMap<PathBuf, usize>>,
    circuit_path: &Path,
    p: &jstprove_circuits::api::CircuitParamsType,
) -> usize {
    if let Some(cached) = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(circuit_path)
    {
        return *cached;
    }

    let count = p
        .inputs
        .iter()
        .filter(|io| dsperse::pipeline::runner::is_activation_placeholder(&io.name))
        .count();

    cache
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(circuit_path.to_path_buf(), count);
    count
}

/// Returns `circuit_path`'s compiled bundle metadata, checking `cache`
/// before ever calling into `backend`. `JstproveBackend::load_params` is
/// already cached internally (`bundle_cache`), but that cache still
/// canonicalizes `circuit_path` (a filesystem syscall) and serializes
/// through a `Mutex` -- not a `RwLock` -- on every call, hit or miss, so
/// concurrent requests for an already-cached circuit still briefly contend
/// on one lock. A hit here skips calling into the backend entirely.
fn load_params_cached(
    cache: &RwLock<HashMap<PathBuf, Option<jstprove_circuits::api::CircuitParamsType>>>,
    backend: &dsperse::backend::jstprove::JstproveBackend,
    circuit_path: &Path,
) -> Result<Option<jstprove_circuits::api::CircuitParamsType>> {
    if let Some(cached) = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(circuit_path)
    {
        return Ok(cached.clone());
    }

    let params = backend
        .load_params(circuit_path)
        .map_err(|e| anyhow::anyhow!("loading circuit params: {e}"))?;

    cache
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(circuit_path.to_path_buf(), params.clone());
    Ok(params)
}

fn extract_input_json(inputs: &serde_json::Value) -> &serde_json::Value {
    for key in &["input_data", "input", "data", "inputs"] {
        if let Some(v) = inputs.get(*key) {
            return v;
        }
    }
    inputs
}

pub struct ProveArtifacts {
    pub proof: Vec<u8>,
    pub witness: Vec<u8>,
    pub computed_outputs: Vec<f64>,
}

fn prove_and_build_response(
    backend: &dsperse::backend::jstprove::JstproveBackend,
    circuit_path: &Path,
    witness_bytes: &[u8],
    effective_input_dims: Option<usize>,
) -> Result<ProveArtifacts> {
    let holographic = circuit_path.join("vk.bin").is_file();
    let proof_bytes = if holographic {
        backend
            .prove_holographic(circuit_path, witness_bytes)
            .map_err(|e| anyhow::anyhow!("holographic proof generation: {e}"))?
    } else {
        backend
            .prove(circuit_path, witness_bytes)
            .map_err(|e| anyhow::anyhow!("proof generation: {e}"))?
    };

    let computed_outputs = if let Some(num_model_inputs) = effective_input_dims {
        match backend.extract_outputs(witness_bytes, num_model_inputs) {
            Ok(outputs) => outputs,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    num_model_inputs,
                    witness_len = witness_bytes.len(),
                    "output extraction failed; validator will extract from verified proof"
                );
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    info!(
        witness_size = witness_bytes.len(),
        proof_size = proof_bytes.len(),
        num_outputs = computed_outputs.len(),
        holographic,
        "witness and proof generated"
    );

    Ok(ProveArtifacts {
        proof: proof_bytes,
        witness: witness_bytes.to_vec(),
        computed_outputs,
    })
}

impl DSperseClient {
    pub fn new(
        cache_dir_override: Option<&str>,
        max_heaviest_n: usize,
        cpu_pressure_threshold: f64,
    ) -> Self {
        let cache_dir = PathBuf::from(
            shellexpand::tilde(cache_dir_override.unwrap_or(sn2_types::CIRCUIT_CACHE_DIR))
                .to_string(),
        );
        info!(cache_dir = %cache_dir.display(), "initialized DSperseClient");

        let backend = Arc::new(dsperse::backend::jstprove::JstproveBackend::new());
        {
            let backend = Arc::clone(&backend);
            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_secs(sn2_types::BUNDLE_CACHE_IDLE_TTL_SECS));
                loop {
                    interval.tick().await;
                    let evicted = backend.evict_idle(Duration::from_secs(
                        sn2_types::BUNDLE_CACHE_IDLE_TTL_SECS,
                    ));
                    if evicted > 0 {
                        info!(evicted, "evicted idle compiled circuit bundles");
                    }
                }
            });
        }

        Self {
            cache_dir,
            backend,
            component_cache: RwLock::new(HashMap::new()),
            onnx_cache: RwLock::new(HashMap::new()),
            onnx_initializers_cache: Arc::new(RwLock::new(HashMap::new())),
            max_heaviest_n,
            cpu_load: crate::cpu_load::CpuLoad::spawn(Duration::from_millis(
                sn2_types::CPU_LOAD_SAMPLE_INTERVAL_MS,
            )),
            cpu_pressure_threshold,
            circuit_estimate_cache: Arc::new(RwLock::new(HashMap::new())),
            activation_mismatch_cache: Arc::new(RwLock::new(HashMap::new())),
            params_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn resolve_slice_onnx(&self, slice_dir: &Path) -> Result<PathBuf> {
        if let Some(cached) = self
            .onnx_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(slice_dir)
            .cloned()
        {
            if cached.exists() {
                return Ok(cached);
            }
        }
        let resolved = find_slice_onnx(slice_dir)?;
        self.onnx_cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(slice_dir.to_path_buf(), resolved.clone());
        Ok(resolved)
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub async fn resolve_component(
        &self,
        component_sha: &str,
        slice_id: &str,
    ) -> Result<Option<PathBuf>> {
        let cache_key = (component_sha.to_string(), slice_id.to_string());

        if let Some(cached) = self
            .component_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&cache_key)
            .cloned()
        {
            if cached.join("jstprove").join("circuit.bundle").is_dir() {
                return Ok(Some(cached));
            }
            // Stale (e.g. evicted from disk since caching) -- drop it and
            // fall through to a full re-resolution below.
            self.component_cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&cache_key);
        }

        let cache_dir = self.cache_dir.clone();
        let scan_component_sha = component_sha.to_string();
        let scan_slice_id = slice_id.to_string();
        let resolved: Option<PathBuf> = tokio::task::spawn_blocking(move || -> Result<Option<PathBuf>> {
            let entries = match std::fs::read_dir(&cache_dir) {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "reading cache directory {}: {e}",
                        cache_dir.display()
                    ))
                }
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with("model_") {
                    continue;
                }
                let stamp_path = entry
                    .path()
                    .join("slices")
                    .join(&scan_slice_id)
                    .join("component.sha");
                if let Ok(stamp) = std::fs::read_to_string(&stamp_path) {
                    if stamp.trim() == scan_component_sha {
                        let slice_dir = entry.path().join("slices").join(&scan_slice_id);
                        if !slice_dir.join("jstprove").join("circuit.bundle").is_dir() {
                            continue;
                        }
                        if find_slice_onnx(&slice_dir).is_err() {
                            continue;
                        }
                        return Ok(Some(slice_dir));
                    }
                }
            }
            Ok(None)
        })
        .await
        .context("component resolution task panicked")??;

        if let Some(ref slice_dir) = resolved {
            self.component_cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(cache_key, slice_dir.clone());
        }

        Ok(resolved)
    }

    pub async fn prove(
        &self,
        model_id: &str,
        inputs: &serde_json::Value,
    ) -> Result<ProveArtifacts> {
        validate_circuit_id(model_id)?;
        let model_dir = self.cache_dir.join(format!("model_{model_id}"));
        let circuit_path = model_dir.join("model.compiled");

        anyhow::ensure!(
            circuit_path.exists(),
            "compiled model not found at {}",
            circuit_path.display()
        );

        info!(
            model_id,
            circuit_path = %circuit_path.display(),
            "generating witness and proof for non-composable model"
        );

        let inputs_clone = inputs.clone();
        let backend = Arc::clone(&self.backend);

        tokio::task::spawn_blocking(move || -> Result<ProveArtifacts> {
            let inputs_bytes = rmp_serde::to_vec_named(&inputs_clone)?;

            let params = backend
                .load_params(&circuit_path)
                .map_err(|e| anyhow::anyhow!("loading circuit params: {e}"))?;

            let witness_bytes = backend
                .witness(&circuit_path, &inputs_bytes, &[])
                .map_err(|e| anyhow::anyhow!("witness generation: {e}"))?;

            let dims = params.as_ref().map(|p| p.effective_input_dims());
            prove_and_build_response(&backend, &circuit_path, &witness_bytes, dims)
        })
        .await
        .context("blocking task panicked")?
    }

    pub async fn prove_slice(
        &self,
        circuit_id: &str,
        slice_num: &str,
        inputs: &serde_json::Value,
        resolved_component_dir: PathBuf,
    ) -> Result<ProveArtifacts> {
        validate_circuit_id(circuit_id)?;
        // Validate slice format; the normalized path is not needed since
        // resolved_component_dir already contains the canonical slice path.
        let _ = normalize_slice_id(slice_num)?;

        let slice_dir = resolved_component_dir;

        anyhow::ensure!(
            slice_dir.is_dir(),
            "resolved component directory not found at {}",
            slice_dir.display()
        );

        let circuit_path = slice_dir.join("jstprove").join("circuit.bundle");
        let onnx_path = self.resolve_slice_onnx(&slice_dir)?;

        anyhow::ensure!(
            circuit_path.is_dir(),
            "bundle directory not found at {}",
            circuit_path.display()
        );
        anyhow::ensure!(
            onnx_path.exists(),
            "onnx model not found at {}",
            onnx_path.display()
        );

        info!(
            circuit_id,
            slice = slice_num,
            circuit_path = %circuit_path.display(),
            "generating witness and proof"
        );

        let input_data = extract_input_json(inputs).clone();
        let backend = Arc::clone(&self.backend);
        let onnx_initializers_cache = Arc::clone(&self.onnx_initializers_cache);
        let circuit_estimate_cache = Arc::clone(&self.circuit_estimate_cache);
        let activation_mismatch_cache = Arc::clone(&self.activation_mismatch_cache);
        let params_cache = Arc::clone(&self.params_cache);
        let max_heaviest_n = self.max_heaviest_n;
        let cpu_load = Arc::clone(&self.cpu_load);
        let cpu_pressure_threshold = self.cpu_pressure_threshold;
        let estimate_onnx_path = onnx_path.clone();

        tokio::task::spawn_blocking(move || -> Result<ProveArtifacts> {
            let input_flat = flatten_json_to_f64(&input_data);
            anyhow::ensure!(
                !input_flat.is_empty(),
                "invalid input tensor: flattened input is empty"
            );

            let params = load_params_cached(&params_cache, &backend, &circuit_path)?;

            // Always enforced, but only meaningful for WAI (donor-sourced)
            // circuits -- non-WAI circuits never populate activations via
            // is_activation_placeholder at all (see the `_ =>` branch
            // below), so checking them would only pay the scan for no
            // possible benefit. The validator only ever sends one
            // activation tensor, so more than one declared placeholder on a
            // WAI circuit is structurally unprovable here -- skip before
            // spending any time on it.
            let weights_as_inputs = params.as_ref().is_some_and(|p| p.weights_as_inputs);
            if weights_as_inputs {
                if let Some(p) = params.as_ref() {
                    let placeholder_count = activation_placeholder_count(
                        &activation_mismatch_cache,
                        &circuit_path,
                        p,
                    );
                    anyhow::ensure!(
                        placeholder_count <= 1,
                        "proof skipped: circuit declares {placeholder_count} distinct activation \
                         placeholders but the validator only provides one activation tensor"
                    );
                }
            }

            // skip_n ramps linearly with CPU pressure: 0 at or below the
            // threshold (never skips -- heavier_count >= 0 is trivially
            // true), up to max_heaviest_n at 100% utilization. Computed
            // fresh every request from the live sample, so admission
            // tightens and relaxes smoothly with load instead of flipping a
            // single check on/off at one fixed line. Named distinctly from
            // the unrelated per-input `n` (shape element count) used later
            // in the WAI fallback below, to avoid two unrelated meanings
            // sharing one identifier in the same function.
            let utilization = cpu_load.utilization();
            let skip_n = if max_heaviest_n == 0 || utilization <= cpu_pressure_threshold {
                0
            } else {
                let fraction =
                    ((utilization - cpu_pressure_threshold) / (1.0 - cpu_pressure_threshold))
                        .clamp(0.0, 1.0);
                ((fraction * max_heaviest_n as f64).round() as usize).min(max_heaviest_n)
            };

            // Warm the estimate cache whenever the feature is configured at
            // all (max_heaviest_n > 0), even while current utilization is
            // below the threshold and skip_n is 0 -- deliberately decoupled
            // from whether we're enforcing anything right now. Deferring the
            // (potentially expensive, ONNX-parsing-on-a-miss) estimate
            // until skip_n actually becomes >0 would mean every circuit
            // first encountered exactly when pressure rises pays that cost
            // simultaneously, right when CPU has the least room to spare.
            // Computing it during idle/light windows instead means most
            // circuits are already warm by the time enforcement kicks in;
            // only a genuinely brand-new circuit first seen mid-pressure
            // still pays the parse cost then. If the feature is fully
            // disabled (max_heaviest_n == 0), skip this too -- there's no
            // future point at which the estimate would ever be used.
            if max_heaviest_n > 0 {
                if skip_n > 0 {
                    let (total, heavier_count) = estimate_circuit_constraints_and_rank(
                        &circuit_estimate_cache,
                        &estimate_onnx_path,
                        weights_as_inputs,
                    )?;
                    anyhow::ensure!(
                        heavier_count >= skip_n,
                        "proof skipped: CPU utilization {:.1}% (threshold {:.1}%) implies \
                         skip-N={skip_n}; circuit ({total} constraints) ranks among the \
                         {skip_n} heaviest circuits observed so far ({heavier_count} known \
                         heavier)",
                        utilization * 100.0,
                        cpu_pressure_threshold * 100.0,
                    );
                } else {
                    // Warming only: cache the estimate for next time without
                    // paying for the O(cache size) ranking scan, which
                    // nothing here would read anyway.
                    estimate_circuit_constraints(
                        &circuit_estimate_cache,
                        &estimate_onnx_path,
                        weights_as_inputs,
                    )?;
                }
            }

            let (activations, inits) = match params.as_ref() {
                Some(p) if p.weights_as_inputs => {
                    match dsperse::pipeline::split_inline_wai_inputs(p, &input_flat) {
                        Some(split) => split,
                        None => {
                            // The validator only sends the true activation
                            // tensor here, assuming every other declared
                            // input (biases, architecture constants) is
                            // independently recoverable locally. For
                            // donor-sourced circuits it isn't: those values
                            // only materialize during this circuit's own
                            // slicing/constant-folding step, which the miner
                            // never runs. Recover known-fixed values (GELU's
                            // constants, frozen backbone biases) from a
                            // static table first; anything not in it falls
                            // back to the existing file-based extraction.
                            let mut cursor = 0usize;
                            let mut activations = Vec::new();
                            let mut inits: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
                            let mut unresolved = Vec::new();

                            for io in &p.inputs {
                                let n: usize = io.shape.iter().product();
                                if dsperse::pipeline::runner::is_activation_placeholder(&io.name) {
                                    let end = (cursor + n).min(input_flat.len());
                                    activations.extend_from_slice(&input_flat[cursor..end]);
                                    cursor = end;
                                } else if let Some((values, shape)) =
                                    crate::wai_known_constants::lookup(&io.name)
                                {
                                    anyhow::ensure!(
                                        values.len() == n,
                                        "known constant '{}' has {} elements but circuit expects {}",
                                        io.name,
                                        values.len(),
                                        n
                                    );
                                    inits.push((values, shape));
                                } else {
                                    unresolved.push(io.name.clone());
                                }
                            }

                            if !unresolved.is_empty() {
                                tracing::warn!(
                                    names = ?unresolved,
                                    "known-constant table missing entries, falling back to file extraction"
                                );
                                let cached_file_inits = onnx_initializers_cache
                                    .read()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .get(&onnx_path)
                                    .cloned();
                                let file_inits = match cached_file_inits {
                                    Some(cached) => cached,
                                    None => {
                                        let extracted = dsperse::pipeline::extract_onnx_initializers(
                                            &onnx_path, p,
                                        )
                                        .map_err(|e| {
                                            anyhow::anyhow!("extracting initializers: {e}")
                                        })?;
                                        onnx_initializers_cache
                                            .write()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .insert(onnx_path.clone(), extracted.clone());
                                        extracted
                                    }
                                };
                                inits.extend(file_inits);
                            }

                            (activations, inits)
                        }
                    }
                }
                _ => (input_flat.clone(), Vec::new()),
            };

            let witness_bytes = backend
                .witness_f64(&circuit_path, &activations, &inits)
                .map_err(|e| anyhow::anyhow!("witness generation: {e}"))?;

            let dims = params.as_ref().map(|p| p.effective_input_dims());
            prove_and_build_response(&backend, &circuit_path, &witness_bytes, dims)
        })
        .await
        .context("blocking task panicked")?
    }
}

#[cfg(test)]
mod wai_fallback_integration_tests {
    use super::*;

    /// Exercises the real `prove_slice` path against slice_216 of the actual
    /// "tail" fine-tuned model (f8121b74...) -- one of the slices confirmed
    /// failing with "activation length mismatch: expected 2050, got 2048" in
    /// production. The fixture directory holds the real, repository-fetched
    /// circuit.bin/manifest.msgpack/witness_solver.bin for this slice, plus
    /// the real donor.onnx as the payload file `find_slice_onnx` requires to
    /// be present. The activation values are synthetic (their numeric
    /// content doesn't matter for this test): the point is confirming the
    /// known-constants fallback supplies the two missing scalar constants so
    /// witness generation no longer fails on element count, regardless of
    /// whether the resulting proof is otherwise meaningful.
    #[tokio::test]
    async fn slice_216_no_longer_reports_activation_length_mismatch() {
        let slice_dir = PathBuf::from(
            "/private/tmp/claude-501/-Volumes-HDD-Develop-Work-Subnet2/67c52f37-ba8f-43ed-a9d9-286ed7ff6dd6/scratchpad/real_slice_216",
        );
        if !slice_dir.is_dir() {
            eprintln!("skipping: fixture directory not present at {}", slice_dir.display());
            return;
        }

        let client = DSperseClient::new(
            None,
            sn2_types::DEFAULT_MAX_HEAVIEST_N,
            sn2_types::DEFAULT_CPU_PRESSURE_THRESHOLD,
        );
        let synthetic_activations: Vec<f64> = (0..2048).map(|i| (i % 7) as f64 * 0.01).collect();
        let inputs = serde_json::json!({ "input_data": synthetic_activations });

        let result = client
            .prove_slice(
                "f8121b74a74514a47bc870dd913b7dbb45c8688e5a0a7328209da6a74d9c7094",
                "slice_216",
                &inputs,
                slice_dir,
            )
            .await;

        match &result {
            Ok(_) => { /* fully succeeded: witness and proof generated */ }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("activation length mismatch"),
                    "known-constants fallback did not resolve the missing WAI inputs: {msg}"
                );
                eprintln!(
                    "prove_slice did not fully succeed, but NOT due to activation length mismatch: {msg}"
                );
            }
        }
    }
}
