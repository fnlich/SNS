use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Background system-wide CPU utilization sampler, backed by `/proc/stat`.
///
/// Utilization is stored as an integer permille (0..=1000) so it can be read
/// lock-free from the hot admission path. On platforms without `/proc/stat`
/// (e.g. local macOS dev builds), sampling is a no-op and utilization reads
/// as 0 -- any CPU-pressure-gated check then never triggers, which is the
/// correct fail-open behavior for a platform this was never meant to run on.
pub struct CpuLoad {
    utilization_permille: AtomicU64,
}

struct ProcStatTotals {
    total: u64,
    idle: u64,
}

fn read_proc_stat_totals() -> Option<ProcStatTotals> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    let mut fields = line.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }
    let values: Vec<u64> = fields.filter_map(|f| f.parse().ok()).collect();
    if values.len() < 4 {
        return None;
    }
    // user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
    let idle = values[3] + values.get(4).copied().unwrap_or(0);
    let total: u64 = values.iter().sum();
    Some(ProcStatTotals { total, idle })
}

impl CpuLoad {
    pub fn spawn(sample_interval: Duration) -> Arc<Self> {
        let this = Arc::new(Self {
            utilization_permille: AtomicU64::new(0),
        });
        let bg = Arc::clone(&this);
        tokio::spawn(async move {
            bg.sample_loop(sample_interval).await;
        });
        this
    }

    /// Current system-wide CPU utilization as a fraction in [0.0, 1.0].
    pub fn utilization(&self) -> f64 {
        self.utilization_permille.load(Ordering::Relaxed) as f64 / 1000.0
    }

    async fn sample_loop(self: Arc<Self>, interval: Duration) {
        let mut prev = read_proc_stat_totals();
        loop {
            tokio::time::sleep(interval).await;
            let Some(prev_totals) = prev.as_ref() else {
                prev = read_proc_stat_totals();
                continue;
            };
            let Some(cur_totals) = read_proc_stat_totals() else {
                continue;
            };
            let total_delta = cur_totals.total.saturating_sub(prev_totals.total);
            let idle_delta = cur_totals.idle.saturating_sub(prev_totals.idle);
            if total_delta > 0 {
                let busy_delta = total_delta.saturating_sub(idle_delta);
                let permille = (busy_delta.saturating_mul(1000) / total_delta).min(1000);
                self.utilization_permille.store(permille, Ordering::Relaxed);
            }
            prev = Some(cur_totals);
        }
    }
}
