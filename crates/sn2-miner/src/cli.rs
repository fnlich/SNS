use clap::{Parser, Subcommand};

/// Rejects out-of-range values before they can reach the CPU-pressure ramp
/// formula, where a negative threshold would silently make the heaviest-N
/// skip partially active even at 0% CPU utilization instead of failing loudly.
fn parse_unit_interval(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("must be between 0.0 and 1.0, got {v}"))
    }
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Emit an nftables ruleset that drops UDP traffic to the QUIC port from any
    /// source IP not currently registered as a validator in the cached metagraph.
    /// Pipe stdout to `sudo nft -f -` to apply atomically.
    Firewall {
        /// Optional path to write the ruleset to instead of stdout.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
}

#[derive(Parser, Debug)]
#[command(name = "sn2-miner", about = "Subnet-2 Miner")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[arg(long, default_value_t = sn2_types::DEFAULT_NETUID)]
    pub netuid: u16,

    #[arg(long, alias = "subtensor.network", default_value = "finney")]
    pub network: String,

    #[arg(long, alias = "subtensor.chain_endpoint")]
    pub subtensor_chain_endpoint: Option<String>,

    #[arg(long, alias = "wallet.name", default_value = "default")]
    pub wallet_name: String,

    #[arg(long, alias = "wallet.hotkey", default_value = "default")]
    pub wallet_hotkey: String,

    #[arg(long, alias = "wallet.path")]
    pub wallet_path: Option<String>,

    #[arg(long, alias = "logging.level", default_value = "info")]
    pub log_level: String,

    #[arg(long, alias = "axon.host", default_value = "0.0.0.0")]
    pub axon_host: String,

    #[arg(long, alias = "axon.port", default_value_t = 8091)]
    pub axon_port: u16,

    #[arg(long, alias = "axon.external_ip")]
    pub external_ip: Option<String>,

    #[arg(long, default_value_t = false)]
    pub no_auto_update: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "[TESTING ONLY] Disable validator permit checks — bypasses all on-chain permit enforcement"
    )]
    pub disable_blacklist: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Do not load the persisted validator roster on startup. New observations still persist unless --no-validator-ip-cache is combined with a read-only wallet path."
    )]
    pub no_validator_ip_cache: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable nftables ruleset emission. The userspace source-IP check at the QUIC listener remains in effect."
    )]
    pub no_nftables: bool,

    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(30..), help = "Metagraph sync interval in seconds")]
    pub metagraph_sync_interval: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Run without chain interaction for local integration testing"
    )]
    pub loopback: bool,

    #[arg(long, value_delimiter = ',')]
    pub additional_circuits: Vec<String>,

    #[arg(
        long,
        env = "SN2_CIRCUIT_CACHE_DIR",
        help = "Directory for the persisted circuit cache. Defaults to \
                ~/.bittensor/subnet-2/circuit_cache when unset. May also be \
                supplied via the SN2_CIRCUIT_CACHE_DIR environment variable; \
                the CLI flag wins when both are present. A leading ~ is \
                expanded to the home directory."
    )]
    pub circuit_cache_dir: Option<String>,

    #[arg(long, default_value_t = sn2_types::CIRCUIT_TIMEOUT_SECONDS, value_parser = clap::value_parser!(u64).range(1..))]
    pub handler_timeout: u64,

    #[arg(
        long,
        default_value_t = sn2_types::DEFAULT_MAX_HEAVIEST_N,
        help = "Ceiling for the heaviest-N proof skip, reached once CPU utilization hits 100%. \
                A request is skipped outright -- before attempting it -- when its circuit ranks \
                among the current N heaviest distinct circuits observed so far (by \
                estimate_circuit's constraint count, ranked against everything seen). The N \
                actually used ramps linearly with live CPU utilization: 0 (never skips) at or \
                below --cpu-pressure-threshold, up to this ceiling at 100%. Set to 0 to disable \
                entirely, at any CPU load. Relative, not an absolute budget: widens on its own \
                as more circuits are observed. Circuits declaring more than one activation \
                placeholder are always skipped regardless of this setting, since the validator \
                only ever sends one activation tensor."
    )]
    pub max_heaviest_n: usize,

    #[arg(
        long,
        default_value_t = sn2_types::DEFAULT_CPU_PRESSURE_THRESHOLD,
        value_parser = parse_unit_interval,
        help = "System-wide CPU utilization fraction (0.0-1.0) at or below which the \
                heaviest-N skip is always off (N=0). Above this, N ramps linearly toward \
                --max-heaviest-n as utilization approaches 100%."
    )]
    pub cpu_pressure_threshold: f64,
}
