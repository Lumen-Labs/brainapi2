use std::env;
use std::time::Duration;

const DEFAULT_URI: &str = "https://glo-matcher.brainapi.lumen-labs.ai/mcp";
const DEFAULT_TIMEOUT_MS: u64 = 60_000;
const DEFAULT_MAX_QUEUE: usize = 10_000;
const MAX_BACKOFF_SECS: u64 = 30;

#[derive(Clone, Debug)]
pub struct Config {
    pub uri: String,
    pub bearer_token: Option<String>,
    pub mcp_name: Option<String>,
    pub timeout: Duration,
    pub max_queue: usize,
    pub max_backoff: Duration,
}

impl Config {
    pub fn from_env() -> Self {
        let uri = env::var("URI")
            .unwrap_or_else(|_| DEFAULT_URI.to_string());
        let bearer_token = env::var("BEARER_TOKEN").ok().filter(|s| !s.is_empty());
        let mcp_name = env::var("MCP_NAME").ok().filter(|s| !s.is_empty());
        let timeout_ms: u64 = env::var("MCP_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_MS);
        let max_queue: usize = env::var("MCP_MAX_QUEUE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_QUEUE);
        Self {
            uri,
            bearer_token,
            mcp_name,
            timeout: Duration::from_millis(timeout_ms),
            max_queue,
            max_backoff: Duration::from_secs(MAX_BACKOFF_SECS),
        }
    }
}
