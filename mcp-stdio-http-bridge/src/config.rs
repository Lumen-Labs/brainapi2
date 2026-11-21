use std::env;
use std::time::Duration;

const DEFAULT_URI: &str = "http://127.0.0.1:8000/mcp";
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
    pub fn from_values(
        uri: Option<String>,
        bearer_token: Option<String>,
        mcp_name: Option<String>,
        timeout_ms: Option<u64>,
        max_queue: Option<usize>,
    ) -> Self {
        let uri = uri
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_URI.to_string());
        Self {
            uri,
            bearer_token: bearer_token.filter(|value| !value.is_empty()),
            mcp_name: mcp_name.filter(|value| !value.is_empty()),
            timeout: Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS)),
            max_queue: max_queue.unwrap_or(DEFAULT_MAX_QUEUE),
            max_backoff: Duration::from_secs(MAX_BACKOFF_SECS),
        }
    }

    pub fn from_env() -> Self {
        Self::from_values(
            env::var("URI").ok(),
            env::var("BEARER_TOKEN").ok(),
            env::var("MCP_NAME").ok(),
            env::var("MCP_TIMEOUT_MS")
                .ok()
                .and_then(|value| value.parse().ok()),
            env::var("MCP_MAX_QUEUE")
                .ok()
                .and_then(|value| value.parse().ok()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uri_is_local_mcp() {
        let cfg = Config::from_values(None, None, None, None, None);
        assert_eq!(cfg.uri, "http://127.0.0.1:8000/mcp");
        assert!(!cfg.uri.contains("glo-matcher"));
        assert!(!cfg.uri.contains("lumen-labs.ai"));
    }

    #[test]
    fn empty_uri_falls_back_to_local() {
        let cfg = Config::from_values(Some("  ".into()), None, None, None, None);
        assert_eq!(cfg.uri, DEFAULT_URI);
    }

    #[test]
    fn custom_uri_is_used() {
        let cfg = Config::from_values(
            Some("http://example.local/mcp".into()),
            None,
            None,
            None,
            None,
        );
        assert_eq!(cfg.uri, "http://example.local/mcp");
    }
}
