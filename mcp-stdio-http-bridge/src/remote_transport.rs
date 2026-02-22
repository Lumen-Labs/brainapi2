use crate::config::Config;
use reqwest::Client;
use std::fmt;
use std::io;
use tracing::{debug, instrument};

#[derive(Debug)]
pub enum TransportError {
    Network(reqwest::Error),
    InvalidUtf8,
    Io(io::Error),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::Network(e) => write!(f, "network: {}", e),
            TransportError::InvalidUtf8 => write!(f, "invalid UTF-8 in response"),
            TransportError::Io(e) => write!(f, "io: {}", e),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<reqwest::Error> for TransportError {
    fn from(e: reqwest::Error) -> Self {
        TransportError::Network(e)
    }
}

impl From<io::Error> for TransportError {
    fn from(e: io::Error) -> Self {
        TransportError::Io(e)
    }
}

pub fn is_retryable(e: &TransportError) -> bool {
    match e {
        TransportError::Network(err) => {
            err.is_connect() || err.is_timeout() || err.is_request()
        }
        TransportError::InvalidUtf8 | TransportError::Io(_) => false,
    }
}

pub fn build_client(config: &Config) -> Client {
    let builder = Client::builder()
        .connect_timeout(config.timeout)
        .timeout(config.timeout);
    builder.build().expect("reqwest client")
}

#[instrument(skip(config, client, body), fields(uri = %config.uri))]
pub async fn send_message(
    config: &Config,
    client: &Client,
    body: &str,
) -> Result<Vec<String>, TransportError> {
    let mut req = client
        .post(&config.uri)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(body.to_string());
    if let Some(ref token) = config.bearer_token {
        req = req.header("Authorization", format!("Bearer {}", token));
    }
    let res = req.send().await.map_err(TransportError::Network)?;
    let status = res.status();
    let content_type = res
        .headers()
        .get("Content-Type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = res.bytes().await.map_err(TransportError::Network)?;
    if status.as_u16() == 202 {
        return Ok(Vec::new());
    }
    let body_str = String::from_utf8(bytes.to_vec()).map_err(|_| TransportError::InvalidUtf8)?;
    if body_str.trim().is_empty() {
        return Ok(Vec::new());
    }
    if content_type.contains("text/event-stream") {
        let messages = parse_sse_to_json_lines(&body_str);
        debug!(count = messages.len(), "parsed SSE response");
        return Ok(messages);
    }
    Ok(vec![body_str])
}

fn parse_sse_to_json_lines(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut data_buf = String::new();
    for line in s.lines() {
        if line.starts_with("data:") {
            let rest = line[5..].trim();
            if rest == "[DONE]" {
                continue;
            }
            if !data_buf.is_empty() {
                data_buf.push('\n');
            }
            data_buf.push_str(rest);
        } else if line.trim().is_empty() {
            if !data_buf.is_empty() {
                out.push(std::mem::take(&mut data_buf));
            }
        }
    }
    if !data_buf.is_empty() {
        out.push(data_buf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sse_multiple_events() {
        let s = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notify\"}\n\n";
        let out = parse_sse_to_json_lines(s);
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("\"result\""));
        assert!(out[1].contains("\"method\""));
    }
}
