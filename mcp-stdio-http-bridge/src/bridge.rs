use crate::config::Config;
use crate::remote_transport::{self, build_client, send_message};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

const INITIAL_BACKOFF_MS: u64 = 500;

pub async fn run_bridge(
    config: Config,
    mut rx: mpsc::Receiver<String>,
    tx_out: mpsc::Sender<String>,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) {
    let client = build_client(&config);
    let name = config
        .mcp_name
        .as_deref()
        .unwrap_or("mcp-stdio-http-bridge");
    let mut backoff_ms = INITIAL_BACKOFF_MS;
    while let Some(msg) = rx.recv().await {
        if shutdown.try_recv().is_ok() {
            debug!("bridge received shutdown, dropping pending");
            break;
        }
        loop {
            match send_message(&config, &client, &msg).await {
                Ok(responses) => {
                    backoff_ms = INITIAL_BACKOFF_MS;
                    for line in responses {
                        if tx_out.send(line).await.is_err() {
                            return;
                        }
                    }
                    break;
                }
                Err(e) if remote_transport::is_retryable(&e) => {
                    warn!(%e, "remote request failed, retrying with backoff");
                    let delay = std::time::Duration::from_millis(backoff_ms);
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = &mut shutdown => {
                            let err_msg = serde_json::json!({"jsonrpc":"2.0","error":{"code":-32603,"message":"bridge shutdown during retry"}}).to_string();
                            if let Err(_) = tx_out.send(err_msg).await {}
                            return;
                        }
                    }
                    backoff_ms = (backoff_ms * 2).min(config.max_backoff.as_millis() as u64);
                }
                Err(e) => {
                    error!(%e, "remote request failed (non-retryable)");
                    let err_body = serde_json::json!({
                        "jsonrpc": "2.0",
                        "error": { "code": -32603, "message": format!("bridge transport error: {}", e) }
                    });
                    let _ = tx_out.send(err_body.to_string()).await;
                    break;
                }
            }
        }
    }
    info!(%name, "bridge finished");
}
