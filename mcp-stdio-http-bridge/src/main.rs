use mcp_stdio_http_bridge::bridge::run_bridge;
use mcp_stdio_http_bridge::config::Config;
use mcp_stdio_http_bridge::stdio;
use tracing::info;
use tracing_subscriber::EnvFilter;

async fn wait_for_shutdown_signal() {
    let sigint = tokio::signal::ctrl_c();
    tokio::pin!(sigint);
    #[cfg(unix)]
    let sigterm = {
        let mut s = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("register SIGTERM");
        async move {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let sigterm = std::future::pending::<()>();
    tokio::select! {
        _ = &mut sigint => {
            info!("received SIGINT, shutting down");
        }
        _ = sigterm => {
            info!("received SIGTERM, shutting down");
        }
    }
}

#[tokio::main]
async fn main() {
    let filter = EnvFilter::from_default_env()
        .add_directive("mcp_stdio_http_bridge=info".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
    let config = Config::from_env();
    let name = config.mcp_name.as_deref().unwrap_or("mcp-stdio-http-bridge");
    info!(%name, uri = %config.uri, "starting bridge");
    let (tx_in, rx_in) = tokio::sync::mpsc::channel(config.max_queue);
    let (tx_out, rx_out) = tokio::sync::mpsc::channel::<String>(config.max_queue);
    let (shutdown_stdin_tx, shutdown_stdin_rx) = tokio::sync::oneshot::channel();
    let (shutdown_bridge_tx, shutdown_bridge_rx) = tokio::sync::oneshot::channel();
    let mut stdin_handle = tokio::spawn(stdio::stdin_reader(tx_in, shutdown_stdin_rx));
    let stdout_handle = tokio::spawn(stdio::stdout_writer(rx_out));
    let mut bridge_handle =
        tokio::spawn(run_bridge(config, rx_in, tx_out, shutdown_bridge_rx));
    let shutdown_fut = wait_for_shutdown_signal();
    tokio::pin!(shutdown_fut);
    tokio::select! {
        _ = &mut shutdown_fut => {}
        _ = &mut stdin_handle => {}
        _ = &mut bridge_handle => {}
    }
    let _ = shutdown_stdin_tx.send(());
    let _ = shutdown_bridge_tx.send(());
    let _ = stdin_handle.await;
    let _ = bridge_handle.await;
    let _ = stdout_handle.await;
}
