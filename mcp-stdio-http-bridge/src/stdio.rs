use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, error, instrument};

#[instrument(skip(rx))]
pub async fn stdout_writer(mut rx: mpsc::Receiver<String>) {
    let mut stdout = tokio::io::stdout();
    while let Some(line) = rx.recv().await {
        if let Err(e) = tokio::io::AsyncWriteExt::write_all(
            &mut stdout,
            format!("{}\n", line).as_bytes(),
        )
        .await
        {
            error!(%e, "stdout write failed");
            break;
        }
        if let Err(e) = stdout.flush().await {
            error!(%e, "stdout flush failed");
            break;
        }
        debug!(len = line.len(), "wrote response line");
    }
}

pub async fn stdin_reader(
    tx: mpsc::Sender<String>,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) {
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();
    loop {
        tokio::select! {
            _ = &mut shutdown => {
                debug!("stdin reader received shutdown");
                break;
            }
            res = reader.read_line(&mut line) => {
                match res {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
                        if trimmed.is_empty() {
                            line.clear();
                            continue;
                        }
                        if tx.send(trimmed.to_string()).await.is_err() {
                            break;
                        }
                        line.clear();
                    }
                    Err(e) => {
                        error!(%e, "stdin read error");
                        break;
                    }
                }
            }
        }
    }
    drop(tx);
}
