//! Sends one Provider HTTP attempt without applying retry policy.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

use super::CallRecorder;

#[derive(Debug)]
pub(crate) enum Attempt {
    Response(reqwest::Response),
    Cancelled,
}

pub(crate) async fn send_once<F>(
    label: &str,
    build: F,
    cancellation: &CancellationToken,
    recorder: Option<&CallRecorder>,
) -> Result<Attempt>
where
    F: FnOnce() -> reqwest::RequestBuilder,
{
    let response = tokio::select! {
        _ = cancellation.cancelled() => return Ok(Attempt::Cancelled),
        response = build().send() => response,
    }?;
    if let Some(recorder) = recorder {
        recorder
            .response_headers(response.status().as_u16())
            .await?;
    }
    if response.status().is_success() {
        return Ok(Attempt::Response(response));
    }
    let status = response.status();
    let retry = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(retry_after_header)
        .map(|delay| format!(" ({RETRY_AFTER} {}ms)", delay.as_millis()))
        .unwrap_or_default();
    let bytes = tokio::select! {
        _ = cancellation.cancelled() => return Ok(Attempt::Cancelled),
        bytes = response.bytes() => bytes,
    }?;
    Err(Error::Provider(format!(
        "{label} {status}{retry}: {}",
        String::from_utf8_lossy(&bytes)
    )))
}

/// Marks a provider's requested retry delay in a failure message's head,
/// before the response body.
const RETRY_AFTER: &str = "retry after";

/// The retry delay a provider asked for, read back from a failure message.
pub(crate) fn retry_after(message: &str) -> Option<Duration> {
    let head = message.split_once(": ").map_or(message, |(head, _)| head);
    let (_, rest) = head.split_once(&format!("({RETRY_AFTER} "))?;
    let millis = rest.split_once("ms)")?.0.parse().ok()?;
    Some(Duration::from_millis(millis))
}

/// Parses a Retry-After header: delay seconds or an HTTP date.
fn retry_after_header(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds.is_finite() && seconds > 0.0).then(|| Duration::from_secs_f64(seconds));
    }
    let date = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    (date.with_timezone(&chrono::Utc) - chrono::Utc::now())
        .to_std()
        .ok()
        .filter(|delay| !delay.is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn server(response: &'static [u8]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = socket.read(&mut request).await;
            socket.write_all(response).await.unwrap();
        });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn non_success_status_is_one_failed_attempt() {
        let url =
            server(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\n\r\ndown").await;
        let client = reqwest::Client::new();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::Provider(message) if message.contains("503") && message.contains("down"))
        );
    }

    #[tokio::test]
    async fn a_requested_retry_delay_is_carried_in_the_failure() {
        let url = server(
            b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 2\r\nContent-Length: 4\r\n\r\nslow",
        )
        .await;
        let client = reqwest::Client::new();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        let Error::Provider(message) = error else {
            panic!("expected a provider failure");
        };
        assert_eq!(
            message,
            "test 429 Too Many Requests (retry after 2000ms): slow"
        );
        assert_eq!(retry_after(&message), Some(Duration::from_secs(2)));
    }

    #[test]
    fn retry_after_headers_are_seconds_or_dates() {
        assert_eq!(
            retry_after_header("1.5"),
            Some(Duration::from_millis(1_500))
        );
        assert_eq!(retry_after_header("0"), None);
        assert_eq!(retry_after_header("Sun, 06 Nov 1994 08:49:37 GMT"), None);
        let soon = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc2822();
        assert!(retry_after_header(&soon).is_some_and(|delay| delay > Duration::from_secs(25)));
        // A body that merely mentions the phrase is not a requested delay.
        assert_eq!(retry_after("test 500: (retry after 5ms) in body"), None);
    }

    #[tokio::test]
    async fn response_body_transport_failure_is_one_failed_attempt() {
        let url =
            server(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\n\r\nshort").await;
        let client = reqwest::Client::new();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Http(_)));
    }

    #[tokio::test]
    async fn request_transport_failure_is_one_failed_attempt() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let url = format!("http://{address}");
        let client = reqwest::Client::new();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Http(_)));
    }
}
