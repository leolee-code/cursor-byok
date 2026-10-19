//! Executes BYOK web search through the fixed Exa Search API.

use std::time::Duration;

use reqwest::{redirect::Policy, StatusCode};
use serde::Deserialize;
use url::Url;

use crate::store::{Store, WebSearchSettingsSecret};

const EXA_SEARCH_URL: &str = "https://api.exa.ai/search";
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const RESULT_LIMIT: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub chunk: String,
}

#[derive(Clone)]
pub struct WebSearch {
    source: SearchSource,
    endpoint: String,
}

#[derive(Clone)]
enum SearchSource {
    Managed(Store),
    Fixed {
        client: reqwest::Client,
        settings: WebSearchSettingsSecret,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("web search failed: {0}")]
pub struct SearchError(String);

#[derive(Deserialize)]
struct ExaSearchResponse {
    #[serde(default)]
    results: Vec<ExaResult>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExaResult {
    #[serde(default)]
    title: Option<String>,
    url: Option<String>,
    #[serde(default)]
    published_date: Option<String>,
    #[serde(default)]
    highlights: Vec<String>,
}

#[derive(Deserialize)]
struct ExaErrorBody {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

impl WebSearch {
    pub fn built_in() -> Self {
        Self {
            source: SearchSource::Fixed {
                client: direct_client(),
                settings: WebSearchSettingsSecret::default(),
            },
            endpoint: EXA_SEARCH_URL.into(),
        }
    }

    pub(crate) fn managed(store: Store) -> Self {
        Self {
            source: SearchSource::Managed(store),
            endpoint: EXA_SEARCH_URL.into(),
        }
    }

    pub async fn requires_confirmation(&self) -> Result<bool, SearchError> {
        let settings = self.settings().await?;
        Ok(settings.enabled && !settings.api_key.is_empty() && settings.require_confirmation)
    }

    pub async fn search(
        &self,
        query: &str,
        confirmation_granted: bool,
    ) -> Result<Vec<SearchHit>, SearchError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(failure("query is empty"));
        }
        let settings = self.settings().await?;
        if !settings.enabled {
            return Err(failure("BYOK WebSearch is disabled"));
        }
        if settings.api_key.is_empty() {
            return Err(failure(
                "Exa API key is not configured; configure it in BYOK WebSearch settings",
            ));
        }
        if settings.require_confirmation && !confirmation_granted {
            return Err(failure("BYOK WebSearch requires confirmation"));
        }
        let client = self.client().await?;
        let response = client
            .post(&self.endpoint)
            .header("x-api-key", &settings.api_key)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&serde_json::json!({
                "query": query,
                "type": "auto",
                "numResults": RESULT_LIMIT,
                "contents": {
                    "highlights": {
                        "highlightsPerUrl": 1
                    }
                }
            }))
            .send()
            .await
            .map_err(request_failure)?;
        let status = response.status();
        if !status.is_success() {
            return Err(http_failure(status, response).await);
        }
        let payload = response
            .json::<ExaSearchResponse>()
            .await
            .map_err(|error| failure(format!("invalid Exa response: {error}")))?;
        Ok(map_results(payload.results))
    }

    async fn settings(&self) -> Result<WebSearchSettingsSecret, SearchError> {
        match &self.source {
            SearchSource::Managed(store) => store
                .web_search_settings_secret()
                .await
                .map_err(config_failure),
            SearchSource::Fixed { settings, .. } => Ok(settings.clone()),
        }
    }

    async fn client(&self) -> Result<reqwest::Client, SearchError> {
        match &self.source {
            SearchSource::Managed(store) => crate::network::client_builder(store)
                .await
                .map_err(config_failure)?
                .redirect(Policy::none())
                .timeout(SEARCH_TIMEOUT)
                .build()
                .map_err(config_failure),
            SearchSource::Fixed { client, .. } => Ok(client.clone()),
        }
    }

    #[cfg(test)]
    fn fixed(endpoint: String, api_key: &str) -> Self {
        Self {
            source: SearchSource::Fixed {
                client: direct_client(),
                settings: WebSearchSettingsSecret {
                    enabled: true,
                    require_confirmation: false,
                    api_key: api_key.into(),
                },
            },
            endpoint,
        }
    }
}

impl Default for WebSearch {
    fn default() -> Self {
        Self::built_in()
    }
}

fn direct_client() -> reqwest::Client {
    reqwest::Client::builder()
        .use_native_tls()
        .redirect(Policy::none())
        .timeout(SEARCH_TIMEOUT)
        .build()
        .expect("static Exa HTTP client configuration")
}

fn map_results(results: Vec<ExaResult>) -> Vec<SearchHit> {
    results
        .into_iter()
        .filter_map(|result| {
            let url = public_url(result.url.as_deref()?)?;
            let highlight = result
                .highlights
                .into_iter()
                .find(|value| !value.trim().is_empty())?;
            let title = result
                .title
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| url.clone());
            let chunk = match result
                .published_date
                .filter(|value| !value.trim().is_empty())
            {
                Some(date) => format!("Published: {date}\n{}", highlight.trim()),
                None => highlight.trim().to_owned(),
            };
            Some(SearchHit { title, url, chunk })
        })
        .take(RESULT_LIMIT)
        .collect()
}

fn public_url(value: &str) -> Option<String> {
    let mut url = Url::parse(value).ok()?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return None;
    }
    url.set_fragment(None);
    Some(url.to_string())
}

async fn http_failure(status: StatusCode, response: reqwest::Response) -> SearchError {
    let message = match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "Exa API key is invalid".into(),
        StatusCode::PAYMENT_REQUIRED => "Exa API credits are exhausted".into(),
        StatusCode::TOO_MANY_REQUESTS => "Exa API rate limit exceeded".into(),
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => {
            "Exa API request timed out".into()
        }
        _ => {
            let detail = response
                .json::<ExaErrorBody>()
                .await
                .ok()
                .and_then(|body| body.error.or(body.message))
                .filter(|value| !value.trim().is_empty());
            match detail {
                Some(detail) => format!("Exa API error (HTTP {status}): {detail}"),
                None => format!("Exa API error (HTTP {status})"),
            }
        }
    };
    failure(message)
}

fn config_failure(error: impl std::fmt::Display) -> SearchError {
    failure(format!("HTTP client configuration failed: {error}"))
}

fn request_failure(error: reqwest::Error) -> SearchError {
    if error.is_timeout() {
        failure("Exa API request timed out")
    } else {
        failure(format!("Exa API request failed: {error}"))
    }
}

fn failure(message: impl Into<String>) -> SearchError {
    SearchError(message.into())
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[test]
    fn maps_only_public_results_with_highlights() {
        let results = map_results(vec![
            ExaResult {
                title: Some("Example".into()),
                url: Some("https://example.com/page#section".into()),
                published_date: Some("2026-09-10".into()),
                highlights: vec!["".into(), "Useful excerpt".into()],
            },
            ExaResult {
                title: Some("No snippet".into()),
                url: Some("https://example.com/empty".into()),
                published_date: None,
                highlights: Vec::new(),
            },
            ExaResult {
                title: Some("Local".into()),
                url: Some("file:///secret".into()),
                published_date: None,
                highlights: vec!["secret".into()],
            },
        ]);

        assert_eq!(
            results,
            vec![SearchHit {
                title: "Example".into(),
                url: "https://example.com/page".into(),
                chunk: "Published: 2026-09-10\nUseful excerpt".into(),
            }]
        );
    }

    #[tokio::test]
    async fn posts_only_the_query_and_fixed_search_options() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let count = socket.read(&mut buffer).await.unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
                let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end + 4]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_owned)
                    })
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap();
                if request.len() >= header_end + 4 + length {
                    break;
                }
            }
            let request = String::from_utf8(request).unwrap();
            assert!(request.starts_with("POST /search HTTP/1.1\r\n"));
            assert!(request
                .to_ascii_lowercase()
                .contains("x-api-key: exa-secret\r\n"));
            assert!(request
                .to_ascii_lowercase()
                .contains("content-type: application/json\r\n"));
            let body = request.split_once("\r\n\r\n").unwrap().1;
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(body).unwrap(),
                serde_json::json!({
                    "query": "rust release",
                    "type": "auto",
                    "numResults": 10,
                    "contents": {"highlights": {"highlightsPerUrl": 1}}
                })
            );
            let response = r#"{"results":[{"title":"Rust","url":"https://www.rust-lang.org/","publishedDate":"2026-09-10","highlights":["Release notes"]}]}"#;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response.len(), response
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let search = WebSearch::fixed(format!("http://{address}/search"), "exa-secret");

        let hits = search.search("rust release", false).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Rust");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn an_empty_key_fails_before_network_access() {
        let search = WebSearch::fixed("http://127.0.0.1:1/search".into(), "");
        let error = search.search("query", false).await.unwrap_err().to_string();
        assert!(error.contains("not configured"));
    }

    #[tokio::test]
    async fn confirmation_is_rechecked_before_network_access() {
        let mut search = WebSearch::fixed("http://127.0.0.1:1/search".into(), "exa-secret");
        let SearchSource::Fixed { settings, .. } = &mut search.source else {
            unreachable!()
        };
        settings.require_confirmation = true;

        let error = search.search("query", false).await.unwrap_err().to_string();

        assert!(error.contains("requires confirmation"));
    }
}
