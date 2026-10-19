//! Exposes the local control API.
mod calls;
mod harness;
mod models;
mod overview;
mod plugins;
mod service;
mod settings;

use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::{header, uri::Authority, Request, Response, StatusCode},
    middleware::{self, Next},
    routing::{any, get, post, put},
    Router,
};
use tower_http::services::ServeDir;
use url::{Host, Url};

pub use service::{
    CallDetail, CallSummary, ControlService, DiscoveredModels, LegacyModelImportPreview,
    LegacyModelImportResult, ModelConnectivityResult, ModelDiscoveryInput, ObservabilitySettings,
};

pub fn web_router(service: ControlService, assets: impl AsRef<std::path::Path>) -> Router {
    local_host_only(
        Router::new()
            .nest_service(
                "/__byok-api__",
                ServeDir::new(assets).append_index_html_on_directories(true),
            )
            .merge(api_routes(service)),
    )
}

pub fn proxy_web_router(service: ControlService, target: Url) -> Router {
    local_host_only(frontend_proxy_router(target).merge(api_routes(service)))
}

pub fn api_router(service: ControlService) -> Router {
    local_host_only(api_routes(service))
}

/// Serves control routes only to requests addressed to `localhost` or an IP literal.
///
/// The control UI is same-origin with this API, so no cross-origin access is
/// granted. A DNS-rebinding page reaches the server under its own domain name,
/// which arrives as the Host; rejecting every other domain keeps such pages away
/// from API keys and settings while desktop, development, and IP-addressed
/// Docker access keep working.
pub fn local_host_only(router: Router) -> Router {
    router.layer(middleware::from_fn(require_local_host))
}

async fn require_local_host(request: Request<Body>, next: Next) -> Response<Body> {
    if request_host(&request).as_ref().is_some_and(local_host) {
        return next.run(request).await;
    }
    tracing::warn!(
        host = ?request.headers().get(header::HOST),
        path = request.uri().path(),
        "rejected control request for a non-local host"
    );
    Response::builder()
        .status(StatusCode::FORBIDDEN)
        .body(Body::from(
            "the control API only accepts localhost or IP address hosts",
        ))
        .expect("static forbidden response")
}

/// Reads the HTTP/1 Host header, or the HTTP/2 `:authority` carried in the URI.
fn request_host(request: &Request<Body>) -> Option<Host> {
    let authority = match request.headers().get(header::HOST) {
        Some(value) => value.to_str().ok()?.parse::<Authority>().ok()?,
        None => request.uri().authority()?.clone(),
    };
    Host::parse(authority.host()).ok()
}

fn local_host(host: &Host) -> bool {
    match host {
        Host::Domain(domain) => domain == "localhost",
        Host::Ipv4(_) | Host::Ipv6(_) => true,
    }
}

fn frontend_proxy_router(target: Url) -> Router {
    let state = FrontendProxy {
        client: reqwest::Client::new(),
        target: target.as_str().trim_end_matches('/').to_string(),
    };
    Router::new()
        .route("/__byok-api__/", any(proxy_frontend))
        .route("/__byok-api__/{*path}", any(proxy_frontend))
        .with_state(state)
}

#[derive(Clone)]
struct FrontendProxy {
    client: reqwest::Client,
    target: String,
}

async fn proxy_frontend(
    State(proxy): State<FrontendProxy>,
    request: Request<Body>,
) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/__byok-api__/");
    let mut upstream = proxy
        .client
        .request(parts.method, format!("{}{path}", proxy.target));
    for (name, value) in &parts.headers {
        if name != header::HOST && name != header::CONNECTION {
            upstream = upstream.header(name, value);
        }
    }
    let body = match to_bytes(body, 64 * 1024 * 1024).await {
        Ok(body) => body,
        Err(error) => return proxy_error(error),
    };
    let upstream = match upstream.body(body).send().await {
        Ok(response) => response,
        Err(error) => return proxy_error(error),
    };
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let body = match upstream.bytes().await {
        Ok(body) => body,
        Err(error) => return proxy_error(error),
    };
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    for (name, value) in &headers {
        if name != header::CONNECTION
            && name != header::TRANSFER_ENCODING
            && name != header::CONTENT_LENGTH
        {
            response.headers_mut().insert(name, value.clone());
        }
    }
    response
}

fn proxy_error(error: impl std::fmt::Display) -> Response<Body> {
    tracing::warn!(%error, "frontend development proxy failed");
    Response::builder()
        .status(StatusCode::BAD_GATEWAY)
        .body(Body::from("frontend development server is unavailable"))
        .expect("static proxy error response")
}

fn api_routes(service: ControlService) -> Router {
    Router::new()
        .route(
            "/__byok-api__/api/models",
            get(models::list).post(models::create),
        )
        .route("/__byok-api__/api/models/discover", post(models::discover))
        .route(
            "/__byok-api__/api/models/import-v0049",
            get(models::preview_v0049).post(models::import_v0049),
        )
        .route("/__byok-api__/api/models/order", put(models::reorder))
        .route("/__byok-api__/api/overview", get(overview::get))
        .route(
            "/__byok-api__/api/models/{model_hash}",
            put(models::update).delete(models::remove),
        )
        .route(
            "/__byok-api__/api/models/{model_hash}/test/{test_id}",
            post(models::test).delete(models::cancel),
        )
        .route("/__byok-api__/api/llm-calls", get(calls::list))
        .route("/__byok-api__/api/llm-calls/{call_id}", get(calls::detail))
        .route("/__byok-api__/api/plugins", get(plugins::list))
        .route(
            "/__byok-api__/api/plugins/runtime",
            get(plugins::runtime_status)
                .post(plugins::initialize_runtime)
                .delete(plugins::cancel_runtime_initialization),
        )
        .route(
            "/__byok-api__/api/plugins/oauth/{session_id}/poll",
            post(plugins::oauth_poll),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}",
            axum::routing::delete(plugins::remove),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/resources/{resource_type}/add/{method_id}/begin",
            post(plugins::oauth_begin),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/resources/{resource_type}/import",
            post(plugins::import),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/resources/{resource_type}/export",
            get(plugins::export_resources),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/resources/{resource_type}/{resource_id}",
            axum::routing::delete(plugins::delete_resource),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/resources/{resource_type}/{resource_id}/actions/{action_id}",
            post(plugins::action),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/resources/{resource_type}/{resource_id}/refresh",
            post(plugins::refresh_resource),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/providers/{provider_id}/models/sync",
            post(plugins::sync_models),
        )
        .route(
            "/__byok-api__/api/plugins/{plugin_id}/providers/{provider_id}/models/enabled",
            put(plugins::set_model_enabled),
        )
        .route(
            "/__byok-api__/api/settings/observability",
            get(settings::get).put(settings::update),
        )
        .route(
            "/__byok-api__/api/settings/ports",
            get(settings::get_ports).put(settings::update_ports),
        )
        .route(
            "/__byok-api__/api/settings/storage/statistics",
            get(settings::get_storage).delete(settings::clear_storage),
        )
        .route(
            "/__byok-api__/api/settings/proxy",
            get(settings::get_proxy).put(settings::update_proxy),
        )
        .route(
            "/__byok-api__/api/settings/web-search",
            get(settings::get_web_search).put(settings::update_web_search),
        )
        .route(
            "/__byok-api__/api/settings/desktop",
            get(settings::get_desktop).put(settings::update_desktop),
        )
        .route(
            "/__byok-api__/api/settings/commit",
            get(settings::get_commit).put(settings::update_commit),
        )
        .route(
            "/__byok-api__/api/harness/cursor/status",
            get(harness::status),
        )
        .route(
            "/__byok-api__/api/harness/cursor/ca/initialize",
            post(harness::initialize_ca),
        )
        .route(
            "/__byok-api__/api/harness/cursor/enabled",
            put(harness::set_enabled),
        )
        .with_state(service)
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{header, Request, StatusCode},
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    use super::local_host_only;

    async fn status(uri: &str, host: Option<&str>) -> StatusCode {
        let routes = Router::new().route("/models", get(|| async { "models" }));
        let mut request = Request::builder().uri(uri);
        if let Some(host) = host {
            request = request.header(header::HOST, host);
        }
        local_host_only(routes)
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn accepts_localhost_and_ip_literal_hosts() {
        for host in [
            "127.0.0.1:3000",
            "localhost:3000",
            "LOCALHOST",
            "[::1]:3000",
            "192.168.1.10:3000",
            "10.0.0.2",
        ] {
            assert_eq!(
                status("/models", Some(host)).await,
                StatusCode::OK,
                "{host}"
            );
        }
    }

    #[tokio::test]
    async fn accepts_http2_authority_without_host_header() {
        assert_eq!(
            status("http://127.0.0.1:3000/models", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn rejects_domain_and_missing_hosts() {
        for host in [
            "rebind.example.com:3000",
            "localhost.example.com",
            "nas.local:3000",
            "not a host",
        ] {
            assert_eq!(
                status("/models", Some(host)).await,
                StatusCode::FORBIDDEN,
                "{host}"
            );
        }
        assert_eq!(status("/models", None).await, StatusCode::FORBIDDEN);
    }
}
