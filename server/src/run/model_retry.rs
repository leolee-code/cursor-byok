//! Defines retry policy for one logical model call.
//!
//! The attempt count, the backoff and the Retry-After rule follow
//! deepseek-harness's default request retry policy (`packages/llm`, MIT
//! licensed).

use std::time::Duration;

use super::{ModelCycleFailure, RunFailure};

pub(super) const MAX_MODEL_RETRIES: u32 = 5;
const INITIAL_DELAY: Duration = Duration::from_millis(500);
const MAX_DELAY: Duration = Duration::from_secs(10);
/// Each delay is scaled by a random factor within this fraction of itself.
const JITTER: f64 = 0.1;

/// How long to wait before retrying, or `None` to report the failure.
///
/// A provider's Retry-After is honored as given. One longer than the maximum
/// delay means the provider will not recover soon, so the failure is reported
/// instead of waited out.
pub(super) fn retry_delay(failure: &ModelCycleFailure, retries: u32) -> Option<Duration> {
    if !failure.retryable || retries >= MAX_MODEL_RETRIES {
        return None;
    }
    let retry_after = match &failure.failure {
        RunFailure::Provider(message) => crate::provider::retry_after(message),
        _ => None,
    };
    match retry_after {
        Some(delay) if delay > MAX_DELAY => None,
        Some(delay) => Some(delay),
        None => Some(backoff(retries + 1, random_unit())),
    }
}

/// Exponential delay for the `retry`-th retry, scaled by `sample` in [0, 1].
fn backoff(retry: u32, sample: f64) -> Duration {
    let exponential = INITIAL_DELAY
        .saturating_mul(2_u32.saturating_pow(retry.saturating_sub(1)))
        .min(MAX_DELAY);
    exponential
        .mul_f64(1.0 - JITTER + 2.0 * JITTER * sample)
        .min(MAX_DELAY)
}

fn random_unit() -> f64 {
    (uuid::Uuid::new_v4().as_u128() >> 75) as f64 / (1_u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure(retryable: bool) -> ModelCycleFailure {
        provider_failure("failed", retryable)
    }

    fn provider_failure(message: &str, retryable: bool) -> ModelCycleFailure {
        ModelCycleFailure {
            failure: RunFailure::Provider(message.into()),
            partial_text: String::new(),
            partial_reasoning: String::new(),
            usage: None,
            retryable,
        }
    }

    #[test]
    fn permits_five_retries_after_the_initial_attempt() {
        let retryable = failure(true);
        for retries in 0..MAX_MODEL_RETRIES {
            assert!(retry_delay(&retryable, retries).is_some());
        }
        assert_eq!(retry_delay(&retryable, MAX_MODEL_RETRIES), None);
    }

    #[test]
    fn terminal_failures_never_retry() {
        assert_eq!(retry_delay(&failure(false), 0), None);
    }

    #[test]
    fn delays_double_from_half_a_second_up_to_ten_seconds() {
        let delays = (1..=6).map(|retry| backoff(retry, 0.5)).collect::<Vec<_>>();
        assert_eq!(
            delays,
            [500, 1_000, 2_000, 4_000, 8_000, 10_000].map(Duration::from_millis)
        );
        assert_eq!(backoff(1, 0.0), Duration::from_millis(450));
        assert_eq!(backoff(1, 1.0), Duration::from_millis(550));
        assert_eq!(backoff(6, 1.0), MAX_DELAY);
    }

    #[test]
    fn a_providers_retry_after_replaces_the_backoff_unless_it_is_too_long() {
        let soon = provider_failure(
            "Test 429 Too Many Requests (retry after 3000ms): slow down",
            true,
        );
        assert_eq!(retry_delay(&soon, 0), Some(Duration::from_secs(3)));
        let late = provider_failure(
            "Test 429 Too Many Requests (retry after 60000ms): slow down",
            true,
        );
        assert_eq!(retry_delay(&late, 0), None);
    }
}
