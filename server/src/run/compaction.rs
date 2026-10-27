//! Decides when and how to compact provider-visible context.
//!
//! The summary instruction, the checkpoint preamble, the retained-tail rule and
//! the tool-result pruning budgets are adapted from deepseek-harness
//! (`packages/compaction`, MIT licensed).

use std::collections::HashSet;

use crate::{
    model::{
        estimate_context_tokens, estimate_projected_messages_tokens, project_messages,
        CanonicalMessage, ContentPart, MessageContent, ModelRequest, ModelSpec, PreparedRun,
        ProjectedContent, ProjectedMessage, PromptSpec, Role,
    },
    store::ContextUsageAnchor,
    Result,
};

const FALLBACK_CHARS: usize = 12_000;

/// Fraction of the context window kept free, as a divisor: 10 = 10%.
///
/// The estimate runs behind the provider: the anchor is what the provider
/// charged for the *previous* call, and the next request re-sends request
/// context and carries provider-side overhead the message-tail estimate does
/// not model. A real conversation measured 948K estimated against 1,017,628
/// actual, a 7% shortfall that landed it over a 1M window while the check said
/// there was room. A proportional reserve absorbs that drift and scales with
/// the model: a 200K window keeps 20K free and a 1M window keeps 100K.
const CONTEXT_RESERVE_DIVISOR: u64 = 10;

/// Most tokens a summary may use.
const SUMMARY_TOKENS: u64 = 65_536;
/// Least room a summary needs; with less, the replayed history is trimmed.
const MIN_SUMMARY_TOKENS: u64 = 4_096;
/// Share of the message budget (window minus output limit) kept verbatim.
const RETAINED_TAIL_PERCENT: u64 = 16;

/// Tool results longer than this many characters are pruned to a head and a
/// tail before the history is summarized.
const PRUNE_THRESHOLD_CHARS: usize = 8_192;
const PRUNE_HEAD_CHARS: usize = 4_096;
const PRUNE_TAIL_CHARS: usize = 1_024;
const PRUNE_MARKER: &str = "\n\n[... tool result middle pruned ...]\n\n";

const INSTRUCTION_MESSAGE_ID: &str = "compaction:instruction";

/// The summary instruction. It is the final user message after the replayed
/// conversation, not a separate system prompt, so the summary call is a
/// prefix of the last main request and the provider reuses its cache.
const INSTRUCTION: &str = "You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.

Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write \"(none)\" for an empty section — never drop a section.

## Primary Request and Intent
- [the user's original and evolving goals; quote verbatim where the exact wording matters]

## Key Technical Concepts
- [technologies, frameworks, patterns, and conventions in play]

## Files and Code
- [exact path: why it matters, key changes or snippets]

## Errors and Fixes
- [error: how it was resolved, plus any related user feedback]

## Pending Jobs
- [explicitly requested work not yet completed]

## Current Work
- [precisely what was in progress at this checkpoint]

## Next Step
- [the single next action, directly in line with the most recent request, or \"(none)\"]

## Critical Context
- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]

Rules:
- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.
- Capture user feedback and explicit instructions faithfully, especially corrections.
- Do NOT mention this summarization request or that the context was compacted.
- Output only the checkpoint text: do not call any tool or take any other action.
- If the conversation already contains a <conversation_summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.";

/// Frames the summary as established context for the turns that follow.
const PREAMBLE: &str = "This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.";

/// Usable prompt budget: the window minus the proportional reserve.
pub(super) fn context_budget(context_window: u64) -> u64 {
    context_window.saturating_sub(context_window / CONTEXT_RESERVE_DIVISOR)
}

pub(super) fn input_budget(prepared: &PreparedRun) -> Option<u64> {
    prepared.model.context_window_tokens.map(context_budget)
}

/// Whether a provider failure means the prompt did not fit.
///
/// Providers report this as a plain 400 with prose, so there is nothing
/// structured to match on. Anthropic says "prompt is too long"; OpenAI-style
/// gateways use `context_length_exceeded` or "maximum context length".
pub(super) fn is_context_overflow(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("prompt is too long")
        || lowered.contains("context window exceeded")
        || lowered.contains("model_context_window_exceeded")
        || lowered.contains("context_length_exceeded")
        || (lowered.contains("maximum context length") && lowered.contains("token"))
}

/// Text of the user message that stands in for the summarized history.
pub(super) fn summary_text(summary: &str) -> String {
    format!("{PREAMBLE}\n\n<conversation_summary>\n{summary}\n</conversation_summary>")
}

/// Builds the summary call.
///
/// It replays the main request unchanged (instructions, tools, reasoning, and
/// the history as a prefix) and appends the instruction as one final user
/// message, so the provider serves the replayed part from its prefix cache.
/// The summary may only use what the window has left, because compaction
/// triggers at the reserve line without setting room aside for an output.
/// When not even a short summary fits, the cache is given up for this one
/// call: the `carried` messages, which survive compaction anyway, are left
/// out, then the oldest turns are dropped.
pub(super) fn summary_request(
    prompt: &PromptSpec,
    model: &ModelSpec,
    history: Vec<ProjectedMessage>,
    carried: &HashSet<&str>,
) -> ModelRequest {
    let instruction = ProjectedMessage {
        message_id: INSTRUCTION_MESSAGE_ID.into(),
        role: Role::User,
        content: ProjectedContent::Parts(vec![ContentPart::Text {
            text: INSTRUCTION.into(),
        }]),
    };
    let ceiling = summary_tokens(model);
    let (mut history, max_output_tokens) = match model
        .context_window_tokens
        .filter(|window| *window > 0)
        .map(context_budget)
    {
        None => (history, ceiling),
        Some(budget) => {
            let fixed = estimate_context_tokens(prompt, std::slice::from_ref(&instruction));
            let room = |history: &[ProjectedMessage]| {
                budget.saturating_sub(
                    fixed.saturating_add(estimate_projected_messages_tokens(history)),
                )
            };
            let floor = MIN_SUMMARY_TOKENS.min(ceiling);
            let history = if room(&history) >= floor {
                history
            } else {
                let mut history = history;
                history.retain(|message| !carried.contains(message.message_id.as_str()));
                trim_to_budget(history, budget.saturating_sub(fixed).saturating_sub(floor))
            };
            let max_output_tokens = room(&history).clamp(floor, ceiling);
            (history, max_output_tokens)
        }
    };
    history.push(instruction);
    let mut model = model.clone();
    model.max_output_tokens = Some(max_output_tokens);
    ModelRequest {
        prompt: prompt.clone(),
        model,
        history,
    }
}

fn summary_tokens(model: &ModelSpec) -> u64 {
    model
        .max_output_tokens
        .map_or(SUMMARY_TOKENS, |max| max.min(SUMMARY_TOKENS))
}

/// Keeps the newest whole user-delimited turns that fit `budget`, so an
/// assistant tool call is never separated from its results.
fn trim_to_budget(mut history: Vec<ProjectedMessage>, budget: u64) -> Vec<ProjectedMessage> {
    if estimate_projected_messages_tokens(&history) <= budget {
        return history;
    }
    let mut kept = 0;
    let mut newest_turn = None;
    for (index, message) in history.iter().enumerate().rev() {
        if !is_turn_boundary(message) {
            continue;
        }
        newest_turn.get_or_insert(index);
        if estimate_projected_messages_tokens(&history[index..]) > budget {
            break;
        }
        kept = history.len() - index;
    }
    if kept == 0 {
        // Not even the newest turn fits. Keep it anyway rather than sending an
        // empty history: an empty summarize call returns a summary of nothing
        // that would then replace the whole conversation.
        let start = newest_turn.unwrap_or(0);
        return history.split_off(start);
    }
    history.split_off(history.len() - kept)
}

fn is_turn_boundary(message: &ProjectedMessage) -> bool {
    message.role == Role::User && matches!(message.content, ProjectedContent::Parts(_))
}

pub(super) fn estimated_tokens(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
    anchor: Option<ContextUsageAnchor>,
) -> u64 {
    anchor
        .filter(|anchor| anchor.message_count <= projected_messages.len())
        .map(|anchor| {
            anchor
                .context_input_tokens
                .saturating_add(estimate_projected_messages_tokens(
                    &projected_messages[anchor.message_count..],
                ))
        })
        .unwrap_or_else(|| estimate_context_tokens(&prepared.prompt, projected_messages))
}

pub(super) fn compaction_estimate(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
    anchor: Option<ContextUsageAnchor>,
) -> Option<u64> {
    let budget = input_budget(prepared)?;
    let estimated = estimated_tokens(prepared, projected_messages, anchor);
    (estimated > budget).then_some(estimated)
}

#[cfg(test)]
pub(super) fn should_compact(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
    anchor: Option<ContextUsageAnchor>,
) -> bool {
    compaction_estimate(prepared, projected_messages, anchor).is_some()
}

pub(super) fn validate_compacted(
    prepared: &PreparedRun,
    projected_messages: &[ProjectedMessage],
) -> std::result::Result<u64, String> {
    let estimated = estimate_context_tokens(&prepared.prompt, projected_messages);
    let Some(budget) = input_budget(prepared) else {
        return Ok(estimated);
    };
    if estimated <= budget {
        return Ok(estimated);
    }
    Err(format!(
        "context overflow after compaction: estimated input {estimated} tokens exceeds budget {budget} tokens"
    ))
}

/// Cuts the middle out of every tool result longer than the threshold,
/// keeping its first and last characters. Returns `None` when no result is
/// that long.
///
/// A pruned result is a new message: the store keeps one payload per message
/// id, so it gets its own id and the original stays in older checkpoints.
pub(super) fn prune_tool_results(messages: &[CanonicalMessage]) -> Option<Vec<CanonicalMessage>> {
    let oversized = |message: &CanonicalMessage| {
        matches!(&message.content, MessageContent::ToolResult(result)
            if result.content.chars().count() > PRUNE_THRESHOLD_CHARS)
    };
    if !messages.iter().any(oversized) {
        return None;
    }
    Some(
        messages
            .iter()
            .map(|message| {
                let mut message = message.clone();
                if let MessageContent::ToolResult(result) = &mut message.content {
                    if let Some(pruned) = prune(&result.content) {
                        result.content = pruned;
                        message.message_id.push_str(":pruned");
                        if let Some(event_id) = &mut message.runtime_event_id {
                            event_id.push_str(":pruned");
                        }
                    }
                }
                message
            })
            .collect(),
    )
}

fn prune(text: &str) -> Option<String> {
    let chars = text.chars().count();
    if chars <= PRUNE_THRESHOLD_CHARS {
        return None;
    }
    let offset = |nth: usize| {
        text.char_indices()
            .nth(nth)
            .map_or(text.len(), |(at, _)| at)
    };
    let head = &text[..offset(PRUNE_HEAD_CHARS)];
    let tail = &text[offset(chars - PRUNE_TAIL_CHARS)..];
    Some(format!("{head}{PRUNE_MARKER}{tail}"))
}

/// A history split for automatic compaction.
pub(super) struct Partition {
    /// The oldest messages, summarized. Always a prefix of the history, so the
    /// summary call can replay it from the provider cache.
    pub head: Vec<CanonicalMessage>,
    /// The newest messages, kept verbatim after the summary.
    tail: Vec<CanonicalMessage>,
    /// The latest request-context, when it is in the head.
    request_context: Option<CanonicalMessage>,
    /// This run's initial messages that are in the head.
    current: Vec<CanonicalMessage>,
}

impl Partition {
    /// Whether the head holds nothing but messages the replacement keeps.
    pub(super) fn is_empty(&self) -> bool {
        self.head.len() == usize::from(self.request_context.is_some()) + self.current.len()
    }

    /// Ids of the head messages the replacement keeps.
    pub(super) fn carried(&self) -> HashSet<&str> {
        self.request_context
            .iter()
            .chain(&self.current)
            .map(|message| message.message_id.as_str())
            .collect()
    }

    /// The history after compaction: the latest request-context, the summary,
    /// this run's initial messages, then the tail.
    pub(super) fn replacement(self, summary: CanonicalMessage) -> Vec<CanonicalMessage> {
        self.request_context
            .into_iter()
            .chain(std::iter::once(summary))
            .chain(self.current)
            .chain(self.tail)
            .collect()
    }
}

pub(super) fn partition(
    prepared: &PreparedRun,
    messages: &[CanonicalMessage],
) -> Result<Partition> {
    let current_ids = prepared
        .initial_messages
        .iter()
        .map(|message| message.message_id.as_str())
        .collect::<HashSet<_>>();
    let split = tail_start(messages, tail_budget(prepared))?;
    let (head, tail) = messages.split_at(split);
    let request_context = head
        .iter()
        .rposition(|message| message.message_id.starts_with("request-context:"))
        .filter(|_| {
            !tail
                .iter()
                .any(|message| message.message_id.starts_with("request-context:"))
        })
        .map(|index| &head[index])
        .filter(|message| !current_ids.contains(message.message_id.as_str()))
        .cloned();
    let current = head
        .iter()
        .filter(|message| current_ids.contains(message.message_id.as_str()))
        .cloned()
        .collect();
    Ok(Partition {
        head: head.to_vec(),
        tail: tail.to_vec(),
        request_context,
        current,
    })
}

/// Token bounds for the verbatim tail.
struct TailBudget {
    /// The tail grows until it holds at least this many tokens.
    min: u64,
    /// A tail over this many tokens would leave no room for the summary, so
    /// the history is summarized whole instead.
    max: u64,
}

fn tail_budget(prepared: &PreparedRun) -> TailBudget {
    let model = &prepared.model;
    let Some(window) = model.context_window_tokens.filter(|window| *window > 0) else {
        return TailBudget { min: 0, max: 0 };
    };
    let messages = window.saturating_sub(model.max_output_tokens.unwrap_or(0));
    TailBudget {
        min: messages * RETAINED_TAIL_PERCENT / 100,
        max: context_budget(window)
            .saturating_sub(summary_tokens(model))
            .saturating_sub(estimate_context_tokens(&prepared.prompt, &[])),
    }
}

/// Where the verbatim tail starts, or the history length for no tail.
///
/// Walks back from the newest message a whole tool round at a time and stops
/// at the first cut that keeps at least `budget.min` tokens.
fn tail_start(messages: &[CanonicalMessage], budget: TailBudget) -> Result<usize> {
    if budget.min == 0 {
        return Ok(messages.len());
    }
    let mut start = messages.len();
    let mut kept = 0_u64;
    for &cut in cut_points(messages).iter().rev() {
        kept = kept.saturating_add(estimate_projected_messages_tokens(&project_messages(
            &messages[cut..start],
        )?));
        start = cut;
        if kept >= budget.min {
            break;
        }
    }
    if start == 0 || kept > budget.max {
        return Ok(messages.len());
    }
    Ok(start)
}

/// Indices where the history can be cut without splitting a tool round:
/// every call before the cut has its result before it, and the message at
/// the cut does not continue the round before it. Projecting either side
/// then gives exactly the messages projecting the whole history gives.
fn cut_points(messages: &[CanonicalMessage]) -> Vec<usize> {
    let mut points = Vec::new();
    let mut open = HashSet::new();
    let mut round = None;
    for (index, message) in messages.iter().enumerate() {
        let calls = match &message.content {
            MessageContent::Assistant {
                tool_round_id: Some(id),
                tool_calls,
                ..
            } if !tool_calls.is_empty() => Some((id, tool_calls)),
            _ => None,
        };
        let continues_round = calls.is_some_and(|(id, _)| round == Some(id));
        if open.is_empty() && !continues_round {
            points.push(index);
        }
        match (&message.content, calls) {
            (_, Some((id, tool_calls))) => {
                round = Some(id);
                open.extend(tool_calls.iter().map(|call| call.call_id.as_str()));
            }
            (MessageContent::ToolResult(result), None) => {
                open.remove(result.call_id.as_str());
            }
            _ => round = None,
        }
    }
    points
}

pub(super) fn fallback_summary(messages: &[CanonicalMessage]) -> String {
    let serialized = serde_json::to_string(messages).unwrap_or_default();
    let start = serialized
        .char_indices()
        .rev()
        .nth(FALLBACK_CHARS.saturating_sub(1))
        .map_or(0, |(index, _)| index);
    format!(
        "Durable recent conversation state:\n{}",
        &serialized[start..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        CheckpointId, ConversationId, Origin, RunAction, RunId, RunKind, ToolCallContent,
        ToolResultContent, ToolRoundId,
    };

    fn prepared(context_window_tokens: u64) -> PreparedRun {
        let mut model = ModelSpec::new("model");
        model.context_window_tokens = Some(context_window_tokens);
        PreparedRun {
            run_id: RunId::new("run"),
            cursor_request_id: None,
            conversation_id: ConversationId::new("conversation"),
            kind: RunKind::Root,
            model,
            prompt: PromptSpec {
                instructions: String::new(),
                tools: Vec::new(),
            },
            initial_messages: Vec::new(),
            action: RunAction::Start,
            base_checkpoint_id: CheckpointId(1),
        }
    }

    #[test]
    fn automatic_compaction_uses_proportional_reserve_for_every_action() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let estimated = estimate_context_tokens(&prepared(1).prompt, &projected);
        // Smallest multiple-of-ten window whose 90% budget covers the estimate.
        let window = estimated.div_ceil(9) * 10;
        let mut prepared = prepared(window);
        assert!(context_budget(window) >= estimated);
        assert!(context_budget(window - 10) < estimated);

        assert!(!should_compact(&prepared, &projected, None));
        prepared.model.context_window_tokens = Some(window - 10);
        assert!(should_compact(&prepared, &projected, None));

        prepared.action = RunAction::Resume {
            pending_tool_round: None,
        };
        assert!(should_compact(&prepared, &projected, None));
    }

    #[test]
    fn the_reserve_leaves_room_for_the_estimate_to_run_behind() {
        // Reproduces a conversation that wedged itself against a 1M window.
        // The anchor said 947,797 tokens, so a fixed 10K reserve found room
        // and let the request through. Anthropic counted 1,017,628 and
        // refused it, and every retry repeated the same arithmetic.
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "hello",
        )];
        let projected = project_messages(&messages).unwrap();
        let anchor = ContextUsageAnchor {
            context_input_tokens: 947_797,
            message_count: 1,
        };
        assert!(should_compact(
            &prepared(1_000_000),
            &projected,
            Some(anchor)
        ));
    }

    #[test]
    fn the_reserve_scales_with_the_window() {
        assert_eq!(context_budget(200_000), 180_000);
        assert_eq!(context_budget(1_000_000), 900_000);
        assert_eq!(context_budget(0), 0);
        assert_eq!(context_budget(1), 1);
    }

    #[test]
    fn provider_refusals_that_mean_the_prompt_did_not_fit_are_recognized() {
        assert!(is_context_overflow(
            "provider error: Anthropic 400 Bad Request: {\"type\":\"error\",\"error\":\
             {\"type\":\"invalid_request_error\",\"message\":\"prompt is too long: \
             1017628 tokens > 1000000 maximum\"}}"
        ));
        assert!(is_context_overflow("model_context_window_exceeded"));
        assert!(is_context_overflow("context_length_exceeded"));
        assert!(is_context_overflow(
            "This model's maximum context length is 128000 tokens"
        ));

        // Unrelated failures must not trigger a compaction, which would
        // destroy history to fix something compaction cannot fix.
        assert!(!is_context_overflow("401 Unauthorized: invalid api key"));
        assert!(!is_context_overflow("429 Too Many Requests"));
        assert!(!is_context_overflow(
            "This model does not support assistant message prefill"
        ));
    }

    fn user(id: &str, text: &str) -> ProjectedMessage {
        ProjectedMessage {
            message_id: id.into(),
            role: Role::User,
            content: ProjectedContent::Parts(vec![ContentPart::Text { text: text.into() }]),
        }
    }

    fn assistant(id: &str, text: &str) -> ProjectedMessage {
        ProjectedMessage {
            message_id: id.into(),
            role: Role::Assistant,
            content: ProjectedContent::Assistant {
                text: text.into(),
                thinking: String::new(),
                replay_state: None,
                calls: Vec::new(),
            },
        }
    }

    fn main_prompt() -> PromptSpec {
        PromptSpec {
            instructions: "You are a coding agent.".into(),
            tools: Vec::new(),
        }
    }

    fn model(window: Option<u64>, max_output_tokens: Option<u64>) -> ModelSpec {
        let mut model = ModelSpec::new("model");
        model.context_window_tokens = window;
        model.max_output_tokens = max_output_tokens;
        model.reasoning.enabled = true;
        model.reasoning.effort = Some("max".into());
        model
    }

    #[test]
    fn summary_request_replays_the_main_request_and_appends_the_instruction() {
        // The summary call must be the last main request plus one message, so
        // the provider serves the replayed history from its prefix cache.
        for history in [
            vec![user("u1", "question"), assistant("a1", "answer")],
            vec![assistant("a1", "answer"), user("u2", "next")],
        ] {
            let request = summary_request(
                &main_prompt(),
                &model(Some(1_000_000), None),
                history.clone(),
                &HashSet::new(),
            );
            assert_eq!(request.prompt, main_prompt());
            assert_eq!(request.history[..history.len()], history[..]);
            assert_eq!(request.history.len(), history.len() + 1);
            let instruction = request.history.last().unwrap();
            assert_eq!(instruction.message_id, "compaction:instruction");
            assert_eq!(instruction.role, Role::User);
            assert!(request.model.reasoning.enabled);
            assert_eq!(request.model.reasoning.effort.as_deref(), Some("max"));
            assert_eq!(request.model.max_output_tokens, Some(SUMMARY_TOKENS));
        }
    }

    #[test]
    fn summary_request_caps_the_summary_by_the_output_limit_and_the_room_left() {
        let history = vec![user("u1", "question"), assistant("a1", "answer")];
        let request = summary_request(
            &main_prompt(),
            &model(None, Some(32_768)),
            history.clone(),
            &HashSet::new(),
        );
        assert_eq!(request.model.max_output_tokens, Some(32_768));

        // A 200K window at its compaction line has less room than the output
        // limit; the summary gets what is left, not more.
        let big = vec![user("u1", &"x".repeat(600_000)), assistant("a1", "answer")];
        let model = model(Some(200_000), Some(32_768));
        let request = summary_request(&main_prompt(), &model, big.clone(), &HashSet::new());
        assert_eq!(request.history[..2], big[..]);
        let room =
            context_budget(200_000) - estimate_context_tokens(&main_prompt(), &request.history);
        assert!(room < 32_768);
        assert_eq!(request.model.max_output_tokens, Some(room));
    }

    #[test]
    fn summary_request_trims_the_oldest_turns_when_no_summary_fits() {
        let big = "x".repeat(400_000);
        let history = vec![
            user("u1", &big),
            assistant("a1", &big),
            user("u2", &big),
            assistant("a2", "recent answer"),
        ];
        let request = summary_request(
            &main_prompt(),
            &model(Some(200_000), None),
            history,
            &HashSet::new(),
        );

        assert!(
            estimate_context_tokens(&main_prompt(), &request.history)
                <= context_budget(200_000) - MIN_SUMMARY_TOKENS
        );
        assert_eq!(
            request
                .model
                .max_output_tokens
                .map(|max| max >= MIN_SUMMARY_TOKENS),
            Some(true)
        );
        // The newest turn survives the trim and is never split.
        let ids = request
            .history
            .iter()
            .map(|message| message.message_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["u2", "a2", "compaction:instruction"]);
    }

    #[test]
    fn a_summary_that_cannot_reuse_the_cache_leaves_out_carried_messages() {
        // Trimming keeps the newest turn; when that turn is the current user
        // message, which survives compaction anyway, the summary would cover
        // nothing else. Carried messages are dropped before trimming.
        let big = "x".repeat(400_000);
        let history = vec![
            user("u1", "old question"),
            assistant("a1", &big),
            user("runtime:current", "continue"),
        ];
        let carried = HashSet::from(["runtime:current"]);
        let request = summary_request(
            &main_prompt(),
            &model(Some(100_000), None),
            history,
            &carried,
        );
        let ids = request
            .history
            .iter()
            .map(|message| message.message_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["u1", "a1", "compaction:instruction"]);
    }

    #[test]
    fn summary_request_keeps_the_newest_turn_even_when_it_is_over_budget() {
        // A history whose newest turn alone exceeds the budget is still sent
        // rather than trimmed to nothing: a summary of nothing would replace
        // the whole conversation.
        let history = vec![
            user("u1", "old"),
            assistant("a1", "old answer"),
            user("u2", &"x".repeat(400_000)),
            assistant("a2", "answer"),
        ];
        let request = summary_request(
            &main_prompt(),
            &model(Some(50_000), None),
            history,
            &HashSet::new(),
        );
        assert_eq!(request.history[0].message_id, "u2");
        assert_eq!(
            request.history.last().unwrap().message_id,
            "compaction:instruction"
        );
        assert_eq!(request.model.max_output_tokens, Some(MIN_SUMMARY_TOKENS));
    }

    #[test]
    fn summaries_carry_the_checkpoint_preamble() {
        let text = summary_text("Durable summary");
        assert!(text.starts_with(PREAMBLE));
        assert!(text.ends_with("<conversation_summary>\nDurable summary\n</conversation_summary>"));
    }

    fn text(id: &str, role: Role, text: &str) -> CanonicalMessage {
        CanonicalMessage::text(id, role, Origin::Runtime, text)
    }

    fn tool_round(round: &str, calls: &[&str], text: &str) -> Vec<CanonicalMessage> {
        let mut messages = vec![CanonicalMessage {
            message_id: format!("{round}:assistant"),
            role: Role::Assistant,
            origin: Origin::Assistant,
            content: MessageContent::Assistant {
                text: String::new(),
                thinking: String::new(),
                tool_round_id: Some(ToolRoundId::new(round)),
                replay_state: None,
                tool_calls: calls
                    .iter()
                    .enumerate()
                    .map(|(index, call)| ToolCallContent {
                        index,
                        call_id: (*call).into(),
                        name: "Read".into(),
                        arguments: serde_json::json!({}),
                    })
                    .collect(),
            },
            runtime_event_id: None,
        }];
        messages.extend(calls.iter().map(|call| CanonicalMessage {
            message_id: format!("{call}:result"),
            role: Role::Tool,
            origin: Origin::Tool,
            content: MessageContent::ToolResult(ToolResultContent {
                call_id: (*call).into(),
                name: "Read".into(),
                content: text.into(),
                is_error: false,
                image: None,
                provider_parts: Vec::new(),
            }),
            runtime_event_id: None,
        }));
        messages
    }

    fn ids(messages: &[CanonicalMessage]) -> Vec<&str> {
        messages
            .iter()
            .map(|message| message.message_id.as_str())
            .collect()
    }

    #[test]
    fn the_tail_keeps_recent_messages_and_never_splits_a_tool_round() {
        // 16% of a 100K window is 16K tokens: the newest round alone is
        // about 10K, so the tail grows to include the round before it.
        let mut messages = vec![
            text("request-context:1", Role::User, "rules"),
            text("u1", Role::User, &"old ".repeat(20_000)),
        ];
        messages.extend(tool_round("r1", &["c1", "c2"], &"y".repeat(20_000)));
        messages.push(text("u2", Role::User, "next"));
        messages.extend(tool_round("r2", &["c3"], &"z".repeat(40_000)));
        let mut prepared = prepared(100_000);
        prepared.model.max_output_tokens = Some(0);

        let partition = partition(&prepared, &messages).unwrap();
        assert_eq!(
            ids(&partition.tail),
            [
                "r1:assistant",
                "c1:result",
                "c2:result",
                "u2",
                "r2:assistant",
                "c3:result"
            ]
        );
        assert_eq!(ids(&partition.head), ["request-context:1", "u1"]);
        assert!(!partition.is_empty());

        let summary = text("runtime:summary:auto:run:1", Role::User, "summary");
        let replacement = partition.replacement(summary);
        assert_eq!(
            ids(&replacement),
            [
                "request-context:1",
                "runtime:summary:auto:run:1",
                "r1:assistant",
                "c1:result",
                "c2:result",
                "u2",
                "r2:assistant",
                "c3:result"
            ]
        );
    }

    #[test]
    fn the_tail_never_starts_inside_a_tool_round() {
        let mut messages = vec![text("u1", Role::User, "start")];
        messages.extend(tool_round("r1", &["c1"], "a"));
        messages.extend(tool_round("r1", &["c2"], "b"));
        messages.extend(tool_round("r2", &["c3"], "c"));
        assert_eq!(cut_points(&messages), [0, 1, 5]);
    }

    #[test]
    fn current_messages_and_the_latest_context_are_carried_past_the_summary() {
        let mut messages = vec![
            text("request-context:old", Role::User, "old rules"),
            text("u1", Role::User, "old question"),
            text("request-context:new", Role::User, "new rules"),
            text("runtime:current", Role::User, "current question"),
        ];
        messages.extend(tool_round("r1", &["c1"], "result"));
        let mut prepared = prepared(1_000_000);
        prepared.initial_messages = vec![messages[3].clone()];

        // The whole history fits in the tail budget, so nothing is kept
        // verbatim and everything is summarized.
        let partition = partition(&prepared, &messages).unwrap();
        assert_eq!(partition.head.len(), messages.len());
        let replacement = partition.replacement(text("summary", Role::User, "summary"));
        assert_eq!(
            ids(&replacement),
            ["request-context:new", "summary", "runtime:current"]
        );
    }

    #[test]
    fn without_a_window_the_whole_history_is_summarized() {
        let messages = vec![text("u1", Role::User, "question")];
        let mut prepared = prepared(1);
        prepared.model.context_window_tokens = None;
        let partition = partition(&prepared, &messages).unwrap();
        assert_eq!(ids(&partition.head), ["u1"]);
        assert_eq!(
            ids(&partition.replacement(text("s", Role::User, "s"))),
            ["s"]
        );
    }

    #[test]
    fn a_history_of_only_carried_messages_has_nothing_to_summarize() {
        let messages = vec![
            text("request-context:1", Role::User, "rules"),
            text("runtime:current", Role::User, "question"),
        ];
        let mut prepared = prepared(1_000_000);
        prepared.initial_messages = vec![messages[1].clone()];
        assert!(partition(&prepared, &messages).unwrap().is_empty());
    }

    #[test]
    fn oversized_tool_results_keep_their_head_and_tail() {
        let long = format!(
            "{}{}{}",
            "h".repeat(4_096),
            "m".repeat(10_000),
            "t".repeat(1_024)
        );
        let mut messages = vec![text("u1", Role::User, &"u".repeat(20_000))];
        messages.extend(tool_round("r1", &["c1"], &long));
        messages.extend(tool_round("r2", &["c2"], "short"));

        let pruned = prune_tool_results(&messages).unwrap();
        assert_eq!(pruned[2].message_id, "c1:result:pruned");
        let MessageContent::ToolResult(result) = &pruned[2].content else {
            panic!("expected a tool result");
        };
        assert_eq!(
            result.content,
            format!("{}{PRUNE_MARKER}{}", "h".repeat(4_096), "t".repeat(1_024))
        );
        // Only tool results are pruned, and only the long ones.
        assert_eq!(pruned[0], messages[0]);
        assert_eq!(pruned[4], messages[4]);
        assert!(prune_tool_results(&pruned).is_none());
    }

    #[test]
    fn pruning_counts_characters_not_bytes() {
        let exact = "字".repeat(PRUNE_THRESHOLD_CHARS);
        assert_eq!(prune(&exact), None);
        let over = format!("{exact}字");
        let pruned = prune(&over).unwrap();
        assert_eq!(
            pruned.chars().count(),
            PRUNE_HEAD_CHARS + PRUNE_MARKER.chars().count() + PRUNE_TAIL_CHARS
        );
    }

    #[test]
    fn provider_usage_anchor_only_estimates_messages_added_after_last_request() {
        let messages = vec![
            CanonicalMessage::text("old", Role::User, Origin::Runtime, "x".repeat(400_000)),
            CanonicalMessage::text("new", Role::User, Origin::Runtime, "short follow-up"),
        ];
        let projected = project_messages(&messages).unwrap();
        let anchor = ContextUsageAnchor {
            context_input_tokens: 103_904,
            message_count: 1,
        };
        let expected = 103_904 + estimate_projected_messages_tokens(&projected[1..]);

        assert_eq!(
            estimated_tokens(&prepared(200_000), &projected, Some(anchor)),
            expected
        );
        assert!(!should_compact(
            &prepared(200_000),
            &projected,
            Some(anchor)
        ));
    }

    #[test]
    fn provider_usage_anchor_triggers_after_new_messages_cross_budget() {
        let messages = vec![
            CanonicalMessage::text("old", Role::User, Origin::Runtime, "old"),
            CanonicalMessage::text("new", Role::User, Origin::Runtime, "x".repeat(80_000)),
        ];
        let projected = project_messages(&messages).unwrap();

        assert!(should_compact(
            &prepared(200_000),
            &projected,
            Some(ContextUsageAnchor {
                context_input_tokens: 180_000,
                message_count: 1,
            })
        ));
    }

    #[test]
    fn missing_anchor_uses_full_fallback() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let prepared = prepared(200_000);

        assert_eq!(
            estimated_tokens(&prepared, &projected, None),
            estimate_context_tokens(&prepared.prompt, &projected)
        );
    }

    #[test]
    fn invalid_anchor_message_count_uses_full_fallback() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let expected = estimate_context_tokens(&prepared(200_000).prompt, &projected);

        assert_eq!(
            estimated_tokens(
                &prepared(200_000),
                &projected,
                Some(ContextUsageAnchor {
                    context_input_tokens: 1,
                    message_count: 2,
                })
            ),
            expected
        );
    }

    #[test]
    fn compacted_history_is_validated_against_the_same_budget() {
        let messages = vec![CanonicalMessage::text(
            "user",
            Role::User,
            Origin::Runtime,
            "x".repeat(40_000),
        )];
        let projected = project_messages(&messages).unwrap();
        let estimated = estimate_context_tokens(&prepared(1).prompt, &projected);
        let window = estimated.div_ceil(9) * 10;
        assert!(context_budget(window) >= estimated);
        assert!(context_budget(window - 10) < estimated);

        assert_eq!(
            validate_compacted(&prepared(window), &projected),
            Ok(estimated)
        );
        assert!(validate_compacted(&prepared(window - 10), &projected)
            .unwrap_err()
            .contains("context overflow after compaction"));
    }
}
