//! Reminds the model when it repeats the exact same tool call.
//!
//! Adapted from deepseek-harness `packages/guard/repeat-tool-reminder` (MIT
//! licensed). The reminder only advises; the call still runs.

use std::collections::HashMap;

use serde_json::Value;

use crate::model::{normalize_tool_name, CanonicalMessage, MessageContent, Role, ToolCall};

/// Consecutive identical calls that trigger a reminder. The first is gentle,
/// later ones name the call.
const THRESHOLDS: [usize; 3] = [3, 5, 8];
/// Characters of the repeated arguments quoted in a detailed reminder.
const ARGUMENTS_PREVIEW_CHARS: usize = 500;

const GENTLE_REMINDER: &str = "You are repeating the exact same tool call with identical arguments. Carefully analyze the previous result before calling again: if the task is not complete, try a different approach or different arguments instead of repeating the call.";

/// Reminders to append to the results of `calls`, by call id.
///
/// The chain of identical calls is rebuilt from the history: it counts calls
/// in order since the last user message, so a new instruction never reads as
/// a loop, and nothing has to survive a restart.
pub(super) fn reminders(
    history: &[CanonicalMessage],
    calls: &[ToolCall],
) -> HashMap<String, String> {
    let mut chain = Chain::default();
    for message in history {
        match &message.content {
            _ if message.role == Role::User => chain = Chain::default(),
            MessageContent::Assistant { tool_calls, .. } => {
                for call in tool_calls {
                    chain.observe(&call.name, &call.arguments);
                }
            }
            _ => {}
        }
    }
    let mut calls = calls.iter().collect::<Vec<_>>();
    calls.sort_by_key(|call| call.index);
    calls
        .into_iter()
        .filter_map(|call| {
            let (count, arguments) = chain.observe(&call.name, &call.arguments);
            let reminder = match THRESHOLDS
                .iter()
                .position(|threshold| *threshold == count)?
            {
                0 => GENTLE_REMINDER.to_string(),
                _ => detailed_reminder(&normalize_tool_name(&call.name), count, &arguments),
            };
            Some((call.call_id.clone(), reminder))
        })
        .collect()
}

/// The last call's identity and how many times in a row it was made.
#[derive(Default)]
struct Chain {
    key: Option<(String, String)>,
    count: usize,
}

impl Chain {
    /// Adds a call and returns the chain length and its canonical arguments.
    fn observe(&mut self, name: &str, arguments: &Value) -> (usize, String) {
        let key = (normalize_tool_name(name), canonical(arguments));
        self.count = if self.key.as_ref() == Some(&key) {
            self.count + 1
        } else {
            1
        };
        let arguments = key.1.clone();
        self.key = Some(key);
        (self.count, arguments)
    }
}

/// Arguments with object keys sorted at every level, so calls that differ only
/// in key order are the same call.
fn canonical(arguments: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut entries = object.iter().collect::<Vec<_>>();
                entries.sort_by_key(|(key, _)| *key);
                Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key.clone(), sorted(value)))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(arguments).to_string()
}

fn detailed_reminder(tool: &str, count: usize, arguments: &str) -> String {
    let total = arguments.chars().count();
    let arguments = if total <= ARGUMENTS_PREVIEW_CHARS {
        arguments.to_string()
    } else {
        let preview = arguments
            .chars()
            .take(ARGUMENTS_PREVIEW_CHARS)
            .collect::<String>();
        format!(
            "{preview}… (+{} more chars)",
            total - ARGUMENTS_PREVIEW_CHARS
        )
    };
    format!(
        "Repeated tool call detected:\n- tool: {tool}\n- consecutive_calls: {count}\n- arguments: {arguments}\nThe repeated calls are not making progress. Do not call this tool with these exact arguments again. Inspect the latest result and choose a different action, different arguments, or finish the task if enough evidence has been gathered."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Origin, ToolCallContent, ToolRoundId};
    use serde_json::json;

    fn round(id: &str, name: &str, arguments: Value) -> CanonicalMessage {
        CanonicalMessage {
            message_id: format!("{id}:assistant"),
            role: Role::Assistant,
            origin: Origin::Assistant,
            content: MessageContent::Assistant {
                text: String::new(),
                thinking: String::new(),
                tool_round_id: Some(ToolRoundId::new(id)),
                replay_state: None,
                tool_calls: vec![ToolCallContent {
                    index: 0,
                    call_id: format!("{id}:call"),
                    name: name.into(),
                    arguments,
                }],
            },
            runtime_event_id: None,
        }
    }

    fn call(index: usize, call_id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            index,
            call_id: call_id.into(),
            model_call_id: "model".into(),
            name: name.into(),
            arguments_text: arguments.to_string(),
            arguments,
            argument_error: None,
        }
    }

    fn history(rounds: usize, arguments: Value) -> Vec<CanonicalMessage> {
        (0..rounds)
            .map(|round_index| round(&format!("r{round_index}"), "Read", arguments.clone()))
            .collect()
    }

    #[test]
    fn the_third_identical_call_gets_a_gentle_reminder() {
        let arguments = json!({"path": "a.rs", "offset": 1});
        let second = reminders(
            &history(1, arguments.clone()),
            &[call(0, "c", "Read", arguments.clone())],
        );
        assert!(second.is_empty());
        // Key order does not make a different call.
        let third = reminders(
            &history(2, arguments),
            &[call(0, "c", "Read", json!({"offset": 1, "path": "a.rs"}))],
        );
        assert_eq!(third["c"], GENTLE_REMINDER);
    }

    #[test]
    fn later_thresholds_name_the_call() {
        let arguments = json!({"path": "a.rs"});
        let fifth = reminders(
            &history(4, arguments.clone()),
            &[call(0, "c", "Read", arguments.clone())],
        );
        assert!(fifth["c"]
            .contains("- tool: Read\n- consecutive_calls: 5\n- arguments: {\"path\":\"a.rs\"}"));
        let sixth = reminders(
            &history(5, arguments.clone()),
            &[call(0, "c", "Read", arguments)],
        );
        assert!(sixth.is_empty());
    }

    #[test]
    fn a_different_call_or_a_user_message_restarts_the_count() {
        let arguments = json!({"path": "a.rs"});
        let mut messages = history(2, arguments.clone());
        messages.push(CanonicalMessage::text(
            "u",
            Role::User,
            Origin::User,
            "try again",
        ));
        assert!(reminders(&messages, &[call(0, "c", "Read", arguments.clone())]).is_empty());

        let mut messages = history(2, arguments.clone());
        messages.push(round("other", "Grep", json!({"pattern": "x"})));
        assert!(reminders(&messages, &[call(0, "c", "Read", arguments)]).is_empty());
    }

    #[test]
    fn parallel_calls_in_one_round_count_in_index_order() {
        let arguments = json!({"path": "a.rs"});
        let calls = [
            call(1, "second", "Read", arguments.clone()),
            call(0, "first", "Read", arguments.clone()),
        ];
        let found = reminders(&history(1, arguments), &calls);
        assert_eq!(found.len(), 1);
        assert_eq!(found["second"], GENTLE_REMINDER);
    }

    #[test]
    fn long_arguments_are_previewed() {
        let reminder = detailed_reminder("Write", 5, &"x".repeat(600));
        assert!(reminder.contains(&format!("{}… (+100 more chars)", "x".repeat(500))));
    }
}
