//! Executes one Run across model cycles, Tool rounds, and message commits.
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::{
    model::{
        CanonicalMessage, MessageContent, Origin, PreparedRun, Role, RunAction, ToolRoundAssistant,
        ToolRoundId, Usage,
    },
    provider::Provider,
    store::{ContextUsageAnchor, RunStatus, Store},
};

use super::{
    consume_model_cycle, CommitBarrier, CommitCause, MessagesCommitted, RunCommand, RunEvent,
    RunFailure, RunOutcome, RunPort,
};

pub struct RunEngine {
    store: Store,
    provider: Arc<dyn Provider>,
}

/// Provider-visible tail appended when the committed history ends with the
/// assistant, which happens when Cursor resumes a turn that had already
/// finished (for example after a cancelled follow-up).
const CONTINUE_MESSAGE_ID: &str = "runtime:continue";
const CONTINUE_INSTRUCTION: &str = "Continue from where you left off.";

impl RunEngine {
    pub fn new(store: Store, provider: Arc<dyn Provider>) -> Self {
        Self { store, provider }
    }

    #[tracing::instrument(
        skip_all,
        fields(run_id = %prepared.run_id, conversation_id = %prepared.conversation_id)
    )]
    pub async fn run(
        &self,
        prepared: PreparedRun,
        mut client: RunPort,
        cancellation: CancellationToken,
    ) -> RunOutcome {
        let claimed = match self.store.claim_run(&prepared).await {
            Ok(claimed) => claimed,
            Err(error) => {
                let outcome = RunOutcome::Failed(error.into());
                client.phase.finish();
                let _ = client.events.send(RunEvent::Ended(outcome.clone())).await;
                tracing::info!(outcome = ?outcome, "Run claim failed");
                return outcome;
            }
        };
        let outcome = self
            .run_claimed(
                &prepared,
                claimed.head_checkpoint_id,
                &mut client,
                &cancellation,
            )
            .await;
        let usage = outcome.1;
        let outcome = outcome.0;
        let (status, failure) = match &outcome {
            RunOutcome::Completed => (RunStatus::Completed, None),
            RunOutcome::Cancelled => (RunStatus::Cancelled, None),
            RunOutcome::Failed(failure) => (
                RunStatus::Failed,
                Some((failure.category(), failure_message(failure))),
            ),
        };
        let failure_ref = failure
            .as_ref()
            .map(|(category, summary)| (*category, summary.as_str()));
        if let Err(error) = self
            .store
            .finish_run(&prepared.run_id, status, usage, failure_ref)
            .await
        {
            tracing::error!(run_id = %prepared.run_id, %error, "failed to persist Run outcome");
        }
        client.phase.finish();
        let _ = client.events.send(RunEvent::Ended(outcome.clone())).await;
        tracing::info!(outcome = ?outcome, usage = ?usage, "Run ended");
        outcome
    }

    async fn run_claimed(
        &self,
        prepared: &PreparedRun,
        mut checkpoint: crate::model::CheckpointId,
        client: &mut RunPort,
        cancellation: &CancellationToken,
    ) -> (RunOutcome, Option<Usage>) {
        let mut usage = None;
        let mut context_usage_anchor = match self
            .store
            .latest_context_usage(prepared.conversation_id.as_str())
            .await
        {
            Ok(anchor) => anchor,
            Err(error) => return (RunOutcome::Failed(error.into()), usage),
        };
        tracing::info!(
            checkpoint_id = checkpoint.0,
            "Run claimed conversation ownership"
        );
        if !prepared.initial_messages.is_empty() {
            let mut changed = false;
            for message in &prepared.initial_messages {
                match self
                    .store
                    .append_message_once(
                        &prepared.conversation_id,
                        &prepared.run_id,
                        checkpoint,
                        message,
                    )
                    .await
                {
                    Ok((next, inserted)) => {
                        checkpoint = next;
                        changed |= inserted;
                    }
                    Err(error) => return (RunOutcome::Failed(error.into()), usage),
                }
            }
            if changed {
                let (barrier, ready) = CommitBarrier::before_continue();
                if emit(
                    client,
                    RunEvent::MessagesCommitted(MessagesCommitted {
                        checkpoint_id: checkpoint,
                        tool_round_version: 0,
                        cause: CommitCause::InitialMessages,
                        barrier,
                    }),
                )
                .await
                .is_err()
                {
                    return (client_failure(), usage);
                }
                if let Err(outcome) = wait_for_state_ready(ready, cancellation).await {
                    return (outcome, usage);
                }
            }
        }

        if let RunAction::Resume {
            pending_tool_round: Some(round),
        } = &prepared.action
        {
            checkpoint = match super::tool_round::execute(
                &self.store,
                prepared,
                client,
                cancellation,
                checkpoint,
                super::tool_round::ToolRound {
                    id: ToolRoundId::new(format!("{}:round:resume", prepared.run_id)),
                    assistant: round.assistant.clone(),
                    calls: round.calls.clone(),
                    recovered_started_at_ms: Some(round.started_at_ms),
                    reminders: Default::default(),
                },
                Vec::new(),
            )
            .await
            {
                Ok(checkpoint) => checkpoint,
                Err(outcome) => return (outcome, usage),
            };
        }

        // A provider refusal for an over-limit prompt triggers one compaction
        // per run. A second refusal after compacting means the current input
        // itself does not fit, and compacting again would only destroy history.
        let mut overflow_compacted = false;
        'model: loop {
            if cancellation.is_cancelled() {
                return (RunOutcome::Cancelled, usage);
            }
            let messages = match self.store.load_checkpoint_messages(checkpoint).await {
                Ok(messages) => messages,
                Err(error) => return (RunOutcome::Failed(error.into()), usage),
            };
            let history = match crate::model::project_messages(&messages) {
                Ok(history) => history,
                Err(error) => return (RunOutcome::Failed(error.into()), usage),
            };
            let compaction_estimate = (prepared.action != RunAction::Compact)
                .then(|| {
                    super::compaction::compaction_estimate(prepared, &history, context_usage_anchor)
                })
                .flatten();
            if let Some(estimated_tokens) = compaction_estimate {
                if emit(
                    client,
                    RunEvent::UsageSnapshot(context_usage_snapshot(estimated_tokens)),
                )
                .await
                .is_err()
                {
                    return (client_failure(), usage);
                }
                match self
                    .auto_compact(
                        prepared,
                        checkpoint,
                        &messages,
                        Some(estimated_tokens),
                        client,
                        cancellation,
                    )
                    .await
                {
                    Ok((next_checkpoint, compaction_usage)) => {
                        checkpoint = next_checkpoint;
                        context_usage_anchor = None;
                        if let Some(compaction_usage) = compaction_usage {
                            accumulate_usage(&mut usage, compaction_usage);
                        }
                        continue 'model;
                    }
                    Err(outcome) => return (outcome, usage),
                }
            }
            let provider_call_index = match self.store.begin_provider_call(&prepared.run_id).await {
                Ok(index) => index,
                Err(error) => return (RunOutcome::Failed(error.into()), usage),
            };
            tracing::debug!(
                provider_call_index,
                checkpoint_id = checkpoint.0,
                "starting model call"
            );
            let mut history = history;
            if let Err(error) = hydrate_tool_images(&self.store, &mut history).await {
                return (RunOutcome::Failed(error.into()), usage);
            }
            // The usage anchor counts persisted messages only; a transient
            // tail is provider-visible but never committed.
            let anchored_messages = history.len();
            let request = if prepared.action == RunAction::Compact {
                super::compaction::summary_request(
                    &prepared.prompt,
                    &prepared.model,
                    history,
                    &Default::default(),
                )
            } else {
                crate::model::ModelRequest {
                    prompt: prepared.prompt.clone(),
                    model: prepared.model.clone(),
                    history: super::history::user_terminated(
                        history,
                        CONTINUE_MESSAGE_ID,
                        CONTINUE_INSTRUCTION,
                    ),
                }
            };
            let mut retries = 0_u32;
            let mut pending_insertions = Vec::new();
            let cycle = 'attempt: loop {
                let call_id = if retries == 0 {
                    format!("{}:{provider_call_index}", prepared.run_id)
                } else {
                    format!("{}:{provider_call_index}:retry-{retries}", prepared.run_id)
                };
                let invocation = crate::model::ModelInvocation {
                    call_id,
                    run_id: prepared.run_id.to_string(),
                    conversation_id: prepared.conversation_id.to_string(),
                    provider_call_index,
                    request: request.clone(),
                };
                let cycle_cancellation = cancellation.child_token();
                let cycle_events = client.events.clone();
                let cycle = consume_model_cycle(
                    self.provider.stream(invocation, cycle_cancellation.clone()),
                    &cycle_events,
                    &cycle_cancellation,
                );
                tokio::pin!(cycle);
                let cycle = loop {
                    tokio::select! {
                        biased;
                        command = client.commands.recv() => {
                            let interruption = match command {
                                Some(RunCommand::InsertMessages(insertion)) => {
                                    pending_insertions.push(insertion);
                                    continue;
                                }
                                Some(RunCommand::BreakMessages(messages)) => messages,
                                Some(RunCommand::Cancel) => {
                                    cycle_cancellation.cancel();
                                    let _ = cycle.await;
                                    let _ = emit(client, RunEvent::CycleInterrupted).await;
                                    return (RunOutcome::Cancelled, usage);
                                }
                                Some(RunCommand::ToolResult(_)) => {
                                    cycle_cancellation.cancel();
                                    let _ = cycle.await;
                                    let _ = emit(client, RunEvent::CycleInterrupted).await;
                                    return (
                                        RunOutcome::Failed(RunFailure::Protocol(
                                            "received a tool result while the model was running".into(),
                                        )),
                                        usage,
                                    );
                                }
                                None => {
                                    cycle_cancellation.cancel();
                                    let _ = cycle.await;
                                    let _ = emit(client, RunEvent::CycleInterrupted).await;
                                    return (client_failure(), usage);
                                }
                            };
                            cycle_cancellation.cancel();
                            let interrupted = cycle.await;
                            match interrupted {
                                Ok(cycle) => {
                                    if let Some(cycle_usage) = cycle.usage {
                                        update_context_usage_anchor(
                                            &mut context_usage_anchor,
                                            cycle_usage,
                                            anchored_messages,
                                        );
                                        accumulate_usage(&mut usage, cycle_usage);
                                    }
                                }
                                Err(failure) => {
                                    if let Some(cycle_usage) = failure.usage {
                                        update_context_usage_anchor(
                                            &mut context_usage_anchor,
                                            cycle_usage,
                                            anchored_messages,
                                        );
                                        accumulate_usage(&mut usage, cycle_usage);
                                    }
                                }
                            }
                            if emit(client, RunEvent::CycleInterrupted).await.is_err() {
                                return (client_failure(), usage);
                            }
                            checkpoint = match super::messages::append_batches(
                                &self.store,
                                prepared,
                                client,
                                cancellation,
                                checkpoint,
                                std::mem::take(&mut pending_insertions),
                            )
                            .await
                            {
                                Ok((checkpoint, _)) => checkpoint,
                                Err(outcome) => return (outcome, usage),
                            };
                            checkpoint = match super::messages::append_batches(
                                &self.store,
                                prepared,
                                client,
                                cancellation,
                                checkpoint,
                                vec![interruption],
                            )
                            .await
                            {
                                Ok((checkpoint, _)) => checkpoint,
                                Err(outcome) => return (outcome, usage),
                            };
                            continue 'model;
                        },
                        result = &mut cycle => break result,
                    }
                };
                match cycle {
                    Ok(cycle) => break 'attempt cycle,
                    Err(cycle_failure) => {
                        if let Some(cycle_usage) = cycle_failure.usage {
                            update_context_usage_anchor(
                                &mut context_usage_anchor,
                                cycle_usage,
                                anchored_messages,
                            );
                            accumulate_usage(&mut usage, cycle_usage);
                        }
                        if cancellation.is_cancelled() {
                            let _ = emit(client, RunEvent::CycleInterrupted).await;
                            return (RunOutcome::Cancelled, usage);
                        }
                        // The estimate that cleared the compaction check can
                        // still land over the real limit: it trails the
                        // provider's own count by whatever the request adds
                        // after the anchor was taken. The provider is the
                        // authority, so treat its refusal as the trigger the
                        // estimate missed. Without this a conversation that
                        // crosses the line is wedged: every retry rebuilds the
                        // same prompt and gets the same refusal.
                        if !overflow_compacted
                            && prepared.action != RunAction::Compact
                            && matches!(&cycle_failure.failure, RunFailure::Provider(message)
                                if super::compaction::is_context_overflow(message))
                        {
                            tracing::warn!(
                                provider_call_index,
                                checkpoint_id = checkpoint.0,
                                "provider rejected the prompt as over-limit; compacting and retrying"
                            );
                            overflow_compacted = true;
                            checkpoint = match super::messages::append_batches(
                                &self.store,
                                prepared,
                                client,
                                cancellation,
                                checkpoint,
                                std::mem::take(&mut pending_insertions),
                            )
                            .await
                            {
                                Ok((checkpoint, _)) => checkpoint,
                                Err(outcome) => return (outcome, usage),
                            };
                            let messages =
                                match self.store.load_checkpoint_messages(checkpoint).await {
                                    Ok(messages) => messages,
                                    Err(error) => return (RunOutcome::Failed(error.into()), usage),
                                };
                            match self
                                .auto_compact(
                                    prepared,
                                    checkpoint,
                                    &messages,
                                    None,
                                    client,
                                    cancellation,
                                )
                                .await
                            {
                                Ok((next_checkpoint, compaction_usage)) => {
                                    checkpoint = next_checkpoint;
                                    context_usage_anchor = None;
                                    if let Some(compaction_usage) = compaction_usage {
                                        accumulate_usage(&mut usage, compaction_usage);
                                    }
                                    continue 'model;
                                }
                                Err(outcome) => return (outcome, usage),
                            }
                        }
                        let Some(delay) = super::model_retry::retry_delay(&cycle_failure, retries)
                        else {
                            return (RunOutcome::Failed(cycle_failure.failure), usage);
                        };
                        retries += 1;
                        let message = failure_message(&cycle_failure.failure);
                        tracing::warn!(
                            provider_call_index,
                            retries,
                            max_retries = super::model_retry::MAX_MODEL_RETRIES,
                            delay_ms = delay.as_millis() as u64,
                            %message,
                            checkpoint_id = checkpoint.0,
                            "model attempt failed; retrying from current checkpoint"
                        );
                        if emit(
                            client,
                            RunEvent::ModelAttemptFailed {
                                attempt: retries,
                                message,
                            },
                        )
                        .await
                        .is_err()
                        {
                            return (client_failure(), usage);
                        }

                        let delay = tokio::time::sleep(delay);
                        tokio::pin!(delay);
                        loop {
                            tokio::select! {
                                biased;
                                command = client.commands.recv() => {
                                    let interruption = match command {
                                        Some(RunCommand::InsertMessages(insertion)) => {
                                            pending_insertions.push(insertion);
                                            continue;
                                        }
                                        Some(RunCommand::BreakMessages(messages)) => messages,
                                        Some(RunCommand::Cancel) => {
                                            let _ = emit(client, RunEvent::CycleInterrupted).await;
                                            return (RunOutcome::Cancelled, usage);
                                        }
                                        Some(RunCommand::ToolResult(_)) => {
                                            return (
                                                RunOutcome::Failed(RunFailure::Protocol(
                                                    "received a tool result while waiting to retry the model".into(),
                                                )),
                                                usage,
                                            );
                                        }
                                        None => return (client_failure(), usage),
                                    };
                                    if emit(client, RunEvent::CycleInterrupted).await.is_err() {
                                        return (client_failure(), usage);
                                    }
                                    checkpoint = match super::messages::append_batches(
                                        &self.store,
                                        prepared,
                                        client,
                                        cancellation,
                                        checkpoint,
                                        std::mem::take(&mut pending_insertions),
                                    )
                                    .await
                                    {
                                        Ok((checkpoint, _)) => checkpoint,
                                        Err(outcome) => return (outcome, usage),
                                    };
                                    checkpoint = match super::messages::append_batches(
                                        &self.store,
                                        prepared,
                                        client,
                                        cancellation,
                                        checkpoint,
                                        vec![interruption],
                                    )
                                    .await
                                    {
                                        Ok((checkpoint, _)) => checkpoint,
                                        Err(outcome) => return (outcome, usage),
                                    };
                                    continue 'model;
                                }
                                _ = cancellation.cancelled() => {
                                    let _ = emit(client, RunEvent::CycleInterrupted).await;
                                    return (RunOutcome::Cancelled, usage);
                                }
                                _ = &mut delay => break,
                            }
                        }
                    }
                }
            };
            if let Some(cycle_usage) = cycle.usage {
                update_context_usage_anchor(
                    &mut context_usage_anchor,
                    cycle_usage,
                    anchored_messages,
                );
                accumulate_usage(&mut usage, cycle_usage);
            }

            if prepared.action == RunAction::Compact {
                // The summary call offers the conversation's tools to keep its
                // prefix cached; any call the model makes anyway is ignored.
                let summary = cycle.text.trim().to_string();
                if summary.is_empty() {
                    return (
                        RunOutcome::Failed(RunFailure::Protocol(
                            "compaction model returned an empty summary".into(),
                        )),
                        usage,
                    );
                }
                let event_id = format!("summary:{}", prepared.run_id);
                let summary_message = CanonicalMessage {
                    message_id: format!("runtime:{event_id}"),
                    role: Role::User,
                    origin: Origin::Runtime,
                    content: MessageContent::Parts {
                        parts: vec![crate::model::ContentPart::Text {
                            text: super::compaction::summary_text(&summary),
                        }],
                    },
                    runtime_event_id: Some(event_id),
                };
                checkpoint = match self
                    .store
                    .replace_checkpoint(
                        &prepared.conversation_id,
                        &prepared.run_id,
                        checkpoint,
                        &[summary_message],
                    )
                    .await
                {
                    Ok(checkpoint) => checkpoint,
                    Err(error) => return (RunOutcome::Failed(error.into()), usage),
                };
                if !pending_insertions.is_empty() {
                    checkpoint = match super::messages::append_batches(
                        &self.store,
                        prepared,
                        client,
                        cancellation,
                        checkpoint,
                        pending_insertions,
                    )
                    .await
                    {
                        Ok((next, _)) => next,
                        Err(outcome) => return (outcome, usage),
                    };
                }
                client.phase.begin_finalizing();
                let closing_insertions = match super::messages::drain_accepted(client) {
                    Ok(insertions) => insertions,
                    Err(outcome) => return (outcome, usage),
                };
                if !closing_insertions.is_empty() {
                    checkpoint = match super::messages::append_batches(
                        &self.store,
                        prepared,
                        client,
                        cancellation,
                        checkpoint,
                        closing_insertions,
                    )
                    .await
                    {
                        Ok((next, _)) => next,
                        Err(outcome) => return (outcome, usage),
                    };
                }
                let (barrier, ready) = CommitBarrier::before_continue();
                if emit(
                    client,
                    RunEvent::MessagesCommitted(MessagesCommitted {
                        checkpoint_id: checkpoint,
                        tool_round_version: 0,
                        cause: CommitCause::Compaction { summary },
                        barrier,
                    }),
                )
                .await
                .is_err()
                {
                    return (client_failure(), usage);
                }
                if let Err(outcome) = wait_for_state_ready(ready, cancellation).await {
                    return (outcome, usage);
                }
                return (RunOutcome::Completed, usage);
            }

            if cycle.calls.is_empty() {
                let assistant = CanonicalMessage {
                    message_id: format!("{}:assistant:{provider_call_index}", prepared.run_id),
                    role: Role::Assistant,
                    origin: Origin::Assistant,
                    content: MessageContent::Assistant {
                        text: cycle.text,
                        thinking: cycle.reasoning,
                        tool_round_id: None,
                        replay_state: cycle.replay_state,
                        tool_calls: Vec::new(),
                    },
                    runtime_event_id: None,
                };
                checkpoint = match self
                    .store
                    .append_checkpoint(
                        &prepared.conversation_id,
                        &prepared.run_id,
                        checkpoint,
                        &[assistant],
                    )
                    .await
                {
                    Ok(checkpoint) => checkpoint,
                    Err(error) => return (RunOutcome::Failed(error.into()), usage),
                };
                if !pending_insertions.is_empty() {
                    let inserted = match super::messages::append_batches(
                        &self.store,
                        prepared,
                        client,
                        cancellation,
                        checkpoint,
                        pending_insertions,
                    )
                    .await
                    {
                        Ok((next, inserted)) => {
                            checkpoint = next;
                            inserted
                        }
                        Err(outcome) => return (outcome, usage),
                    };
                    if inserted {
                        continue 'model;
                    }
                }
                client.phase.begin_finalizing();
                let closing_insertions = match super::messages::drain_accepted(client) {
                    Ok(insertions) => insertions,
                    Err(outcome) => return (outcome, usage),
                };
                if !closing_insertions.is_empty() {
                    checkpoint = match super::messages::append_batches(
                        &self.store,
                        prepared,
                        client,
                        cancellation,
                        checkpoint,
                        closing_insertions,
                    )
                    .await
                    {
                        Ok((next, _)) => next,
                        Err(outcome) => return (outcome, usage),
                    };
                    client.phase.resume_running();
                    continue 'model;
                }
                let (barrier, ready) = CommitBarrier::before_continue();
                if emit(
                    client,
                    RunEvent::MessagesCommitted(MessagesCommitted {
                        checkpoint_id: checkpoint,
                        tool_round_version: 0,
                        cause: CommitCause::FinalTurn,
                        barrier,
                    }),
                )
                .await
                .is_err()
                {
                    return (client_failure(), usage);
                }
                if let Err(outcome) = wait_for_state_ready(ready, cancellation).await {
                    return (outcome, usage);
                }
                return (RunOutcome::Completed, usage);
            }

            let round_id =
                ToolRoundId::new(format!("{}:round:{provider_call_index}", prepared.run_id));
            let reminders = super::repeat_guard::reminders(&messages, &cycle.calls);
            checkpoint = match super::tool_round::execute(
                &self.store,
                prepared,
                client,
                cancellation,
                checkpoint,
                super::tool_round::ToolRound {
                    id: round_id,
                    assistant: ToolRoundAssistant {
                        text: cycle.text,
                        thinking: cycle.reasoning,
                        model_call_id: cycle.model_call_id,
                        replay_state: cycle.replay_state,
                    },
                    calls: cycle.calls,
                    recovered_started_at_ms: None,
                    reminders,
                },
                pending_insertions,
            )
            .await
            {
                Ok(checkpoint) => checkpoint,
                Err(outcome) => return (outcome, usage),
            };
        }
    }

    /// Compacts the history before a model call that would not fit.
    ///
    /// `estimated_tokens` is the estimate that triggered compaction; it is
    /// `None` when the provider refused the prompt as too long, and then
    /// pruning alone is never trusted to make it fit.
    async fn auto_compact(
        &self,
        prepared: &PreparedRun,
        checkpoint: crate::model::CheckpointId,
        messages: &[CanonicalMessage],
        estimated_tokens: Option<u64>,
        client: &mut RunPort,
        cancellation: &CancellationToken,
    ) -> std::result::Result<(crate::model::CheckpointId, Option<Usage>), RunOutcome> {
        let failed = |error: crate::Error| RunOutcome::Failed(error.into());
        let pruned = super::compaction::prune_tool_results(messages);
        if let (Some(pruned), Some(estimated_tokens), Some(budget)) = (
            pruned.as_deref(),
            estimated_tokens,
            super::compaction::input_budget(prepared),
        ) {
            let before = crate::model::estimate_context_tokens(
                &prepared.prompt,
                &crate::model::project_messages(messages).map_err(failed)?,
            );
            let after = crate::model::estimate_context_tokens(
                &prepared.prompt,
                &crate::model::project_messages(pruned).map_err(failed)?,
            );
            let remaining = estimated_tokens.saturating_sub(before.saturating_sub(after));
            if remaining <= budget {
                tracing::info!(
                    estimated_tokens,
                    remaining,
                    "pruned oversized tool results instead of summarizing"
                );
                return self
                    .commit_pruned(
                        prepared,
                        checkpoint,
                        pruned,
                        remaining,
                        client,
                        cancellation,
                    )
                    .await
                    .map(|checkpoint| (checkpoint, None));
            }
        }
        let messages = pruned.as_deref().unwrap_or(messages);
        let partition = super::compaction::partition(prepared, messages).map_err(failed)?;
        if partition.is_empty() {
            let projected = crate::model::project_messages(messages).map_err(failed)?;
            let message = super::compaction::validate_compacted(prepared, &projected)
                .err()
                .unwrap_or_else(|| {
                    "context overflow after compaction: no conversation history can be compacted"
                        .into()
                });
            return Err(RunOutcome::Failed(RunFailure::Protocol(message)));
        }

        emit(client, RunEvent::AutoCompactionStarted)
            .await
            .map_err(|_| client_failure())?;
        let provider_call_index = self
            .store
            .begin_provider_call(&prepared.run_id)
            .await
            .map_err(failed)?;
        let mut history = crate::model::project_messages(&partition.head).map_err(failed)?;
        hydrate_tool_images(&self.store, &mut history)
            .await
            .map_err(failed)?;
        let invocation = crate::model::ModelInvocation {
            call_id: format!("{}:{provider_call_index}", prepared.run_id),
            run_id: prepared.run_id.to_string(),
            conversation_id: prepared.conversation_id.to_string(),
            provider_call_index,
            request: super::compaction::summary_request(
                &prepared.prompt,
                &prepared.model,
                history,
                &partition.carried(),
            ),
        };
        let cycle_cancellation = cancellation.child_token();
        let (silent_events, mut discarded_events) = tokio::sync::mpsc::channel(256);
        let drain = tokio::spawn(async move { while discarded_events.recv().await.is_some() {} });
        let mut pending_insertions = Vec::new();
        let mut break_messages = None;
        let cycle = {
            let cycle = consume_model_cycle(
                self.provider.stream(invocation, cycle_cancellation.clone()),
                &silent_events,
                &cycle_cancellation,
            );
            tokio::pin!(cycle);
            loop {
                tokio::select! {
                    biased;
                    command = client.commands.recv() => match command {
                        Some(RunCommand::InsertMessages(insertion)) => {
                            pending_insertions.push(insertion);
                        }
                        Some(RunCommand::BreakMessages(messages)) => {
                            cycle_cancellation.cancel();
                            break_messages = Some(messages);
                            break cycle.await;
                        }
                        Some(RunCommand::Cancel) => {
                            cycle_cancellation.cancel();
                            let _ = cycle.await;
                            return Err(RunOutcome::Cancelled);
                        }
                        Some(RunCommand::ToolResult(_)) => {
                            cycle_cancellation.cancel();
                            let _ = cycle.await;
                            return Err(RunOutcome::Failed(RunFailure::Protocol(
                                "received a tool result while automatic compaction was running".into(),
                            )));
                        }
                        None => {
                            cycle_cancellation.cancel();
                            let _ = cycle.await;
                            return Err(client_failure());
                        }
                    },
                    result = &mut cycle => break result,
                }
            }
        };
        drop(silent_events);
        let _ = drain.await;
        let (summary, compaction_usage) = match (break_messages.is_some(), cycle) {
            (true, Ok(cycle)) => (
                super::compaction::fallback_summary(&partition.head),
                cycle.usage,
            ),
            (true, Err(failure)) => (
                super::compaction::fallback_summary(&partition.head),
                failure.usage,
            ),
            // Tool calls are ignored: the call offers the conversation's tools
            // only to keep its prefix cached.
            (false, Ok(cycle)) if !cycle.text.trim().is_empty() => {
                (cycle.text.trim().to_string(), cycle.usage)
            }
            (false, Ok(cycle)) => {
                tracing::warn!("automatic compaction returned no usable summary; using fallback");
                (
                    super::compaction::fallback_summary(&partition.head),
                    cycle.usage,
                )
            }
            (false, Err(failure)) => {
                tracing::warn!(error = ?failure.failure, "automatic compaction model failed; using fallback");
                (
                    super::compaction::fallback_summary(&partition.head),
                    failure.usage,
                )
            }
        };
        let event_id = format!("summary:auto:{}:{provider_call_index}", prepared.run_id);
        let summary_message = CanonicalMessage {
            message_id: format!("runtime:{event_id}"),
            role: Role::User,
            origin: Origin::Runtime,
            content: MessageContent::Parts {
                parts: vec![crate::model::ContentPart::Text {
                    text: super::compaction::summary_text(&summary),
                }],
            },
            runtime_event_id: Some(event_id),
        };
        let replacement = partition.replacement(summary_message);
        let projected_replacement = crate::model::project_messages(&replacement).map_err(failed)?;
        super::compaction::validate_compacted(prepared, &projected_replacement)
            .map_err(|message| RunOutcome::Failed(RunFailure::Protocol(message)))?;
        let mut checkpoint = self
            .store
            .replace_checkpoint(
                &prepared.conversation_id,
                &prepared.run_id,
                checkpoint,
                &replacement,
            )
            .await
            .map_err(|error| RunOutcome::Failed(error.into()))?;
        let (barrier, ready) = CommitBarrier::before_continue();
        emit(
            client,
            RunEvent::MessagesCommitted(MessagesCommitted {
                checkpoint_id: checkpoint,
                tool_round_version: 0,
                cause: CommitCause::Compaction { summary },
                barrier,
            }),
        )
        .await
        .map_err(|_| client_failure())?;
        wait_for_state_ready(ready, cancellation).await?;
        emit(client, RunEvent::AutoCompactionCompleted)
            .await
            .map_err(|_| client_failure())?;
        emit(client, RunEvent::UsageSnapshot(context_usage_snapshot(0)))
            .await
            .map_err(|_| client_failure())?;
        checkpoint = super::messages::append_batches(
            &self.store,
            prepared,
            client,
            cancellation,
            checkpoint,
            pending_insertions,
        )
        .await?
        .0;
        if let Some(messages) = break_messages {
            checkpoint = super::messages::append_batches(
                &self.store,
                prepared,
                client,
                cancellation,
                checkpoint,
                vec![messages],
            )
            .await?
            .0;
        }
        Ok((checkpoint, compaction_usage))
    }

    /// Replaces the history with its pruned copy; no summary is made.
    async fn commit_pruned(
        &self,
        prepared: &PreparedRun,
        checkpoint: crate::model::CheckpointId,
        pruned: &[CanonicalMessage],
        estimated_tokens: u64,
        client: &mut RunPort,
        cancellation: &CancellationToken,
    ) -> std::result::Result<crate::model::CheckpointId, RunOutcome> {
        let checkpoint = self
            .store
            .replace_checkpoint(
                &prepared.conversation_id,
                &prepared.run_id,
                checkpoint,
                pruned,
            )
            .await
            .map_err(|error| RunOutcome::Failed(error.into()))?;
        let (barrier, ready) = CommitBarrier::before_continue();
        emit(
            client,
            RunEvent::MessagesCommitted(MessagesCommitted {
                checkpoint_id: checkpoint,
                tool_round_version: 0,
                cause: CommitCause::ToolResultsPruned,
                barrier,
            }),
        )
        .await
        .map_err(|_| client_failure())?;
        wait_for_state_ready(ready, cancellation).await?;
        emit(
            client,
            RunEvent::UsageSnapshot(context_usage_snapshot(estimated_tokens)),
        )
        .await
        .map_err(|_| client_failure())?;
        Ok(checkpoint)
    }
}

async fn hydrate_tool_images(
    store: &Store,
    messages: &mut [crate::model::ProjectedMessage],
) -> crate::Result<()> {
    use crate::{
        model::{ContentPart, ProjectedContent},
        store::BlobId,
        Error,
    };

    for message in messages {
        let ProjectedContent::ToolResult(result) = &mut message.content else {
            continue;
        };
        let Some(image) = &result.image else {
            continue;
        };
        let id = BlobId::from_base64(&image.blob_id)?;
        let data = store.get_blob(&id).await?.ok_or_else(|| {
            Error::Protocol(format!("Read image Blob is missing: {}", image.blob_id))
        })?;
        result.provider_parts = vec![
            ContentPart::Text {
                text: result.content.clone(),
            },
            ContentPart::Image {
                mime_type: image.mime_type.clone(),
                data,
            },
        ];
    }
    Ok(())
}

fn context_usage_snapshot(tokens: u64) -> Usage {
    Usage {
        input_tokens: Some(tokens),
        context_input_tokens: Some(tokens),
        output_tokens: Some(0),
        total_tokens: Some(tokens),
        cache_read_tokens: Some(0),
        cache_write_tokens: Some(0),
        reasoning_tokens: Some(0),
    }
}

fn update_context_usage_anchor(
    anchor: &mut Option<ContextUsageAnchor>,
    usage: Usage,
    message_count: usize,
) {
    if let Some(context_input_tokens) = usage.context_input_tokens {
        *anchor = Some(ContextUsageAnchor {
            context_input_tokens,
            message_count,
        });
    }
}

fn accumulate_usage(total: &mut Option<Usage>, usage: Usage) {
    match total {
        Some(total) => *total += usage,
        None => *total = Some(usage),
    }
}

pub(super) async fn wait_for_state_ready(
    ready: tokio::sync::oneshot::Receiver<std::result::Result<(), String>>,
    cancellation: &CancellationToken,
) -> std::result::Result<(), RunOutcome> {
    let result = tokio::select! {
        biased;
        result = ready => result,
        _ = cancellation.cancelled() => return Err(RunOutcome::Cancelled),
    };
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(RunOutcome::Failed(RunFailure::Client(error))),
        Err(_) => Err(client_failure()),
    }
}

pub(super) async fn emit(client: &RunPort, event: RunEvent) -> Result<(), ()> {
    client.events.send(event).await.map_err(|_| ())
}

pub(super) fn client_failure() -> RunOutcome {
    RunOutcome::Failed(RunFailure::Client("client event channel closed".into()))
}

fn failure_message(failure: &RunFailure) -> String {
    match failure {
        RunFailure::Protocol(message)
        | RunFailure::Provider(message)
        | RunFailure::Store(message)
        | RunFailure::Client(message) => message.clone(),
    }
}
