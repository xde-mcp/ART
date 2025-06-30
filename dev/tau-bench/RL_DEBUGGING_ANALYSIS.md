# RL Training Debugging Analysis: tau-bench vs art-e

## Executive Summary

After analyzing both `dev/tau-bench/run_rl.py` (non-working RL) and `examples/art-e/art_e/train.py` (working RL), I've identified several potential subtle bugs and architectural differences that could cause the tau-bench RL training to run without improving the model.

## Key Findings and Theories

### 1. **Trajectory Finishing and Duration Tracking**

**Issue**: The tau-bench implementation calls `traj.finish()` BEFORE logging to OpenPipe, while art-e properly tracks duration throughout the rollout.

**tau-bench (problematic)**:
```python
traj.finish()  # Called immediately after try/except

# Log to langfuse/openpipe
try:
    await log_trajectory_to_openpipe(traj, result.messages)
except Exception as e:
    print(f"Error logging trajectory to openpipe: {e}")
```

**art-e (working)**:
```python
# Records start time at beginning
rollout_start_time = datetime.now()

# ... rollout happens ...

# Records end time and computes duration
rollout_end_time = datetime.now()
duration_seconds = (rollout_end_time - rollout_start_time).total_seconds()
traj.metrics["duration"] = duration_seconds
```

**Theory**: The `finish()` method adds duration metrics that might be important for the training algorithm. Calling it too early could result in incorrect duration measurements (near-zero), potentially affecting reward scaling or trajectory selection.

### 2. **Message and Choice Structure Discrepancy**

**Issue**: tau-bench creates `messages_and_choices` differently than art-e, potentially missing critical training signals.

**tau-bench approach**:
```python
# Uses a separate method after the fact
traj.messages_and_choices = agent.create_messages_and_choices(result.messages)
```

**art-e approach**:
```python
# Builds messages_and_choices incrementally during rollout
if model.trainable:
    traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))
else:
    traj.messages_and_choices.append(choice.message.to_dict())
```

**Theory**: The tau-bench approach might not properly preserve the Choice objects with their logprobs and other metadata needed for RL training. The `create_messages_and_choices` method in tau-bench only adds choices for assistant messages, potentially missing important structural information.

### 3. **Reward Assignment Timing**

**Issue**: Different reward assignment patterns between implementations.

**tau-bench**:
```python
# Assigns reward from result
traj.reward = result.reward

# Later potentially overwritten by general_rm
if config.reward_type == "general_rm":
    # Creates new trajectory with different reward
    new_trajectory.reward = response.rollout_scores[idx].score
```

**art-e**:
```python
# Calculates reward at the end of rollout
reward = calculate_reward(model.config, rubric, traj)
traj.reward = reward
```

**Theory**: The tau-bench implementation might have issues with reward propagation, especially when using general_rm. The reward might not be properly associated with the trajectory during training.

### 4. **Async Handling and Concurrency Issues**

**Issue**: tau-bench has complex async wrapping that might introduce subtle bugs.

**tau-bench**:
```python
# Has both rollout_tau_bench_task and async_rollout_tau_bench_task
async def async_rollout_tau_bench_task(...):
    """Direct alias for rollout_tau_bench_task since it's now truly async."""
    return await rollout_tau_bench_task(...)
```

**art-e**:
```python
# Simple async function with retry decorator
@retry(stop=stop_after_attempt(3))
@weave.op()
async def rollout(...):
    # Direct implementation
```

**Theory**: The extra async wrapping layer might cause issues with trajectory collection or introduce race conditions in the training loop.

### 5. **Tool Response Format**

**Issue**: tau-bench uses a different tool response format that might not align with what the art library expects.

**tau-bench**:
```python
{
    "role": "tool",
    "tool_call_id": next_message["tool_calls"][0]["id"],
    "name": next_message["tool_calls"][0]["function"]["name"],
    "content": env_response.observation,
}
```

**art-e**:
```python
{
    "role": "tool",
    "tool_call_id": message.tool_calls[0].id,
    "content": json.dumps(response),  # JSON serialized
}
```

**Theory**: The tau-bench version includes a "name" field and doesn't JSON-serialize the content, which might cause the model to learn incorrect patterns or fail to process tool responses properly.

### 6. **Trajectory Metadata and Metrics**

**Issue**: Different metadata structures might affect training.

**tau-bench metadata**:
```python
metadata={
    "task_index": str(task_index),  # String
    "env": config.env,
    "training_step": str(step),      # String
    "phase": phase,
    "model": model.name,
    "reward_type": config.reward_type,
}
```

**art-e metadata**:
```python
metadata={
    "email_inbox": scenario.inbox_address,
    "scenario_id": scenario.id
}
```

**Theory**: Converting indices to strings and including training-specific metadata in the trajectory might interfere with the training process.

### 7. **Error Handling and Trajectory Completion**

**Issue**: tau-bench catches exceptions but still returns trajectories with zero reward.

**tau-bench**:
```python
except Exception as e:
    print(f"Error in rollout for task {task_index}: {e}")
    traj.reward = 0.0
    traj.metadata["error"] = str(e)

traj.finish()  # Still finishes the trajectory
```

**art-e**:
```python
# Uses retry decorator
@retry(stop=stop_after_attempt(3))
# Lets exceptions propagate up
```

**Theory**: Including failed trajectories with zero reward might poison the training data, especially if the failures are due to infrastructure issues rather than poor model behavior.

### 8. **Learning Rate and Training Configuration**

**Issue**: Default learning rates differ significantly.

- tau-bench: `1.2e-5`
- art-e: `1.2e-5` (config) but TrainConfig default is `5e-6`

**Theory**: The learning rate might be too high for the tau-bench task complexity, causing training instability.

### 9. **Max Tokens and Completion Settings**

**Issue**: tau-bench sets explicit max_tokens while art-e uses max_completion_tokens.

**tau-bench**:
```python
max_tokens=1024,
logprobs=True,
```

**art-e**:
```python
max_completion_tokens=model.config.max_tokens,  # 2048 by default
```

**Theory**: The lower token limit might truncate important reasoning steps, preventing the model from learning complex behaviors.

### 10. **Training Step Updates**

**Issue**: tau-bench uses a custom step update mechanism for OpenPipe logs.

```python
await update_steps_for_openpipe_logs(trajectory_groups, global_step)
```

**Theory**: This external step update might interfere with the art library's internal step tracking, causing misalignment in the training process.

## Recommendations for Debugging

1. **Add detailed logging** to compare trajectory structures between working and non-working implementations
2. **Verify Choice objects** contain proper logprobs and metadata
3. **Check reward distribution** before and after general_rm processing
4. **Monitor learning curves** to see if gradients are being computed correctly
5. **Test with simpler reward functions** (disable general_rm) to isolate issues
6. **Compare serialized trajectories** between both implementations
7. **Verify the model is actually updating** by checking parameter changes
8. **Test with smaller learning rates** (e.g., 5e-6 or 1e-6)
9. **Ensure proper async context** throughout the rollout process
10. **Check if the base model** is compatible with the training setup

## Most Likely Culprits

Based on the analysis, the most likely issues are:

1. **Improper Choice object preservation** in `messages_and_choices`
2. **Early trajectory finishing** affecting duration metrics
3. **Tool response format mismatch** causing learning issues
4. **Failed trajectory inclusion** with zero rewards
5. **Async handling complexity** introducing subtle bugs

These issues could individually or collectively prevent the model from learning effectively, even though the training loop appears to run correctly.