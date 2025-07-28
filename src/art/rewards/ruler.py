"""
RULER (Relative Universal LLM-Elicited Rewards) - A general-purpose reward function for RL agents.

RULER uses an LLM-as-judge to rank multiple agent trajectories relative to each other,
requiring no labeled data or hand-crafted reward functions. It leverages the insight
that relative scoring is easier than absolute scoring, and GRPO only needs relative
scores within each group.

For detailed documentation and examples, see: https://art.openpipe.ai/fundamentals/ruler
"""

import art
from typing import List
import json
from litellm import acompletion
from litellm.types.utils import ModelResponse
from textwrap import dedent
from pydantic import BaseModel, Field
from rich import print
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class TrajectoryScore(BaseModel):
    """Individual score for a single trajectory."""

    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(
        description="A short description of the trajectory's performance."
    )
    score: float = Field(description="A score between 0 and 1.")


class Response(BaseModel):
    """Response format expected from the LLM judge."""

    scores: List[TrajectoryScore] = Field(description="The scores for each trajectory.")


DEFAULT_RUBRIC = dedent(
    """         
        - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
        - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
        - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
    """
)
"""Default rubric used by RULER. This generic rubric works well for most tasks,
as RULER extracts task understanding from the system prompts in the trajectories."""


async def ruler(
    message_lists: list[list[ChatCompletionMessageParam]],
    judge_model: str = "openai/o3",
    extra_litellm_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """Core RULER implementation that scores a list of message trajectories.

    This is the low-level API that works with raw message lists. For integration
    with ART's training loop, use `ruler_score_group` instead.

    RULER works by:
    1. Extracting common prefixes from trajectories to save tokens
    2. Passing all trajectories to an LLM judge for relative scoring
    3. Returning scores that can be used directly as rewards in GRPO

    The key insight is that relative scores within a group are all that matters
    for GRPO, which normalizes them anyway.

    Args:
        message_lists: A list where each item is a list of ChatCompletionMessageParam
            dicts representing a single trajectory.
        judge_model: The model to use for judging. Common options:
            - "openai/gpt-4o-mini" - Fast and cost-effective
            - "openai/o3" - Most capable but expensive (default)
            - "anthropic/claude-3-opus-20240229" - Alternative judge
        extra_litellm_params: Additional parameters to pass to LiteLLM completion.
            Can include temperature, max_tokens, etc.
        rubric: The grading rubric. The default rubric works well for most tasks.
        debug: If True, pretty-print the judge's reasoning to help understand scores.

    Returns:
        A list of TrajectoryScore objects with scores and explanations.

    Example:
        >>> message_lists = [
        ...     [{"role": "system", "content": "You are helpful."},
        ...      {"role": "user", "content": "What is 2+2?"},
        ...      {"role": "assistant", "content": "4"}],
        ...     [{"role": "system", "content": "You are helpful."},
        ...      {"role": "user", "content": "What is 2+2?"},
        ...      {"role": "assistant", "content": "I don't know"}]
        ... ]
        >>> scores = await ruler(message_lists, debug=True)
        >>> print(scores[0].score)  # Higher score for correct answer
        0.9
    """

    # Short-circuit for the trivial case
    if not message_lists:
        return []

    # Determine the length of the longest common prefix shared by all trajectories.
    # This optimization reduces token usage when all trajectories share the same
    # system prompt or initial messages.
    message_lists = message_lists
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(
            len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists
        ):
            common_prefix_len += 1
        else:
            break

    # If there is a non-empty common prefix, serialize it once to save tokens.
    user_text = ""
    if common_prefix_len > 0:
        common_prefix_messages = message_lists[0][:common_prefix_len]
        user_text += (
            "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"
        )

    # Serialize each trajectory (minus the common prefix) for the judge.
    serialized_trajectories: List[str] = []
    for idx, full_messages in enumerate(message_lists, start=1):
        trimmed_messages = full_messages[common_prefix_len:]
        serialized_trajectories.append(
            f'<trajectory id="{idx}">\n'
            + json.dumps(trimmed_messages)
            + "\n</trajectory>"
        )

    user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)

    judge_prompt = dedent(
        f"""
        All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
        """
    )

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]

    response = await acompletion(
        model=judge_model,
        messages=messages,
        response_format=Response,
        caching=False,
        **extra_litellm_params if extra_litellm_params else {},
    )
    assert isinstance(response, ModelResponse)

    if len(response.choices) == 0:
        raise ValueError(f"No choices in response: {response}")
    first_choice = response.choices[0]

    if debug:
        raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
        try:
            print("\n[RULER] Pretty-printed LLM choice JSON:")
            print(json.loads(raw_content))
        except json.JSONDecodeError as e:
            print(f"[RULER] Could not parse choice content as JSON: {e}")
            print(f"[RULER] Raw choice content: {raw_content}")

    content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
    parsed = Response.model_validate_json(content)
    assert len(parsed.scores) == len(message_lists)

    return parsed.scores


async def ruler_score_group(
    group: art.TrajectoryGroup,
    judge_model: str = "openai/o3",
    extra_litellm_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    swallow_exceptions: bool = False,
    debug: bool = False,
) -> art.TrajectoryGroup | None:
    """Score a trajectory group using RULER for use in training loops.

    This is the recommended API for using RULER with ART. It integrates seamlessly
    with `gather_trajectory_groups` via the `after_each` callback.

    Key features:
    - Works with TrajectoryGroup objects
    - Preserves original rewards in metrics["independent_reward"]
    - Adds RULER scores to metrics["ruler_score"]
    - Supports graceful error handling with swallow_exceptions
    - Returns a new TrajectoryGroup with updated rewards

    Args:
        group: A TrajectoryGroup containing trajectories to score.
        judge_model: The model to use for judging. See `ruler` for options.
        extra_litellm_params: Additional parameters to pass to LiteLLM completion.
        rubric: Custom rubric or use the default which works well for most tasks.
        swallow_exceptions: If True, returns None on errors instead of raising.
            This is recommended for production to handle API failures gracefully.
        debug: If True, prints the judge's reasoning.

    Returns:
        A new TrajectoryGroup with updated rewards, or None if swallow_exceptions=True
        and an error occurred.

    Example:
        >>> # In your training loop
        >>> groups = await art.gather_trajectory_groups(
        ...     (art.TrajectoryGroup(rollout(model, scenario) for _ in range(4))
        ...      for scenario in scenarios),
        ...     after_each=lambda g: ruler_score_group(g, "openai/o3",
        ...                                               swallow_exceptions=True)
        ... )

    For complete documentation and examples, see: https://art.openpipe.ai/fundamentals/ruler
    """

    # Validate that we don't have additional histories (not yet supported)
    for traj in group.trajectories:
        if len(traj.additional_histories) > 0:
            raise ValueError("Additional histories are not supported by RULER yet.")

    # Create deep copies to avoid modifying the original trajectories
    # First create shallow copies to avoid issues with unpicklable objects
    new_trajectories = []
    for t in group.trajectories:
        # Create a new trajectory with the same data but fresh objects
        new_traj = t.__class__(
            messages_and_choices=t.messages_and_choices.copy(),
            tools=t.tools.copy() if t.tools else None,
            additional_histories=[h.model_copy(deep=True) for h in t.additional_histories],
            reward=t.reward,
            metrics=t.metrics.copy(),
            metadata=t.metadata.copy(),
            logs=t.logs.copy(),
        )
        new_trajectories.append(new_traj)

    # Extract message lists and preserve original rewards for comparison
    message_lists: list[list[ChatCompletionMessageParam]] = []
    for traj in new_trajectories:
        message_lists.append(traj.messages())
        traj.metrics["independent_reward"] = traj.reward

    try:
        # Call the core ruler function to get scores
        scores = await ruler(
            message_lists,
            judge_model=judge_model,
            extra_litellm_params=extra_litellm_params,
            rubric=rubric,
            debug=debug,
        )
    except Exception as e:
        if swallow_exceptions:
            # In production, it's often better to skip failed groups than crash
            print(f"[art_ruler] Swallowed exception: {e}")
            return None
        else:
            raise

    # Update each trajectory with its RULER score
    for traj, score in zip(new_trajectories, scores):
        traj.metrics["ruler_score"] = score.score
        traj.reward = score.score  # Replace reward with RULER score
        traj.log(f"RULER explanation: {score.explanation}")

    return art.TrajectoryGroup(new_trajectories)
