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
    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(
        description="A short description of the trajectory's performance."
    )
    score: float = Field(description="A score between 0 and 1.")


class Response(BaseModel):
    scores: List[TrajectoryScore] = Field(description="The scores for each trajectory.")


DEFAULT_RUBRIC = dedent(
    """         
        - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
        - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
        - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
    """
)


async def ruler(
    message_lists: list[list[ChatCompletionMessageParam]],
    litellm_completion_params: dict = {"model": "openai/o3"},
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """Core RULER implementation that scores a list of message trajectories.

    Args:
        trajectories: A list where each item is the message list (ChatCompletionMessageParam)
            representing a single trajectory.
        litellm_completion_params: Parameters forwarded to `litellm.acompletion`.
        rubric: The grading rubric to present to the RULER model.
        debug: If True, pretty-print the raw RULER response.

    Returns:
        A list of `TrajectoryScore` objects – one for each provided trajectory.
    """

    # Short-circuit for the trivial case
    if not message_lists:
        return []

    # Determine the length of the longest common prefix shared by all trajectories.
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
        **litellm_completion_params,
        messages=messages,
        response_format=Response,
        caching=False,
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


async def art_ruler(
    trajectories: list[art.Trajectory],
    litellm_completion_params: dict = {"model": "openai/o3"},
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[art.Trajectory]:
    """Wrapper around :func:`ruler` that works with :class:`art.Trajectory` objects.

    This function preserves the previous public interface, extracting the raw
    message lists, delegating the scoring to :func:`ruler`, and then writing the
    results back onto the provided ``art.Trajectory`` instances.
    """

    # Fast-path for empty input
    if not trajectories:
        return trajectories

    for traj in trajectories:
        if len(traj.additional_histories) > 0:
            raise ValueError("Additional histories are not supported by RULER yet.")

    # Gather message lists and preserve the original reward for later inspection.
    message_lists: list[list[ChatCompletionMessageParam]] = []
    for traj in trajectories:
        message_lists.append(traj.messages())
        traj.metrics["independent_reward"] = traj.reward

    # Delegate the heavy lifting to the core implementation.
    scores = await ruler(
        message_lists,
        litellm_completion_params=litellm_completion_params,
        rubric=rubric,
        debug=debug,
    )

    assert len(scores) == len(trajectories)

    # Update the trajectories with the returned scores.
    for traj, score in zip(trajectories, scores):
        traj.metrics["ruler_score"] = score.score
        traj.reward = (
            score.score if traj.metrics.get("failed_format_validation", 0) == 0 else 0
        )
        traj.log(f"RULER explanation: {score.explanation}")

    return trajectories
