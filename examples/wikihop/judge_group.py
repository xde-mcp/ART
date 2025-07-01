from openai.types.chat import ChatCompletionMessageParam
import tenacity
import json
import os
from textwrap import dedent
import art
from typing import List
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from litellm import acompletion

judge_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


class JudgeGroupScore(BaseModel):
    rollout_id: str = Field(description="The id of the rollout being scored.")
    explanation: str = Field(
        description="A short explanation of why you gave this score."
    )
    score: float = Field(description="A score between 0 and 1.")


class JudgeGroupResponse(BaseModel):
    scores: List[JudgeGroupScore]


@tenacity.retry(stop=tenacity.stop_after_attempt(10))
async def judge_group(
    _model_name: str,  # Just included for observability
    trajectories: list[art.Trajectory],
    judge_model_name: str = "openai/o4-mini",
    *,
    debug: bool = False,
) -> list[art.Trajectory]:
    """Judge a list of trajectories with an LLM-as-a-judge.

    This keeps the original trajectories but overwrites ``reward`` with the
    score returned by the judge (0–1).  The original reward is preserved in
    ``traj.metrics['independent_reward']`` and the new score is written to
    ``traj.metrics['judge_group_reward']``.
    """

    # Serialize each rollout's messages (keeping tool_calls as-is)
    serialized_rollouts: List[str] = []
    # Keep structured messages for nicer debug printing
    debug_rollouts: List[tuple[int, list]] = [] if debug else []
    for idx, traj in enumerate(trajectories, start=1):
        # Save the original reward
        traj.metrics["independent_reward"] = traj.reward
        # Flatten messages to regular OpenAI format (role/content/…)
        messages = traj.messages()
        if debug:
            debug_rollouts.append((idx, messages))
        serialized_rollouts.append(
            f'<rollout id="{idx}">\n' + json.dumps(messages) + "\n</rollout>"
        )

    if debug:
        print("\n[judge_group] Serialized rollouts (pretty JSON):")
        for idx, msg_list in debug_rollouts:
            print(f"\nRollout {idx}:")
            print(json.dumps(msg_list, indent=2, ensure_ascii=False))

        print("\n[judge_group] Rollout metrics:")
        for idx, traj in enumerate(trajectories, start=1):
            print(f"\nRollout {idx} metrics:")
            print(json.dumps(traj.metrics, indent=2, ensure_ascii=False))

    rubric_text = dedent(
        """
        All of the rollouts below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

        Grading standards:
            - A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
            - A rollout that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a rollout that achieves its goal less efficiently.
            - If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
            - You may give some partial credit for a rollout that makes progress towards its goal but does not complete it.
        """
    )

    user_text = "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

    # Decide which LLM should act as the judge.  TrainingConfig now carries
    # a `judge_group_model_name` with a default of "openai/o3" so existing
    # runs do not have to set anything.  If `training_config` is None, we also
    # fall back to "openai/o3".

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": rubric_text},
        {"role": "user", "content": user_text},
    ]

    response = await acompletion(
        model=judge_model_name,
        messages=messages,
        response_format=JudgeGroupResponse,
        caching=True,
    )

    first_choice = response.choices[0]  # type: ignore[attr-defined]

    if debug:
        raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
        print("\n[judge_group] Raw LLM choice content:")
        print(raw_content)

        try:
            print("\n[judge_group] Pretty-printed LLM choice JSON:")
            print(json.dumps(json.loads(raw_content), indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"[judge_group] Could not parse choice content as JSON: {e}")

    content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
    parsed = JudgeGroupResponse.model_validate_json(content)
    assert len(parsed.scores) == len(trajectories)

    for idx, (traj, score) in enumerate(zip(trajectories, parsed.scores)):
        traj.metrics["judge_group_reward"] = score.score
        traj.reward = score.score
        if traj.metrics.get("failed_format_validation", 0) > 0:
            traj.reward = 0
        traj.metadata["judge_group_explanation"] = score.explanation

    return trajectories
