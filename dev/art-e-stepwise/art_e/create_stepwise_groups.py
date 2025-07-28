import art
from art.trajectories import Trajectory, TrajectoryGroup
from tqdm.asyncio import tqdm
from art_e.project_types import ProjectPolicyConfig
from art.types import Messages, Tools
import litellm
from art.utils.litellm import convert_litellm_choice_to_openai
from litellm.types.utils import ModelResponse
from art.rewards import ruler_score_group


async def create_group_from_prefix(
    prefix: Messages,
    tools: Tools | None,
    model: art.Model[ProjectPolicyConfig],
) -> TrajectoryGroup:
    """
    Create a group from a prefix.
    """

    completions = await litellm.acompletion(
        **model.litellm_completion_params(),
        messages=prefix,
        max_completion_tokens=model.config.max_tokens,
        tools=tools,
        n=4,
    )

    assert isinstance(completions, ModelResponse)

    # Convert choices to OpenAI format immediately to avoid pickling issues
    openai_choices = [
        convert_litellm_choice_to_openai(choice) for choice in completions.choices
    ]

    group = TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=prefix + [openai_choice],
                tools=tools,
                reward=0,
            )
            for openai_choice in openai_choices
        ]
    )

    scored_group = await ruler_score_group(group, model.config.ruler_judge_model)
    return scored_group


async def create_stepwise_groups(
    model: art.Model[ProjectPolicyConfig],
    trajectory: Trajectory,
) -> list[TrajectoryGroup]:
    """
    Create stepwise groups from a trajectory.
    """
    prefixes = []
    messages = trajectory.messages()
    
    # Ensure messages are clean dictionaries without any litellm objects
    clean_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Create a clean copy of the message
            clean_msg = {"role": msg["role"]}
            
            # Handle different message types
            if msg["role"] == "tool":
                # Tool messages need tool_call_id
                clean_msg["tool_call_id"] = msg.get("tool_call_id", "unknown")
                clean_msg["content"] = msg.get("content", "")
            else:
                # Regular messages
                if "content" in msg:
                    clean_msg["content"] = msg["content"]
                if "tool_calls" in msg and msg["tool_calls"]:
                    clean_msg["tool_calls"] = msg["tool_calls"]
            
            clean_messages.append(clean_msg)
        else:
            # This shouldn't happen but let's be safe
            clean_messages.append({"role": "user", "content": str(msg)})
    
    for i in range(len(clean_messages)):
        if clean_messages[i]["role"] == "assistant":
            prefix = clean_messages[:i]
            prefixes.append(prefix)

    return await tqdm.gather(
        *(
            create_group_from_prefix(prefix, trajectory.tools, model)
            for prefix in prefixes
        )
    )
