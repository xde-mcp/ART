import os
from typing import Any, Dict, List, Optional
import art
from openpipe.client import AsyncOpenPipe, UpdateLogTagsRequestFiltersItem
from langfuse import Langfuse
from art.trajectories import MetadataValue
import uuid
import time
from datetime import datetime
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


def log_trajectory_to_langfuse(
    traj: art.Trajectory, messages: List[Dict[str, Any]]
) -> None:
    """
    Push one trajectory to Langfuse with task_idx and step for comparison.
    """
    # Initialize langfuse
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    phase = traj.metadata.get("phase", "unknown")
    step = traj.metadata.get("training_step", 0)
    task_idx = traj.metadata.get("task_index", 0)
    env = traj.metadata.get("env", "unknown")

    trace_name = f"rl-{phase}-step-{step}-task-{task_idx}"

    # Create trace with trajectory data
    trace = langfuse.trace(
        name=trace_name,
        input={
            "task_idx": task_idx,
            "step": step,
            "phase": phase,
            "metadata": traj.metadata,
        },
        output={"messages": messages, "reward": traj.reward, "metadata": traj.metadata},
        metadata={
            "task_idx": task_idx,
            "training_step": step,
            "phase": phase,
            "env": env,
        },
    )

    # Add reward as a score
    trace.score(name="reward", value=traj.reward)


def string_to_string_dict(metadata: dict[str, Any]) -> dict[str, str]:
    string_dict = {}
    for key, value in metadata.items():
        if isinstance(value, MetadataValue):
            string_dict[key] = str(value)
    return string_dict


def create_response_payload(response_str: Optional[str] = None) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-dummy-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "dummy-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": response_str or "Dummy Response",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def log_trajectory_to_openpipe(
    traj: art.Trajectory,
    messages: List[Dict[str, Any]],
    response_str: Optional[str] = None,
) -> None:
    """
    Push one trajectory to Langfuse with task_idx and step for comparison.
    """
    # Initialize langfuse
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])
    report_payload_metrics = string_to_string_dict(traj.metadata)
    resp_payload = create_response_payload(response_str=response_str)
    traj.metadata["completion_id"] = resp_payload["id"]

    await op_client.report(
        req_payload={
            "model": traj.metadata["model"],
            "messages": messages,
            "tools": traj.tools,
            "metadata": report_payload_metrics,
        },
        resp_payload=resp_payload,
        status_code=200,
    )
    await op_client.base_client._client_wrapper.httpx_client.aclose()


async def update_openpipe_log(traj: art.Trajectory):
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])

    update_log_metadata_response = await op_client.update_log_metadata(
        filters=[
            UpdateLogTagsRequestFiltersItem(
                field="completionId",
                equals=traj.metadata["completion_id"],  # type: ignore
            ),
        ],
        metadata={
            "reward": str(traj.reward),
            "judge_explanation": traj.metadata["judge_explanation"],
        },
    )
    print(f"update_log_metadata_response: {update_log_metadata_response}")
    await op_client.base_client._client_wrapper.httpx_client.aclose()


async def update_steps_for_openpipe_logs(
    trajectory_groups: List[art.TrajectoryGroup], global_step: int
):
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])
    for trajectory_group in trajectory_groups:
        for trajectory in trajectory_group:
            completion_id = trajectory.metadata["completion_id"]
            await op_client.update_log_metadata(
                filters=[
                    UpdateLogTagsRequestFiltersItem(
                        field="completionId",
                        equals=completion_id,  # type: ignore
                    ),
                ],
                metadata={
                    "training_step": str(global_step),
                },
            )
    print(f"Updated {len(trajectory_groups)} trajectory groups with step {global_step}")
    await op_client.base_client._client_wrapper.httpx_client.aclose()


class ErrorAnalysis(BaseModel):
    """Model representing the error analysis for a task with failed rollouts"""

    task_id: int = Field(description="The task ID that was analyzed")
    reasoning: str = Field(
        description="Detailed reasoning about why the rollouts failed"
    )
    blame_assignment: str = Field(
        description="Assignment of blame: 'scenario', 'assistant', 'user', or 'combination'"
    )
    specific_issues: List[str] = Field(
        description="List of specific issues identified in the rollouts"
    )
    recommendations: str = Field(
        description="Recommendations for addressing the identified issues"
    )


ERROR_ANALYSIS_PROMPT = """You are an expert at analyzing AI assistant performance in customer service scenarios. You will be given:

1. A user objective/task
2. The correct set of tools that should be used to accomplish the task
3. The system message given to the assistant
4. Multiple rollouts where the assistant failed to complete the task correctly

Your job is to analyze why the assistant failed and assign blame appropriately.

**Blame Categories:**
- "scenario": The task setup, tools, or environment is problematic/unclear
- "assistant": The assistant made poor decisions or mistakes
- "user": The user's instructions were unclear or problematic
- "combination": Multiple factors contributed to the failure

**Analysis Guidelines:**
1. Look for patterns across the failed rollouts
2. Identify if the assistant consistently makes the same mistakes
3. Check if the tools provided are sufficient for the task
4. Evaluate if the system message gives clear guidance
5. Consider if the user's objective is achievable with the given tools

Provide detailed reasoning, specific issues found, and recommendations for improvement.
"""


def keep_only_messages(
    messages_and_choices: art.MessagesAndChoices,
) -> List[Dict[str, Any]]:
    """Keep only the messages from the messages_and_choices."""
    only_messages = []
    for message_and_choice in messages_and_choices:
        if hasattr(message_and_choice, "message"):
            only_messages.append(message_and_choice.message.model_dump())
        else:
            only_messages.append(message_and_choice)
    return only_messages


def create_correct_tools_description(actions: List[Dict[str, Any]]) -> str:
    """Create a string representation of the correct tools for the task."""
    tools_desc = ""
    for action in actions:
        tools_desc += f"TOOL: {action['name']} with arguments: {action['kwargs']}\n"
    return tools_desc


def format_rollout_messages(messages: List[Dict[str, Any]]) -> str:
    """Format rollout messages for analysis."""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            tool_call_str = ""
            for tool_call in tool_calls:
                tool_call_str += f"TOOL CALL: {tool_call['function']['name']}: {tool_call['function']['arguments']}\n"
            formatted += f"{role.upper()}: {tool_call_str}\n"
        else:
            formatted += f"{role.upper()}: {content}\n"
    return formatted


async def analyze_failed_task(
    task_id: int,
    failed_rollouts: List[art.Trajectory],
    env_task: Dict[str, Any],
    analyzer_model: str = "o3",
) -> Optional[tuple[ErrorAnalysis, str]]:
    """Analyze why a task failed across multiple rollouts. Returns (analysis, prompt) tuple."""

    try:
        # Extract task information
        user_objective = env_task["instruction"]
        correct_tools = create_correct_tools_description(env_task["actions"])

        # Get system message from first rollout
        system_message = ""
        if failed_rollouts and failed_rollouts[0].messages_and_choices:
            messages = keep_only_messages(failed_rollouts[0].messages_and_choices)
            if messages:
                system_message = messages[0].get("content", "")

        # Build analysis prompt
        prompt = ERROR_ANALYSIS_PROMPT + "\n\n"
        prompt += f"**Task ID:** {task_id}\n\n"
        prompt += f"**User Objective:**\n{user_objective}\n\n"
        prompt += f"**Correct Tools:**\n{correct_tools}\n\n"
        prompt += f"**System Message:**\n{system_message}\n\n"
        prompt += f"**Failed Rollouts ({len(failed_rollouts)} total):**\n"

        # Add each failed rollout
        for i, rollout in enumerate(failed_rollouts):
            prompt += f"\n--- ROLLOUT {i + 1} ---\n"
            messages = keep_only_messages(rollout.messages_and_choices)
            # Skip system message for rollout display
            rollout_messages = messages[1:] if len(messages) > 1 else messages
            prompt += format_rollout_messages(rollout_messages)

        # Make LLM call for analysis
        async with AsyncOpenAI() as client:
            response = await client.beta.chat.completions.parse(
                model=analyzer_model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ErrorAnalysis,
            )

            return response.choices[0].message.parsed, prompt

    except Exception as e:
        print(f"Error analyzing task {task_id}: {e}")
        return None


async def log_analysis_to_openpipe(
    task_id: int, analysis: ErrorAnalysis, prompt: str, response: str
) -> None:
    """Log the analysis prompt and response to OpenPipe."""

    try:
        # Create a dummy trajectory for OpenPipe logging
        dummy_trajectory = art.Trajectory(
            messages_and_choices=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            reward=0.0,
            metadata={
                "task_id": task_id,
                "analysis_type": "error_analysis",
                "blame_assignment": analysis.blame_assignment,
                "model": "error_analyzer",
                "phase": "analysis",
                "completion_id": f"analysis-{task_id}-{int(datetime.now().timestamp())}",
            },
            tools=[],
        )

        messages = [
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": response},
        ]

        await log_trajectory_to_openpipe(
            dummy_trajectory, messages, response_str=response
        )
        print(f"Logged analysis for task {task_id} to OpenPipe")

    except Exception as e:
        print(f"Error logging analysis for task {task_id} to OpenPipe: {e}")
