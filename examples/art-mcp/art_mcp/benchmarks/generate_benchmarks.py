import art
from art.local import LocalBackend
from dotenv import load_dotenv
import random
import asyncio
import os
import json
from typing import List
import weave

from art.rewards.ruler import ruler_score_group

from ..rollout import McpScenario, rollout
from servers.python.mcp_alphavantage.server_params import server_params

load_dotenv()

random.seed(42)

weave.init("mcp-agent-training")

# Initialize the server
backend = LocalBackend()

# comparison models
gpt_4o_mini = art.Model(
    name="gpt-4o-mini",
    project="mcp-agent-training",
    inference_model_name="openai/gpt-4o-mini",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
gpt_4o = art.Model(
    name="gpt-4o",
    project="mcp-agent-training",
    inference_model_name="openai/gpt-4o",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
gpt_41 = art.Model(
    name="gpt-4.1",
    project="mcp-agent-training",
    inference_model_name="openai/gpt-4.1",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
o3 = art.Model(
    name="o3",
    project="mcp-agent-training",
    inference_model_name="o3",
    inference_base_url="https://api.openai.com/v1",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
)
o4_mini = art.Model(
    name="o4-mini",
    project="mcp-agent-training",
    inference_model_name="o4-mini",
    inference_base_url="https://api.openai.com/v1",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
)
sonnet_4 = art.Model(
    name="sonnet-4",
    project="mcp-agent-training",
    inference_model_name="anthropic/claude-sonnet-4",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)


async def log_comparison_model(
    comparison_model: art.Model,
    val_scenarios: List[McpScenario],
    control_groups: list[art.TrajectoryGroup] | None = None,
) -> list[art.TrajectoryGroup]:
    groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(comparison_model, val_scenarios[i]) for _ in range(4)
            )
            for i in range(len(val_scenarios))
        ),
        pbar_desc=f"gather {comparison_model.name}",
        max_exceptions=1,
    )

    if control_groups is not None:
        for i in range(len(groups)):
            for j in range(len(groups[i].trajectories)):
                group = art.TrajectoryGroup(
                    [
                        control_groups[i].trajectories[j],
                        groups[i].trajectories[j],
                    ]
                )
                await ruler_score_group(
                    group,
                    judgement_model="openai/o4-mini",
                )
                control_score = group.trajectories[0].metrics["reward"]
                benchmark_score = group.trajectories[1].metrics["reward"]

                if benchmark_score > control_score:
                    groups[i].trajectories[j].metrics["beat_comp"] = 1
                elif benchmark_score < control_score:
                    groups[i].trajectories[j].metrics["beat_comp"] = 0
                else:
                    groups[i].trajectories[j].metrics["beat_comp"] = 0.5

    await comparison_model.log(
        groups,
        split="val",
    )
    await backend._experimental_push_to_s3(
        comparison_model,
    )

    return groups


async def run_benchmarks():
    with open("servers/python/mcp_alphavantage/scenarios/val.jsonl") as f:
        raw_val_scenarios = [json.loads(line.strip()) for line in f if line.strip()]
    val_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=5,
        )
        for scenario in raw_val_scenarios
    ]
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_41.register(backend)
    await o3.register(backend)
    await o4_mini.register(backend)
    await sonnet_4.register(backend)

    control_groups = await log_comparison_model(gpt_4o_mini, val_scenarios)

    for comparison_model in [
        gpt_4o_mini,
        gpt_4o,
        gpt_41,
        # o3,
        # o4_mini,
        # sonnet_4,
    ]:
        await log_comparison_model(comparison_model, val_scenarios, control_groups)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
