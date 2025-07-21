"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio
import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
from dotenv import load_dotenv
import os
import weave
from art.skypilot.backend import SkyPilotBackend
import json
import litellm

from servers.python.mcp_alphavantage.server_params import server_params
from .rollout import rollout, McpScenario

load_dotenv()
# litellm._turn_on_debug()

# Model configuration
model = art.TrainableModel(
    name="mcp-006",
    project="mcp-agent-training",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

gpt_4o_mini = art.Model(
    name="gpt-4o-mini",
    project="mcp-agent-training",
    inference_model_name="gpt-4o-mini",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
    inference_base_url="https://api.openai.com/v1",
)

if os.getenv("WANDB_API_KEY"):
    print("Initializing Weave")
    weave.init(model.project)


async def train_mcp_agent():
    """Example training function that creates AlphaMcpServer and passes it in scenarios."""
    load_dotenv()

    # Load pre-split scenarios from scenarios directory
    scenarios_dir = "servers/python/mcp_alphavantage/scenarios"

    with open(f"{scenarios_dir}/train.jsonl") as f:
        raw_train_scenarios = [json.loads(line.strip()) for line in f if line.strip()]

    with open(f"{scenarios_dir}/val.jsonl") as f:
        raw_val_scenarios = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(raw_train_scenarios)} training scenarios")
    print(f"Loaded {len(raw_val_scenarios)} validation scenarios")

    backend = await SkyPilotBackend().initialize_cluster(
        cluster_name="mcp-agent-training", gpu="H100"
    )
    await model.register(backend)

    train_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=5,
        )
        for scenario in raw_train_scenarios
    ]

    # Create validation scenarios from pre-split data (used for periodic evaluation)
    val_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=5,
        )
        for scenario in raw_val_scenarios
    ]

    # Create dataset iterator using raw scenarios (not McpScenario objects)
    train_iterator = iterate_dataset(
        train_scenarios,  # Use raw data, create McpScenario objects in batches
        groups_per_step=2,  # Batch size of 2
        num_epochs=3,  # Multiple epochs over the dataset
        initial_step=await model.get_step(),  # Resume from checkpoint
    )

    # Main training loop using iterate_dataset
    for batch in train_iterator:
        print("Gathering trajectory groups with RULER scoring...")

        # Use gather_trajectory_groups with ruler_score_group
        # Create groups of 4 trajectories per scenario for better evaluation
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, scenario, False) for _ in range(4))
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
            after_each=lambda group: ruler_score_group(
                group,
                judge_model="openrouter/google/gemini-2.5-flash",  # Cost-effective judge model
                debug=True,  # Show judge reasoning
                swallow_exceptions=True,
            ),
        )

        print("train groups finished")

        if batch.step % 5 == 0:
            print("starting comparison train gather")
            comparison_train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        [
                            rollout(model, scenario, False)
                            if i % 2 == 0
                            else rollout(gpt_4o_mini, scenario, False),
                            rollout(gpt_4o_mini, scenario, False)
                            if i % 2 == 0
                            else rollout(model, scenario, False),
                        ]
                    )
                    for i, scenario in enumerate(train_scenarios)
                ),
                pbar_desc=f"comparison train gather step {batch.step}",
                after_each=lambda group: ruler_score_group(
                    group,
                    judge_model="openrouter/google/gemini-2.5-flash",
                    debug=True,
                    swallow_exceptions=True,
                ),
            )
            for i in range(len(comparison_train_groups)):
                group = comparison_train_groups[i]
                # reverse every other group
                if i % 2 == 1:
                    group.trajectories = group.trajectories[::-1]
                group.trajectories[0].metrics["beat_comp"] = (
                    group.trajectories[0].reward > group.trajectories[1].reward
                )

            await model.log(comparison_train_groups, split="train")

            print("starting comparison val gather")
            comparison_val_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        [
                            rollout(model, scenario, False)
                            if i % 2 == 0
                            else rollout(gpt_4o_mini, scenario, False),
                            rollout(gpt_4o_mini, scenario, False)
                            if i % 2 == 0
                            else rollout(model, scenario, False),
                        ]
                    )
                    for i, scenario in enumerate(val_scenarios)
                ),
                pbar_desc=f"val gather step {batch.step}",
                after_each=lambda group: ruler_score_group(
                    group,
                    judge_model="openrouter/google/gemini-2.5-flash",
                    debug=True,
                    swallow_exceptions=True,
                ),
            )
            for i in range(len(comparison_val_groups)):
                group = comparison_val_groups[i]
                # reverse every other group
                if i % 2 == 1:
                    group.trajectories = group.trajectories[::-1]
                group.trajectories[0].metrics["beat_comp"] = (
                    group.trajectories[0].reward > group.trajectories[1].reward
                )
            await model.log(comparison_val_groups, split="val")

        print("starting train")
        await model.train(groups)


def main():
    """Main training entry point."""
    print("Starting MCP agent training...")
    try:
        asyncio.run(train_mcp_agent())
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise the exception to ensure it reaches the user


if __name__ == "__main__":
    main()
