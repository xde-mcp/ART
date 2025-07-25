"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio
import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
from dotenv import load_dotenv
import os
import weave
import json

from servers.python.mcp_alphavantage.server_params import server_params
from .rollout import rollout, McpScenario
from .benchmarks.generate_benchmarks import generate_val_groups, calculate_beat_comp

load_dotenv()

# Model configuration
model = art.TrainableModel(
    name="mcp-017",
    project="mcp-agent-training",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

gpt_4o = art.Model(
    name="gpt-4o",
    project="mcp-agent-training",
    inference_model_name="gpt-4o",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
    inference_base_url="https://api.openai.com/v1",
)

if os.getenv("WANDB_API_KEY"):
    print("Initializing Weave")
    weave.init(model.project)


async def train_mcp_agent(use_skypilot: bool = False):
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

    if use_skypilot:
        from art.skypilot.backend import SkyPilotBackend

        backend = await SkyPilotBackend().initialize_cluster(
            cluster_name="mcp-agent-training", gpu="H100-SXM"
        )
    else:
        from art.local.backend import LocalBackend

        backend = LocalBackend()

    await backend._experimental_pull_from_s3(
        model,
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
        groups_per_step=4,  # Batch size of 4
        num_epochs=20,  # Multiple epochs over the dataset
        initial_step=await model.get_step(),  # Resume from checkpoint
    )

    control_groups = await generate_val_groups(gpt_4o, val_scenarios)

    # Main training loop using iterate_dataset
    for batch in train_iterator:
        print("Gathering trajectory groups with RULER scoring...")

        # Use gather_trajectory_groups with ruler_score_group
        # Create groups of 7 trajectories per scenario for better evaluation
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, scenario, False) for _ in range(7))
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
            after_each=lambda group: ruler_score_group(
                group,
                judge_model="openrouter/openai/o4-mini",  # Cost-effective judge model
                debug=True,  # Show judge reasoning
                swallow_exceptions=True,
            ),
        )

        print("train groups finished")

        if batch.step % 5 == 0:
            print("starting comparison val gather")
            val_groups = await generate_val_groups(model, val_scenarios)
            await calculate_beat_comp(val_groups, control_groups)

            await model.log(val_groups, split="val")

        print("starting train")
        await model.train(groups, config=art.TrainConfig(learning_rate=5e-8))

        await backend._experimental_push_to_s3(
            model,
        )


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
