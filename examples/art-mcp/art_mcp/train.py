"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio
import art
from art.rewards import ruler_score_group
from dotenv import load_dotenv
import os
import weave

from art.skypilot.backend import SkyPilotBackend
from mcp_server import AlphaMcpServer
from rollout import rollout, McpScenario

load_dotenv()


# Model configuration
model = art.TrainableModel(
    name="mcp-002",
    project="mcp-agent-training",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

if os.getenv("WANDB_API_KEY"):
    print("Initializing Weave")
    weave.init(model.project)


async def train_mcp_agent():
    """Example training function that creates AlphaMcpServer and passes it in scenarios."""
    load_dotenv()

    task_descriptions = [
        "Get the current stock price for AAPL",
        "Search for stocks related to technology companies",
        "Get company overview for Microsoft (MSFT)",
        "Find the latest earnings report for Tesla (TSLA)",
        "Get technical analysis indicators for Amazon (AMZN)",
    ]

    # Create AlphaVantage MCP server
    mcp_server = AlphaMcpServer(api_key=os.getenv("ALPHAVANTAGE_API_KEY", "demo"))

    # Start the server
    await mcp_server.start()

    try:
        # Define training scenarios with the server
        train_scenarios = [
            McpScenario(
                task_description=task_description,
                mcp_server=mcp_server,
                max_turns=3,
            )
            for task_description in task_descriptions[:4]
        ]

        backend = await SkyPilotBackend().initialize_cluster(
            cluster_name="mcp-agent-training", gpu="H100"
        )
        await model.register(backend)

        print("Gathering trajectory groups with RULER scoring...")

        print(f"Num train scenarios: {len(train_scenarios)}")

        # Use gather_trajectory_groups with ruler_score_group
        # Create groups of 4 trajectories per scenario for better evaluation
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, False)
                    for _ in range(4)  # 4 trajectories per scenario
                )
                for scenario in train_scenarios
            ),
            pbar_desc="Training MCP Agent",
            after_each=lambda group: ruler_score_group(
                group,
                judge_model="openai/gpt-4o-mini",  # Cost-effective judge model
                debug=True,  # Show judge reasoning
            ),
        )

        await model.train(groups)

    finally:
        # Clean up the server session
        await mcp_server.stop()


def main():
    """Main training entry point."""
    print("Starting MCP agent training...")
    asyncio.run(train_mcp_agent())


if __name__ == "__main__":
    main()
