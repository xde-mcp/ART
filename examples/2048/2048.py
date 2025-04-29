import asyncio
import art
from art.skypilot import SkyPilotAPI
from dotenv import load_dotenv
from openpipe.client import AsyncOpenPipe
import random
import openai
import time
import math
import requests

from utils import (
    apply_agent_move,
    check_game_finished,
    generate_game,
    max_cell_value,
    total_board_value,
    render_board,
    WINNING_VALUE,
)

load_dotenv()

random.seed(42)

# Declare the model
model = art.TrainableModel(
    name="001",
    project="2048",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)
# To run on a T4, we need to override some config defaults.
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=8192,
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        num_scheduler_steps=1,
    ),
)


# Optional logging client
op_client = AsyncOpenPipe()


@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(model: art.Model, step: int, is_validation: bool) -> art.Trajectory:
    game = generate_game()

    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent 2048 player. Always choose the move most likely to lead to combine cells to eventually reach the number 2048. Optional moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>",
            }
        ],
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        requested_at = int(time.time() * 1000)
        messages = trajectory.messages()

        async def get_completion():
            client = model.openai_client()
            return await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.name,
            )

        try:
            chat_completion = await get_completion()
            last_completion = chat_completion
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        try:
            if op_client.api_key:
                await op_client.report(
                    requested_at=requested_at,
                    received_at=int(time.time() * 1000),
                    req_payload={
                        "model": model.name,
                        "messages": messages,
                        "metadata": {
                            "game_id": game["id"],
                            "notebook-id": "2048",
                            "step": str(step),
                            "validation": str(is_validation),
                            "move_number": str(move_number),
                        },
                    },
                    resp_payload=chat_completion,
                    status_code=200,
                )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            trajectory.reward = -1
            break

        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics["max_value"] = max_value
            trajectory.metrics["board_value"] = board_value

            if max_value < WINNING_VALUE:
                # scale max value logarithmically between 0 for 2 and 1 for WINNING_VALUE
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                # scale board value logarithmically between 0 for 2 * 16 and 1 for WINNING_VALUE * 16
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                # combine the two rewards, with max value having a higher weight
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
            else:
                # double reward if the agent wins
                trajectory.reward = 2
            break

    try:
        if op_client.api_key:
            await op_client.update_log_metadata(
                filters=[
                    {
                        "field": "completionId",
                        "equals": last_completion.id,
                    }
                ],
                metadata={
                    "reward": str(trajectory.reward),
                    "reward_assigned": "true",
                },
            )
    except Exception as e:
        print(f"Error updating log metadata: {e}")

    return trajectory


async def main():
    # Initialize the server
    api = await SkyPilotAPI.initialize_cluster(
        cluster_name="art", gpu="H100", art_version=".", env_path=".env"
    )

    # Register the model with the local API (sets up logging, inference, and training)
    await model.register(api)

    for i in range(await model.get_step(), 100):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(48)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )
        await model.delete_checkpoints()
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=3e-5),
            # Lowering the logprob_calculation_chunk_size is a memory saving measure
            # to allow longer sequences (up to 4096 tokens) to be processed on a T4.
            _config={"logprob_calculation_chunk_size": 8},
        )


if __name__ == "__main__":
    asyncio.run(main())
