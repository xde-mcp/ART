import art
import asyncio
from dotenv import load_dotenv
import os
from typing import List
from rollout import rollout
from dataset import get_jokes, RedditJoke
from project_types import PolicyConfig
from art.local import LocalBackend
from benchmark import benchmark_model
import argparse
import json
from tqdm.auto import tqdm  # Progress bar

load_dotenv()


async def run_group_rollouts(
    model: art.TrainableModel,
    scenario: RedditJoke,
    buf: asyncio.Queue,
    progress_bar: tqdm,
):
    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None
    group = await art.TrajectoryGroup.from_rollout(
        lambda: rollout(model, scenario),
        model.config.training_config.trajectories_per_group,
        max_exceptions=10,
    )
    progress_bar.update(1)
    await buf.put(group)


async def producer(
    model: art.TrainableModel,
    dataset: list[RedditJoke],
    sem_backlog: asyncio.Semaphore,
    buf: asyncio.Queue,
):
    """Iterate over dataset for cfg.epochs and push completed groups into buf."""

    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    total_groups = len(dataset) * model.config.training_config.num_epochs

    current_step = await model.get_step()
    groups_to_skip = current_step * model.config.training_config.groups_per_step

    total_groups_seen = 0

    with tqdm(
        total=total_groups,
        initial=min(groups_to_skip, total_groups),
        desc="Training progress",
        unit="group",
    ) as progress_bar:
        # Iterate over the data for the configured number of epochs
        for _ in range(model.config.training_config.num_epochs):
            for scenario in dataset:
                # Skip the first `groups_to_skip` groups
                if total_groups_seen < groups_to_skip:
                    total_groups_seen += 1
                    continue

                await sem_backlog.acquire()
                asyncio.create_task(
                    run_group_rollouts(model, scenario, buf, progress_bar)
                )
                total_groups_seen += 1

    # signal completion
    await buf.put(None)


async def trainer(
    model: art.TrainableModel,
    buf: asyncio.Queue,
    sem_backlog: asyncio.Semaphore,
    lock: asyncio.Lock,
    backup_bucket: str | None,
    backend: LocalBackend,
):
    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    while True:
        groups = []
        while len(groups) < model.config.training_config.groups_per_step:
            item = await buf.get()
            if item is None:
                break
            groups.append(item)

        # Ensure only one train at a time
        async with lock:
            await model.train(
                groups,
                config=art.TrainConfig(
                    learning_rate=model.config.training_config.learning_rate
                ),
            )
            step = await model.get_step()
            if step % model.config.training_config.eval_steps == 0:
                await benchmark_model(model)
                await model.delete_checkpoints()
                if backup_bucket:
                    await backend._experimental_push_to_s3(
                        model, prefix="roflbot", s3_bucket=backup_bucket
                    )

        for _ in groups:
            sem_backlog.release()


async def run_training(model: art.TrainableModel):
    assert isinstance(model.config, PolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set for the model")

    training_config = model.config.training_config
    backend = LocalBackend()

    backup_bucket = os.environ.get("REMOTE_BUCKET")
    if backup_bucket:
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=backup_bucket,
            prefix="roflbot",
            verbose=True,
        )
    await model.register(
        backend,
        # _openai_client_config={
        #     "engine_args": {"speculative_config": '{"method": "suffix"}'}
        # },
    )

    print("Loading training data...")
    train_scenarios: List[RedditJoke] = get_jokes(
        split="train", limit=training_config.training_dataset_size
    )
    print("Loading validation data...")
    # Using 'test' split for validation as per typical dataset splits
    val_scenarios: List[RedditJoke] = get_jokes(
        split="test", limit=training_config.val_set_size
    )

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    initial_step = await model.get_step()
    print(f"Starting training from step: {initial_step}")

    buf = asyncio.Queue(maxsize=training_config.max_untrained_groups)
    sem_backlog = asyncio.Semaphore(training_config.max_untrained_groups)
    lock = asyncio.Lock()

    await asyncio.gather(
        producer(model, train_scenarios, sem_backlog, buf),
        trainer(model, buf, sem_backlog, lock, backup_bucket, backend),
    )

    await benchmark_model(model)
    if backup_bucket:
        await backend._experimental_push_to_s3(
            model, prefix="roflbot", s3_bucket=backup_bucket
        )
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", help="JSON string serialization of the Model")
    args = parser.parse_args()

    print("Model JSON: ", args.model_json)

    model_dict = json.loads(args.model_json)
    model = art.TrainableModel(**model_dict)
    model.config = PolicyConfig(**model_dict["config"])
    asyncio.run(run_training(model))
