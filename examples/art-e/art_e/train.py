import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List
from rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art_e.project_types import PolicyConfig
from art_e.evaluate.benchmark import benchmark_model
from tqdm import tqdm
import os

load_dotenv()


async def run_group(
    model: art.TrainableModel,
    scenario: SyntheticQuery,
    finished_groups: asyncio.Queue[art.TrajectoryGroup | None],
    sem_rollout_concurrency: asyncio.Semaphore,
    progress_bar: tqdm,
):
    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    async def rollout_fn():
        async with sem_rollout_concurrency:
            return await rollout(model, scenario)

    trajectories = await asyncio.gather(
        *[
            rollout_fn()
            for _ in range(model.config.training_config.trajectories_per_group)
        ]
    )
    progress_bar.update(1)
    await finished_groups.put(art.TrajectoryGroup(trajectories))


async def producer(
    model: art.TrainableModel,
    dataset: list[SyntheticQuery],
    finished_groups: asyncio.Queue[art.TrajectoryGroup | None],
    sem_groups_in_flight: asyncio.Semaphore,
):
    """Iterate over dataset for cfg.epochs and push completed groups into queue."""

    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    total_groups = len(dataset) * model.config.training_config.num_epochs

    current_step = await model.get_step()
    groups_to_skip = current_step * model.config.training_config.groups_per_step

    total_groups_seen = 0

    sem_rollout_concurrency = asyncio.Semaphore(
        model.config.training_config.rollout_concurrency
    )

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

                await sem_groups_in_flight.acquire()
                asyncio.create_task(
                    run_group(
                        model,
                        scenario,
                        finished_groups,
                        sem_rollout_concurrency,
                        progress_bar,
                    )
                )
                total_groups_seen += 1

    # signal completion
    await finished_groups.put(None)


async def trainer(
    model: art.TrainableModel,
    finished_groups: asyncio.Queue[art.TrajectoryGroup | None],
    sem_groups_in_flight: asyncio.Semaphore,
    backup_bucket: str | None,
    backend: LocalBackend,
):
    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    while True:
        groups = []
        while len(groups) < model.config.training_config.groups_per_step:
            item = await finished_groups.get()
            if item is None:
                break
            groups.append(item)

        await model.train(
            groups,
            config=art.TrainConfig(
                learning_rate=model.config.training_config.learning_rate
            ),
        )
        for _ in groups:
            sem_groups_in_flight.release()

        step = await model.get_step()

        if step % model.config.training_config.eval_steps == 0:
            await benchmark_model(model)
            await model.delete_checkpoints()
            if backup_bucket:
                await backend._experimental_push_to_s3(model, s3_bucket=backup_bucket)


async def run_training(model: art.TrainableModel):
    generate_database()

    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    backend = LocalBackend()
    backup_bucket = os.environ["BACKUP_BUCKET"]
    print(f"Pulling from S3 bucket: `{backup_bucket}`")

    await backend._experimental_pull_from_s3(
        model,
        s3_bucket=backup_bucket,
        verbose=True,
    )
    await model.register(backend)

    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=model.config.training_config.training_dataset_size
    )
    print("Loading validation data...")
    val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="test", limit=model.config.training_config.val_set_size
    )

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    if model.config.max_groups_in_flight < model.config.training_config.groups_per_step:
        raise ValueError(
            f"max_groups_in_flight ({model.config.max_groups_in_flight}) must be greater than or equal to groups_per_step ({model.config.training_config.groups_per_step}) or else a training batch can never be formed. I recommend setting it to at least 2x groups_per_step to maximize throughput. Setting `max_groups_in_flight=groups_per_step` will ensure your model is always training with on-policy data, while setting it to a larger number will increase throughput at the cost of using off-policy data. In general this tradeoff seems to be worth it."
        )
    sem_groups_in_flight = asyncio.Semaphore(model.config.max_groups_in_flight)
    finished_groups: asyncio.Queue[art.TrajectoryGroup | None] = asyncio.Queue()

    await asyncio.gather(
        producer(model, train_scenarios, finished_groups, sem_groups_in_flight),
        trainer(model, finished_groups, sem_groups_in_flight, backup_bucket, backend),
    )

    await benchmark_model(model)
    if backup_bucket:
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=backup_bucket,
        )
    print("Training finished.")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", help="JSON string serialization of the Model")
    args = parser.parse_args()

    print("Model JSON: ", args.model_json)

    model_dict = json.loads(args.model_json)
    model = art.TrainableModel(**model_dict)
    model.config = PolicyConfig(**model_dict["config"])
    asyncio.run(run_training(model))
