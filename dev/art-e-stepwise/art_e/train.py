import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List
from rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art.utils import iterate_dataset
from art_e.project_types import ProjectPolicyConfig
from art_e.evaluate.benchmark import benchmark_model
import os
from art_e.create_stepwise_groups import create_stepwise_groups

load_dotenv()


async def train(model: art.TrainableModel[ProjectPolicyConfig]):
    generate_database()

    with LocalBackend() as backend:
        print(
            f"Pulling latest checkpoint from S3 bucket: `{os.environ['BACKUP_BUCKET']}`"
        )
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
            verbose=True,
            only_step="latest",  # Only pull the latest checkpoint
            exclude=["trajectories"],  # Exclude trajectories to save space/time
        )

        # Handle fork configuration if specified
        if model.config.fork_from_model:
            print(f"Forking from model: {model.config.fork_from_model}")
            await backend._experimental_fork_checkpoint(
                model,
                from_model=model.config.fork_from_model,
                from_s3_bucket=os.environ["BACKUP_BUCKET"],
                not_after_step=model.config.fork_not_after_step,
                verbose=True,
            )

        await model.register(backend)

        print("Loading training data...")
        # Load the training data with deterministic shuffling if a seed is provided.
        train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="train",
            limit=model.config.training_dataset_size,
        )
        print("Loading validation data...")
        val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="test", limit=model.config.val_set_size
        )

        print(f"Training data size: {len(train_scenarios)}")
        print(f"Validation data size: {len(val_scenarios)}")

        train_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=model.config.groups_per_step,
            num_epochs=model.config.num_epochs,
            initial_step=await model.get_step(),
        )

        for batch in train_iterator:
            if batch.step % model.config.eval_steps == 0:
                print(f"\n--- Evaluating at Iteration {batch.step} ---")
                await benchmark_model(model, logging_split="train")
                await model.delete_checkpoints()
                await backend._experimental_push_to_s3(
                    model,
                    s3_bucket=os.environ["BACKUP_BUCKET"],
                )

            original_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            rollout(model, scenario)
                            for _ in range(model.config.trajectories_per_group)
                        )
                    )
                    for scenario in batch.items
                ),
            )

            training_groups = create_stepwise_groups(original_groups)
            await model.log(original_groups, split="train_full")

            await model.train(
                training_groups,
                config=art.TrainConfig(learning_rate=model.config.learning_rate),
                _config=art.dev.TrainConfig(
                    scale_rewards=model.config.scale_rewards,
                ),
            )

        await benchmark_model(model, logging_split="train")
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
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
    model_dict["config"] = ProjectPolicyConfig(**model_dict["config"])

    model: art.TrainableModel[ProjectPolicyConfig] = art.TrainableModel(
        **model_dict,
    )
    asyncio.run(train(model))
