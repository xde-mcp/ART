# Usage: uv run run_training.py --models=003
import art
from art_e.project_types import PolicyConfig, TrainingConfig
import argparse
import sky
import textwrap
from dotenv import load_dotenv, dotenv_values
from sky import ClusterStatus
import concurrent.futures
import json
import traceback

load_dotenv()

models = {}

models["002"] = art.TrainableModel(
    name="email-agent-002",
    project="email_agent",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    config=PolicyConfig(
        max_turns=10,
        log_to_langfuse=True,
        training_config=TrainingConfig(
            trajectories_per_group=6,
            groups_per_step=8,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=4000,
            num_epochs=1,
        ),
    ),
)

models["004"] = models["002"].model_copy(deep=True)
assert isinstance(models["004"].config, PolicyConfig)
models["004"].name = "email-agent-004"
models["004"].config.max_turns = 30

models["005"] = models["002"].model_copy(deep=True)
assert isinstance(models["005"].config, PolicyConfig)
models["005"].name = "email-agent-005"

models["006"] = models["005"].model_copy(deep=True)
models["006"].name = "email-agent-006"

models["007"] = models["005"].model_copy(deep=True)
models["007"].name = "email-agent-007"
assert isinstance(models["007"].config, PolicyConfig)
models["007"].config.use_tools = True

models["008"] = models["005"].model_copy(deep=True)
models["008"].name = "email-agent-008"
assert isinstance(models["008"].config, PolicyConfig)
assert models["008"].config.training_config is not None
models["008"].config.use_tools = True
models["008"].config.training_config.trajectories_per_group = 4
models["008"].config.training_config.groups_per_step = 12
models["008"].config.training_config.num_epochs = 3

models["011"] = models["008"].model_copy(deep=True)
models["011"].name = "email-agent-011"
assert isinstance(models["011"].config, PolicyConfig)
assert models["011"].config.training_config is not None
models["011"].config.training_config.num_epochs = 4

models["012"] = models["008"].model_copy(deep=True)
models["012"].name = "email-agent-012"

models["013"] = models["002"].model_copy(deep=True)
models["013"].name = "email-agent-013"
assert isinstance(models["013"].config, PolicyConfig)
assert models["013"].config.training_config is not None
models["013"].config.training_config.num_epochs = 4
models["013"].config.training_config.trajectories_per_group = 4
models["013"].config.training_config.groups_per_step = 24

models["014"] = models["008"].model_copy(deep=True)
models["014"].name = "email-agent-014"
assert isinstance(models["014"].config, PolicyConfig)
models["014"].config.stupid_simple_reward_fn = True

models["015"] = models["008"].model_copy(deep=True)
models["015"].name = "email-agent-015"

models["016"] = models["008"].model_copy(deep=True)
models["016"].name = "email-agent-016"

models["017"] = models["008"].model_copy(deep=True)
models["017"].name = "email-agent-017"
assert isinstance(models["017"].config, PolicyConfig)
assert models["017"].config.training_config is not None
models["017"].config.training_config.art_location = "0.3.6"

models["018"] = models["008"].model_copy(deep=True)
models["018"].name = "email-agent-018"
assert isinstance(models["018"].config, PolicyConfig)
assert models["018"].config.training_config is not None
models["018"].config.training_config.art_location = "0.3.6"

models["019"] = models["008"].model_copy(deep=True)
models["019"].name = "email-agent-019"
assert isinstance(models["019"].config, PolicyConfig)
assert models["019"].config.training_config is not None
models["019"].config.training_config.rollout_concurrency = 100
models["019"].config.max_groups_in_flight = 48
models["019"].config.training_config.art_location = "0.3.6"

models["020"] = models["008"].model_copy(deep=True)
models["020"].name = "email-agent-020"
assert isinstance(models["020"].config, PolicyConfig)
assert models["020"].config.training_config is not None
models[
    "020"
].config.training_config.art_location = (
    "git+https://github.com/OpenPipe/ART.git@potential_fix"
)

parser = argparse.ArgumentParser(
    description="Train one or more art-e models (comma separated)."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. 003).",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Whether to use fast launch (skip setup).",
)
args = parser.parse_args()

# Parse and validate the requested model keys
requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
unknown = [m for m in requested_models if m not in models]
if unknown:
    raise ValueError(
        f"Unknown model keys requested: {', '.join(unknown)}. Valid keys: {', '.join(models.keys())}"
    )


def launch_model(model_key: str):
    model = models[model_key]
    print(f"Launching {model_key} ({model.name}) on SkyPilot…")

    if (
        not model.config
        or not isinstance(model.config, PolicyConfig)
        or not model.config.training_config
    ):
        raise ValueError(
            f"Training config not found or is invalid for model {model_key}"
        )

    # Assert type for linter
    assert isinstance(model.config, PolicyConfig)
    assert model.config.training_config is not None

    training_config = model.config.training_config

    setup_script = textwrap.dedent(
        f"""
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            uv remove openpipe-art
            uv add {model.config.training_config.art_location}
        """
    )

    model_dict = model.model_dump()
    model_dict["config"] = model.config.model_dump()
    # Pass the model key as run_name and the serialized model
    run_script = f"uv run art_e/train.py '{json.dumps(model_dict)}'"

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"art-e-train-{model_key}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )

    task.set_resources(sky.Resources(accelerators=training_config.gpus))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    cluster_name = f"kyle-art-e-{model_key}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs…")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch the task; stream_and_get blocks until the task starts running, but
    # running this in its own thread means all models run in parallel.
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
            fast=args.fast,
        )
    )

    print(f"Job submitted for {model_key} (ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} for {model_key} finished with exit code {exit_code}.")


# Launch all requested models in parallel threads
with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(requested_models)
) as executor:
    futures = [executor.submit(launch_model, key) for key in requested_models]
    for future in concurrent.futures.as_completed(futures):
        # Propagate any exceptions raised inside threads
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            print(f"Traceback: {traceback.format_exc()}")
