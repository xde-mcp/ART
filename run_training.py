import argparse
import sky
import json
import textwrap
import concurrent.futures
import traceback
from dotenv import dotenv_values
from sky import ClusterStatus

# Usage:
# uv run run_training.py 001 --fast
# uv run run_training.py 001,002,003 --fast

# Define model configurations for tau-bench RL experiments
models = {
    "001": {
        "model_name": "tau-bench-rl-001",
        "base_model": "Qwen/Qwen2.5-14B-Instruct", 
        "env": "retail",
        "model": "gpt-4o-mini",
        "model_provider": "openai",
        "user_model": "gpt-4o",
        "user_model_provider": "openai",
        "agent_strategy": "tool-calling",
        "temperature": 0.0,
        "task_split": "train",
        "start_index": 0,
        "end_index": 500,
        "trajectories_per_group": 6,
        "groups_per_step": 8,
        "learning_rate": 1.2e-5,
        "eval_steps": 30,
        "val_set_size": 100,
        "training_dataset_size": 500,
        "num_epochs": 1,
    }
}

# Retail environment variants
models["002"] = models["001"].copy()
models["002"]["model_name"] = "tau-bench-rl-002"
models["002"]["trajectories_per_group"] = 4
models["002"]["groups_per_step"] = 12
models["002"]["num_epochs"] = 2

models["003"] = models["001"].copy()
models["003"]["model_name"] = "tau-bench-rl-003"
models["003"]["learning_rate"] = 2.0e-5
models["003"]["training_dataset_size"] = 1000
models["003"]["end_index"] = 1000

# Airline environment variants
models["004"] = models["001"].copy()
models["004"]["model_name"] = "tau-bench-rl-004"
models["004"]["env"] = "airline"

models["005"] = models["004"].copy()
models["005"]["model_name"] = "tau-bench-rl-005"
models["005"]["trajectories_per_group"] = 4
models["005"]["groups_per_step"] = 12
models["005"]["num_epochs"] = 2

# ReAct strategy variants
models["006"] = models["001"].copy()
models["006"]["model_name"] = "tau-bench-rl-006"
models["006"]["agent_strategy"] = "react"

models["007"] = models["004"].copy()
models["007"]["model_name"] = "tau-bench-rl-007"
models["007"]["agent_strategy"] = "react"

# Few-shot strategy variants
models["008"] = models["001"].copy()
models["008"]["model_name"] = "tau-bench-rl-008"
models["008"]["agent_strategy"] = "few-shot"

models["009"] = models["004"].copy()
models["009"]["model_name"] = "tau-bench-rl-009"
models["009"]["agent_strategy"] = "few-shot"

# Higher temperature experiments
models["010"] = models["001"].copy()
models["010"]["model_name"] = "tau-bench-rl-010"
models["010"]["temperature"] = 0.2

parser = argparse.ArgumentParser(
    description="Train one or more tau-bench RL models (comma separated)."
)
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Comma-separated list of model keys to train (e.g. 001,002,003).",
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
    model_config = models[model_key]
    print(f"Launching {model_key} ({model_config['model_name']}) on SkyPilot…")

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt update && apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            # Install project in editable mode
            uv remove openpipe-art
            uv add --editable ~/workspace

            # Sync dependencies
            uv sync
            
            # Set up tau-bench specific dependencies if needed
            cd ~/workspace/dev/tau-bench
        """
    )

    # Construct the run_rl.py command with all necessary arguments
    run_args = [
        f"--model-name {model_config['model_name']}",
        f"--base-model {model_config['base_model']}",
        f"--env {model_config['env']}",
        f"--model {model_config['model']}",
        f"--model-provider {model_config['model_provider']}",
        f"--user-model {model_config['user_model']}",
        f"--user-model-provider {model_config['user_model_provider']}",
        f"--agent-strategy {model_config['agent_strategy']}",
        f"--temperature {model_config['temperature']}",
        f"--task-split {model_config['task_split']}",
        f"--start-index {model_config['start_index']}",
        f"--end-index {model_config['end_index']}",
        f"--trajectories-per-group {model_config['trajectories_per_group']}",
        f"--groups-per-step {model_config['groups_per_step']}",
        f"--learning-rate {model_config['learning_rate']}",
        f"--eval-steps {model_config['eval_steps']}",
        f"--val-set-size {model_config['val_set_size']}",
        f"--training-dataset-size {model_config['training_dataset_size']}",
        f"--num-epochs {model_config['num_epochs']}",
    ]

    run_script = textwrap.dedent(f"""
        echo 'Starting tau-bench RL training...'
        cd ~/workspace/dev/tau-bench
        
        # Ensure project is installed in editable mode
        uv remove openpipe-art
        uv add --editable ~/workspace
        
        # Run the RL training
        uv run run_rl.py {' '.join(run_args)}
    """)

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"tau-bench-rl-{model_key}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),
    )
    task.set_resources(sky.Resources(accelerators="H100-SXM:1"))
    task.set_file_mounts({"~/workspace": "."})

    # Generate cluster name
    cluster_name = f"tau-bench-rl-{model_key}"
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