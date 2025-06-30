import argparse
import sky
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
        "base_model": "Qwen/Qwen2.5-14B-Instruct", 
        "env": "retail",
        "model": "tau-bench-rl-001-final",
        "model_provider": "hosted_vllm",
        "user_model": "gpt-4o",
        "user_model_provider": "openai",
        "agent_strategy": "tool-calling-rl",
        "temperature": 1.0,
        "task_split": "test",
        "start_index": 0,
        "end_index": -1,
        "trajectories_per_group": 6,
        "groups_per_step": 10,
        "learning_rate": 1.2e-5,
        "eval_steps": 10,
        "val_set_size": 85,
        "training_dataset_size": 30,
        "num_epochs": 50,
        "reward_type": "real",
        "max_num_steps": 30,
        "train_mode": "sync_rl",
        "skip_eval": False,
    }
}

# # Retail environment variants
models["002"] = models["001"].copy()
models["002"]["model"] = "tau-bench-rl-002-final"
models["002"]["reward_type"] = "general_rm"

models["003"] = models["001"].copy()
models["003"]["model"] = "tau-bench-rl-003-final"
models["003"]["reward_type"] = "general_rm"
models["003"]["learning_rate"] = 5e-6

models["004"] = models["001"].copy()
models["004"]["model"] = "tau-bench-rl-004-final"
models["004"]["reward_type"] = "general_rm"
models["004"]["learning_rate"] = 5e-6
models["004"]["max_num_steps"] = 14

models["005"] = models["003"].copy()
models["005"]["model"] = "tau-bench-rl-005-final-4"
models["005"]["train_mode"] = "async_rl"
models["005"]["val_set_size"] = 30

models["006"] = models["001"].copy()
models["006"]["model"] = "tau-bench-rl-006-final-4"
models["006"]["train_mode"] = "async_rl"
models["006"]["learning_rate"] = 5e-6
models["006"]["val_set_size"] = 30

models["007"] = models["005"].copy()
models["007"]["model"] = "tau-bench-rl-007"
models["007"]["base_model"] = "Qwen/Qwen2.5-32B-Instruct"

models["008"] = models["001"].copy()
models["008"]["model"] = "tau-bench-rl-debug-1"
models["008"]["base_model"] = "Qwen/Qwen2.5-7B-Instruct"
models["008"]["learning_rate"] = 5e-6
models["008"]["trajectories_per_group"] = 3
models["008"]["groups_per_step"] = 3
models["008"]["val_set_size"] = 4
models["008"]["training_dataset_size"] = 6
models["008"]["train_mode"] = "async_rl"
models["008"]["reward_type"] = "general_rm"
models["008"]["skip_eval"] = True

models["009"] = models["003"].copy()
models["009"]["model"] = "tau-bench-rl-009-2"
models["009"]["train_mode"] = "async_rl"
models["009"]["val_set_size"] = 30
models["009"]["trajectories_per_group"] = 10

models["010"] = models["001"].copy()
models["010"]["model"] = "tau-bench-rl-010-2"
models["010"]["train_mode"] = "async_rl"
models["010"]["learning_rate"] = 5e-6
models["010"]["val_set_size"] = 30
models["010"]["trajectories_per_group"] = 10

models["011"] = models["005"].copy()
models["011"]["model"] = "tau-bench-rl-011"
models["011"]["base_model"] = "Qwen/Qwen2.5-32B-Instruct"
models["011"]["trajectories_per_group"] = 10

models["012"] = models["001"].copy()
models["012"]["model"] = "tau-bench-rl-012"
models["012"]["learning_rate"] = 5e-6
models["012"]["val_set_size"] = 30
models["012"]["trajectories_per_group"] = 10
models["012"]["reward_type"] = "general_rm"

# same as 012 but this one has slightly more graded rewards since max token trajs get -1 reward
models["013"] = models["001"].copy()
models["013"]["model"] = "tau-bench-rl-013"
models["013"]["learning_rate"] = 5e-6
models["013"]["val_set_size"] = 30
models["013"]["trajectories_per_group"] = 10
models["013"]["reward_type"] = "general_rm"

# models["013"] = models["001"].copy()
# models["013"]["model"] = "tau-bench-rl-013"
# models["013"]["base_model"] = "Qwen/Qwen2.5-32B-Instruct"
# models["013"]["learning_rate"] = 5e-6
# models["013"]["val_set_size"] = 30
# models["013"]["trajectories_per_group"] = 10
# models["013"]["reward_type"] = "general_rm"

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
    print(f"Launching {model_key} ({model_config['model']}) on SkyPilot…")

    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt update && apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            # Install project in editable mode
            uv remove openpipe-art
            uv add --editable ~/ART

            # Sync dependencies
            uv sync
        """
    )

    # Construct the run_rl.py command with all necessary arguments
    run_args = [
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
        f"--reward-type {model_config['reward_type']}",
        f"--max-num-steps {model_config['max_num_steps']}",
        f"--train-mode {model_config['train_mode']}",
        f"{'--skip-eval' if model_config['skip_eval'] else ''}",
    ]

    run_script = textwrap.dedent(f"""
        # Run the RL training
        uv add "accelerate==1.7.0"
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
    task.set_file_mounts({"~/ART": "../.."})

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