# Usage: uv run debug_art.py --configs=0.7.3
import argparse
import sky
import textwrap
from dotenv import load_dotenv, dotenv_values
from sky import ClusterStatus
import concurrent.futures
import traceback

load_dotenv()

configs = {}

configs["0.8.0-tool-calls"] = {
    "art_location": "git+https://github.com/OpenPipe/ART.git@vllm_0_8_0_new_unsloth",
    "run_script": "source .venv/bin/activate && python logprob_check_tool_calls.py",
    "vllm_commit": "v0.8.0",
}

configs["ef640440791a461a181e6d497965701462c166b3-tool-calls"] = {
    "art_location": "git+https://github.com/OpenPipe/ART.git@vllm_0_8_0_new_unsloth",
    "run_script": "source .venv/bin/activate && python logprob_check_tool_calls.py",
    "vllm_commit": "ef640440791a461a181e6d497965701462c166b3",
}


parser = argparse.ArgumentParser(
    description="Train one or more art-e models (comma separated)."
)
parser.add_argument(
    "--configs",
    type=str,
    required=True,
    help="The configs to use (e.g. 0.7.3,0.8.0).",
)
args = parser.parse_args()

# Parse and validate the requested model keys
requested_configs = [m.strip() for m in args.configs.split(",") if m.strip()]
unknown = [m for m in requested_configs if m not in configs]
if unknown:
    raise ValueError(
        f"Unknown config requested: {', '.join(unknown)}. Valid configs: {', '.join(configs.keys())}"
    )

def launch_model(config_str: str):
    config = configs[config_str]
    print(f"Launching {config} on SkyPilot…")

    setup_script = textwrap.dedent(
        f"""
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            uv remove openpipe-art
            uv add {config["art_location"]}
            uv remove vllm
            uv pip install vllm --extra-index-url https://wheels.vllm.ai/{config["vllm_commit"]}
        """
    )
    print(setup_script)

    run_script = config["run_script"]

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"art-debug-{config_str}",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )

    task.set_resources(sky.Resources(accelerators={"H100-SXM": 1}, cloud=sky.clouds.RunPod()))
    task.set_file_mounts({"~/ART": "../.."})

    # Generate cluster name
    # cluster_name = f"saumya-art-debug-{config_str}"
    cluster_name = "saumya-art-debug-cluster"
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
            fast=False,
        )
    )

    print(f"Job submitted for {config_str} (ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} for {config_str} finished with exit code {exit_code}.")


# Launch all requested models in parallel threads
with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(requested_configs)
) as executor:
    futures = [executor.submit(launch_model, config_str) for config_str in requested_configs]
    for future in concurrent.futures.as_completed(futures):
        # Propagate any exceptions raised inside threads
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred during config training: {e}")
            print(f"Traceback: {traceback.format_exc()}")
