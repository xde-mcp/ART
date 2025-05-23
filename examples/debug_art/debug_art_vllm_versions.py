# Usage: uv run debug_art.py --configs=0.7.3
import argparse
import os
import re
import sky
import textwrap
from dotenv import load_dotenv, dotenv_values
from sky import ClusterStatus
import concurrent.futures
import traceback

load_dotenv()

configs = {}

configs["0.8.0-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "uv run logprob_check_tool_calls.py",
    "vllm_commit": "v0.8.0",
}

configs["ef640440791a461a181e6d497965701462c166b3-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "ef640440791a461a181e6d497965701462c166b3",
}

configs["128bf7528370d792099c66f301c6c5deef8f4110-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "128bf7528370d792099c66f301c6c5deef8f4110",
}

configs["432d6dad15e74ba2d5f2f97f9693706b0930b2f0-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "432d6dad15e74ba2d5f2f97f9693706b0930b2f0",
}

configs["a21076ed3a4077e79afe0a3b422f89f9a920154d-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "a21076ed3a4077e79afe0a3b422f89f9a920154d",
}

configs["0b7f06b447e513dabfb87f490713516943c7c371-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "0b7f06b447e513dabfb87f490713516943c7c371",
}

configs["b8b0ccbd2dd4325a916dd5c2735d3320da1600ad-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "b8b0ccbd2dd4325a916dd5c2735d3320da1600ad",
}

configs["333681408feabb97193880303b23f6571ba39045-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "333681408feabb97193880303b23f6571ba39045",
}

configs["47512b3200156cc0db4a62791d2f7fefcaecb6ad-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "47512b3200156cc0db4a62791d2f7fefcaecb6ad",
}

configs["9f3bc0f58c431404f02372e22b4050460e2be448-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "9f3bc0f58c431404f02372e22b4050460e2be448",
}

configs["3b9c6c69476fa29bc3c719a431564579a48e8b17-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "3b9c6c69476fa29bc3c719a431564579a48e8b17",
}

configs["4aae667668e59509ab152b08c5eb617b8ae1187a-tool-calls"] = {
    "art_location": "openpipe-art",
    "run_script": "python logprob_check_tool_calls.py",
    "vllm_commit": "4aae667668e59509ab152b08c5eb617b8ae1187a",
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
parser.add_argument(
    "--cluster-name",
    type=str,
    required=False,
    default="saumya-art-debug-cluster",
    help="The cluster name to use.",
)
args = parser.parse_args()

# Parse and validate the requested model keys
requested_configs = [m.strip() for m in args.configs.split(",") if m.strip()]
unknown = [m for m in requested_configs if m not in configs]
if unknown:
    raise ValueError(
        f"Unknown config requested: {', '.join(unknown)}. Valid configs: {', '.join(configs.keys())}"
    )

def _update_pyproject(vllm_commit: str, art_location: str = ".") -> None:
    """
    Replace the existing vllm wheels URL in pyproject.toml with one that ends
    in the supplied commit/tag string.

    Args:
        vllm_commit: e.g. "ef640440791a461a181e6d497965701462c166b3"
        art_location: directory that contains pyproject.toml
    """
    pyproject = os.path.join(art_location, "pyproject.toml")
    if not os.path.exists(pyproject):        # fall back to repo root
        raise RuntimeError("pyproject.toml not found")

    with open(pyproject, "r") as f:
        text = f.read()

    # Match ANY current wheels.vllm.ai fragment and replace it
    new_url = f"https://wheels.vllm.ai/{vllm_commit}"
    text, n = re.subn(r"https://wheels\.vllm\.ai/[^\s\"']+", new_url, text, count=1)

    if n == 0:
        raise RuntimeError("vllm wheels URL not found in pyproject.toml")

    with open(pyproject, "w") as f:
        f.write(text)
    print(f"✅  Updated {pyproject} → {new_url}")

def launch_model(config_str: str):
    config = configs[config_str]
    print(f"Launching {config} on SkyPilot…")

    setup_script = textwrap.dedent(
        f"""
            echo 'Setting up environment...'
            apt install -y nvtop
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env

            uv pip uninstall openpipe-art
            uv pip install -e ~/ART --force-reinstall
        """
    )

    _update_pyproject(config["vllm_commit"], "../..")

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
    cluster_name = args.cluster_name
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
