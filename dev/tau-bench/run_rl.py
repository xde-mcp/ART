# Copyright Sierra

import argparse
import asyncio
import os
from dotenv import load_dotenv

import art
from art.local import LocalBackend
from art.utils import iterate_dataset

from tau_bench.types import RunConfig
from tau_bench.envs import get_env
from tau_bench.run import agent_factory
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
from tau_bench.agents.tool_calling_agent import ToolCallingRLAgent
from tau_bench.types import TauBenchPolicyConfig, TauBenchTrainingConfig
from langfuse import Langfuse

# Load environment variables
load_dotenv()

def log_trajectory_to_langfuse(
    langfuse: Langfuse,
    traj: art.Trajectory,
    task_idx: int,
    step: int,
    phase: str = "train"
) -> None:
    """
    Push one trajectory to Langfuse with task_idx and step for comparison.
    """
    trace_name = f"rl-{phase}-step-{step}-task-{task_idx}"
    
    # Create trace with trajectory data
    trace = langfuse.trace(
        name=trace_name,
        input={
            "task_idx": task_idx,
            "step": step,
            "phase": phase,
            "metadata": traj.metadata
        },
        output={
            "messages": [
                {"role": choice.role, "content": choice.content} 
                for msg_and_choice in traj.messages_and_choices 
                for choice in msg_and_choice.choices
            ] if traj.messages_and_choices else [],
            "reward": traj.reward,
            "metadata": traj.metadata
        },
        metadata={
            "task_idx": task_idx,
            "training_step": step,
            "phase": phase,
            "env": traj.metadata.get("env", "unknown")
        }
    )
    
    # Add reward as a score
    trace.score(name="reward", value=traj.reward)
    
    # Add step as a score for easy filtering
    trace.score(name="training_step", value=step)

def rollout_tau_bench_task(
    model: art.Model[TauBenchPolicyConfig],
    task_index: int,
    langfuse: Langfuse = None,
    step: int = 0,
    phase: str = "train",
) -> art.Trajectory:
    """
    Generate a trajectory for a single tau-bench task using the given model.
    This adapts the tau-bench evaluation loop for RL trajectory generation.
    Now synchronous to match the tau-bench architecture.
    """
    print(f"Rolling out task {task_index} (step {step}, phase {phase})")
    config = model.config.run_config
    
    # Get isolated environment for this task
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        task_index=task_index,
    )
    
    # Create agent with the trainable model
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )

    if not isinstance(agent, ToolCallingRLAgent):
        raise ValueError("Agent must be a ToolCallingRLAgent")
    
    # Create trajectory object
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={
            "task_index": task_index, 
            "env": config.env,
            "training_step": step,
            "phase": phase
        }
    )
    
    try:
        # Run the agent on the task (synchronous call)
        result = agent.solve(
            env=env,
            task_index=task_index,
        )
        
        # Convert result to trajectory format
        traj.reward = result.reward
        traj.metadata.update(result.info)
        
        traj.messages_and_choices = agent.create_messages_and_choices(result.messages)
    except Exception as e:
        print(f"Error in rollout for task {task_index}: {e}")
        traj.reward = 0.0
        traj.metadata["error"] = str(e)
    
    traj.finish()
    
    # Log to langfuse if provided
    if langfuse:
        log_trajectory_to_langfuse(langfuse, traj, task_index, step, phase)
    
    print(f"Finished rolling out task {task_index} (reward: {traj.reward})")
    return traj


async def async_rollout_tau_bench_task(
    model: art.Model[TauBenchPolicyConfig],
    task_index: int,
    langfuse: Langfuse = None,
    step: int = 0,
    phase: str = "train",
) -> art.Trajectory:
    """
    Async wrapper for rollout_tau_bench_task using asyncio.to_thread().
    This allows the sync tau-bench infrastructure to work with the async ART framework.
    """
    return await asyncio.to_thread(rollout_tau_bench_task, model, task_index, langfuse, step, phase)


def parse_args() -> tuple[RunConfig, TauBenchTrainingConfig, argparse.Namespace]:
    """Parse command line arguments for RL training"""
    parser = argparse.ArgumentParser(description="Train an agent on tau-bench using ART RL")
    
    # tau-bench arguments (reuse from original run.py)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
        required=True,
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
        required=True,
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling-rl",
        choices=["tool-calling-rl"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="train",  # Default to train for RL
        choices=["train", "test", "dev"],
        help="The split of tasks to run",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=100, help="End index for training tasks")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="rl_results")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    
    # RL-specific arguments
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Base model for training")
    parser.add_argument("--trajectories-per-group", type=int, default=6, help="Number of trajectories per group")
    parser.add_argument("--groups-per-step", type=int, default=8, help="Number of groups per training step")
    parser.add_argument("--learning-rate", type=float, default=1.2e-5, help="Learning rate for training")
    parser.add_argument("--eval-steps", type=int, default=30, help="Evaluate every N steps")
    parser.add_argument("--val-set-size", type=int, default=100, help="Validation set size")
    parser.add_argument("--training-dataset-size", type=int, default=1000, help="Training dataset size")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    
    args = parser.parse_args()
    print(args)
    
    # Create RunConfig for tau-bench
    run_config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=1,  # RL training is sequential
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
    )
    
    # Create training config
    training_config = TauBenchTrainingConfig(
        trajectories_per_group=args.trajectories_per_group,
        groups_per_step=args.groups_per_step,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        val_set_size=args.val_set_size,
        training_dataset_size=args.training_dataset_size,
        num_epochs=args.num_epochs,
    )
    
    return run_config, training_config, args


async def evaluate_model(
    model: art.TrainableModel[TauBenchPolicyConfig],
    config: RunConfig,
    langfuse: Langfuse,
    step: int,
    num_eval_tasks: int = 50
) -> float:
    """Evaluate the model on a subset of tasks"""
    print(f"Evaluating model on {num_eval_tasks} tasks...")
    
    # Get environment for evaluation
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    
    total_reward = 0.0
    eval_tasks = min(num_eval_tasks, len(env.tasks))

    trajectories = await art.gather_trajectories(
        (
            async_rollout_tau_bench_task(model, i, langfuse, step, "eval")
            for i in range(eval_tasks)
        )
    )
    await model.log(trajectories=trajectories, split="val")
    
    for traj in trajectories:
        total_reward += traj.reward
        print(f"Eval task {traj.metadata['task_index']}: reward={traj.reward}")
    
    avg_reward = total_reward / eval_tasks
    print(f"Average evaluation reward: {avg_reward}")
    return avg_reward


async def train(model: art.TrainableModel[TauBenchPolicyConfig], langfuse: Langfuse):
    """Main training loop adapted from art-e example"""
    config = model.config.run_config
    training_config = model.config.training_config
    
    if training_config is None:
        raise ValueError("Training config is not set")
    
    with LocalBackend() as backend:
        # Setup model with backend
        await model.register(backend)
        config.api_key = model.inference_api_key
        config.base_url = model.inference_base_url
        
        print("Loading training tasks...")
        # Get environment to access tasks
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
        )
        
        # Create list of task indices for training
        end_index = min(config.end_index, len(env.tasks)) if config.end_index != -1 else len(env.tasks)
        if config.task_ids:
            train_task_indices = config.task_ids
        else:
            train_task_indices = list(range(config.start_index, min(end_index, training_config.training_dataset_size)))
        
        # Validation task indices
        val_task_indices = list(range(len(train_task_indices), len(train_task_indices) + training_config.val_set_size))
        
        print(f"Training on {len(train_task_indices)} tasks")
        print(f"Validation on {len(val_task_indices)} tasks")
        
        # Training iterator
        train_iterator = iterate_dataset(
            train_task_indices,
            groups_per_step=training_config.groups_per_step,
            num_epochs=training_config.num_epochs,
            initial_step=await model.get_step(),
        )
        
        for batch, epoch, global_step, epoch_step in train_iterator:
            print(f"\n--- Training Step {global_step} (Epoch {epoch}, Step {epoch_step}) ---")
            
            # Evaluation
            if global_step % training_config.eval_steps == 0:
                print(f"\n--- Evaluating at Step {global_step} ---")
                await evaluate_model(model, config, langfuse, global_step, num_eval_tasks=len(val_task_indices))
                await model.delete_checkpoints()
            
            # Generate trajectory groups
            print(f"Generating trajectories for {len(batch)} tasks...")
            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            async_rollout_tau_bench_task(model, task_index, langfuse, global_step, "train")
                            for _ in range(training_config.trajectories_per_group)
                        )
                    )
                    for task_index in batch
                )
            )
            
            # Training step
            print(f"Training on {len(groups)} trajectory groups...")
            await model.train(
                groups,
                config=art.TrainConfig(
                    learning_rate=training_config.learning_rate
                ),
            )
            
            # Log progress
            total_reward = sum(
                sum(traj.reward for traj in group.trajectories) 
                for group in groups
            )
            num_trajectories = sum(len(group.trajectories) for group in groups)
            avg_reward = total_reward / num_trajectories if num_trajectories > 0 else 0
            print(f"Step {global_step}: Average training reward = {avg_reward}")
        
        # Final evaluation
        print("\n--- Final Evaluation ---")
        final_step = await model.get_step()
        final_reward = await evaluate_model(model, config, langfuse, final_step, num_eval_tasks=len(val_task_indices))
        print(f"Final average reward: {final_reward}")
        
        print("Training completed!")


def main():
    """Main function"""
    run_config, training_config, args = parse_args()
    
    # Initialize langfuse
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    
    # Create trainable model
    model = art.TrainableModel(
        name=args.model,
        project="tau_bench_rl",
        base_model=args.base_model,
        config=TauBenchPolicyConfig(
            training_config=training_config,
            run_config=run_config,
        ),
    )
    
    print(f"Starting RL training for model: {model.name}")
    print(f"Base model: {model.base_model}")
    print(f"Environment: {run_config.env}")
    print(f"Task split: {run_config.task_split}")
    
    try:
        # Run training
        asyncio.run(train(model, langfuse))
    finally:
        # Ensure langfuse flushes all traces
        langfuse.flush()
        print("Langfuse traces flushed.")


if __name__ == "__main__":
    main()