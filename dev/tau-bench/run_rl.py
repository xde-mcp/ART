# Copyright Sierra

import argparse
import asyncio
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import art
from tau_bench.types import RunConfig, EnvRunResult
from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.run import agent_factory
try:
    from litellm import provider_list
except ImportError:
    provider_list = ["openai", "anthropic"]  # fallback
from tau_bench.envs.user import UserStrategy
from pydantic import BaseModel


class TauBenchRLConfig(BaseModel):
    """Configuration for tau-bench RL training"""
    max_turns: int = 10
    temperature: float = 0.1
    agent_strategy: str = "tool-calling"
    user_strategy: str = "llm"
    user_model: str = "gpt-4o"
    user_model_provider: str = "openai"


@dataclass
class RolloutMetrics:
    """Metrics for a single rollout"""
    success: bool = False
    reward: float = 0.0
    num_turns: int = 0
    error: Optional[str] = None
    task_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def rollout_tau_bench_task(
    model: art.TrainableModel,
    env_name: str,
    task_split: str,
    task_index: int,
    config: TauBenchRLConfig,
    step: int,
) -> art.Trajectory:
    """
    Run a single tau-bench task and return a trajectory for training.
    """
    metrics = RolloutMetrics(task_id=task_index)
    
    try:
        # Create isolated environment for this task
        env = get_env(
            env_name,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=task_split,
            task_index=task_index,
        )
        
        # Create RunConfig for the agent factory
        run_config = RunConfig(
            model_provider="openai",  # placeholder, will be overridden by ART model
            user_model_provider=config.user_model_provider,
            model=model.name,  # Use ART model name
            user_model=config.user_model,
            num_trials=1,
            env=env_name,
            agent_strategy=config.agent_strategy,
            temperature=config.temperature,
            task_split=task_split,
            start_index=task_index,
            end_index=task_index + 1,
            task_ids=[task_index],
            log_dir="rl_results",
            max_concurrency=1,
            seed=42,
            shuffle=0,
            user_strategy=config.user_strategy,
            few_shot_displays_path=None,
        )
        
        # Create agent using tau-bench's agent factory
        agent = create_art_agent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            run_config=run_config,
            art_model=model,
        )
        
        # Run the agent on the task
        result = agent.solve(env=env, task_index=task_index)
        
        # Update metrics
        metrics.success = result.reward > 0.9  # Consider >0.9 as success
        metrics.reward = result.reward
        metrics.num_turns = len([m for m in result.messages if m.get("role") == "assistant"])
        
        # Create trajectory from the result
        trajectory = art.Trajectory(
            messages_and_choices=[
                {
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", "")),
                }
                for msg in result.messages
            ],
            reward=result.reward,
            metadata={
                "task_id": task_index,
                "env": env_name,
                "step": step,
                "model_name": model.name,
                **metrics.to_dict(),
            }
        )
        
        return trajectory
        
    except Exception as e:
        # Create a failure trajectory
        metrics.error = str(e)
        return art.Trajectory(
            messages_and_choices=[
                {"role": "user", "content": f"Task {task_index}"},
                {"role": "assistant", "content": f"Error: {str(e)}"},
            ],
            reward=0.0,
            metadata={
                "task_id": task_index,
                "env": env_name,
                "step": step,
                "model_name": model.name,
                **metrics.to_dict(),
            }
        )


def create_art_agent(
    tools_info: List[Dict[str, Any]], 
    wiki: Any, 
    run_config: RunConfig,
    art_model: art.TrainableModel
) -> Agent:
    """
    Create an agent that uses the ART trainable model.
    This wraps the existing tau-bench agent with ART model integration.
    """
    # Create base agent using tau-bench's factory
    base_agent = agent_factory(tools_info, wiki, run_config)
    
    class ARTAgent(Agent):
        def __init__(self, base_agent: Agent, art_model: art.TrainableModel):
            self.base_agent = base_agent
            self.art_model = art_model
        
        def solve(self, env, task_index: int):
            """
            Solve method that uses ART model for inference.
            """
            # Replace the base agent's model with our ART model's client
            if hasattr(self.base_agent, 'client'):
                setattr(self.base_agent, 'client', self.art_model.openai_client())
            if hasattr(self.base_agent, 'model'):
                setattr(self.base_agent, 'model', self.art_model.get_inference_name())
            
            # Use the base agent's solve method
            return self.base_agent.solve(env, task_index)
    
    return ARTAgent(base_agent, art_model)


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="ART RL Training for Tau-Bench")
    
    # Basic model and training parameters
    parser.add_argument("--model-name", type=str, required=True, help="Name for the trainable model")
    parser.add_argument("--project-name", type=str, default="tau-bench-rl", help="Project name")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base model to fine-tune")
    
    # Training parameters
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--trajectories-per-step", type=int, default=32, help="Number of trajectories to collect per step")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--eval-frequency", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--eval-tasks", type=int, default=20, help="Number of tasks to use for evaluation")
    
    # Environment parameters (similar to run.py)
    parser.add_argument("--env", type=str, choices=["retail", "airline"], default="retail", help="Environment to train on")
    parser.add_argument("--task-split", type=str, default="train", choices=["train", "test", "dev"], help="Task split for training")
    parser.add_argument("--user-model", type=str, default="gpt-4o", help="Model for user simulation")
    parser.add_argument("--user-model-provider", type=str, choices=provider_list, default="openai", help="Provider for user model")
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy], help="User strategy")
    parser.add_argument("--agent-strategy", type=str, default="tool-calling", choices=["tool-calling", "act", "react", "few-shot"], help="Agent strategy")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    
    # Backend parameters
    parser.add_argument("--backend", choices=["skypilot", "local"], default="local", help="Backend to use for training")
    parser.add_argument("--restart", action="store_true", help="Restart the ART server")
    parser.add_argument("--cluster-name", type=str, default="tau-bench-rl", help="Cluster name for SkyPilot")
    
    # Logging and checkpointing
    parser.add_argument("--log-dir", type=str, default="rl_results", help="Directory for logging results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    print(f"Training arguments: {args}")
    return vars(args)


async def gather_trajectories(
    model: art.TrainableModel,
    env_name: str,
    task_split: str,
    config: TauBenchRLConfig,
    num_trajectories: int,
    step: int,
    task_start_index: int = 0,
) -> art.TrajectoryGroup:
    """
    Gather trajectories by running the model on tau-bench tasks.
    Similar to art-e pattern but adapted for tau-bench.
    """
    # Get environment to determine available tasks
    env = get_env(
        env_name, 
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        task_split=task_split,
        user_provider=config.user_model_provider
    )
    num_tasks = len(env.tasks)
    
    trajectories = []
    for i in range(num_trajectories):
        # Cycle through available tasks
        task_index = task_start_index + (i % num_tasks)
        trajectory = await rollout_tau_bench_task(
            model=model,
            env_name=env_name,
            task_split=task_split,
            task_index=task_index,
            config=config,
            step=step,
        )
        trajectories.append(trajectory)
    
    return art.TrajectoryGroup(trajectories=trajectories)


async def evaluate_model(
    model: art.TrainableModel,
    env_name: str,
    config: TauBenchRLConfig,
    num_eval_tasks: int,
    step: int,
) -> float:
    """
    Evaluate the model on validation tasks.
    """
    eval_trajectories = await gather_trajectories(
        model=model,
        env_name=env_name,
        task_split="dev",  # Use dev split for evaluation
        config=config,
        num_trajectories=num_eval_tasks,
        step=step,
        task_start_index=0,
    )
    
    # Log evaluation trajectories
    await model.log(eval_trajectories.trajectories, split="val")
    
    # Calculate metrics
    rewards = [t.reward for t in eval_trajectories.trajectories]
    success_rate = sum(1 for r in rewards if r > 0.9) / len(rewards)
    avg_reward = sum(rewards) / len(rewards)
    
    print(f"📊 Step {step} - Evaluation: avg_reward={avg_reward:.3f}, success_rate={success_rate:.3f}")
    
    return avg_reward


async def main():
    args = parse_args()
    random.seed(args["seed"])
    
    # Initialize backend (following art-e pattern)
    if args["backend"] == "skypilot":
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=args["cluster_name"],
            art_version=".",
            env_path=".env",
            gpu="H100",
            force_restart=args["restart"],
        )
    else:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    
    # Create RL config
    rl_config = TauBenchRLConfig(
        max_turns=10,
        temperature=args["temperature"],
        agent_strategy=args["agent_strategy"],
        user_strategy=args["user_strategy"],
        user_model=args["user_model"],
        user_model_provider=args["user_model_provider"],
    )
    
    # Create trainable model (following art-e pattern)
    model = art.TrainableModel(
        name=args["model_name"],
        project=args["project_name"],
        base_model=args["base_model"],
        config=rl_config,
    )
    
    # Register model with backend
    print("🔧 Registering model with backend...")
    await model.register(backend)
    
    # Create log directory
    log_dir = args["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f"🚀 Starting RL training for {args['num_steps']} steps...")
    
    # Training loop (following art-e pattern)
    for step in range(await model.get_step(), args["num_steps"]):
        print(f"\n📈 Training Step {step}")
        
        # Gather trajectories
        print(f"📝 Collecting {args['trajectories_per_step']} trajectories...")
        trajectory_group = await gather_trajectories(
            model=model,
            env_name=args["env"],
            task_split=args["task_split"],
            config=rl_config,
            num_trajectories=args["trajectories_per_step"],
            step=step,
            task_start_index=step * args["trajectories_per_step"],
        )
        
        # Calculate step metrics
        rewards = [t.reward for t in trajectory_group.trajectories]
        step_avg_reward = sum(rewards) / len(rewards)
        step_success_rate = sum(1 for r in rewards if r > 0.9) / len(rewards)
        print(f"📊 Step {step} - Training: avg_reward={step_avg_reward:.3f}, success_rate={step_success_rate:.3f}")
        
        # Train model (following art-e pattern)
        print("🎯 Training model...")
        await model.train(
            trajectory_groups=[trajectory_group],
            config=art.TrainConfig(learning_rate=args["learning_rate"]),
            verbose=True,
        )
        
        # Evaluation
        if step % args["eval_frequency"] == 0:
            print("🔍 Running evaluation...")
            eval_reward = await evaluate_model(
                model=model,
                env_name=args["env"],
                config=rl_config,
                num_eval_tasks=args["eval_tasks"],
                step=step,
            )
            
            # Save checkpoint info
            checkpoint_info = {
                "step": step,
                "training_reward": step_avg_reward,
                "training_success_rate": step_success_rate,
                "eval_reward": eval_reward,
                "timestamp": datetime.now().isoformat(),
                "args": args,
            }
            
            checkpoint_path = os.path.join(log_dir, f"checkpoint_step_{step}.json")
            with open(checkpoint_path, "w") as f:
                import json
                json.dump(checkpoint_info, f, indent=2)
            
            print(f"💾 Checkpoint saved to {checkpoint_path}")
    
    print(f"✅ Training completed! Model: {args['model_name']}")


if __name__ == "__main__":
    asyncio.run(main())