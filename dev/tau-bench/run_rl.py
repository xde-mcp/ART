# Copyright Sierra

import argparse
import asyncio
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import art
from tau_bench.types import RunConfig, EnvRunResult
from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
try:
    from litellm import provider_list
except ImportError:
    provider_list = ["openai", "anthropic"]  # fallback
from tau_bench.envs.user import UserStrategy
from pydantic import BaseModel


class TauBenchConfig(BaseModel):
    """Configuration for tau-bench RL training"""
    max_turns: int = 10
    temperature: float = 0.1
    agent_strategy: str = "tool-calling"


async def rollout_tau_bench_task(
    model: art.TrainableModel,
    env_name: str,
    task_split: str,
    task_index: int,
    user_strategy: str,
    user_model: str,
    user_model_provider: str,
    step: int,
) -> art.Trajectory:
    """
    Run a single tau-bench task and return a trajectory for training.
    """
    # Create isolated environment for this task
    env = get_env(
        env_name,
        user_strategy=user_strategy,
        user_model=user_model,
        user_provider=user_model_provider,
        task_split=task_split,
        task_index=task_index,
    )
    
    # Create agent with the trainable model
    agent = art_agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=model,
    )
    
    try:
        # Run the agent on the task
        result = agent.solve(env=env, task_index=task_index)
        
        # Create trajectory from the result
        trajectory = art.Trajectory(
            messages_and_choices=[
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": str(msg.get("content", "")),
                }
                for i, msg in enumerate(result.messages)
            ],
            reward=result.reward,
            metadata={
                "task_id": task_index,
                "env": env_name,
                "step": step,
                "model_name": model.name,
                "agent_strategy": model.config.agent_strategy if hasattr(model.config, 'agent_strategy') else "tool-calling",
            }
        )
        
        return trajectory
        
    except Exception as e:
        # Create a failure trajectory
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
                "error": str(e),
            }
        )


def art_agent_factory(
    tools_info: List[Dict[str, Any]], 
    wiki, 
    model: art.TrainableModel
) -> Agent:
    """
    Create an agent that uses the ART trainable model.
    This creates a simplified agent wrapper for the tau-bench environment.
    """
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent
    
    class ARTAgent(Agent):
        def __init__(self, tools_info, wiki, art_model: art.TrainableModel):
            self.art_model = art_model
            self.tools_info = tools_info
            self.wiki = wiki
            self.temperature = getattr(art_model.config, 'temperature', 0.1)
        
        def solve(self, env, task_index: int):
            """
            Simplified solve method that interacts with the environment.
            This is a basic implementation - you may need to enhance it
            based on your specific requirements.
            """
            # Reset environment
            response = env.reset(task_index)
            messages = [{"role": "user", "content": response.observation}]
            
            reward = 0.0
            done = False
            conversation = []
            
            while not done:
                try:
                    # Get model response (simplified - you may need to adapt this)
                    # For now, we'll use a basic response format
                    client = self.art_model.openai_client()
                    completion = client.chat.completions.create(
                        model=self.art_model.get_inference_name(),
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=512,
                    )
                    
                    response_content = completion.choices[0].message.content
                    messages.append({"role": "assistant", "content": response_content})
                    conversation.append({"role": "assistant", "content": response_content})
                    
                    # Parse action from response (this is environment-specific)
                    # For now, we'll assume the response is a direct action
                    action = {"name": "respond", "kwargs": {"content": response_content}}
                    
                    # Step environment
                    env_response = env.step(action)
                    reward = env_response.reward
                    done = env_response.done
                    
                    if not done:
                        messages.append({"role": "user", "content": env_response.observation})
                        conversation.append({"role": "user", "content": env_response.observation})
                
                except Exception as e:
                    # Handle errors gracefully
                    return type('Result', (), {
                        'reward': 0.0,
                        'messages': conversation + [{"role": "system", "content": f"Error: {str(e)}"}],
                        'info': {"error": str(e)}
                    })()
            
            return type('Result', (), {
                'reward': reward,
                'messages': conversation,
                'info': {"task_id": task_index}
            })()
    
    return ARTAgent(tools_info, wiki, model)


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
    
    # Environment parameters
    parser.add_argument("--env", type=str, choices=["retail", "airline"], default="retail", help="Environment to train on")
    parser.add_argument("--task-split", type=str, default="train", choices=["train", "test", "dev"], help="Task split for training")
    parser.add_argument("--user-model", type=str, default="gpt-4o", help="Model for user simulation")
    parser.add_argument("--user-model-provider", type=str, choices=provider_list, help="Provider for user model")
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy], help="User strategy")
    parser.add_argument("--agent-strategy", type=str, default="tool-calling", choices=["tool-calling", "act", "react"], help="Agent strategy")
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
    user_strategy: str,
    user_model: str,
    user_model_provider: str,
    num_trajectories: int,
    step: int,
    task_start_index: int = 0,
) -> art.TrajectoryGroup:
    """
    Gather trajectories by running the model on tau-bench tasks.
    """
    trajectories = []
    
    for i in range(num_trajectories):
        task_index = task_start_index + (i % 100)  # Cycle through tasks if needed
        trajectory = await rollout_tau_bench_task(
            model=model,
            env_name=env_name,
            task_split=task_split,
            task_index=task_index,
            user_strategy=user_strategy,
            user_model=user_model,
            user_model_provider=user_model_provider,
            step=step,
        )
        trajectories.append(trajectory)
    
    return art.TrajectoryGroup(trajectories=trajectories)


async def evaluate_model(
    model: art.TrainableModel,
    env_name: str,
    user_strategy: str,
    user_model: str,
    user_model_provider: str,
    num_eval_tasks: int,
    step: int,
) -> float:
    """
    Evaluate the model on a set of validation tasks.
    """
    eval_trajectories = await gather_trajectories(
        model=model,
        env_name=env_name,
        task_split="dev",  # Use dev split for evaluation
        user_strategy=user_strategy,
        user_model=user_model,
        user_model_provider=user_model_provider,
        num_trajectories=num_eval_tasks,
        step=step,
        task_start_index=0,
    )
    
    # Log evaluation trajectories
    await model.log(eval_trajectories.trajectories, split="val")
    
    # Calculate average reward
    avg_reward = sum(t.reward for t in eval_trajectories.trajectories) / len(eval_trajectories.trajectories)
    print(f"📊 Step {step} - Evaluation avg reward: {avg_reward:.3f}")
    
    return avg_reward


async def main():
    args = parse_args()
    random.seed(args["seed"])
    
    # Initialize backend
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
    
    # Create trainable model
    model = art.TrainableModel(
        name=args["model_name"],
        project=args["project_name"],
        base_model=args["base_model"],
        config=TauBenchConfig(
            max_turns=10,
            temperature=args["temperature"],
            agent_strategy=args["agent_strategy"],
        ),
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(
                num_scheduler_steps=1,
            ),
        ),
    )
    
    # Register model with backend
    print("🔧 Registering model with backend...")
    await model.register(backend)
    
    # Create log directory
    log_dir = args["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f"🚀 Starting RL training for {args['num_steps']} steps...")
    
    # Training loop
    for step in range(await model.get_step(), args["num_steps"]):
        print(f"\n📈 Training Step {step}")
        
        # Gather trajectories
        print(f"📝 Collecting {args['trajectories_per_step']} trajectories...")
        trajectory_group = await gather_trajectories(
            model=model,
            env_name=args["env"],
            task_split=args["task_split"],
            user_strategy=args["user_strategy"],
            user_model=args["user_model"],
            user_model_provider=args["user_model_provider"],
            num_trajectories=args["trajectories_per_step"],
            step=step,
            task_start_index=step * args["trajectories_per_step"],
        )
        
        # Calculate average reward for this step
        step_avg_reward = sum(t.reward for t in trajectory_group.trajectories) / len(trajectory_group.trajectories)
        print(f"📊 Step {step} - Training avg reward: {step_avg_reward:.3f}")
        
        # Train model
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
                user_strategy=args["user_strategy"],
                user_model=args["user_model"],
                user_model_provider=args["user_model_provider"],
                num_eval_tasks=args["eval_tasks"],
                step=step,
            )
            
            # Save checkpoint info
            checkpoint_info = {
                "step": step,
                "training_reward": step_avg_reward,
                "eval_reward": eval_reward,
                "timestamp": datetime.now().isoformat(),
            }
            
            checkpoint_path = os.path.join(log_dir, f"checkpoint_step_{step}.json")
            with open(checkpoint_path, "w") as f:
                import json
                json.dump(checkpoint_info, f, indent=2)
            
            print(f"💾 Checkpoint saved to {checkpoint_path}")
    
    print(f"✅ Training completed! Model: {args['model_name']}")
    

if __name__ == "__main__":
    asyncio.run(main())