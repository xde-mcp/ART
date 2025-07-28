from pydantic import BaseModel


class ProjectPolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    log_to_openpipe: bool = False
    stupid_simple_reward_fn: bool = False
    include_qwen3_nothink: bool = False

    # Training configuration fields (previously in TrainingConfig)
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4
    ruler_judge_model: str = "openrouter/qwen/qwen3-32b"

    # Fork configuration
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # Training configuration
    scale_rewards: bool = True  # Whether to scale rewards during training
