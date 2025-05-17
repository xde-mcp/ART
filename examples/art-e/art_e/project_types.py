from pydantic import BaseModel


class TrainingConfig(BaseModel):
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4
    gpus: str = "H100-SXM:1"
    art_location: str = "--editable ~/ART"
    rollout_concurrency: int = 100


class PolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    log_to_langfuse: bool = False
    use_tools: bool = True
    stupid_simple_reward_fn: bool = False
    max_groups_in_flight: int = 4

    training_config: TrainingConfig | None = None
