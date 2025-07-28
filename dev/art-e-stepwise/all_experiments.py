import art
from art_e.project_types import ProjectPolicyConfig


models: dict[str, art.TrainableModel[ProjectPolicyConfig]] = {
    "001": art.TrainableModel(
        name="ae-stepwise-001",
        project="email_agent",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=ProjectPolicyConfig(
            max_turns=10,
            log_to_openpipe=True,
            trajectories_per_group=4,
            groups_per_step=4,
            learning_rate=1.2e-5,
            eval_steps=10,
            val_set_size=100,
            training_dataset_size=4000,
            num_epochs=3,
            ruler_judge_model="openrouter/qwen/qwen3-32b",
        ),
    )
}
