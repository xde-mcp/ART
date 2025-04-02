import unsloth
from vllm import SamplingParams
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb
from transformers import TrainerCallback, PreTrainedTokenizer, AutoTokenizer
import numpy as np
from typing import Callable, Tuple, Coroutine, Any
import asyncio
from utils import score_title
from utils import pull_data, cache, prompt_for_title
from panza import limit_concurrency

load_dotenv()

# --- Hardcoded Configuration ---
RUN_NAME = "model7"
LORA_RANK = 16
BASE_MODEL = "unsloth/Qwen2.5-32B-Instruct"  # Updated from comment
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]  # Updated from comment
LOAD_IN_4BIT = True
FAST_INFERENCE = True
GPU_MEMORY_UTILIZATION = 0.6  # Updated from comment
LEARNING_RATE = 5e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "constant"  # Updated from comment
OPTIM = "paged_adamw_8bit"
LOGGING_STEPS = 5  # Updated from comment
PER_DEVICE_TRAIN_BATCH_SIZE = 6  # Updated from comment
GRADIENT_ACCUMULATION_STEPS = 1
NUM_GENERATIONS = 6
MAX_PROMPT_LENGTH = 8192  # Updated from comment
MAX_COMPLETION_LENGTH = 512  # Updated from comment
MAX_STEPS = -1  # Default, will use num_epochs
SAVE_STEPS = 250
MAX_GRAD_NORM = 0.1
OUTPUT_DIR = "outputs"
TRAINING_DATASET_SIZE = 5000  # Updated from comment
VAL_SET_SIZE = 100  # Updated from comment
VAL_SAMPLES_TO_LOG = 30  # Updated from comment
EVAL_STEPS = 50  # Updated from comment
NUM_EPOCHS = 1
BETA = 0.04
MIN_SCORE_FOR_DATA = 20  # From original load_data call
# --- End Hardcoded Configuration ---


def filter_on_length(data: Dataset, max_length: int, tokenizer_name: str) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def check_length(x):
        # Ensure x["prompt"] is a list of dicts as expected by apply_chat_template
        prompt_list = (
            x["prompt"]
            if isinstance(x["prompt"], list)
            else [{"role": "user", "content": x["prompt"]}]
        )
        return (
            len(
                tokenizer.apply_chat_template(
                    prompt_list, tokenize=True, add_generation_prompt=True
                )
            )
            <= max_length
        )

    len_before = len(data)
    data = data.filter(check_length)
    print(f"Samples before: {len_before}, samples after: {len(data)}")
    return data


@cache.cache()
async def load_titles_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
    max_length: int = 8192,
    tokenizer_name: str = "unsloth/Qwen2.5-14B-Instruct",
) -> Dataset:
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(x["scraped_body"]),
            "row": x,
        }
    )
    return filter_on_length(data, max_length, tokenizer_name)


RewardAndMetrics = Callable[
    [list[dict], list[list[dict]], list[dict]],
    Coroutine[Any, Any, list[Tuple[float, dict[str, float]]]],
]


class ValidationCallback(TrainerCallback):
    def __init__(
        self,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        model: FastLanguageModel,  # Pass model directly
        max_completion_length: int,
        reward_func: RewardAndMetrics,
        val_generations_to_log_to_wandb: int,
        eval_steps: int,
    ):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer  # Store tokenizer
        self.model = model  # Store model
        self.val_generations_to_log_to_wandb = val_generations_to_log_to_wandb
        self.validation_table = None
        self.inputs = tokenizer.apply_chat_template(
            self.val_dataset["prompt"], tokenize=False, add_generation_prompt=True
        )
        self.max_completion_length = max_completion_length
        self.reward_func = reward_func
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):  # Model is already stored
        if state.global_step % self.eval_steps != 0:
            return control

        # Use the stored model directly
        # No need to save/load LoRA for validation generation if using the trainer's model state
        outputs = self.model.fast_generate(  # type: ignore
            self.inputs,
            sampling_params=SamplingParams(max_tokens=self.max_completion_length),
            use_tqdm=False,
            # Assuming fast_generate uses the current PEFT adapter state correctly
        )
        outputs = [o.outputs[0].text for o in outputs]

        scores_and_metrics = asyncio.run(
            self.reward_func(
                self.val_dataset["prompt"],
                [
                    [{"content": o}] for o in outputs
                ],  # adapt to the shape the reward function expects
                self.val_dataset["row"],
            )
        )
        scores = [s[0] for s in scores_and_metrics]

        wandb.log(
            {
                "val/reward/mean": np.mean(scores),
                "val/reward/p5": np.percentile(scores, 5),
                "val/reward/median": np.percentile(scores, 50),
                "val/reward/p95": np.percentile(scores, 95),
                "val/reward/std_dev": np.std(scores),
            },
            step=state.global_step,
        )

        all_metrics: dict[str, list[float]] = {}
        for s_and_m in scores_and_metrics:
            for k, v in s_and_m[1].items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        for k, v in all_metrics.items():
            wandb.log(
                {
                    f"val/metrics/{k}/mean": np.mean(v),
                    f"val/metrics/{k}/p5": np.percentile(v, 5),
                    f"val/metrics/{k}/median": np.percentile(v, 50),
                    f"val/metrics/{k}/p95": np.percentile(v, 95),
                    f"val/metrics/{k}/std_dev": np.std(v),
                },
                step=state.global_step,
            )

        generations_to_log = self.val_generations_to_log_to_wandb
        if generations_to_log == 0:
            return control  # Return control object
        # Create tuples of (input, output, score) and sort by input text
        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [
                    f"input_{i + 1}",
                    f"output_{i + 1}",
                    f"score_{i + 1}",
                    f"metrics_{i + 1}",
                ]
                for i in range(generations_to_log)
            ],
            [],
        )

        if self.validation_table is None:
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = [state.global_step]

        samples = [x for x in zip(self.inputs, outputs, scores_and_metrics)][
            :generations_to_log
        ]
        for input_text, output_text, (score, metrics) in samples:  # Renamed variables
            row_data.extend(
                [
                    input_text,
                    output_text,
                    score,
                    "\n".join([f"{k}={v}" for k, v in metrics.items()]),
                ]
            )
        new_table.add_data(*row_data)

        # Log the table and update reference
        wandb.log({"val/generations": new_table}, step=state.global_step)
        self.validation_table = new_table
        return control  # Return control object


# --- Titles Task Specific Reward Logic ---
async def titles_reward(
    prompts: list[dict],
    completions: list[list[dict]],  # GRPOTrainer provides list of lists
    rows: list[dict],
    tokenizer: PreTrainedTokenizer,  # Need tokenizer and model for matching check
    model: FastLanguageModel,
) -> list[Tuple[float, dict[str, float]]]:
    responses = [completion[0]["content"] for completion in completions]
    updated_rows = [{**r, "title": response} for r, response in zip(rows, responses)]

    @limit_concurrency(10)
    async def score_title_async(r):
        return await score_title(r, "rm")

    def check_if_titles_match_bodies(bodies, titles):
        inputs = []
        for body, title in zip(bodies, titles):
            inputs.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a moderator for Hacker News. You are given the body of an article, as well as a proposed title. You are to determine whether the title makes any claims that are not substantiated by the article body. If there are any unsubstantiated claims, you should return False. Otherwise, you should return True. Only return False or True, no other text.",
                        },
                        {
                            "role": "user",
                            "content": f"<article>{body}</article>\n<proposed_title>{title}</proposed_title>",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        outputs = model.fast_generate(  # type: ignore
            inputs,
            sampling_params=SamplingParams(max_tokens=2),
            use_tqdm=False,  # Added for consistency
        )
        outputs = [o.outputs[0].text for o in outputs]
        results = []
        for output in outputs:
            clean_output = output.strip().lower()
            if clean_output.startswith("true"):
                results.append(1)
            elif clean_output.startswith("false"):
                results.append(0)
            else:
                print(
                    f"Warning: Invalid output from check_if_title_matches_scraped_body: {output}"
                )
                results.append(0)  # Penalize invalid output
        return results

    # Kick this off first so the RM can be scoring while we're doing the matching check locally
    rm_task_coros = [score_title_async(r) for r in updated_rows]

    # Run the matching check locally
    matching_scores = check_if_titles_match_bodies(
        [r["scraped_body"] for r in updated_rows],
        [r["title"] for r in updated_rows],
    )

    rm_scores = await asyncio.gather(*rm_task_coros)

    rewards = []
    for r, rm_score, title_matches in zip(updated_rows, rm_scores, matching_scores):
        if title_matches == 0:
            score = 0
        else:
            score = rm_score
        rewards.append(
            (
                score,
                {
                    "length": len(r["title"]),
                    "matches": title_matches,
                    "rm": rm_score,
                },
            )
        )
    return rewards


# --- Main Execution Logic ---
def main():
    wandb.init(
        project="hn_title_generation",  # Hardcoded project name
        name=RUN_NAME,
    )

    # Load model and tokenizer using the hardcoded hyperparameters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",  # type: ignore
        random_state=3407,
    )

    # Load data using the hardcoded function and parameters
    dataset = asyncio.run(
        load_titles_data(
            split="train",
            max_items=TRAINING_DATASET_SIZE,
            min_score=MIN_SCORE_FOR_DATA,
            max_length=MAX_PROMPT_LENGTH,
            tokenizer_name=BASE_MODEL,
        )
    )
    val_dataset = asyncio.run(
        load_titles_data(
            split="val",
            max_items=VAL_SET_SIZE,
            min_score=MIN_SCORE_FOR_DATA,
            max_length=MAX_PROMPT_LENGTH,
            tokenizer_name=BASE_MODEL,
        )
    )

    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Define the reward function wrapper for GRPOTrainer
    def grpo_trainer_reward_func(prompts, completions, **kwargs):
        # Need to pass tokenizer and model to the underlying reward function
        rewards_and_metrics = asyncio.run(
            titles_reward(
                prompts,
                completions,
                rows=kwargs["row"],
                tokenizer=tokenizer,
                model=model,
            )
        )
        return [r[0] for r in rewards_and_metrics]

    # Define the reward function wrapper for ValidationCallback
    async def validation_reward_func(prompts, completions, rows):
        # Need to pass tokenizer and model to the underlying reward function
        return await titles_reward(
            prompts, completions, rows, tokenizer=tokenizer, model=model
        )

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIM,
        logging_steps=LOGGING_STEPS,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        report_to="wandb",
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        beta=BETA,
    )

    trainer = GRPOTrainer(
        model=model,
        # Pass tokenizer instance directly if needed by GRPOTrainer internally,
        # otherwise it might infer from the model. Check TRL docs if issues arise.
        tokenizer=tokenizer,
        reward_funcs=grpo_trainer_reward_func,  # Use the wrapper
        args=training_args,
        train_dataset=dataset,
    )

    validation_callback = ValidationCallback(
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        model=model,  # Pass the model instance
        max_completion_length=MAX_COMPLETION_LENGTH,
        reward_func=validation_reward_func,  # Use the async wrapper
        val_generations_to_log_to_wandb=VAL_SAMPLES_TO_LOG,
        eval_steps=EVAL_STEPS,
    )
    trainer.add_callback(validation_callback)
    trainer.train()

    model.save_lora(RUN_NAME)


if __name__ == "__main__":
    main()
