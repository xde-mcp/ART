import art
import asyncio
import openai
from openai.types.chat import ChatCompletionMessageParam
import os
import numpy as np
import wandb
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Iterable
import random

from utils import score_title, pull_data, cache, prompt_for_title
from panza import limit_concurrency

load_dotenv()

RUN_NAME = "model7-art"
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAX_COMPLETION_LENGTH = 100
MAX_PROMPT_LENGTH = 8192 - MAX_COMPLETION_LENGTH
LEARNING_RATE = 5e-6
ENTRIES_PER_ITERATION = 1
EVAL_STEPS = 50
VAL_SET_SIZE = 100
VAL_SAMPLES_TO_LOG = 30
TRAINING_DATASET_SIZE = 5000
WANDB_PROJECT = "hn_title_generation"
NUM_EPOCHS = 1
NUM_GENERATIONS = 6


# --- Data Loading ---
def filter_on_length(data: Dataset, max_length: int, tokenizer_name: str) -> Dataset:
    """Filters dataset based on tokenized prompt length."""
    print(
        f"Filtering dataset for max prompt length: {max_length} using tokenizer: {tokenizer_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def check_length(x):
        # Ensure 'prompt' is a list of dicts
        if not isinstance(x.get("prompt"), list):
            print(f"Warning: Skipping row with invalid prompt format: {x}")
            return False
        try:
            tokenized_len = len(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            return tokenized_len <= max_length
        except Exception as e:
            print(
                f"Warning: Error tokenizing prompt, skipping row. Error: {e}, Prompt: {x['prompt']}"
            )
            return False

    len_before = len(data)
    data = data.filter(check_length)
    len_after = len(data)
    print(f"Samples before length filtering: {len_before}, samples after: {len_after}")
    if len_after == 0 and len_before > 0:
        print(
            "Warning: All samples were filtered out. Check MAX_PROMPT_LENGTH and tokenizer."
        )
    elif len_after < len_before * 0.5:
        print(
            f"Warning: More than 50% of samples filtered out ({len_before - len_after} samples)."
        )
    return data


@cache.cache()
async def load_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
    max_length: int = 8192,
    tokenizer_name: str = BASE_MODEL,
) -> Dataset:
    """Loads, preprocesses, and filters the dataset."""
    print(
        f"Loading data for split: {split}, max_items: {max_items}, tokenizer: {tokenizer_name}"
    )
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    if not data:
        raise ValueError(f"No data loaded for split {split}. Check pull_data function.")

    # Ensure 'scraped_body' exists and is text
    def check_scraped_body(x):
        body = x.get("scraped_body")
        return isinstance(body, str) and len(body.strip()) > 0

    data = data.filter(check_scraped_body)
    if not data:
        raise ValueError(
            f"No data remaining after filtering for valid 'scraped_body' in split {split}."
        )

    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(
                x["scraped_body"]
            ),  # Creates the list of messages
            "row": x,  # Keep original row data
        }
    )
    return filter_on_length(data, max_length, tokenizer_name)


# --- Rollout Function ---
@limit_concurrency(10)  # Limit concurrent calls to external RM service
async def call_score_title(row_with_title: Dict[str, Any]) -> float:
    """Async wrapper for scoring."""
    return await score_title(row_with_title, "rm")


async def check_title_matches_body(
    client: openai.AsyncOpenAI, body: str, title: str
) -> int:
    """Uses the LLM itself to check if the title makes unsubstantiated claims."""
    system_prompt = "You are a moderator for Hacker News. You are given the body of an article, as well as a proposed title. You are to determine whether the title makes any claims that are not substantiated by the article body. If there are any unsubstantiated claims, you should return False. Otherwise, you should return True. Only return False or True, no other text."
    user_prompt = f"<article>{body}</article>\n<proposed_title>{title}</proposed_title>"
    messages: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await client.chat.completions.create(
            model=BASE_MODEL,
            messages=messages,
            max_tokens=5,  # Should only need 1 token for True/False
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content:
            # Be robust to variations like "True.", " False", etc.
            content_cleaned = content.strip().lower()
            if content_cleaned.startswith("true"):
                return 1
            elif content_cleaned.startswith("false"):
                return 0
            else:
                print(
                    f"Warning: Unexpected validation response: '{content}'. Defaulting to mismatch (0)."
                )
                return 0
        else:
            print(f"Warning: Empty validation response. Defaulting to mismatch (0).")
            return 0
    except Exception as e:
        print(
            f"Error during title validation API call: {e}. Defaulting to mismatch (0)."
        )
        return 0


async def rollout(
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt: Iterable[ChatCompletionMessageParam],
    row: Dict[str, Any],
) -> art.Trajectory:
    """Generates a title, validates it, scores it, and returns a trajectory."""
    metrics = {}
    try:
        # 1. Generate Title
        chat_completion = await client.chat.completions.create(
            messages=prompt,
            model=model_name,
            max_tokens=MAX_COMPLETION_LENGTH,
            temperature=0.7,
        )
        choice = chat_completion.choices[0]
        generated_title = choice.message.content
        if not generated_title:
            print("Warning: Empty title generated.")
            generated_title = ""  # Assign empty string if None or empty

        metrics["length"] = len(generated_title)

        # 2. Validate Title against Body using the LLM itself
        title_matches = await check_title_matches_body(
            client, row["scraped_body"], generated_title
        )
        metrics["matches"] = title_matches

        # 3. Score Title using external RM
        row_with_title = {**row, "title": generated_title}
        rm_score = await call_score_title(row_with_title)
        metrics["rm"] = rm_score

        # 4. Calculate Final Reward
        # If the title doesn't match the body (makes unsubstantiated claims), reward is 0
        final_reward = 0.0 if title_matches == 0 else rm_score

        # Ensure messages_and_choices includes the actual response choice
        messages_and_choices = [*prompt, choice]

        return art.Trajectory(
            messages_and_choices=messages_and_choices,
            reward=final_reward,
            metrics=metrics,
        )

    except Exception as e:
        print(f"Error during rollout: {e}.")
        return art.Trajectory(
            messages_and_choices=[*prompt],
            reward=0.0,
            metrics=metrics,
        )


# --- Main Training Loop ---
async def main():
    # Initialize ART API and Model
    api = art.UnslothAPI(wandb_project=WANDB_PROJECT)
    model = await api.get_or_create_model(
        name=RUN_NAME,
        base_model=BASE_MODEL,
    )

    # Load Data
    print("Loading training data...")
    train_dataset = await load_data(
        split="train",
        max_items=TRAINING_DATASET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
        tokenizer_name=BASE_MODEL,
    )
    print("Loading validation data...")
    val_dataset = await load_data(
        split="val",
        max_items=VAL_SET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
        tokenizer_name=BASE_MODEL,
    )

    if not train_dataset or not val_dataset:
        raise ValueError("Failed to load datasets. Exiting.")

    val_data_list: List[Dict[str, Any]] = list(val_dataset)
    train_data_list: List[Dict[str, Any]] = list(train_dataset)

    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")

    # Get OpenAI Client from ART Model
    openai_client = await model.openai_client()

    # Training Loop
    start_iteration = await model.get_iteration()
    print(f"Starting training from iteration {start_iteration}")

    for epoch in range(NUM_EPOCHS):
        random.shuffle(train_data_list)

        for i in tqdm(
            range(len(train_dataset) // ENTRIES_PER_ITERATION),
            desc="Training Iterations",
        ):
            batch_inputs = train_data_list[
                i * ENTRIES_PER_ITERATION : (i + 1) * ENTRIES_PER_ITERATION
            ]

            train_groups = await art.gather_trajectories(
                (
                    (
                        rollout(openai_client, RUN_NAME, bi["prompt"], bi["row"])
                        for _ in range(NUM_GENERATIONS)
                    )
                    for bi in batch_inputs
                )
            )

            await model.tune(
                train_groups,
                config=art.TuneConfig(
                    lr=LEARNING_RATE,
                    sequence_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
                ),
            )
            print("Tune step finished.")

            # --- Evaluation Step ---
            if i % EVAL_STEPS == 0:
                print(f"\n--- Evaluating at Iteration {i} ---")
                val_rewards = []
                val_metrics_agg: Dict[str, List[float]] = {
                    "length": [],
                    "matches": [],
                    "rm": [],
                }
                val_samples_for_log = []

                # Sample validation data (or use all if VAL_SET_SIZE is small)
                val_batch_indices = np.random.choice(
                    len(val_data_list),
                    min(VAL_SET_SIZE, len(val_data_list)),
                    replace=False,
                )
                val_batch = [val_data_list[i] for i in val_batch_indices]

                print(f"Running validation rollouts on {len(val_batch)} samples...")
                val_rollout_tasks = [
                    rollout(openai_client, RUN_NAME, item["prompt"], item["row"])
                    for item in val_batch
                ]
                val_trajectories = await tqdm.gather(
                    *val_rollout_tasks, desc="Validation Rollouts"
                )

                valid_val_trajectories = [
                    t for t in val_trajectories if t and "error" not in t.metrics
                ]

                if not valid_val_trajectories:
                    print(
                        "Warning: No valid validation trajectories generated. Skipping eval logging."
                    )
                    continue

                print(f"Processing {len(valid_val_trajectories)} validation results...")
                for i, traj in enumerate(valid_val_trajectories):
                    val_rewards.append(traj.reward)
                    for key in val_metrics_agg.keys():
                        if key in traj.metrics:
                            val_metrics_agg[key].append(traj.metrics[key])

                    # Log samples
                    if i < VAL_SAMPLES_TO_LOG:
                        # Extract user prompt content and assistant response
                        user_prompt_content = next(
                            (
                                m["content"]
                                for m in traj.messages_and_choices
                                if m["role"] == "user"
                            ),
                            "N/A",
                        )
                        assistant_response = next(
                            (
                                m.message.content
                                for m in traj.messages_and_choices
                                if hasattr(m, "message")
                                and m.message.role == "assistant"
                            ),
                            "N/A",
                        )

                        val_samples_for_log.append(
                            [
                                i,
                                user_prompt_content,  # Or format the whole prompt list if needed
                                assistant_response,
                                traj.reward,
                                str(traj.metrics),
                            ]
                        )

                # Log average metrics to WandB
                log_dict: Dict[str, Any] = {
                    "eval/iteration": i,
                    "eval/avg_reward": np.mean(val_rewards) if val_rewards else 0,
                    "eval/reward_std": np.std(val_rewards) if val_rewards else 0,
                    "eval/reward_median": np.median(val_rewards) if val_rewards else 0,
                }
                for key, values in val_metrics_agg.items():
                    if values:
                        log_dict[f"eval/avg_{key}"] = np.mean(values)
                        log_dict[f"eval/std_{key}"] = np.std(values)
                        log_dict[f"eval/median_{key}"] = np.median(values)
                    else:
                        log_dict[f"eval/avg_{key}"] = 0
                        log_dict[f"eval/std_{key}"] = 0
                        log_dict[f"eval/median_{key}"] = 0

                # Log validation samples table
                if val_samples_for_log:
                    columns = ["iteration", "prompt", "completion", "reward", "metrics"]
                    val_table = wandb.Table(columns=columns, data=val_samples_for_log)
                    log_dict["eval/validation_samples"] = val_table

                wandb.log(log_dict)
                print("Evaluation results logged to WandB.")

    print("Training finished.")
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
