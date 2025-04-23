import art
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel
import asyncio

from art.utils.litellm import convert_litellm_choice_to_openai

load_dotenv()


class ModelConfig(BaseModel):
    litellm_model_name: str | None


async def rollout(model: art.Model, prompt: str) -> art.Trajectory:
    assert isinstance(model.config, ModelConfig)

    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    model_id = (
        model.config.litellm_model_name
        if model.config.litellm_model_name
        else f"hosted_vllm/{model.name}"
    )
    chat_completion = await acompletion(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model_id,
        messages=messages,
        max_completion_tokens=100,
    )
    choice = convert_litellm_choice_to_openai(chat_completion.choices[0])
    content = choice.message.content
    assert isinstance(content, str)
    if content == "yes":
        reward = 0.5
    elif content == "no":
        reward = 0.75
    elif content == "maybe":
        reward = 1.0
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


prompts = [
    f"{prefix} with {', '.join([f"'{w}'" if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
    for prefix in ["respond", "just respond"]
    for use_quotes in [True, False]
    for words in [
        ["yes", "no", "maybe"],
        ["maybe", "yes", "no"],
        ["no", "yes", "maybe"],
        ["yes", "maybe", "no"],
        ["yes", "no"],
        ["maybe", "no"],
        ["no", "maybe"],
        ["no", "yes"],
        ["yes", "no"],
    ]
]


async def main():
    api = art.LocalAPI()

    model = art.TrainableModel(
        name="001",
        project="yes-no-maybe",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        config=ModelConfig(),
    )
    await model.register(api)

    gpt4o = art.Model(
        name="gpt4o",
        project="yes-no-maybe",
        base_model="gpt-4o",
        config=ModelConfig(litellm_model_name="openai/gpt-4o"),
    )
    await gpt4o.register(api)

    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, prompt) for _ in range(64))
                for prompt in prompts
            ),
            pbar_desc="gather",
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
        )

    val_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(gpt4o, prompt) for _ in range(64))
            for prompt in prompts
        ),
        pbar_desc="gather",
    )
    await gpt4o.log(val_groups, split="val")


if __name__ == "__main__":
    asyncio.run(main())
