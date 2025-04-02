import art
from dotenv import load_dotenv
import openai
import asyncio

load_dotenv()


MODEL_NAME = "yes-or-no-unsloth-001"


async def rollout(client: openai.AsyncOpenAI, prompt: str) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    chat_completion = await client.chat.completions.create(
        messages=messages, model=MODEL_NAME, max_tokens=100
    )
    choice = chat_completion.choices[0]
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


async def main():
    api = art.UnslothAPI(wandb_project="agent-reinforcement-training")
    model = await api.get_or_create_model(
        name=MODEL_NAME, base_model="Qwen/Qwen2.5-7B-Instruct"
    )

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

    openai_client = await model.openai_client()
    for _ in range(await model.get_iteration(), 1_000):
        train_groups = await art.gather_trajectories(
            ((rollout(openai_client, prompt) for _ in range(32)) for prompt in prompts),
            pbar_desc="train",
            stream_chat_completions=8,
        )
        await model.tune(
            train_groups,
            config=art.TuneConfig(lr=1e-4, sequence_length=32768),
        )


if __name__ == "__main__":
    asyncio.run(main())
