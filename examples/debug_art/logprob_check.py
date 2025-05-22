import art
import asyncio
from art.local import LocalBackend



async def main():
    model = art.TrainableModel(
        name="logprob-check-14b",
        project="logprob_check",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )

    with LocalBackend() as backend:
        await model.register(backend)
        client = model.openai_client()
        response = await client.chat.completions.create(
            model=model.name,
            messages=[{"role": "user", "content": "What is the capital of Germany?"}],
            temperature=0.0,
            logprobs=True,
        )
        print(response)

        logprobs = response.choices[0].logprobs.content
        for token in logprobs:
            print(f"{token.token}- {token.logprob}")

if __name__ == "__main__":
    asyncio.run(main())