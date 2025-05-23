import os
import art
import asyncio
from art.local import LocalBackend



async def main():
    tools: art.Tools = [
        {
            "type": "function",
            "function": {
                "name": "play_move",
                "description": "Play a move in rock-paper-scissors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "string",
                            "enum": ["rock", "paper", "scissors"],
                            "description": "The move to play",
                        }
                    },
                    "required": ["move"],
                },
            },
        }
    ]

    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Current folder path: {current_folder_path}")

    model = art.TrainableModel(
        name="logprob-check-14b",
        project="logprob_check",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        _internal_config={
            "engine_args": {
                "generation_config": current_folder_path,
            }
        }
    )

    with LocalBackend() as backend:
        await model.register(backend)
        client = model.openai_client()
        response = await client.chat.completions.create(
            model=model.name,
            messages=[{"role": "system", "content": "You are a rock-paper-scissors playing agent. Use the play_move function tool to declare your moves."},
                      {"role": "user", "content": "What will your first move be?"}],
            temperature=0.8,
            logprobs=True,
            tools=tools,
        )
        print(response)

        logprobs = response.choices[0].logprobs.content
        for token in logprobs:
            print(f"{token.token}- {token.logprob}")

if __name__ == "__main__":
    asyncio.run(main())