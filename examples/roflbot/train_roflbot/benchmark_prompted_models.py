# To run:
# uv run scripts/benchmark_prompted_models.py

import art
import asyncio
import polars as pl
from art.local import LocalBackend
from dotenv import load_dotenv
from project_types import PolicyConfig
from benchmark import benchmark_model
import os

load_dotenv()

MODELS_TO_BENCHMARK = [
    # ("gpt-4o", "openai/gpt-4o"),
    ("gpt-4.1", "openai/gpt-4.1"),
    ("qwen3-32b", "openrouter/qwen/qwen3-32b"),
    ("o4-mini", "openai/o4-mini"),
    ("o3", "openai/o3"),
    ("gemini-2.0-flash", "gemini/gemini-2.0-flash"),
    ("gemini-2.5-pro", "gemini/gemini-2.5-pro-preview-03-25"),
    ("qwen3-235b-a22b", "openrouter/qwen/qwen3-235b-a22b"),
    # ("deepseek-r1", "together_ai/deepseek-ai/DeepSeek-R1"),
]

TEST_SET_ENTRIES = 100
LOG_TO_LANGFUSE = False


async def main():
    backend = LocalBackend()
    models = []
    for model_name, model_id in MODELS_TO_BENCHMARK:
        model = art.Model(
            name=model_name,
            project="roflbot",
            inference_model_name=model_id,
            config=PolicyConfig(
                log_to_langfuse=LOG_TO_LANGFUSE,
                max_tokens=16000,
            ),
        )
        await model.register(backend)
        models.append(model)

    results = await asyncio.gather(
        *[
            benchmark_model(model, TEST_SET_ENTRIES, swallow_exceptions=False)
            for model in models
        ]
    )
    for model in models:
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["REMOTE_BUCKET"],
            prefix="roflbot",
        )

    df: pl.DataFrame = pl.concat(results, how="diagonal")
    df = df.transpose(include_header=True)

    col_names = {"column": "metric"}
    for i, model in enumerate(MODELS_TO_BENCHMARK):
        col_names[f"column_{i}"] = model[0]

    df = df.rename(col_names)
    with open("data/benchmark_prompted_models.html", "w") as f:
        f.write(df.to_pandas().to_html())

    print("View the results in data/benchmark_prompted_models.html")
    print(df.to_pandas().to_markdown())


asyncio.run(main())
