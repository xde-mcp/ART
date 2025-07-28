import art
from art_e.rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
import polars as pl
from tqdm.asyncio import tqdm
from art_e.project_types import ProjectPolicyConfig


async def benchmark_model(
    model: art.Model,
    limit: int = 100,
    logging_split: str = "val",
) -> pl.DataFrame:
    val_scenarios = load_synthetic_queries(split="test", limit=limit)

    val_trajectories = await tqdm.gather(
        *(rollout(model, scenario) for scenario in val_scenarios),
        desc=f"validation {model.name}",
    )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

    if model._backend is not None:
        await model.log(valid_trajectories, split=logging_split)

    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))

    return avg_metrics
