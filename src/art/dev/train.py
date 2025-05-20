from typing_extensions import TypedDict


class TrainConfig(TypedDict, total=False):
    clip_epsilon_low: float
    clip_epsilon_high: float
    logprob_calculation_chunk_size: int
