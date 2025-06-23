import asyncio
import contextvars
import contextlib
from collections import Counter
from dataclasses import dataclass, field
from openai.types.chat.chat_completion import Choice
from tqdm import auto as tqdm
from typing import Awaitable, Iterable, Iterator, Literal, overload

from .trajectories import Trajectory, TrajectoryGroup


async def gather_trajectory_groups(
    groups: Iterable[Awaitable[TrajectoryGroup]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: int | float = 0,
    max_metrics: int | None = None,
) -> list[TrajectoryGroup]:
    groups = list(groups)
    context = GatherContext(
        pbar=None,
        pbar_total_completion_tokens=pbar_total_completion_tokens,
        max_exceptions=max_exceptions,
        max_metrics=max_metrics,
    )
    with set_gather_context(context):
        future = asyncio.gather(*[wrap_group_awaitable(g) for g in groups])
        total = sum(getattr(g, "_num_trajectories", 1) for g in groups)
        context.pbar = tqdm.tqdm(desc=pbar_desc, total=total)
        result_groups = await future
    if context.pbar is not None:
        context.pbar.close()
    return [g for g in result_groups if g is not None]


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Trajectory]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: Literal[0] = 0,
) -> list[Trajectory]: ...


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Trajectory]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: int | float,
) -> list[Trajectory | BaseException]: ...


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Iterable[Trajectory]]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: Literal[0] = 0,
) -> list[list[Trajectory]]: ...


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Iterable[Trajectory]]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: int | float,
) -> list[list[Trajectory] | BaseException]: ...


async def gather_trajectories(
    trajectories: (
        Iterable[Awaitable[Trajectory]] | Iterable[Awaitable[Iterable[Trajectory]]]
    ),
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = True,
    max_exceptions: int | float = 0,
) -> (
    list[Trajectory]
    | list[Trajectory | BaseException]
    | list[list[Trajectory]]
    | list[list[Trajectory] | BaseException]
):
    trajectories_list = list(trajectories)
    context = GatherContext(
        pbar=tqdm.tqdm(desc=pbar_desc, total=len(trajectories_list)),
        pbar_total_completion_tokens=pbar_total_completion_tokens,
        max_exceptions=max_exceptions,
    )
    with set_gather_context(context):
        results = await asyncio.gather(
            *[wrap_trajectories_awaitable(t) for t in trajectories_list]
        )
    if context.pbar is not None:
        context.pbar.close()
    return results  # type: ignore


async def wrap_group_awaitable(
    awaitable: Awaitable[TrajectoryGroup],
) -> TrajectoryGroup | None:
    if hasattr(awaitable, "_num_trajectories"):
        return await awaitable
    context = get_gather_context()
    try:
        group = await awaitable
        for trajectory in group:
            record_metrics(context, trajectory)
        context.update_pbar(n=len(group))
        return group
    except BaseException:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        if context.too_many_exceptions():
            raise


async def wrap_trajectories_awaitable(
    awaitable: Awaitable[Trajectory] | Awaitable[Iterable[Trajectory]],
) -> Trajectory | list[Trajectory] | BaseException:
    context = get_gather_context()
    try:
        result = await awaitable
        if isinstance(result, Trajectory):
            record_metrics(context, result)
            context.update_pbar(n=1)
            return result
        result = list(result)
        for trajectory in result:
            record_metrics(context, trajectory)
        context.update_pbar(n=1)
        return result
    except BaseException as e:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        if context.too_many_exceptions():
            raise
        else:
            return e


def record_metrics(context: "GatherContext", trajectory: Trajectory) -> None:
    logprobs = [
        message_or_choice.logprobs
        for message_or_choice in trajectory.messages_and_choices
        if isinstance(message_or_choice, Choice)
        if message_or_choice.logprobs
    ]
    if logprobs:
        trajectory.metrics["completion_tokens"] = sum(
            len(l.content or l.refusal or []) for l in logprobs
        ) / len(logprobs)
    context.metric_sums["reward"] += trajectory.reward  # type: ignore
    context.metric_divisors["reward"] += 1
    context.metric_sums.update(trajectory.metrics)
    context.metric_divisors.update(trajectory.metrics.keys())


@dataclass
class GatherContext:
    pbar: tqdm.tqdm | None = None
    metric_sums: Counter[str] = field(default_factory=Counter)
    metric_divisors: Counter[str] = field(default_factory=Counter)
    max_metrics: int | None = None
    pbar_total_completion_tokens: bool = False
    max_exceptions: int | float = 0
    increment_pbar: bool = True

    def update_pbar(self, n: int) -> None:
        if self.pbar is None:
            return
        if self.increment_pbar:
            self.pbar.update(n)
        postfix = {}
        included_metrics = self.metric_sums.keys()
        if self.max_metrics is not None:
            included_metrics = list(self.metric_sums.keys())[: self.max_metrics]
        for metric in included_metrics:
            sum_value = self.metric_sums[metric]
            divisor = max(1, self.metric_divisors[metric])
            avg_value = sum_value / divisor
            
            # Check if this metric appears to be boolean-like
            # (values close to 0 or 1, indicating it's likely a rate/percentage metric)
            if self._is_boolean_metric(metric, avg_value):
                # Format as percentage for better readability
                if avg_value >= 0.995:
                    postfix[metric] = "100%"
                elif avg_value <= 0.005:
                    postfix[metric] = "0%"
                else:
                    postfix[metric] = f"{avg_value:.1%}"
            else:
                # For non-boolean metrics, keep the original formatting
                postfix[metric] = avg_value
        
        # move token metrics to the end
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_completion_tokens",
        ):
            if key in postfix:
                postfix[key] = postfix.pop(key)
        self.pbar.set_postfix(postfix)
    
    def _is_boolean_metric(self, metric_name: str, avg_value: float) -> bool:
        """
        Determine if a metric should be displayed as a boolean/percentage.
        
        This heuristic identifies metrics that are likely boolean-based by:
        1. Checking common boolean metric names
        2. Checking if the average value is between 0 and 1 (inclusive)
        """
        # Common boolean metric patterns
        boolean_patterns = [
            "win", "loss", "invalid", "error", "success", "fail", "correct", "wrong",
            "match", "hit", "miss", "valid", "timeout", "exception"
        ]
        
        # Check if metric name contains boolean patterns
        metric_lower = metric_name.lower()
        has_boolean_pattern = any(pattern in metric_lower for pattern in boolean_patterns)
        
        # Check if value is in [0, 1] range (typical for rates/percentages)
        is_rate_like = 0 <= avg_value <= 1
        
        # For now, be conservative and only format as percentage if:
        # 1. It has a boolean pattern in the name, OR
        # 2. It's clearly a rate (between 0 and 1) AND not a token count metric
        is_token_metric = any(token in metric_lower for token in ["token", "prompt", "completion"])
        
        return (has_boolean_pattern or (is_rate_like and not is_token_metric))

    def too_many_exceptions(self) -> bool:
        if (
            0 < self.max_exceptions < 1
            and self.pbar is not None
            and self.metric_sums["exceptions"] / self.pbar.total <= self.max_exceptions
        ) or self.metric_sums["exceptions"] <= self.max_exceptions:
            return False
        return True

    def reset(self) -> None:
        self.pbar = None
        self.metric_sums = Counter()
        self.metric_divisors = Counter()
        self.pbar_total_completion_tokens = False
        self.max_exceptions = 0


gather_context_var = contextvars.ContextVar("gather_context", default=GatherContext())


@contextlib.contextmanager
def set_gather_context(context: GatherContext) -> Iterator[None]:
    token = gather_context_var.set(context)
    try:
        yield
    finally:
        gather_context_var.reset(token)


def get_gather_context() -> GatherContext:
    return gather_context_var.get()
