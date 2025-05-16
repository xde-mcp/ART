import asyncio
import pydantic
import traceback
from typing import Awaitable, Any, cast, Iterable, Iterator, overload, Callable
from openai.types.chat.chat_completion import Choice
from .types import Messages, MessagesAndChoices, Tools
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from datetime import datetime


MetadataValue = float | int | str | bool | None


class PydanticException(pydantic.BaseModel):
    type: str
    message: str
    traceback: str


class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    tools: Tools | None = None
    reward: float
    metrics: dict[str, float | int | bool] = {}
    metadata: dict[str, MetadataValue] = {}
    logs: list[str] = []
    start_time: datetime = pydantic.Field(default_factory=datetime.now, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.start_time = datetime.now()

    def finish(self) -> "Trajectory":
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics["duration"] = duration
        return self

    @asynccontextmanager
    async def track_duration(self, metric_name: str) -> AsyncGenerator[None, None]:
        start_time = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            metric_key = f"{metric_name}_duration"
            self.metrics[metric_key] = self.metrics.get(metric_key, 0.0) + duration

    def __str__(self) -> str:
        return f"Trajectory(reward={self.reward}, metrics={self.metrics}, metadata={self.metadata})"

    def messages(self) -> Messages:
        return [
            (
                {
                    "role": "assistant",
                    "content": message_or_choice.message.content,
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": tool_call.type,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in message_or_choice.message.tool_calls
                            ]
                        }
                        if message_or_choice.message.tool_calls
                        else {}
                    ),  # type: ignore
                }
                if isinstance(message_or_choice, Choice)
                else message_or_choice
            )
            for message_or_choice in self.messages_and_choices
        ]

    # Used for logging to console
    def for_logging(self) -> dict[str, Any]:
        loggable_dict = {
            "reward": self.reward,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": [],
            "tools": self.tools,
            "logs": self.logs,
        }
        for message_or_choice in self.messages_and_choices:
            trainable = isinstance(message_or_choice, Choice)
            message = (
                message_or_choice.message.to_dict() if trainable else message_or_choice
            )
            loggable_dict["messages"].append({**message, "trainable": trainable})
        return loggable_dict


class TrajectoryGroup(pydantic.BaseModel):
    trajectories: list[Trajectory]
    metadata: dict[str, MetadataValue] = {}
    exceptions: list[PydanticException] = []

    def __init__(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> None:
        super().__init__(
            trajectories=[
                trajectory
                for trajectory in trajectories
                if isinstance(trajectory, Trajectory)
            ]
            or getattr(self, "trajectories", []),
            metadata=metadata,
            exceptions=[
                PydanticException(
                    type=str(type(exception)),
                    message=str(exception),
                    traceback="\n".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    ),
                )
                for exception in (
                    [
                        exception
                        for exception in trajectories
                        if isinstance(exception, BaseException)
                    ]
                    + exceptions
                )
            ],
        )

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException],
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup": ...

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Awaitable[Trajectory]],
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> Awaitable["TrajectoryGroup"]: ...

    def __new__(
        cls,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup | Awaitable[TrajectoryGroup]":
        ts = list(trajectories)
        if any(hasattr(t, "__await__") for t in ts):

            async def _(exceptions: list[BaseException]):
                from .gather import get_gather_context, record_metrics

                context = get_gather_context()
                trajectories = []
                for future in asyncio.as_completed(
                    cast(list[Awaitable[Trajectory]], ts)
                ):
                    try:
                        trajectory = await future
                        trajectories.append(trajectory)
                        record_metrics(context, trajectory)
                        context.update_pbar(n=1)
                    except BaseException as e:
                        exceptions.append(e)
                        context.metric_sums["exceptions"] += 1
                        context.update_pbar(n=0)
                        if context.too_many_exceptions():
                            raise
                return TrajectoryGroup(
                    trajectories=trajectories,
                    exceptions=exceptions,
                    metadata=metadata,
                )

            class CoroutineWithMetadata:
                def __init__(self, coro, num_trajectories):
                    self.coro = coro
                    self._num_trajectories = num_trajectories

                def __await__(self):
                    return self.coro.__await__()

            coro = _(exceptions.copy())
            return CoroutineWithMetadata(coro, len(ts))
        else:
            group = super().__new__(cls)
            group.__init__(
                trajectories=cast(list[Trajectory | BaseException], ts),
                metadata=metadata,
                exceptions=exceptions,
            )
            return group

    @staticmethod
    async def from_rollout(
        rollout: Callable[[], Awaitable[Trajectory]],
        group_size: int,
        max_exceptions: int = 0,
    ) -> "TrajectoryGroup":
        """
        Execute `rollout` `group_size` times concurrently and collect the
        resulting `Trajectory` objects. If a rollout raises an exception it is
        recorded and a new rollout is started until `group_size` successful
        trajectories have been gathered. If the number of exceptions ever
        exceeds `max_exceptions`, the most recent exception is re-raised.
        """
        successes: list[Trajectory] = []
        exceptions: list[BaseException] = []

        # Kick off the initial batch of rollout tasks.
        pending: set[asyncio.Task[Trajectory]] = {
            asyncio.create_task(rollout()) for _ in range(group_size)
        }

        try:
            while len(successes) < group_size:
                # Wait for at least one task to finish.
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    try:
                        successes.append(task.result())
                    except BaseException as e:
                        exceptions.append(e)
                        if len(exceptions) > max_exceptions:
                            # Cancel any ongoing tasks before propagating error.
                            for t in pending:
                                t.cancel()
                            raise

                # Maintain the concurrency level by launching replacement tasks
                # so that we eventually reach the desired number of successes.
                while len(pending) + len(successes) < group_size:
                    pending.add(asyncio.create_task(rollout()))

            # Enough successful trajectories collected; cancel remaining tasks.
            for task in pending:
                task.cancel()
        finally:
            # Await all pending tasks (even if cancelled) to silence warnings.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return TrajectoryGroup(
            trajectories=successes,
            exceptions=exceptions,
        )
