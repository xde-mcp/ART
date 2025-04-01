import asyncio
import httpx
from openai import AsyncOpenAI
import os

from .config.openai_server import OpenAIServerConfig
from .model import Model
from .types import BaseModel, Trajectory, TuneConfig, Verbosity


class API:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
    ) -> None:
        """
        Initializes an Agent Reinforcement Trainer (ART) API interface.

        Args:
            api_key: The API key to use. Will check the `ART_API_KEY` and
                `OPENPIPE_API_KEY` environment variables if not provided.
            base_url: The base API URL. Defaults to OpenPipe's ART API. Will check
                the `ART_BASE_URL` environment variable if not provided.

        Example:
            ```python
            import art

            api = art.API()
            model = await api.get_or_create_model(
                name="my-model",
                base_model="Qwen/Qwen2.5-7B-Instruct",
            )
            ```
        """
        if api_key is None:
            api_key = os.environ.get("ART_API_KEY")
        if api_key is None:
            api_key = os.environ.get("OPENPIPE_API_KEY")
        assert (
            api_key is not None
        ), "The api_key option must be set either by passing api_key to the API constructor or by setting the ART_API_KEY or OPENPIPE_API_KEY environment variable"
        if base_url is None:
            base_url = os.environ.get("ART_BASE_URL")
        if base_url is None:
            base_url = f"https://api.openpipe.ai/art/v1"
        self._client = httpx.AsyncClient(base_url=base_url)

    async def get_or_create_model(self, name: str, base_model: BaseModel) -> Model:
        """
        Retrieves an existing model or creates a new one.

        Args:
            name: The model's name.
            base_model: The model's base model.

        Returns:
            Model: A model instance.
        """
        response = await self._client.post(
            "/models",
            json={"name": name, "base_model": base_model},
        )
        response.raise_for_status()
        return Model(api=self, name=name, base_model=base_model)

    async def _get_iteration(self, model: Model) -> int:
        response = await self._client.get(
            f"/models/{model.name}/iteration",
        )
        response.raise_for_status()
        return response.json()["iteration"]

    async def _clear_iterations(
        self,
        model: Model,
        benchmark: str,
        benchmark_smoothing: float = 1.0,
        verbosity: Verbosity = 1,
    ) -> None:
        response = await self._client.post(
            f"/models/{model.name}/clear_iterations",
            json={"benchmark": benchmark, "benchmark_smoothing": benchmark_smoothing},
        )
        response.raise_for_status()

    async def _get_openai_client(
        self,
        model: Model,
        estimated_completion_tokens: int,
        tool_use: bool,
        verbosity: Verbosity,
        config: OpenAIServerConfig | None,
    ) -> tuple[AsyncOpenAI, asyncio.Semaphore]:
        response = await self._client.post(
            f"/openai_clients",
            json={
                "model": model.name,
                "estimated_completion_tokens": estimated_completion_tokens,
                "tool_use": tool_use,
            },
        )
        response.raise_for_status()
        data = response.json()
        return AsyncOpenAI(**data["client_kwargs"]), asyncio.Semaphore(
            data["max_concurrent_requests"]
        )

    async def _close_openai_client(self, client: AsyncOpenAI) -> None:
        response = await self._client.post(
            f"/openai_clients/close",
            json={
                "api_key": client.api_key,
                "organization": client.organization,
                "project": client.project,
                "base_url": client.base_url,
            },
        )
        response.raise_for_status()

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory | BaseException]],
        name: str,
    ) -> None:
        response = await self._client.post(
            "/evals",
            json={
                "model": model.name,
                "trajectory_groups": trajectory_groups,
                "name": name,
            },
        )
        response.raise_for_status()

    async def _tune_model(
        self,
        model: Model,
        trajectory_groups: list[list[Trajectory | BaseException]],
        config: TuneConfig,
    ) -> None:
        response = await self._client.post(
            "/models/tune",
            json={
                "model": model.name,
                "trajectory_groups": trajectory_groups,
            },
        )
        response.raise_for_status()
