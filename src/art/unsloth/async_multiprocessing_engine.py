import asyncio
import pickle
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import zmq.asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.multiprocessing import (
    IPC_DATA_EXT,
    IPC_HEALTH_EXT,
    IPC_INPUT_EXT,
    IPC_OUTPUT_EXT,
    RPCAbortRequest,
    RPCAdapterLoadedResponse,
    RPCError,
    # RPCIsSleepingRequest,
    # RPCIsSleepingResponse,
    RPCLoadAdapterRequest,
    RPCProcessRequest,
    RPCResetPrefixCacheRequest,
    RPCSleepRequest,
    RPCStartupRequest,
    RPCStartupResponse,
    RPCWakeUpRequest,
    VLLM_RPC_SUCCESS_STR,
)
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestOutput

logger = init_logger(__name__)

HEARTBEAT_INTERVAL = 10  # Seconds between heartbeat messages


class MQAsyncLLMEngine:
    """
    A multiprocessing wrapper for an existing AsyncLLMEngine, designed to be non-blocking.

    This class uses ZeroMQ with asyncio for inter-process communication, handling requests
    asynchronously and streaming outputs back to the client. It relies on an already
    initialized AsyncLLMEngine instance for processing requests.

    Args:
        ipc_path (str): Base path for ZeroMQ interprocess messaging.
        async_engine (AsyncLLMEngine): An already initialized AsyncLLMEngine instance.
    """

    def __init__(self, ipc_path: str, async_engine: AsyncLLMEngine) -> None:
        self.async_engine = async_engine
        self.ctx = zmq.asyncio.Context()

        # Set up ZeroMQ sockets
        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")
        self.output_socket = self.ctx.socket(zmq.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")
        self.heartbeat_socket = self.ctx.socket(zmq.PUSH)
        self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        self._errored_with: Optional[BaseException] = None
        self._running = False

    async def run(self) -> None:
        """
        Start the asynchronous engine loop, including startup, heartbeat, and request handling.
        """
        try:
            self._running = True
            # Start the AsyncLLMEngine's background loop
            self.async_engine.start_background_loop()

            # Run the startup loop
            await self.run_startup_loop()

            # Start the heartbeat task
            heartbeat_task = asyncio.create_task(self.run_heartbeat_loop())

            # Run the main engine loop
            await self.run_engine_loop()

            # Cleanup on normal exit
            heartbeat_task.cancel()
            await heartbeat_task
        except asyncio.CancelledError:
            logger.debug("MQAsyncLLMEngine terminated via cancellation.")
        except Exception as e:
            logger.exception(f"Error in MQAsyncLLMEngine: {repr(e)}")
            self._errored_with = e
        finally:
            await self.cleanup()
            self._running = False

    async def cleanup(self) -> None:
        """
        Shut down the AsyncLLMEngine's background loop and clean up ZeroMQ resources.
        """
        self.async_engine.shutdown_background_loop()
        self.ctx.destroy(linger=0)

    @asynccontextmanager
    async def make_data_socket(self):
        """
        Context manager for creating a temporary ROUTER socket for startup communication.
        """
        socket = self.ctx.socket(zmq.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def run_startup_loop(self) -> None:
        """
        Handle the initial startup request from the client.
        """
        async with self.make_data_socket() as socket:
            identity, message = await socket.recv_multipart()
            request = pickle.loads(message)
            if request == RPCStartupRequest.IS_SERVER_READY:
                tracing_enabled = await self.async_engine.is_tracing_enabled()
                response = RPCStartupResponse(tracing_enabled=tracing_enabled)
            else:
                response = ValueError("Unknown startup request")
            await socket.send_multipart([identity, pickle.dumps(response)])

    async def run_heartbeat_loop(self) -> None:
        """
        Periodically send heartbeat messages to indicate engine health.
        """
        while self._running:
            await self.send_heartbeat()
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def send_heartbeat(self) -> None:
        """
        Send a heartbeat message indicating the engine's health status.
        """
        if self._errored_with is not None:
            await self.heartbeat_socket.send_multipart(
                [pickle.dumps(self._errored_with)]
            )
        else:
            try:
                await self.async_engine.check_health()
                await self.heartbeat_socket.send_multipart(
                    [pickle.dumps(VLLM_RPC_SUCCESS_STR)]
                )
            except Exception as e:
                self._errored_with = e
                await self.heartbeat_socket.send_multipart([pickle.dumps(e)])

    async def run_engine_loop(self) -> None:
        """
        Main asynchronous loop to handle incoming requests.
        """
        while self._running:
            if await self.input_socket.poll(timeout=0):
                frames = await self.input_socket.recv_multipart()
                request = pickle.loads(frames[0])
                asyncio.create_task(self.handle_request(request))
            await asyncio.sleep(0.001)  # Yield control to other tasks

    async def handle_request(
        self,
        request: (
            RPCProcessRequest
            | RPCAbortRequest
            | RPCLoadAdapterRequest
            | RPCResetPrefixCacheRequest
            | RPCSleepRequest
            | RPCWakeUpRequest
        ),
    ) -> None:
        """
        Handle incoming requests by delegating to the AsyncLLMEngine.

        Args:
            request: The request object received from the input socket.
        """
        try:
            if isinstance(request, RPCProcessRequest):
                generator = await self.async_engine.add_request(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    params=request.params,
                    lora_request=request.lora_request,
                    trace_headers=request.trace_headers,
                    prompt_adapter_request=request.prompt_adapter_request,
                    priority=request.priority,
                )
                asyncio.create_task(self.consume_generator(generator))
            elif isinstance(request, RPCAbortRequest):
                await self.async_engine.abort(request.request_id)
                logger.info(f"Aborted request {request.request_id}.")
            elif isinstance(request, RPCLoadAdapterRequest):
                await self.async_engine.add_lora(request.lora_request)
                response = RPCAdapterLoadedResponse(request_id=request.request_id)
                await self.send_output(response)
            elif isinstance(request, RPCResetPrefixCacheRequest):
                await self.async_engine.reset_prefix_cache()
            elif isinstance(request, RPCSleepRequest):
                await self.async_engine.sleep(request.value)
            elif isinstance(request, RPCWakeUpRequest):
                await self.async_engine.wake_up()
            # elif isinstance(request, RPCIsSleepingRequest):
            #     is_sleeping = await self.async_engine.is_sleeping()
            #     response = RPCIsSleepingResponse(request_id=request.request_id, is_sleeping=is_sleeping)
            #     await self.send_output(response)
            else:
                raise ValueError(f"Unknown RPCRequest Type: {type(request)}")
        except Exception as e:
            logger.exception(f"Error handling request {request}: {e}")
            self._errored_with = e
            if hasattr(request, "request_id"):
                rpc_err = RPCError(
                    request_id=request.request_id,  # type: ignore
                    is_engine_errored=True,
                    exception=e,
                )
                await self.send_output(rpc_err)

    async def consume_generator(
        self,
        generator: AsyncGenerator[
            RequestOutput | PoolingRequestOutput[PoolingOutput], None
        ],
    ) -> None:
        """
        Consume the asynchronous generator returned by AsyncLLMEngine.add_request.

        Args:
            generator: Async generator yielding request outputs.
        """
        try:
            async for output in generator:
                await self.send_output([output])
        except Exception as e:
            logger.exception(f"Error consuming generator: {e}")
            self._errored_with = e

    async def send_output(
        self,
        output: (
            list[RequestOutput | PoolingRequestOutput[PoolingOutput]]
            | RPCAdapterLoadedResponse
            | RPCError
        ),
    ) -> None:
        """
        Send an output back to the client via the output socket.

        Args:
            output: The output object to send (e.g., RequestOutput, RPC response).
        """
        output_bytes = pickle.dumps(output)
        await self.output_socket.send_multipart([output_bytes])
