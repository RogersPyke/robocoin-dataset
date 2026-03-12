import asyncio
import json
import logging
import socket
import traceback
from abc import ABC, abstractmethod

from websockets.exceptions import ConnectionClosed
from websockets.legacy.client import connect

from .constant import (
    CLIENT_ID,
    CLIENT_IP,
    DATASET_UUID,
    ERR_MSG,
    ERROR,
    ERROR_MSG,
    MSG_CONTENT,
    MSG_TYPE,
    NO_TASK,
    PING,
    PONG,
    REGISTER,
    REGISTERED,
    REQUEST_TASK,
    TASK,
    TASK_CATEGORY,
    TASK_CONTENT,
    TASK_FAILED,
    TASK_ID,
    TASK_RESULT,
    TASK_RESULT_CONTENT,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)


def get_client_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        raise e


class TaskClient(ABC):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        heartbeat_interval: float = 10.0,
        request_task_timeout: float | None = 15.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.server_uri = server_uri
        self.heartbeat_interval = heartbeat_interval
        self.websocket = None
        self.connected = False
        self.client_id: str | None = None
        self.local_ip: str = get_client_ip()

        self._heartbeat_task: asyncio.Task | None = None
        self._receiver_task: asyncio.Task | None = None
        self._response_future: asyncio.Future | None = None  # May be None
        # None or <= 0 means: wait indefinitely for task response
        self.request_task_timeout: float | None = request_task_timeout
        self.logger = logger
        # Retry config for submit_result so failed tasks get marked FAILED on server (backward compatible)
        self._submit_result_max_attempts: int = 4
        self._submit_result_retry_delay: float = 1.0

    async def connect_to_server(self, max_retries: int = 5, delay: float = 3.0) -> None:
        for attempt in range(max_retries):
            try:
                if self.logger:
                    self.logger.debug(f"[CONNECT_ATTEMPT] Connecting to {self.server_uri} (attempt {attempt + 1}/{max_retries})")
                self.websocket = await connect(self.server_uri)
                self.connected = True
                if self.logger:
                    self.logger.info(f"[CONNECT_SUCCESS] Connected to server: {self.server_uri}")
                return
            except Exception as e:  # noqa: PERF203
                error_type = type(e).__name__
                error_detail = str(e)
                if self.logger:
                    self.logger.warning(
                        f"[CONNECT_FAILED] Attempt {attempt + 1}/{max_retries} failed | "
                        f"Error Type: {error_type} | Detail: {error_detail} | "
                        f"Server URI: {self.server_uri}"
                    )
                if attempt < max_retries - 1:
                    if self.logger:
                        self.logger.debug(f"[CONNECT_RETRY] Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    error_msg = (
                        f"[CONNECT_FATAL] All {max_retries} retries failed, unable to connect to server\n"
                        f"  Server URI: {self.server_uri}\n"
                        f"  Last Error: {error_type}: {error_detail}\n"
                        f"  Diagnosis Hints:\n"
                        f"    - Ensure server is running on {self.server_uri}\n"
                        f"    - Check firewall/network connectivity\n"
                        f"    - Verify host and port are correct"
                    )
                    if self.logger:
                        self.logger.error(error_msg)
                    raise ConnectionError(error_msg)

    async def _start_heartbeat(self) -> None:
        async def send_ping() -> None:
            try:
                while True:
                    if self.connected and self.websocket and not self.websocket.closed:
                        try:
                            await self.websocket.send(json.dumps({MSG_TYPE: PING}))
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Heartbeat sending failed: {e}")
                            break
                    await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Heartbeat task exception: {e}")

        self._heartbeat_task = asyncio.create_task(send_ping())

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _message_receiver(self) -> None:
        try:
            async for message in self.websocket:
                try:
                    msg = json.loads(message)
                    msg_type = msg.get(MSG_TYPE)
                    if self.logger:
                        self.logger.debug(f"[MSG_RECV] Message received: {msg}")
                    msg_content = msg.get(MSG_CONTENT, {})

                    if msg_type == REGISTERED:
                        self.client_id = str(msg_content[CLIENT_ID])
                        if self.logger:
                            self.logger.info(f"Registered | Client ID: {self.client_id}")
                        # Wake up the waiting registration future
                        if self._response_future is not None and not self._response_future.done():
                            self._response_future.set_result(msg)

                    elif msg_type == PING:
                        await self.websocket.send(json.dumps({MSG_TYPE: PONG}))
                        if self.logger:
                            self.logger.debug("[PING_REPLY] Client reply PONG")

                    elif msg_type == PONG:
                        # Server heartbeat response acknowledged
                        if self.logger:
                            self.logger.debug("[PONG_RECV] Server heartbeat response received")

                    elif msg_type in (TASK, NO_TASK):
                        if self._response_future is not None and not self._response_future.done():
                            self._response_future.set_result(msg)
                        else:
                            if self.logger:
                                self.logger.warning(
                                    f"[WARN] Received task response but no pending request: {msg_type}"
                                )

                    # elif msg_type == ACK:
                    #     self._logger.info(f"[ACK] Server acknowledged: {msg}")

                    elif msg_type == ERROR:
                        if self.logger:
                            self.logger.error(f"[ERROR] Server error: {msg.get(ERROR_MSG)}")

                    else:
                        if self.logger:
                            self.logger.debug(f"[MSG_UNKNOWN] Unknown message type: {msg_type}")

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to process message: {e}")

        except ConnectionClosed as e:
            if self.logger:
                self.logger.warning(f"[CONN_CLOSED] The connection to the server has been closed: {e}")
            self.connected = False
            if self._response_future is not None and not self._response_future.done():
                self._response_future.set_exception(e)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Message receiving exception: {e}")
            self.connected = False
            if self._response_future is not None and not self._response_future.done():
                self._response_future.set_exception(e)

    async def register(self) -> bool:
        if not self.connected:
            await self.connect_to_server()

        if self.client_id:
            if self.logger:
                self.logger.info("[INFO] Client already registered, skipping")
            return True

        # Use Future to wait for registration response
        old_future = self._response_future
        if old_future is not None and not old_future.done():
            old_future.cancel()  # Cancel the old one

        self._response_future = asyncio.Future()
        try:
            register_msg = {
                MSG_TYPE: REGISTER,
                MSG_CONTENT: {
                    CLIENT_IP: get_client_ip(),
                    TASK_CATEGORY: self.get_task_category(),
                },
            }
            await self.websocket.send(json.dumps(register_msg))
            if self.logger:
                self.logger.info(f"[REGISTER_SENT] Registration request sent | IP: {self.local_ip}")

            try:
                result = await asyncio.wait_for(self._response_future, timeout=10.0)
                return result is not None
            except asyncio.TimeoutError:
                if self.logger:
                    self.logger.error(
                        "[ERROR] Registration timeout, no response received from the server"
                    )
                if self._response_future and not self._response_future.done():
                    self._response_future.cancel()
                return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Registration failed: {e}")
            self.connected = False
            return False
        # Do not set to None here, handled by cleanup uniformly

    @abstractmethod
    def get_task_category(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_task_request_desc(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _sync_process_task(self, task: dict) -> dict:
        raise NotImplementedError

    async def request_task(self) -> dict | None:
        if not self.client_id:
            if self.logger:
                self.logger.warning("[WARN] Registration failed")
            return None

        # Check and cancel old future (if exists)
        if self._response_future is not None and not self._response_future.done():
            if self.logger:
                self.logger.warning("[WARN] There are unfinished task requests, canceling old request")
            self._response_future.set_exception(
                RuntimeError("Old request overridden by new request")
            )

        self._response_future = asyncio.Future()

        try:
            task_request_desc = self.generate_task_request_desc()
            task_request_dict = {
                MSG_TYPE: REQUEST_TASK,
                MSG_CONTENT: {
                    TASK_CATEGORY: self.get_task_category(),
                    TASK_CONTENT: task_request_desc,
                },
            }
            await self.websocket.send(json.dumps(task_request_dict))
            if self.logger:
                self.logger.debug("Task request sent")

            # Wait for server to respond with either TASK or NO_TASK.
            # If `request_task_timeout` is None or <= 0, wait indefinitely and
            # rely on heartbeat / connection errors to break the loop.
            # Otherwise, use asyncio.wait_for with the configured timeout.
            if self.request_task_timeout is not None and self.request_task_timeout > 0:
                try:
                    task_msg = await asyncio.wait_for(
                        self._response_future,
                        timeout=self.request_task_timeout,
                    )
                except asyncio.TimeoutError:
                    if self.logger:
                        self.logger.warning("Request task timeout")
                    if self._response_future is not None and not self._response_future.done():
                        self._response_future.cancel()
                    return None
            else:
                task_msg = await self._response_future

            # Parse response
            msg_type = task_msg.get(MSG_TYPE)
            if msg_type == TASK:
                task_content = task_msg[MSG_CONTENT]
                task_id = task_msg.get(TASK_ID, "unknown")
                dataset_uuid = task_content.get("dataset_uuid", "")
                hardlink_path = task_content.get("leformat_path") or task_content.get("hardlink_path") or ""
                if self.logger:
                    if dataset_uuid or hardlink_path:
                        self.logger.info(f"Task received | Task: {task_id} | UUID: {dataset_uuid} | Path: {hardlink_path}")
                    else:
                        self.logger.info(f"Task received | Task: {task_id}")
                return task_content

            if msg_type == NO_TASK:
                if self.logger:
                    self.logger.debug("No task available")
                return None

            if self.logger:
                self.logger.warning(f"[WARN] Unexpected response type: {msg_type}")
            return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"Request task failed: {e}")
            return None

    async def submit_result(self, result: dict) -> None:
        if not self.client_id:
            if self.logger:
                self.logger.error("[ERROR] Not registered, cannot submit result")
            return

        last_exc = None
        for attempt in range(self._submit_result_max_attempts):
            try:
                if self.websocket and not self.websocket.closed:
                    await self.websocket.send(json.dumps(result))
                    if self.logger:
                        self.logger.info(
                            "[RESULT_SUBMIT] Task result submitted, task_id=%s",
                            result.get(TASK_ID),
                        )
                    return
            except Exception as e:
                last_exc = e
                if self.logger:
                    self.logger.warning(
                        "[submit_result] Send failed (attempt %d/%d): %s",
                        attempt + 1,
                        self._submit_result_max_attempts,
                        e,
                    )
                if attempt < self._submit_result_max_attempts - 1:
                    await asyncio.sleep(self._submit_result_retry_delay)
        if self.logger and last_exc is not None:
            self.logger.error(
                "[submit_result] Could not send result after %d attempts; "
                "database may still show PROCESSING. Last error: %s",
                self._submit_result_max_attempts,
                last_exc,
            )

    async def process_task(self, task_data: dict) -> dict:
        loop = asyncio.get_event_loop()
        task_id = task_data.get(TASK_ID)
        try:
            task_result_content = await loop.run_in_executor(
                None, self._sync_process_task, task_data
            )
            return {TASK_RESULT_STATUS: TASK_SUCCESS, TASK_RESULT_CONTENT: task_result_content}
        except Exception:
            err_detail = f"Task {task_id} failed.\n{traceback.format_exc()}"
            if self.logger:
                self.logger.error("[process_task] %s", err_detail, exc_info=True)
            failure_result = {
                TASK_RESULT_STATUS: TASK_FAILED,
                ERR_MSG: err_detail,
                TASK_RESULT_CONTENT: {},
            }
            # Include DATASET_UUID so server can update status when task_content is missing (e.g. restart)
            if task_data.get(DATASET_UUID) is not None:
                failure_result[DATASET_UUID] = task_data.get(DATASET_UUID)
            return failure_result


    async def run_until_no_task(self) -> None:
        try:
            if not self.connected:
                await self.connect_to_server()

                # Start receiver and heartbeat
                self._receiver_task = asyncio.create_task(self._message_receiver())
                await self.register()
                if not self.client_id:
                    if self.logger:
                        self.logger.error("[ERROR] Registration failed, exiting")
                    return

                await self._start_heartbeat()
                if self.logger:
                    self.logger.info(f"[SUCCESS] Client {self.client_id} is ready, starting task loop")

            while True:
                task = await self.request_task()
                if task is None:
                    if self.logger:
                        self.logger.info("[EXIT] No task from server, client exiting")
                    break

                if self.logger:
                    self.logger.info(f"[START] Starting to process task: {task.get(TASK_ID)}")
                result_content = await self.process_task(task)
                result = {
                    MSG_TYPE: TASK_RESULT,
                    MSG_CONTENT: result_content,
                }
                result[TASK_ID] = task.get(TASK_ID)
                result[CLIENT_ID] = self.client_id
                await self.submit_result(result)
                if self.logger:
                    self.logger.info("[NEXT] Task result submitted, preparing to request next task...")

        except Exception as e:
            error_type = type(e).__name__
            if self.logger:
                self.logger.error(
                    f"[CLIENT_RUNTIME_ERROR] {error_type}: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Unified resource cleanup"""
        self.connected = False

        # Clean up response_future
        if self._response_future is not None and not self._response_future.done():
            self._response_future.set_exception(asyncio.CancelledError())
        self._response_future = None

        # Cancel receiver task
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        # Stop heartbeat
        await self._stop_heartbeat()

        # Close websocket
        if self.websocket:
            try:
                await self.websocket.close()
                if self.logger:
                    self.logger.info("[CLOSED] Client connection closed")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to close connection: {e}")

    async def run(self) -> None:
        try:
            await self.run_until_no_task()
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("[INTERRUPT] Client interrupted by user")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Client terminated abnormally: {e}")
        finally:
            await self._cleanup()
