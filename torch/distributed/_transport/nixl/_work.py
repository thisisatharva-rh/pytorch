from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work

from .._work import _validate_timeout


if TYPE_CHECKING:
    from ._memory import NIXLMemoryView, NIXLRemoteBuffer
    from ._transport import NIXLTransport


# Registered memory can still receive remote DMA after the last local transfer.
# Retain its owner until explicit unregister/close, not merely Work completion.
_live_transports: set[NIXLTransport] = set()


class _FutureProgress:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._works: dict[int, _PollingWork] = {}
        self._thread: threading.Thread | None = None

    def add(self, work: _PollingWork, loop: asyncio.AbstractEventLoop | None) -> None:
        with self._condition:
            if id(work) in self._works:
                return
            work._future_loop = loop
            self._works[id(work)] = work
            if self._thread is None:
                try:
                    self._thread = threading.Thread(
                        target=self._run, name="nixl-future-progress", daemon=True
                    )
                    self._thread.start()
                except Exception:
                    self._thread = None
                    del self._works[id(work)]
                    raise
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._works:
                    self._thread = None
                    return
                works = list(self._works.values())
            # Polling resolves futures and runs user callbacks. Neither the
            # progress lock nor the transport lock may be held for callbacks.
            completed = [id(work) for work in works if work._progress_future()]
            with self._condition:
                for key in completed:
                    self._works.pop(key, None)
                if self._works:
                    self._condition.wait(timeout=0.001)


class _PollingWork(Work):
    """Work that resolves a future from backend status checks.

    Subclasses implement a nonblocking, thread-safe ``_poll`` and record a
    terminal error in ``_error``. A failed status query must not report completion
    unless it establishes that the backend has stopped accessing memory.
    """

    def __init__(self, progress: _FutureProgress, timeout: float | None = None) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._timeout = timeout
        self._error: BaseException | None = None
        self._future: torch.futures.Future[Any] = torch.futures.Future()
        self._future_lock = threading.Lock()
        self._future_completed = False
        self._future_loop: asyncio.AbstractEventLoop | None = None
        self._future_poll_scheduled = False
        self._progress = progress

    def _poll(self) -> bool:
        raise NotImplementedError

    def _progress_future(self) -> bool:
        with self._future_lock:
            if self._future_completed:
                return True
            loop = self._future_loop
            # Preserve polling and callback affinity while the requesting loop runs.
            if loop is not None and loop.is_running():
                if self._future_poll_scheduled:
                    return False
                self._future_poll_scheduled = True
                try:
                    loop.call_soon_threadsafe(self._poll_on_loop)
                except RuntimeError:
                    self._future_poll_scheduled = False
                    if not loop.is_closed():
                        raise
                else:
                    return False
        # A stopped loop may still have a queued poll that will never execute.
        return self.is_completed()

    def _poll_on_loop(self) -> None:
        try:
            self.is_completed()
        finally:
            with self._future_lock:
                self._future_poll_scheduled = False

    def is_completed(self) -> bool:
        if not self._poll():
            return False
        with self._future_lock:
            notify = not self._future_completed
            self._future_completed = True
        # Future callbacks may synchronously call is_completed()/get_future() again.
        # Notify outside _future_lock to avoid deadlocking on that reentrancy.
        if notify:
            if self._error is None:
                self._future.set_result([])
            else:
                error = self._error
                if not isinstance(error, Exception):
                    error = RuntimeError(str(error))
                self._future.set_exception(error)
        return True

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        seconds = seconds or self._timeout
        deadline = None if seconds is None else time.monotonic() + seconds
        while not self.is_completed():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "transport wait timed out; operation remains pending"
                )
            # Yielding alone would immediately poll again and consume a CPU core.
            time.sleep(0.001)
        if self._error is not None:
            raise self._error
        return True

    def is_success(self) -> bool:
        return self.is_completed() and self._error is None

    def exception(self) -> BaseException | None:
        return self._error if self.is_completed() else None

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        if not self.is_completed():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            self._progress.add(self, loop)
        return self._future

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()


class _NIXLWork(_PollingWork):
    def __init__(
        self,
        transport: NIXLTransport,
        local: NIXLMemoryView,
        remote: NIXLRemoteBuffer,
        timeout: float,
    ) -> None:
        super().__init__(transport._future_progress, timeout)
        self._transport = transport
        self._buffers = (local, remote)
        self._descriptors: Any = None
        self._handle: Any = None
        self._state = "PROC"
        self._done = False

    def _poll(self) -> bool:
        transport = self._transport
        if not transport._operation_lock.acquire(blocking=False):
            return False
        try:
            if self._done:
                return True
            if self._state == "PROC":
                try:
                    self._state = transport._agent.check_xfer_state(self._handle)
                except BaseException as error:
                    if self._error is None:
                        self._error = error
                    return False
            if self._state == "PROC":
                return False
            if self._state != "DONE" and self._error is None:
                self._error = RuntimeError(f"NIXL transfer failed: {self._state}")
            try:
                transport._agent.release_xfer_handle(self._handle)
                transport._transfers.pop(id(self))
            except BaseException as error:
                # Keep unreleased handles for close; disallow new requests so
                # an old Work's identity cannot be reused as a new handle key.
                # This is a one-way shutdown latch: close may be retried, but
                # the transport must never accept new transfers after this error.
                transport._closing = True
                if self._error is None:
                    self._error = error
            # Each Work represents one request; completed Works are never reset.
            self._done = True
            transport._pending.pop(id(self))
            # Completion ends this Work, not the lifetime of exposed registrations.
            # unregister_memory()/close() remove the transport's retention root.
            return True
        finally:
            transport._operation_lock.release()
